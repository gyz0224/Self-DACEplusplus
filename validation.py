"""SIMI-compatible epoch evaluation and independent best-model selection.

Reference images are used ONLY here, never in the self-supervised training loss.
Metrics use original-size floating outputs: no resize, clamp or PNG quantization.
"""

import csv
from copy import deepcopy
from hashlib import sha256
from importlib.metadata import version
import io
import json
import math
from pathlib import Path
import shutil
from uuid import uuid4

import numpy as np
from PIL import Image
import torch
from pytorch_msssim import SSIM

from training_utils import (
    IMAGE_SUFFIXES, PROJECT_DIR, as_bool, atomic_save_checkpoint,
    load_checkpoint, restore_rng, rng_state,
)


DEFAULT_VALIDATION_ROOT = PROJECT_DIR.parent / "model" / "dataset"
DATASETS = {
    "LOLv1": ("LOLv1/Test/input", "LOLv1/Test/target"),
    "LOLv2real": ("LOLv2/Real_captured/Test/Low", "LOLv2/Real_captured/Test/Normal"),
    "LOLv2syn": ("LOLv2/Synthetic/Test/Low", "LOLv2/Synthetic/Test/Normal"),
    "LSRWhuawei": ("LSRW/Eval/Huawei/low", "LSRW/Eval/Huawei/high"),
    "LSRWNikon": ("LSRW/Eval/Nikon/low", "LSRW/Eval/Nikon/high"),
    "SICE": ("SICE/SIMI/part2_eval512/low", "SICE/SIMI/part2_eval512/high"),
}
METRICS = ("psnr", "ssim", "lpips")


def add_validation_arguments(parser):
    parser.add_argument("--validation", type=as_bool, nargs="?", const=True, default=True,
                        help="Evaluate all six datasets every epoch; --validation false disables it")
    parser.add_argument("--validation_root", default=str(DEFAULT_VALIDATION_ROOT),
                        help="Root containing LOLv1, LOLv2, LSRW and SICE (same layout as SIMI)")
    parser.add_argument("--lpips_net", choices=("alex", "vgg", "squeeze"), default="alex")


def _paired_images(low_dir, high_dir):
    if low_dir.resolve() == high_dir.resolve():
        raise ValueError("Validation input and reference directories must be different")
    named = []
    for folder in (low_dir, high_dir):
        if not folder.is_dir():
            raise FileNotFoundError(f"Validation directory not found: {folder}. Set --validation_root.")
        images = {path.name: path for path in folder.iterdir()
                  if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES | {".webp"}}
        if not images:
            raise ValueError(f"Validation directory is empty: {folder}")
        named.append(images)
    low, high = named
    if set(low) != set(high):
        raise ValueError(
            f"Validation requires identical filenames: {low_dir} vs {high_dir}; "
            f"missing reference={sorted(set(low) - set(high))[:10]}, "
            f"missing input={sorted(set(high) - set(low))[:10]}"
        )
    pairs, records = [], []
    for name in sorted(low):
        source, target = low[name], high[name]
        if source.resolve() == target.resolve():
            raise ValueError(f"Validation reference points to its input: {source}")
        with Image.open(source) as source_image, Image.open(target) as target_image:
            dimensions = source_image.size
            if dimensions != target_image.size:
                raise ValueError(f"Validation pair has different dimensions: {source}, {target}")
        source_stat, target_stat = source.stat(), target.stat()
        records.append({
            "name": name, "dimensions": dimensions,
            "low": [source_stat.st_size, source_stat.st_mtime_ns],
            "high": [target_stat.st_size, target_stat.st_mtime_ns],
        })
        pairs.append((source, target))
    fingerprint = sha256(json.dumps(records, sort_keys=True).encode("utf-8")).hexdigest()
    return pairs, {"num_images": len(pairs), "pairs_fingerprint": fingerprint}


def _read_image(path, device):
    with Image.open(path) as image:
        pixels = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).to(device)


def _create_lpips_model(net, device):
    # Pretrained-network construction initializes temporary weights; preserve training RNG.
    previous_rng = rng_state()
    try:
        try:
            import lpips
        except ImportError as error:
            raise RuntimeError("Epoch validation requires lpips: pip install lpips") from error
        return lpips.LPIPS(
            net=net, version="0.1", pretrained=True, pnet_rand=False,
            pnet_tune=False, eval_mode=True, verbose=False,
        ).to(device).eval().requires_grad_(False)
    finally:
        restore_rng(previous_rng)


def _write_text(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class EpochValidation:
    """One frozen LPIPS evaluator, six paired datasets and 18 independent bests."""

    def __init__(self, config, device, stage_dir, resume=None):
        self.device = torch.device(device)
        self.folder = Path(config.snapshots_folder).expanduser().resolve()
        self.history_path = Path(stage_dir) / "data" / "validation_metrics.csv"
        root = Path(config.validation_root).expanduser().resolve()
        self.pairs, self.signature = {}, {}
        # Check ALL pairs before constructing LPIPS or starting the first training epoch.
        for name, (low, high) in DATASETS.items():
            self.pairs[name], self.signature[name] = _paired_images(root / low, root / high)
        protocol = {
            "psnr_data_range": 1.0, "ssim_data_range": 1.0,
            "ssim_library": "pytorch_msssim", "ssim_version": version("pytorch-msssim"),
            "lpips_net": config.lpips_net, "lpips_version": "0.1",
            "lpips_package_version": version("lpips"), "lpips_pretrained": True,
            "lpips_input_normalization": "2*image-1",
            "resize": False, "clamp": False, "average": "per_image",
        }
        for signature in self.signature.values():
            signature["metrics"] = deepcopy(protocol)
        self.run_id = uuid4().hex
        self.best_records = {name: dict.fromkeys(METRICS) for name in DATASETS}
        self.last_validation, self.history, self.improved = {}, [], []
        selection = resume.get("selection_state") if resume else None
        if selection:
            if selection["dataset_signature"] != self.signature:
                raise ValueError(
                    "Validation datasets/metric protocol changed since the resume checkpoint. "
                    "Restore the original data and --lpips_net, or use --load_pretrain for a new run."
                )
            self.run_id = selection["run_id"]
            self.best_records = deepcopy(selection["best_records"])
            self.last_validation = deepcopy(selection["last_validation"])
            self.history = deepcopy(selection["validation_history"])
            self._restore_best_files(config.resume)
            self._write_reports()
        elif resume:
            print("Resume checkpoint has no validation history; best-model selection starts now.", flush=True)
        self.lpips_metric = _create_lpips_model(config.lpips_net, self.device)
        self.ssim_metric = SSIM(data_range=1.0, size_average=True, channel=3).to(self.device)
        counts = ", ".join(f"{name}={len(pairs)}" for name, pairs in self.pairs.items())
        print(f"Epoch validation enabled | {counts} | LPIPS={config.lpips_net} | "
              "original-size float outputs, no clamp/resize", flush=True)

    def _restore_best_files(self, resume_path):
        """Keep historical best weights on resume, including a changed snapshot folder."""
        resume_path = Path(resume_path).expanduser().resolve()
        source_root = resume_path.parent
        if source_root.name in DATASETS:
            source_root = source_root.parent
        copies = []
        for dataset, bests in self.best_records.items():
            for metric, record in bests.items():
                if record is None:
                    continue
                relative = Path(dataset) / f"best_{metric}.pth"
                destination = self.folder / relative
                candidates = dict.fromkeys((destination, source_root / relative, resume_path))
                for candidate in candidates:
                    if not candidate.is_file():
                        continue
                    payload = load_checkpoint(candidate)
                    selection = payload.get("selection_state") or {}
                    scores = selection.get("last_validation", {}).get(dataset, {})
                    if (selection.get("run_id") == self.run_id
                            and selection.get("dataset_signature") == self.signature
                            and payload.get("epoch") == record["epoch"]
                            and payload.get("global_step") == record["global_step"]
                            and scores == {key: record[key] for key in (*METRICS, "num_images")}):
                        if candidate != destination:
                            copies.append((candidate, destination))
                        break
                else:
                    raise FileNotFoundError(
                        f"Cannot recover historical {dataset}/{relative.name} at epoch {record['epoch'] + 1}. "
                        "Keep/copy the dataset best-checkpoint directories together with latest.pth. "
                        "Use --load_pretrain instead of --resume to start fresh selection history."
                    )
        for source, destination in copies:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + ".tmp")
            try:
                shutil.copyfile(source, temporary)
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)

    def evaluate(self, light_net, epoch, global_step, denoise_net=None):
        """Evaluate the fused enhancer, optionally followed by the trained denoiser."""
        networks = [light_net] + ([denoise_net] if denoise_net is not None else [])
        modes = [(module, module.training) for net in networks for module in net.modules()]
        previous_rng = rng_state()
        results = {}
        try:
            for network in networks:
                network.eval()
            self.lpips_metric.eval()
            with torch.inference_mode(), torch.autocast(self.device.type, enabled=False):
                for dataset, pairs in self.pairs.items():
                    totals = dict.fromkeys(METRICS, 0.0)
                    for source, reference in pairs:
                        image = _read_image(source, self.device)
                        target = _read_image(reference, self.device)
                        output = light_net(image)[0]
                        if denoise_net is not None:
                            output = denoise_net(output)
                        if output.shape != target.shape or not torch.isfinite(output).all():
                            raise ValueError(f"Invalid validation output shape/values: {dataset}/{source.name}")
                        mse = (output - target).square().mean()
                        measured = {
                            "psnr": float(20 * torch.log10(1.0 / torch.sqrt(mse))),
                            "ssim": float(self.ssim_metric(output, target)),
                            "lpips": float(self.lpips_metric(output * 2 - 1, target * 2 - 1).mean()),
                        }
                        if (math.isnan(measured["psnr"]) or measured["psnr"] == -math.inf
                                or not all(math.isfinite(measured[key]) for key in ("ssim", "lpips"))):
                            raise ValueError(f"Non-finite validation metric: {dataset}/{source.name}")
                        for metric in METRICS:
                            totals[metric] += measured[metric]
                    results[dataset] = {**{key: value / len(pairs) for key, value in totals.items()},
                                        "num_images": len(pairs)}
        finally:
            for module, was_training in modes:
                module.training = was_training
            restore_rng(previous_rng)

        # Commit selection only after ALL six datasets have completed successfully.
        self.improved = []
        for dataset, scores in results.items():
            for metric in METRICS:
                old = self.best_records[dataset][metric]
                better = old is None or (scores[metric] < old[metric] if metric == "lpips"
                                         else scores[metric] > old[metric])
                if better:
                    self.best_records[dataset][metric] = {"epoch": epoch, "global_step": global_step, **scores}
                    self.improved.append((dataset, metric))
        self.last_validation = results
        self.history.append({"epoch": epoch, "global_step": global_step, "datasets": deepcopy(results)})
        return results

    def state_dict(self):
        return deepcopy({
            "run_id": self.run_id, "dataset_signature": self.signature,
            "best_records": self.best_records, "last_validation": self.last_validation,
            "validation_history": self.history,
        })

    def save_best(self, checkpoint):
        # Full training state, not merely inference weights: bests also support --resume.
        for dataset, metric in self.improved:
            atomic_save_checkpoint(checkpoint, self.folder / dataset / f"best_{metric}.pth")

    def report(self, epoch, epochs, remaining):
        self._write_reports()
        for dataset, values in self.last_validation.items():
            updated = ",".join(metric for name, metric in self.improved if name == dataset) or "none"
            print(f"[validation:{dataset}] Epoch {epoch + 1}/{epochs} images={values['num_images']} "
                  f"PSNR={values['psnr']:.4f} SSIM={values['ssim']:.6f} LPIPS={values['lpips']:.6f} "
                  f"best_updated={updated} remaining={remaining}", flush=True)

    def _write_reports(self):
        stream = io.StringIO(newline="")
        writer = csv.writer(stream)
        writer.writerow(("epoch", "global_step", "dataset", "num_images", *METRICS))
        for record in self.history:
            for dataset, values in record["datasets"].items():
                writer.writerow((record["epoch"] + 1, record["global_step"], dataset,
                                 values["num_images"], *(values[key] for key in METRICS)))
        # Rebuild from checkpoint history rather than append: no duplicate rows on resume.
        _write_text(self.history_path, stream.getvalue())
        for dataset, records in self.best_records.items():
            lines = [f"Dataset: {dataset}", "PSNR/SSIM: higher is better; LPIPS: lower is better",
                     "Original-size float outputs; no clamp, resize or quantization.", ""]
            for metric, record in records.items():
                if record is not None:
                    lines.extend([f"Model: best_{metric}.pth", f"Epoch: {record['epoch'] + 1}",
                                  f"Global step: {record['global_step']}",
                                  f"PSNR: {record['psnr']:.6f} dB", f"SSIM: {record['ssim']:.6f}",
                                  f"LPIPS: {record['lpips']:.6f}", f"Images: {record['num_images']}", ""])
            _write_text(self.folder / dataset / "best_models_metrics.txt", "\n".join(lines) + "\n")
