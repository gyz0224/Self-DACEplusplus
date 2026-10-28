"""Inference shared by Stage-I variants and the corrected Standard pipeline."""

import argparse
import importlib
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torchvision.utils import save_image

from training_utils import (
    DEFAULT_TEST_DIR, IMAGE_SUFFIXES, PROJECT_DIR, get_device,
    load_checkpoint, load_light_state, state_dict_from, unfolded_standard_layout,
)


def main(stage_dir, variant, denoising=False):
    stage_dir = Path(stage_dir)
    parser = argparse.ArgumentParser(description="Self-DACE++ enhancement with fused LightNet")
    parser.add_argument("--input_dir", default=str(DEFAULT_TEST_DIR))
    parser.add_argument("--output_dir", default=str(stage_dir / "data" / "result" / "LOLv1"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--layers", type=int, help="Override iteration count; must match checkpoint metadata")
    parser.add_argument("--layers_high", type=int)
    if denoising:
        parser.add_argument("--checkpoint", "--denoise_checkpoint", dest="checkpoint",
                            default=str(stage_dir / "snapshots_denoise" / "latest.pth"))
        parser.add_argument("--stage1_dir", help="Optional override; normally use the LightNet embedded in Stage-II")
    else:
        latest = stage_dir / "snapshots_light" / "latest.pth"
        released = stage_dir / "snapshots_light" / ("Epoch70.pth" if variant == "small" else "Epoch100.pth")
        parser.add_argument("--checkpoint", default=str(latest if latest.is_file() else released))
    config = parser.parse_args()
    device = get_device(config.device)
    input_dir = Path(config.input_dir).expanduser().resolve()
    output_dir = Path(config.output_dir).expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if input_dir == output_dir or input_dir in output_dir.parents:
        raise ValueError("Output must be outside the input directory to protect the original dataset")
    images = sorted(path for path in input_dir.rglob("*")
                    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise FileNotFoundError(f"No images found in {input_dir}")
    destinations = [path.relative_to(input_dir).with_suffix(".png") for path in images]
    if len(set(destinations)) != len(destinations):
        raise ValueError("Input filenames collide after PNG conversion; use separate input directories")

    checkpoint = load_checkpoint(config.checkpoint)
    denoise_net = None
    if denoising:
        model = importlib.import_module("stage_2_denoising.model")
        if config.stage1_dir:
            light_checkpoint = load_checkpoint(config.stage1_dir)
        elif "light_net" in checkpoint:
            light_checkpoint = {"net_decay": checkpoint["light_net"],
                                "light_config": checkpoint["light_config"]}
        else:
            light_checkpoint = load_checkpoint(PROJECT_DIR / "stage_1_lit" / "snapshots_light" / "latest.pth")
        denoise_net = model.denoise_net().to(device)
        denoise_net.load_state_dict(state_dict_from(checkpoint, "net"), strict=True)
        denoise_net.eval()
    else:
        model = importlib.import_module(f"{stage_dir.name}.model")
        light_checkpoint = checkpoint

    saved_config = light_checkpoint.get("light_config") or {}
    layers = config.layers if config.layers is not None else saved_config.get("layers", 12)
    high = config.layers_high if config.layers_high is not None else saved_config.get(
        "layers_high", 1 if variant == "small" else 3,
    )
    light_net = model.light_net("test", layers, high).to(device)
    load_light_state(light_net, light_checkpoint, variant)
    light_net.eval()
    layout = unfolded_standard_layout(light_checkpoint) if variant == "small" else None
    if not saved_config and layout and layout != (light_net.layers_low, light_net.layers_high):
        print(
            f"Released checkpoint has {layout[0]}+{layout[1]} unfolded modules and no inference metadata; "
            f"using the requested {light_net.layers_low}+{light_net.layers_high} inference configuration. "
            f"Use --layers {sum(layout)} --layers_high {layout[1]} to match the unfolded counts."
        )
    print(f"Checkpoint: {config.checkpoint} | LAEC={light_net.layers_low}, HASC={light_net.layers_high} | device={device}")
    with torch.inference_mode():
        for source_path, relative_path in zip(images, destinations):
            with Image.open(source_path) as source:
                array = np.asarray(source.convert("RGB"), dtype=np.float32) / 255.0
            image = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)
            enhanced, _, _ = light_net(image)
            if denoise_net is not None:
                enhanced = denoise_net(enhanced)
            destination = output_dir / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            save_image(enhanced.clamp(0, 1), str(destination))
    print(f"Saved {len(images)} PNG images to {output_dir}", flush=True)
