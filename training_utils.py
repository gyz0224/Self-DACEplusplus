"""Shared curve fusion, checkpoint and runtime helpers for Self-DACE++."""

import argparse
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_TRAIN_DIR = PROJECT_DIR.parent / "model" / "dataset" / "SICE" / "SIMI" / "train"
DEFAULT_TEST_DIR = PROJECT_DIR.parent / "model" / "dataset" / "LOLv1" / "Test" / "input"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def as_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {"true", "1", "yes"}:
        return True
    if value.lower() in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError("Expected true or false")


def get_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu")
    return device


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def rng_state():
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    if not state:
        return
    random.setstate(state["python"])
    name, keys, pos, has_gauss, cached = state["numpy"]
    np.random.set_state((name, np.array(keys, dtype=np.uint32), pos, has_gauss, cached))
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state["cuda"]:
        if len(state["cuda"]) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def apply_curve(image, amplitude, beta, high=False):
    """Author's LAEC/HASC equations; HASC stores the complement of its threshold."""
    if high:
        gate = 1 / (1 + torch.exp(15 * (-image + (1 - beta) + 0.1)))
        return image + amplitude * gate * (1 - image) * (image - (1 - beta)) * (1 / beta)
    gate = 1 / (1 + torch.exp(-15 * (-image + beta - 0.1)))
    return image + amplitude * gate * image * (beta - image) * (1 / beta)


def apply_curves(image, amplitudes, betas, layers_low):
    """Replay the SAME predicted curves on the noise-free self-reference image."""
    if amplitudes.shape != betas.shape or amplitudes.shape[1] % 3:
        raise ValueError("Curve maps must have matching shapes with 3 channels per step")
    for step in range(amplitudes.shape[1] // 3):
        channels = slice(3 * step, 3 * step + 3)
        image = apply_curve(image, amplitudes[:, channels], betas[:, channels], step >= layers_low)
    return image


@torch.no_grad()
def fuse_network(training_net, inference_net):
    """Average corresponding DMs separately for LAEC and HASC, as in save_network."""
    for name, destination in inference_net.named_children():
        if not isinstance(destination, nn.Conv2d):
            continue
        source = getattr(training_net, name)
        if isinstance(source, nn.ModuleList):
            count = training_net.layers_high if name.endswith("_1") else training_net.layers_low
            source = list(source)[:count]
            destination.weight.copy_(torch.stack([layer.weight for layer in source]).mean(0))
            destination.bias.copy_(torch.stack([layer.bias for layer in source]).mean(0))
        else:
            destination.load_state_dict(source.state_dict())
    return inference_net


def light_config(network, variant):
    return {
        "variant": variant,
        "layers": network.layers_low + network.layers_high,
        "layers_high": network.layers_high,
    }


def load_checkpoint(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=True)


def state_dict_from(checkpoint, key):
    if key in checkpoint:
        return checkpoint[key]
    if checkpoint and all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        return checkpoint
    raise ValueError(f"Checkpoint does not contain '{key}'; available keys: {list(checkpoint)}")


def unfolded_standard_layout(checkpoint):
    """Released fused weights lack metadata, but intact ordinary DMs expose their counts."""
    state = checkpoint.get("net_intact", checkpoint)
    def count_modules(prefix):
        return sum(
            name.startswith(prefix + ".") and name.endswith(".weight")
            and name.split(".")[1].isdigit() for name in state
        )
    low = count_modules("backbone")
    high = count_modules("backbone_1")
    return (low, high) if low and high else None


def load_light_state(network, checkpoint, variant, training=False):
    expected = light_config(network, variant)
    saved = checkpoint.get("light_config")
    if saved and saved != expected:
        raise ValueError(f"LightNet configuration mismatch: checkpoint={saved}, requested={expected}")
    layout = unfolded_standard_layout(checkpoint) if variant == "small" else None
    if training and layout and layout != (network.layers_low, network.layers_high):
        low, high = layout
        raise ValueError(
            f"The unfolded checkpoint contains {low} LAEC + {high} HASC modules, but the requested "
            f"network uses {network.layers_low}+{network.layers_high}. To initialize these weights, "
            f"set --layers {low + high} --layers_high {high}; train without --load_pretrain for the defaults."
        )
    key = "net_intact" if training else "net_decay"
    state = state_dict_from(checkpoint, key)
    if any(name.startswith("e_conv") for name in state):
        raise ValueError("This is an original seven-step Self-DACE checkpoint. Use a Self-DACE++ net_decay/net_intact checkpoint.")
    if training and variant == "tiny":
        # Older Tiny checkpoints included six HASC modules never used in forward().
        state = {
            name: value for name, value in state.items()
            if not (name.startswith("conv_curve_1.") and int(name.split(".")[1]) >= network.layers_high)
        }
    network.load_state_dict(state, strict=True)


def atomic_save_checkpoint(state, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        torch.save(state, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def save_checkpoint(state, folder, epoch, interval, final_epoch):
    folder = Path(folder).expanduser()
    atomic_save_checkpoint(state, folder / "latest.pth")
    if (interval > 0 and (epoch + 1) % interval == 0) or epoch + 1 == final_epoch:
        atomic_save_checkpoint(state, folder / f"Epoch{epoch + 1}.pth")


def resume_training(checkpoint, optimizer):
    required = {"epoch", "global_step", "optimizer"}
    if not required.issubset(checkpoint):
        raise ValueError("This checkpoint lacks resume metadata. Use --load_pretrain --pretrain_dir PATH for weights-only initialization.")
    optimizer.load_state_dict(checkpoint["optimizer"])
    restore_rng(checkpoint.get("rng"))
    return checkpoint["epoch"] + 1, checkpoint["global_step"]


class Progress:
    def __init__(self, total_steps):
        self.total_steps = total_steps
        self.steps = 0
        self.start = time.perf_counter()

    def update(self):
        self.steps += 1

    def remaining(self):
        # Called after epoch validation too, so its wall time contributes to ETA.
        elapsed = time.perf_counter() - self.start
        remaining = elapsed / max(self.steps, 1) * max(self.total_steps - self.steps, 0)
        return (f"{int(remaining // 3600):02d}:"
                f"{int(remaining % 3600 // 60):02d}:{int(remaining % 60):02d}")

    def log(self, epoch, epochs, iteration, iterations, loss):
        print(
            f"Epoch {epoch + 1}/{epochs} | batch {iteration + 1}/{iterations} | "
            f"loss {loss:.6f} | remaining {self.remaining()}",
            flush=True,
        )
