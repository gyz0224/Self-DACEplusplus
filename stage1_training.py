"""Training entry shared by the Small and Tiny illumination networks."""

import argparse
import importlib
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from training_utils import (
    DEFAULT_TRAIN_DIR, Progress, as_bool, fuse_network, get_device, light_config,
    load_checkpoint, load_light_state, resume_training, rng_state,
    save_checkpoint, seed_everything, seed_worker,
)
from validation import EpochValidation, add_validation_arguments


def train(config, stage_dir, variant):
    stage_dir = Path(stage_dir)
    package = stage_dir.name
    model = importlib.import_module(f"{package}.model")
    dataloader = importlib.import_module(f"{package}.dataloader")
    losses = importlib.import_module(f"{package}.Myloss")
    device = get_device(config.device)
    seed_everything(config.seed)

    network = model.light_net("train", config.layers, config.layers_high).to(device)
    fused = model.light_net("test", config.layers, config.layers_high).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    dataset = dataloader.lowlight_loader(config.lowlight_images_path)
    loader = DataLoader(
        dataset, batch_size=config.train_batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=device.type == "cuda", worker_init_fn=seed_worker,
    )
    start_epoch, global_step = 0, 0
    checkpoint = None
    if config.resume or config.load_pretrain:
        checkpoint = load_checkpoint(config.resume or config.pretrain_dir)
        load_light_state(network, checkpoint, variant, training=True)
        if config.resume:
            start_epoch, global_step = resume_training(checkpoint, optimizer)
        print(f"Loaded: {config.resume or config.pretrain_dir}", flush=True)

    resume = checkpoint if config.resume else None
    validator = EpochValidation(config, device, stage_dir, resume) if config.validation else None
    L_exp, L_smo = losses.L_exp(4, 0.8), losses.L_SMO()
    L_local, L_color = losses.L_localcolor(), losses.L_color()
    # Preserve the effective loss weights in each released version.
    smooth_weight, local_weight, color_weight = (
        (5000, 20000, 50) if variant == "small" else (100000, 1000, 10)
    )
    progress = Progress((config.num_epochs - start_epoch) * len(loader))
    print(
        f"{variant} | device={device} | LAEC={network.layers_low}, HASC={network.layers_high} | "
        f"lr={optimizer.param_groups[0]['lr']} | snapshots={config.snapshots_folder}", flush=True,
    )
    network.train()
    for epoch in range(start_epoch, config.num_epochs):
        for iteration, sample in enumerate(loader):
            image = sample["data_lowlight"].to(device, non_blocking=True)
            enhanced, amplitudes, betas = network(image)
            loss = (
                smooth_weight * (L_smo(amplitudes) + L_smo(betas))
                + 5 * L_exp(enhanced, image).mean()
                + local_weight * L_local(enhanced, image).mean()
                + color_weight * L_color(enhanced).mean()
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), config.grad_clip_norm)
            optimizer.step()
            global_step += 1
            progress.update()
            if iteration + 1 == len(loader) or (
                config.display_iter > 0 and (iteration + 1) % config.display_iter == 0
            ):
                progress.log(epoch, config.num_epochs, iteration, len(loader), loss.item())

        fuse_network(network, fused)
        if validator is not None:
            # Release the last training batch's curve maps before full-resolution evaluation.
            del image, enhanced, amplitudes, betas, loss
            validator.evaluate(fused, epoch, global_step)
        state = {
            "stage": "illumination", "epoch": epoch, "global_step": global_step,
            "net_intact": network.state_dict(), "net_decay": fused.state_dict(),
            "optimizer": optimizer.state_dict(), "light_config": light_config(network, variant),
            "rng": rng_state(), "config": vars(config),
            "selection_state": validator.state_dict() if validator is not None
            else (resume.get("selection_state") if resume else None),
        }
        if validator is not None:
            validator.save_best(state)
        save_checkpoint(state, config.snapshots_folder, epoch, config.checkpoint_interval, config.num_epochs)
        if validator is not None:
            validator.report(epoch, config.num_epochs, progress.remaining())


def main(stage_dir, variant):
    stage_dir = Path(stage_dir)
    parser = argparse.ArgumentParser(description=f"Train Self-DACE++ {variant} Stage-I")
    parser.add_argument("--lowlight_images_path", default=str(DEFAULT_TRAIN_DIR))
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip_norm", type=float, default=0.1)
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--display_iter", type=int, default=0, help="0 logs only at each epoch end")
    parser.add_argument("--checkpoint_interval", type=int, default=50, help="Keep latest plus every N epochs")
    parser.add_argument("--snapshots_folder", default=str(stage_dir / "snapshots_light"))
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--layers_high", type=int, default=1 if variant == "small" else 3)
    parser.add_argument("--seed", type=int, default=1143)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", metavar="CHECKPOINT", help="Resume weights, optimizer, epoch and RNG")
    parser.add_argument("--load_pretrain", type=as_bool, nargs="?", const=True, default=False,
                        help="Initialize unfolded weights only; use --resume to continue training")
    parser.add_argument("--pretrain_dir", default=str(stage_dir / "snapshots_light" / "latest.pth"))
    add_validation_arguments(parser)
    config = parser.parse_args()
    if config.resume and config.load_pretrain:
        parser.error("Choose either --resume or --load_pretrain")
    if config.num_epochs < 1 or config.train_batch_size < 1 or config.num_workers < 0:
        parser.error("Epochs/batch size must be positive and num_workers must be non-negative")
    train(config, stage_dir, variant)
