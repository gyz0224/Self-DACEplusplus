"""Train DenoiseNet independently with a frozen, fused Self-DACE++ LightNet."""
import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from pytorch_msssim import ssim

STAGE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(STAGE_DIR.parent))
from stage_2_denoising import dataloader, model, Myloss
from training_utils import (
    DEFAULT_TRAIN_DIR, PROJECT_DIR, Progress, as_bool, get_device, light_config,
    load_checkpoint, load_light_state, resume_training, rng_state, save_checkpoint,
    seed_everything, seed_worker, state_dict_from,
)
from validation import EpochValidation, add_validation_arguments


def simulate_noise(image):
    # Preserve the released noise distribution, including the shared batch pattern.
    sigma = float(0.009 * np.random.rand() + 0.001)
    height, width = image.shape[-2:]
    noise = np.concatenate([
        np.random.normal(0.0, sigma, (1, height, width)),
        np.random.normal(0.0, sigma / 2, (1, height, width)),
        np.random.normal(0.0, sigma, (1, height, width)),
    ], axis=0).astype(np.float32)
    noise = torch.from_numpy(noise).to(device=image.device, dtype=image.dtype).unsqueeze(0)
    return (image + (1 - image) * noise).clamp(1e-10, 1)


def train(config):
    device = get_device(config.device)
    seed_everything(config.seed)
    resume = load_checkpoint(config.resume) if config.resume else None
    if resume and (resume.get("stage") != "denoising" or "light_net" not in resume):
        raise ValueError("Resume requires a corrected Stage-II checkpoint containing its frozen LightNet.")

    light_net = model.light_net("test", config.layers, config.layers_high).to(device)
    if resume:
        load_light_state(light_net, {
            "net_decay": resume["light_net"], "light_config": resume["light_config"],
        }, "small")
        print("Restored the paired frozen LightNet from the Stage-II checkpoint.", flush=True)
    else:
        load_light_state(light_net, load_checkpoint(config.stage1_dir), "small")
        print(f"Frozen Stage-I checkpoint: {config.stage1_dir}", flush=True)
    light_net.eval()
    light_net.requires_grad_(False)

    denoise_net = model.denoise_net().to(device)
    optimizer = torch.optim.Adam(
        denoise_net.parameters(), lr=config.lr, weight_decay=config.weight_decay,
    )
    dataset = dataloader.lowlight_loader(config.lowlight_images_path)
    loader = DataLoader(
        dataset, batch_size=config.train_batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=device.type == "cuda", worker_init_fn=seed_worker,
    )
    start_epoch, global_step = 0, 0
    if resume:
        denoise_net.load_state_dict(resume["net"], strict=True)
        start_epoch, global_step = resume_training(resume, optimizer)
    elif config.load_pretrain:
        denoise_net.load_state_dict(state_dict_from(load_checkpoint(config.pretrain_dir), "net"), strict=True)
        print(f"DenoiseNet weights-only initialization: {config.pretrain_dir}", flush=True)

    validator = EpochValidation(config, device, STAGE_DIR, resume) if config.validation else None
    l1_loss, gradient_loss = torch.nn.L1Loss(), Myloss.L_gradient()
    progress = Progress((config.num_epochs - start_epoch) * len(loader))
    denoise_net.train()
    print(
        f"Standard | device={device} | frozen LAEC={light_net.layers_low}, HASC={light_net.layers_high} | "
        f"denoising epochs={config.num_epochs} | lr={optimizer.param_groups[0]['lr']}", flush=True,
    )
    for epoch in range(start_epoch, config.num_epochs):
        for iteration, image in enumerate(loader):
            image = image.to(device, non_blocking=True)
            with torch.no_grad():
                noisy_input = simulate_noise(image)
                enhanced_noisy, amplitudes, betas = light_net(noisy_input)
                # Use the noisy image's maps on the ORIGINAL input, including HASC.
                reference = light_net.apply_curves(image, amplitudes, betas)
            output = denoise_net(enhanced_noisy)
            loss = (
                10 * l1_loss(output, reference)
                + 10 * (1 - ssim(output, reference, data_range=1.0, size_average=True))
                + 40 * gradient_loss(output, reference)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(denoise_net.parameters(), config.grad_clip_norm)
            optimizer.step()
            global_step += 1
            progress.update()
            if iteration + 1 == len(loader) or (
                config.display_iter > 0 and (iteration + 1) % config.display_iter == 0
            ):
                progress.log(epoch, config.num_epochs, iteration, len(loader), loss.item())

        if validator is not None:
            del image, noisy_input, enhanced_noisy, amplitudes, betas, reference, output, loss
            validator.evaluate(light_net, epoch, global_step, denoise_net=denoise_net)
        state = {
            "stage": "denoising", "epoch": epoch, "global_step": global_step,
            "net": denoise_net.state_dict(), "optimizer": optimizer.state_dict(),
            "light_net": light_net.state_dict(), "light_config": light_config(light_net, "small"),
            "stage1_checkpoint": str(Path(config.stage1_dir).resolve()),
            "rng": rng_state(), "config": vars(config),
            "selection_state": validator.state_dict() if validator is not None
            else (resume.get("selection_state") if resume else None),
        }
        if validator is not None:
            validator.save_best(state)
        save_checkpoint(state, config.snapshots_folder, epoch, config.checkpoint_interval, config.num_epochs)
        if validator is not None:
            validator.report(epoch, config.num_epochs, progress.remaining())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lowlight_images_path", default=str(DEFAULT_TRAIN_DIR))
    parser.add_argument("--stage1_dir", default=str(PROJECT_DIR / "stage_1_lit" / "snapshots_light" / "latest.pth"))
    parser.add_argument("--lr", type=float, default=1e-5, help="Released denoising optimizer's effective learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip_norm", type=float, default=0.1)
    parser.add_argument("--num_epochs", type=int, default=200, help="Author-confirmed denoising training length")
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--display_iter", type=int, default=0, help="0 logs only at each epoch end")
    parser.add_argument("--checkpoint_interval", type=int, default=50)
    parser.add_argument("--snapshots_folder", default=str(STAGE_DIR / "snapshots_denoise"))
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--layers_high", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1143)
    parser.add_argument("--resume", metavar="CHECKPOINT")
    parser.add_argument("--load_pretrain", type=as_bool, nargs="?", const=True, default=False)
    parser.add_argument("--pretrain_dir", default=str(STAGE_DIR / "snapshots_denoise" / "pre_train_2.pth"))
    add_validation_arguments(parser)
    config = parser.parse_args()
    if config.resume and config.load_pretrain:
        parser.error("Choose either --resume or --load_pretrain")
    if config.num_epochs < 1 or config.train_batch_size < 1 or config.num_workers < 0:
        parser.error("Epochs/batch size must be positive and num_workers must be non-negative")
    train(config)


if __name__ == "__main__":
    main()








