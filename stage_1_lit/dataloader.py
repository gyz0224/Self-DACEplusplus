"""Load unpaired training images; preserve this variant's original preprocessing."""
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def populate_train_list(lowlight_images_path):
    folder = Path(lowlight_images_path).expanduser()
    if not folder.is_dir():
        raise FileNotFoundError(f"Training directory not found: {folder}. Set --lowlight_images_path.")
    images = sorted(
        path for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not images:
        raise FileNotFoundError(f"No training images found in {folder}")
    random.shuffle(images)
    return images


class lowlight_loader(Dataset):
    def __init__(self, lowlight_images_path):
        self.data_list = populate_train_list(lowlight_images_path)
        self.train_list = self.data_list
        self.size = 512
        print(f"Total training examples: {len(self.data_list)}", flush=True)
        self.transform = transforms.Compose([
            transforms.RandomCrop((512, 512)),
            transforms.ToTensor(),
        ])
        self.aug = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
        ])

    def __getitem__(self, index):
        path = self.data_list[index]
        with Image.open(path) as source:
            image = source.convert("RGB")
        if min(image.size) < self.size:
            raise ValueError(f"{path}: image size {image.size} is smaller than the {self.size}x{self.size} crop")
        image = self.transform(image)
        image = self.aug(image)
        if np.random.uniform(0, 1) > 0.5:
            image = image ** np.random.uniform(1, 2)
        return {"data_lowlight": image, "path": str(path)}

    def __len__(self):
        return len(self.data_list)

