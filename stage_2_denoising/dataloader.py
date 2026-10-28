"""Load unpaired training images; preserve this variant's original preprocessing."""
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset



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
        self.size = 256
        print(f"Total training examples: {len(self.data_list)}", flush=True)

    def __getitem__(self, index):
        path = self.data_list[index]
        with Image.open(path) as source:
            image = source.convert("RGB")
        image = image.resize((self.size, self.size), Image.Resampling.LANCZOS)
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1)

    def __len__(self):
        return len(self.data_list)

