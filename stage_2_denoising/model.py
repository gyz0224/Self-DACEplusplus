"""Standard Self-DACE++ = fused Stage-I LightNet followed by DenoiseNet."""
from pathlib import Path
import sys

import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage_1_lit.model import light_net


class denoise_net(nn.Module):
    def __init__(self):
        super().__init__()
        # Keep the released 19-layer, 64-channel denoising architecture and keys.
        num_layers, num_features = 19, 64
        layers = [nn.Sequential(
            nn.Conv2d(3, num_features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )]
        for _ in range(num_layers - 2):
            layers.append(nn.Sequential(
                nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            ))
        layers.append(nn.Conv2d(num_features, 3, kernel_size=3, padding=1))
        self.layers = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)

    def forward(self, image):
        # Self-DACE++ uses an unconstrained residual, unlike the original tanh.
        return image + self.layers(image)
