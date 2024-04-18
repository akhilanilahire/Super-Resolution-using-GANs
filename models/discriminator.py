import torch
import torch.nn as nn
from models.utils import Convolution

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        features = [64, 64, 128, 128, 256, 256, 512, 512]
        blocks = []
        for i, feature in enumerate(features):
            blocks.append(
                Convolution(
                    in_channels = in_channels,
                    out_channels = feature,
                    kernel_size = 3,
                    stride = 1 + (i % 2),
                    padding = 1,
                    is_discriminator = True,
                    use_activation = True,
                    batch_norm = True if i != 0 else False,
                )
            )
            in_channels = feature

        self.classifier = nn.Sequential(
            *[(block) for block in blocks],
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(512*6*6, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
        )

    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.classifier(x).view(batch_size))
