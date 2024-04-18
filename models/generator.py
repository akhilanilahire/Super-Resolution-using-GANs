import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils import Convolution, Upsample, Residual


class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, num_blocks=10, scale_factor=4):
        super().__init__()

        self.initial = Convolution(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = 9,
            stride = 1,
            padding = 4,
            batch_norm=False,

        )

        self.residuals = nn.Sequential(
            *[Residual(out_channels) for _ in range(num_blocks)]
        )

        self.post_residual = Convolution(
            in_channels = out_channels,
            out_channels = out_channels,
            kernel_size = 3,
            stride = 1,
            padding = 1,
            use_activation = False,
        )

        self.upsample = nn.Sequential(
            *[Upsample(out_channels, 2) for _ in range(scale_factor//2)],
            nn.Conv2d(
                in_channels = out_channels,
                out_channels = in_channels,
                kernel_size = 9,
                stride = 1,
                padding = 4,
            ),
        )


    def forward(self, x):
        initial = self.initial(x)
        residuals = self.residuals(initial)
        post_residual = self.post_residual(residuals) + initial
        upsample = self.upsample(post_residual)
        return F.tanh(upsample)
