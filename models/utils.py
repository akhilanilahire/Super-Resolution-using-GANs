import torch
import torch.nn as nn

class Convolution(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        batch_norm = True,
        is_discriminator = False,
        use_activation = True,
        ):
        super().__init__()
        self.use_activation = use_activation
        self.is_discriminator = is_discriminator
        self.out_channels = out_channels

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels = in_channels,
                out_channels = out_channels,
                bias = not batch_norm,
                kernel_size = kernel_size,
                stride = stride,
                padding = padding,
                ),
            nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity(),
        )
        self.activation = (
            nn.LeakyReLU(0.2, inplace=True)
            if is_discriminator
            else nn.PReLU(num_parameters=out_channels)
        )
        self.weight = nn.Parameter(torch.Tensor(out_channels))

    def forward(self, x):
        return self.activation(self.conv(x)) if self.use_activation else self.conv(x)

class Upsample(nn.Module):
    def __init__(self, in_channels, scale_factor = 4):
        super().__init__()
        self.in_channels = in_channels
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels = in_channels,
                out_channels = in_channels * scale_factor ** 2,
                kernel_size = 3,
                stride = 1,
                padding = 1,
            ),
            nn.PixelShuffle(scale_factor),
        )
        self.activation = nn.PReLU(num_parameters=in_channels)

    def forward(self, x):
        x= self.conv(x)
        return self.activation(x)

class Residual(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.residual = nn.Sequential(
            Convolution(
                in_channels = in_channels,
                out_channels = in_channels,
                kernel_size = 3,
                stride = 1,
                padding = 1,
            ),
            Convolution(
                in_channels = in_channels,
                out_channels = in_channels,
                kernel_size = 3,
                stride = 1,
                padding = 1,
                use_activation = False,
            ),
        )

    def forward(self, x):
        out = self.residual(x)
        return out + x
