import torch
import torch.nn as nn
import torch.nn.functional as F


class DualConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int | None = None,
    ):
        super(DualConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.sequential(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(Down, self).__init__()
        self.sequential = nn.Sequential(
            nn.MaxPool2d(2),
            DualConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.sequential(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super(Up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DualConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels,
                in_channels // 2,
                kernel_size=2,
                stride=2,
            )
            self.conv = DualConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        x1: from the previous layer - decoder
        x2: from the skip connection - encoder
        """
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]  # height
        diffX = x2.size()[3] - x1.size()[3]  # width

        # pad function: (L, R, T, B)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        # Concatenate along the channels axis
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNetBaseline(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super(UNetBaseline, self).__init__()

        # Encoder
        # self.inc = DualConv(in_channels, 64)
        # self.down1 = Down(64, 128)
        # self.down2 = Down(128, 256)
        # self.down3 = Down(256, 512)

        # # Bottleneck
        # self.down4 = Down(512, 1024)

        # # Decoder
        # self.up1 = Up(1024, 512, bilinear=False)
        # self.up2 = Up(512, 256, bilinear=False)
        # self.up3 = Up(256, 128, bilinear=False)
        # self.up4 = Up(128, 64, bilinear=False)

        # Output layer
        # self.outc = nn.Conv2d(64, num_classes, kernel_size=1)

        # Smaller UNet for faster training
        self.inc = DualConv(in_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)  # Bottleneck
        self.up1 = Up(512, 256, bilinear=False)
        self.up2 = Up(256, 128, bilinear=False)
        self.up3 = Up(128, 64, bilinear=False)
        self.up4 = Up(64, 32, bilinear=False)
        self.outc = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder with skip connections
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)  # Bottleneck

        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)

        return x
