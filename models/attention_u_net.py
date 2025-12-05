import torch
from torch import nn
from torch.nn import functional as F


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
            nn.Conv2d(
                in_channels,
                mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                mid_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sequential(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.dual_conv = DualConv(in_channels, out_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.dual_conv(x)
        down = self.maxpool(skip)

        return skip, down


class UpConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


class AttentionGate(nn.Module):
    def __init__(self, g_channels: int, s_channels: int, out_channels: int):
        super(AttentionGate, self).__init__()
        self.Wg = nn.Sequential(
            nn.Conv2d(g_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
        )

        self.Ws = nn.Sequential(
            nn.Conv2d(s_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
        )

        self.psi = nn.Sequential(
            nn.Conv2d(out_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, g, s):
        # g: gating signal (from decoder)
        # s: skip connection (from encoder)

        # Decoder features
        g1 = self.Wg(g)

        # Skip connection features
        s1 = self.Ws(s)

        # Merge signals
        out = F.relu(g1 + s1)

        # Attention map (0 to 1)
        psi = self.psi(out)

        # Filtered skip
        return s * psi


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super(DecoderBlock, self).__init__()
        self.attention_gate = AttentionGate(
            g_channels=in_channels,
            s_channels=skip_channels,
            out_channels=out_channels,
        )

        self.conv = DualConv(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        # x: from the previous layer - decoder
        # skip: from the skip connection - encoder

        # Upsample decoder features
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)

        # Apply attention gate
        skip = self.attention_gate(g=x, s=skip)

        # Concatenate along channel dimension
        x = torch.cat([x, skip], dim=1)  # Merge

        # Apply dual convolution
        return self.conv(x)


class AttentionUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        filters=None,
    ):
        super(AttentionUNet, self).__init__()
        if filters is None:
            filters = [32, 64, 128, 256, 512]

        # Encoder
        self.enc1 = EncoderBlock(in_channels, filters[0])
        self.enc2 = EncoderBlock(filters[0], filters[1])
        self.enc3 = EncoderBlock(filters[1], filters[2])
        self.enc4 = EncoderBlock(filters[2], filters[3])

        # Bottleneck
        self.bottleneck = DualConv(filters[3], filters[4])

        # Decoder
        self.dec4 = DecoderBlock(filters[4], filters[3], filters[3])
        self.dec3 = DecoderBlock(filters[3], filters[2], filters[2])
        self.dec2 = DecoderBlock(filters[2], filters[1], filters[1])
        self.dec1 = DecoderBlock(filters[1], filters[0], filters[0])

        # Output layer
        self.out_conv = nn.Conv2d(filters[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor):
        # Encoder
        skip1, down1 = self.enc1(x)
        skip2, down2 = self.enc2(down1)
        skip3, down3 = self.enc3(down2)
        skip4, down4 = self.enc4(down3)

        # Bottleneck
        bottleneck = self.bottleneck(down4)

        # Decoder
        up4 = self.dec4(bottleneck, skip4)
        up3 = self.dec3(up4, skip3)
        up2 = self.dec2(up3, skip2)
        up1 = self.dec1(up2, skip1)
        out = self.out_conv(up1)

        return out
