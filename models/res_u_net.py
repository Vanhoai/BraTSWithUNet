from typing import List, Tuple

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Residual Block as used in ResNet architectures.
    Consists of two convolutional layers with a skip connection.
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int): Stride for the first convolutional layer.
    Returns:
        torch.Tensor: Output tensor after applying the residual block.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(identity)
        out = self.relu(out)

        return out


class ResidualConvBlock(nn.Module):
    """
    ResidualConvBlock used in ResUNet architecture.
    Consists of two convolutional layers with batch normalization and ReLU activation,
    along with a skip connection.
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    Returns:
        torch.Tensor: Output tensor after applying the residual convolutional block.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(ResidualConvBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_block(x) + self.conv_skip(x)
        return out


class EncoderBlock(nn.Module):
    """
    Encoder Block used in ResUNet architecture.
    Consists of a ResidualConvBlock followed by a MaxPooling layer.
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
        - down (torch.Tensor): Output tensor after downsampling.
        - skip (torch.Tensor): Output tensor for skip connection.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(EncoderBlock, self).__init__()
        self.conv = ResidualConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        skip = self.conv(x)
        down = self.pool(skip)

        return down, skip


class DecoderBlock(nn.Module):
    """
    Decoder Block used in ResUNet architecture.
    Consists of a transposed convolution for upsampling followed by a ResidualConvBlock.
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        skip_channels (int): Number of channels from the skip connection.
    Returns:
        torch.Tensor: Output tensor after applying the decoder block.
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super(DecoderBlock, self).__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )

        self.conv = ResidualConvBlock(
            in_channels=out_channels + skip_channels,
            out_channels=out_channels,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)

        # dim = 1 => channel dimension
        x = torch.cat((x, skip), dim=1)
        x = self.conv(x)
        return x


class Bridge(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            ResidualConvBlock(in_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        )

    def forward(self, x):
        return self.conv(x)


class ResUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        filters=None,
    ):
        super(ResUNet, self).__init__()

        # Initial convolution
        if filters is None:
            filters = [32, 64, 128, 256, 512]

        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels, filters[0], 3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(inplace=True),
        )

        # Encoder
        self.encoder1 = EncoderBlock(filters[0], filters[0])
        self.encoder2 = EncoderBlock(filters[0], filters[1])
        self.encoder3 = EncoderBlock(filters[1], filters[2])
        self.encoder4 = EncoderBlock(filters[2], filters[3])

        # Bridge (Bottleneck)
        self.bridge = Bridge(filters[3], filters[4])

        # RuntimeError: Sizes of tensors must match except in dimension 1. Expected size 256 but got size 64 for tensor number 1 in the list.

        # Decoder
        self.decoder4 = DecoderBlock(filters[4], filters[3], filters[3])
        self.decoder3 = DecoderBlock(filters[3], filters[2], filters[2])
        self.decoder2 = DecoderBlock(filters[2], filters[1], filters[1])
        self.decoder1 = DecoderBlock(filters[1], filters[0], filters[0])

        # Output layer
        self.output_layer = nn.Conv2d(filters[0], num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial conv
        x = self.input_layer(x)

        # Encoder
        x, skip1 = self.encoder1(x)
        x, skip2 = self.encoder2(x)
        x, skip3 = self.encoder3(x)
        x, skip4 = self.encoder4(x)

        # Bridge
        x = self.bridge(x)
        print(f"Bridge shape: {x.shape}")

        # Decoder
        x = self.decoder4(x, skip4)
        x = self.decoder3(x, skip3)
        x = self.decoder2(x, skip2)
        x = self.decoder1(x, skip1)

        # Output
        x = self.output_layer(x)

        return x
