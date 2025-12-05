import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat((x, skip), dim=1)
        x = self.conv(x)
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embedding_dim):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size

        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # (B, embedding_dim, H/P, W/P) => [8, 768, 16, 16]
        x = x.flatten(2)  # (B, embedding_dim, n_patches) => [8, 768, 256]
        x = x.transpose(1, 2)  # (B, n_patches, embedding_dim) => [8, 256, 768]
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(embedding_dim, embedding_dim * 3)
        self.proj = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, mlp_ratio=4.0):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.attn = MultiHeadAttention(embedding_dim, num_heads)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.mlp = MLP(embedding_dim, int(embedding_dim * mlp_ratio), embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TransUNet(nn.Module):
    def __init__(
        self,
        img_size=128,
        in_channels=3,
        num_classes=1,
        patch_size=16,
        embedding_dim=768,
        depth=12,
        num_heads=12,
    ):
        super(TransUNet, self).__init__()
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim

        # CNN Encoder
        self.conv1 = ConvBlock(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)  # Shape: [8, 32, 64, 64]

        self.conv2 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2)  # [8, 64, 32, 32]

        self.conv3 = ConvBlock(64, 128)
        self.pool3 = nn.MaxPool2d(2)  # [8, 128, 16, 16]

        # Patch Embedding
        bottleneck_size = img_size // 8  # 16
        self.patch_embedding = PatchEmbedding(
            img_size=bottleneck_size,
            patch_size=1,
            in_channels=128,
            embedding_dim=embedding_dim,
        )

        # Positional Embedding
        self.position_embedding = nn.Parameter(
            torch.zeros(
                1,
                (bottleneck_size**2),
                embedding_dim,
            )
        )

        # Transformer Encoder
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(embedding_dim, num_heads) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embedding_dim)

        # Decoder
        self.decoder3 = DecoderBlock(embedding_dim, 128, 128)
        self.decoder2 = DecoderBlock(128, 64, 64)
        self.decoder1 = DecoderBlock(64, 32, 32)

        # Final Convolution
        self.final = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, 1),
        )

        # Initialize weights
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN Encoder with skip connections
        skip1 = self.conv1(x)
        x = self.pool1(skip1)

        skip2 = self.conv2(x)
        x = self.pool2(skip2)

        skip3 = self.conv3(x)
        x = self.pool3(skip3)

        # Transformer Encoder
        B, C, H, W = x.shape
        x = self.patch_embedding(x)  # [8, 256, 768]
        x = x + self.position_embedding  # [8, 256, 768]

        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        x = self.norm(x)  # [8, 256, 768]

        # Reshape back to spatial
        x = x.transpose(1, 2).reshape(B, self.embedding_dim, H, W)  # [8, 768, 16, 16]

        # Decoder with skip connections
        x = self.decoder3(x, skip3)
        x = self.decoder2(x, skip2)
        x = self.decoder1(x, skip1)

        # Final output
        x = self.final(x)
        return x
