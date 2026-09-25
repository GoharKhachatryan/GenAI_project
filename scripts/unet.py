import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.blocks import (
    ResBlock,
    Downsample,
    Upsample,
    AttentionBlock,
)

class UNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        base_channels=64,
        emb_dim=256,
    ):

        super().__init__()

        # Initial projection from RGB to feature space
        self.init_conv = nn.Conv2d(
            in_channels,
            base_channels,
            kernel_size=3,
            padding=1,
        )

        # Encoder:
        self.enc1 = ResBlock(
            base_channels,
            base_channels,
            emb_dim,
        )
        self.down1 = Downsample(base_channels)


        self.enc2 = ResBlock(
            base_channels,
            base_channels * 2,
            emb_dim,
        )
        self.down2 = Downsample(base_channels * 2)


        self.enc3 = ResBlock(
            base_channels * 2,
            base_channels * 4,
            emb_dim,
        )
        self.down3 = Downsample(base_channels * 4)


        # Bottleneck:
        self.mid1 = ResBlock(
            base_channels * 4,
            base_channels * 4,
            emb_dim,
        )

        self.attn = AttentionBlock(base_channels * 4)

        self.mid2 = ResBlock(
            base_channels * 4,
            base_channels * 4,
            emb_dim,
        )

        # Decoder

        self.up3 = Upsample(base_channels * 4)
        # concat with encoder skip: 256 + 256 = 512
        self.dec3 = ResBlock(
            base_channels * 8,
            base_channels * 4,
            emb_dim,
        )

        self.up2 = Upsample(base_channels * 4)
        # concat with encoder skip: 256 + 128 = 384
        self.dec2 = ResBlock(
            base_channels * 6,
            base_channels * 2,
            emb_dim,
        )

        self.up1 = Upsample(base_channels * 2)
        # concat with encoder skip: 128 + 64 = 192
        self.dec1 = ResBlock(
            base_channels * 3,
            base_channels,
            emb_dim,
        )


        # Final conv before the output
        self.out_norm = nn.GroupNorm(8, base_channels)
        self.out_conv = nn.Conv2d(
            base_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x, emb):
        # Initial conv
        x = self.init_conv(x)

        # Encoder
        x1 = self.enc1(x, emb)
        x = self.down1(x1)

        x2 = self.enc2(x, emb)
        x = self.down2(x2)

        x3 = self.enc3(x, emb)
        x = self.down3(x3)

        # Bottleneck
        x = self.mid1(x, emb)
        x = self.attn(x)
        x = self.mid2(x, emb)

        # Decoder
        x = self.up3(x)
        x = torch.cat([x, x3], dim=1)
        x = self.dec3(x, emb)

        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.dec2(x, emb)

        x = self.up1(x)
        x = torch.cat([x, x1], dim=1)
        x = self.dec1(x, emb)

        # Output
        x = self.out_norm(x)
        x = F.silu(x)
        x = self.out_conv(x)

        return x
