import torch
import torch.nn as nn
import torch.nn.functional as F

# Classical ResBlock implementation for UNet
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim):
        super().__init__()

        # Implement ResBlock's normalization and convolution layers
        # (usage of torch.nn is possible)
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )

        # For projecting the embedding with a linear layer
        self.emb_proj = nn.Linear(
            emb_dim,
            out_channels
        )

        # Normalization and convolution layers after embedding injection
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )

        # Define the skip connection based on the in/out channels
        if in_channels != out_channels:
            self.skip = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1
            )
        
        else:
            self.skip = nn.Identity()

    # Forward pass of the ResBlock
    def forward(self, x, emb):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        emb_out = self.emb_proj(F.silu(emb))

        # Inject the embedding
        h = h + emb_out[:, :, None, None]

        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        return h + self.skip(x)


# Downsample blocks for UNet
class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()

        # Downsample with convolution with stride > 1
        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=2,
            padding=1
        )

    def forward(self, x):
        return self.conv(x)


# Upsample blocks for UNet
class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1
        )

    def forward(self, x):
        # Upsample with nearest neighbour interpolation
        x = F.interpolate(
            x,
            scale_factor=2,
            mode="nearest"
        )

        return self.conv(x)

# Attention block
class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.norm = nn.GroupNorm(8, channels)

        self.q = nn.Conv2d(channels, channels, kernel_size=1)
        self.k = nn.Conv2d(channels, channels, kernel_size=1)
        self.v = nn.Conv2d(channels, channels, kernel_size=1)

        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

        self.scale = channels ** (-0.5)

    def forward(self, x):
        b, c, h, w = x.shape

        h_in = self.norm(x)

        q = self.q(h_in)
        k = self.k(h_in)
        v = self.v(h_in)

        q = q.reshape(b, c, h * w).permute(0, 2, 1)
        k = k.reshape(b, c, h * w)
        v = v.reshape(b, c, h * w).permute(0, 2, 1)

        # Implementation of the classical self-attention
        attn = torch.bmm(q, k) * self.scale
        attn = torch.softmax(attn, dim=-1)
        out = torch.bmm(attn, v)

        out = out.permute(0, 2, 1).reshape(b, c, h, w)
        out = self.proj(out)

        return x + out
