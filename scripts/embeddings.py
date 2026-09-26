import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Classical Sinusoidal Embedding for the timestep
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2

        frequencies = torch.exp(
            -math.log(10000)
            * torch.arange(
                half_dim,
                device=t.device,
                dtype=torch.float32,
            )
            / (half_dim - 1)
        )

        angles = t.float()[:, None] * frequencies[None, :]

        emb = torch.cat(
            [
                torch.sin(angles),
                torch.cos(angles),
            ],
            dim=1,
        )

        return emb


class TimeEmbedding(nn.Module):
    def __init__(
        self,
        time_dim=256,
        emb_dim=256,
    ):
        super().__init__()

        self.sinusoidal = SinusoidalTimeEmbedding(time_dim)

        self.mlp = nn.Sequential(
            nn.Linear(time_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, t):
        emb = self.sinusoidal(t)
        emb = self.mlp(emb)

        return emb


class ConditionEmbedding(nn.Module):
    def __init__(
        self,
        cond_dim=16,
        emb_dim=256,
    ):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, cond):
        return self.mlp(cond)