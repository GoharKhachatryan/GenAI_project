import torch
import torch.nn as nn

from scripts.embeddings import TimeEmbedding, ConditionEmbedding

class ConditionalDenoiser(nn.Module):
    def __init__(
        self,
        unet,
        time_dim=256,
        cond_dim=16,
        emb_dim=256,
    ):
        super().__init__()

        self.unet = unet

        self.time_embedding = TimeEmbedding(
            time_dim=time_dim,
            emb_dim=emb_dim,
        )

        self.cond_embedding = ConditionEmbedding(
            cond_dim=cond_dim,
            emb_dim=emb_dim,
        )

    def forward(self, x, t, cond):
        time_emb = self.time_embedding(t)
        cond_emb = self.cond_embedding(cond)

        emb = time_emb + cond_emb

        return self.unet(x, emb)