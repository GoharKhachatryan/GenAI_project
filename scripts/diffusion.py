import torch
import torch.nn.functional as F

from scripts.helpers import extract, cosine_beta_schedule


class DDPM:
    def __init__(
        self,
        timesteps=1000,
        schedule="cosine",
        device="cpu",
    ):

        self.timesteps = timesteps
        self.device = device

        if schedule == "cosine":
            self.betas = cosine_beta_schedule(
                timesteps,
                device=device,
            )

        elif schedule == "linear":
            self.betas = torch.linspace(
                1e-4,
                2e-2,
                timesteps,
                device=device,
            )

        else:
            raise ValueError(
                f"Unknown noise schedule: {schedule}"
            )

        # alphas
        self.alphas = 1.0 - self.betas

        # alpha bars
        self.alpha_bars = torch.cumprod(
            self.alphas,
            dim=0,
        )

    # Forward diffusion process
    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)

        # Extract the sequence of alpha_bars for the given t
        alpha_bar_t = extract(self.alpha_bars, t, x0.shape)

        # Sqrt for the forward process' formula
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)

        # Forward diffusion formula
        x_t = (sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise)

        return x_t, noise

    # Training objective
    def training_loss(
        self,
        model,
        x_0,
        t,
        cond,
    ):

        noise = torch.randn_like(x_0)

        x_t, _ = self.q_sample(x_0, t, noise=noise)

        noise_pred = model(
            x_t,
            t,
            cond,
        )

        loss = F.mse_loss(
            noise_pred,
            noise,
        )

        return loss
