import torch
import math 

def extract(values, t, x_shape):
    batch_size = t.shape[0]

    # Take the coefficient corresponding to the timestep t
    out = values.gather(0, t)

    # Reshape them
    return out.reshape(
        batch_size,
        *((1,) * (len(x_shape) - 1))
    )

def cosine_beta_schedule(
    timesteps,
    s=0.008,
    device="cpu",
):

    steps = timesteps + 1

    x = torch.linspace(
        0,
        timesteps,
        steps,
        device=device,
    )

    alpha_bars = torch.cos(
        ((x / timesteps + s) / (1 + s)) * math.pi * 0.5
    ) ** 2

    alpha_bars = alpha_bars / alpha_bars[0]

    betas = 1.0 - (
        alpha_bars[1:] / alpha_bars[:-1]
    )

    return torch.clamp(
        betas,
        max=0.999
    )