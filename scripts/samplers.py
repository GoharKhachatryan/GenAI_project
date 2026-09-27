import torch

from scripts.helpers import extract

# For one step sampling
@torch.no_grad()
def ddpm_step(
    model,
    ddpm,
    x_t,
    t,
    cond,
):

    # Predict the noise with the trained model
    eps_pred = model(x_t, t, cond)

    # Extract the coefficients for the reverse step
    beta_t = extract(
        ddpm.betas,
        t,
        x_t.shape,
    )

    alpha_t = extract(
        ddpm.alphas,
        t,
        x_t.shape,
    )

    alpha_bar_t = extract(
        ddpm.alpha_bars,
        t,
        x_t.shape,
    )

    posterior_var_t = extract(
        ddpm.posterior_variance,
        t,
        x_t.shape,
    )

    # Calculate the mean of the x_t's distribution
    mean = (
        1.0 / torch.sqrt(alpha_t)
    ) * (
        x_t
        - beta_t
        / torch.sqrt(1.0 - alpha_bar_t)
        * eps_pred
    )

    # Sample a noise
    noise = torch.randn_like(x_t)

    # Add noise for any timestep, except for the t=0
    mask = (
        (t != 0)
        .float()
        .reshape(
            x_t.shape[0],
            *((1,) * (x_t.ndim - 1))
        )
    )

    # Sample x_t from a distribution with mean and variance
    x_prev = mean + mask * torch.sqrt(posterior_var_t) * noise

    return x_prev


# For the full generation
@torch.no_grad()
def ddpm_sample(
    model,
    ddpm,
    cond,
    image_shape=(3, 32, 32),
):

    model.eval()

    batch_size = cond.shape[0]

    # Sample an initial noise for each image in the batch
    x = torch.randn(
        batch_size,
        *image_shape,
        device=cond.device,
    )

    # Perform a step-by-step denoising x_t -> x_{t-1} for all of the conditions in the batch
    for timestep in reversed(range(ddpm.timesteps)):
        t = torch.full(
            (batch_size,),
            timestep,
            device=cond.device,
            dtype=torch.long,
        )

        x = ddpm_step(
            model,
            ddpm,
            x,
            t,
            cond,
        )

    return x