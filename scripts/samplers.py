import torch

from scripts.helpers import extract, make_sampling_timesteps

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


# DDPM step for x_t -> x_{prev_t}
@torch.no_grad()
def ddpm_step_respaced(
    model,
    ddpm,
    x_t,
    t,
    prev_t,
    cond,
):
    
    batch_size = x_t.shape[0]

    t_batch = torch.full(
        (batch_size,),
        t,
        device=x_t.device,
        dtype=torch.long,
    )

    # Predict the noise for the original timestep t
    eps_pred = model(
        x_t,
        t_batch,
        cond,
    )

    alpha_bar_t = ddpm.alpha_bars[t]

    # Based on the predicted noise, predict x_0
    x_0_pred = (
        x_t
        - torch.sqrt(1.0 - alpha_bar_t) * eps_pred
    ) / torch.sqrt(alpha_bar_t)

    x_0_pred = x_0_pred.clamp(-1.0, 1.0)

    # If the step is last, return x_0_pred
    if prev_t < 0:
        return x_0_pred

    # For other steps find the coefficients of the predicted x_0 and given x_t
    alpha_bar_prev = ddpm.alpha_bars[prev_t]

    alpha_ratio = alpha_bar_t / alpha_bar_prev
    beta_eff = 1.0 - alpha_ratio

    posterior_var = (
        beta_eff
        * (1.0 - alpha_bar_prev)
        / (1.0 - alpha_bar_t)
    )
    
    # clip the negative values in case of incorrect rounding
    posterior_var = posterior_var.clamp(min=0.0)

    coef_x_0 = (
        torch.sqrt(alpha_bar_prev)
        * beta_eff
        / (1.0 - alpha_bar_t)
    )

    coef_x_t = (
        torch.sqrt(alpha_ratio)
        * (1.0 - alpha_bar_prev)
        / (1.0 - alpha_bar_t)
    )

    mean = coef_x_0 * x_0_pred + coef_x_t * x_t

    noise = torch.randn_like(x_t)

    # Sample from this distribution x_{prev_t}
    x_prev = mean + torch.sqrt(posterior_var) * noise

    return x_prev


# Step of DDIM from x_t to x_{prev_t}
@torch.no_grad()
def ddim_step(
    model,
    ddpm,
    x_t,
    t,
    prev_t,
    cond,
    eta=0.0,
):
    # Same way like for ddpm respaced
    batch_size = x_t.shape[0]

    t_batch = torch.full(
        (batch_size,),
        t,
        device=x_t.device,
        dtype=torch.long,
    )

    eps_pred = model(
        x_t,
        t_batch,
        cond,
    )

    alpha_bar_t = ddpm.alpha_bars[t]

    x_0_pred = (
        x_t
        - torch.sqrt(1.0 - alpha_bar_t) * eps_pred
    ) / torch.sqrt(alpha_bar_t)

    x_0_pred = x_0_pred.clamp(-1.0, 1.0)

    # After clamping x_0_pred recompute eps_pred
    # to get 2 mutually consistent quantities
    eps_pred = (
        x_t - torch.sqrt(alpha_bar_t) * x_0_pred
    ) / torch.sqrt(1.0 - alpha_bar_t)

    # If the final step
    if prev_t < 0:
        return x_0_pred

    # Implement the formula for DDIM
    alpha_bar_prev = ddpm.alpha_bars[prev_t]

    # DDIM's stochastic term
    # If eta = 0.0 => no stochastic term => eliminates the random noise term,
    # making the reverse trajectory deterministic for a given initial noise
    sigma_term = (
            (1.0 - alpha_bar_prev)
            / (1.0 - alpha_bar_t)
    ) * (
            1.0 - alpha_bar_t / alpha_bar_prev
    )

    sigma = eta * torch.sqrt(sigma_term.clamp(0.0))

    direction_var = (
        1.0
        - alpha_bar_prev
        - sigma ** 2
    ).clamp(min=0.0)

    direction = torch.sqrt(direction_var) * eps_pred

    if eta > 0:
        noise = torch.randn_like(x_t)
    
    else:
        noise = torch.zeros_like(x_t)

    x_prev = (
        torch.sqrt(alpha_bar_prev) * x_0_pred
        + direction
        + sigma * noise
    )

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


# Full reverse sampling using fewer than T inference steps
@torch.no_grad()
def ddpm_sample_respaced(
    model,
    ddpm,
    cond,
    num_steps,
    image_shape=(3, 32, 32),
    x_init=None,
):

    model.eval()

    batch_size = cond.shape[0]

    # For the fair comparision of the models, use the same initial noise
    if x_init is None:
        x = torch.randn(
            batch_size,
            *image_shape,
            device=cond.device,
        )

    else:
        x = x_init.clone()

    sampling_timesteps = make_sampling_timesteps(
        total_timesteps=ddpm.timesteps,
        num_steps=num_steps,
        device=cond.device,
    )

    sampling_timesteps = sampling_timesteps.flip(0)

    for i in range(len(sampling_timesteps)):
        t = sampling_timesteps[i].item()

        # For the last step, make sure x_0_pred is returned
        if i == len(sampling_timesteps) - 1:
            prev_t = -1

        else:
            prev_t = sampling_timesteps[i + 1].item()

        x = ddpm_step_respaced(
            model=model,
            ddpm=ddpm,
            x_t=x,
            t=t,
            prev_t=prev_t,
            cond=cond,
        )

    return x


# DDIM full sampling
@torch.no_grad()
def ddim_sample(
    model,
    ddpm,
    cond,
    num_steps,
    eta=0.0,
    image_shape=(3,32,32),
    x_init=None,
):

    model.eval()

    batch_size = cond.shape[0]

    # For the deterministic evaluation with a fixed initial noise
    if x_init is None:
        x = torch.randn(
            batch_size,
            *image_shape,
            device=cond.device,
        )

    else:
        x = x_init.clone()

    # Use the same function for timesteps, since the logic here is not changing
    sampling_timesteps = make_sampling_timesteps(
        total_timesteps=ddpm.timesteps,
        num_steps=num_steps,
        device=cond.device,
    )

    sampling_timesteps = sampling_timesteps.flip(0)

    for i in range(len(sampling_timesteps)):
        t = sampling_timesteps[i].item()

        if i == len(sampling_timesteps) - 1:
            prev_t = -1
        else:
            prev_t = sampling_timesteps[i + 1].item()

        x = ddim_step(
            model=model,
            ddpm=ddpm,
            x_t=x,
            t=t,
            prev_t=prev_t,
            cond=cond,
            eta=eta,
        )

    return x