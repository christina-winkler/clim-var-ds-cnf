"""
Denoising Diffusion Probabilistic Model.

Conditioned on a specified denoising architecture a Denoising Diffusion Model is
defined and can be used for single image super-resolution.

This code is based on https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement.

Christina Winkler, Oct 2023
"""

import numpy as np
import torch
import math
import torch.nn as nn
from typing import Union, Tuple
from models.architectures import RRDBNet_arch, unet
from functools import partial

def make_beta_schedule(schedule: str, n_timestep: int, linear_start: float = 1e-4,
                       linear_end: float = 2e-2, cosine_s: float = 8e-3) -> \
        Union[np.ndarray, torch.Tensor]:
    """Defines Gaussian noise variance beta schedule that is gradually added
    to the data during the diffusion process.

    Args:
        schedule: Defines the type of beta schedule. Possible types are const,
            linear, warmup10, warmup50, quad, jsd and cosine.
        n_timestep: Number of diffusion timesteps.
        linear_start: Minimum value of the linear schedule.
        linear_end: Maximum value of the linear schedule.
        cosine_s: An offset to prevent beta to be smaller at timestep 0.

    Returns:
        Beta values for each timestep starting from 1 to n_timestep.
    """
    if schedule == "const":  # Constant beta schedule.
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == "linear":  # Linear beta schedule.
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == "warmup10":  # Linear beta schedule with warmup fraction of 0.10.
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == "warmup50":  # Linear beta schedule with warmup fraction of 0.50.
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == "quad":  # Quadratic beta schedule.
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == "jsd":  # Multiplicative inverse beta schedule: 1/T, 1/(T-1), 1/(T-2), ..., 1.
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":  # Cosine beta schedule [formula 17, arxiv:2102.09672].
        timesteps = (torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s)
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas

class CondDiffusion(nn.Module):
    """
    Conditioned Denoising Diffusion Probabilistic Model with gaussian noise
    process.
    """
    def __init__(self, input_shape, bsz, s, nb, cond_channels, trainmode, device,
                 linear_start=1e-6, linear_end=1e-2, noise_sched='cosine', T=2000):
        super().__init__()

        c,w,h = input_shape
        self.T = T # gaussianization steps
        self.noise_sched = noise_sched
        self.linear_start = linear_start
        self.linear_end = linear_end
        self.sqrt_alphas_cumprod_prev = None
        self.trainmode = trainmode
        self.s = s
        self.loss = nn.MSELoss(reduction='sum').to(args.device)

        self.denoise_net = unet.UNet(c+1, 1, inner_channel=16, norm_groups=4,
                                     channel_mults=[1,2,4,8],
                                     attn_res=[8], res_blocks=5, dropout=0.7)

        if self.trainmode:
            self.set_new_noise_schedule(noise_sched, T, linear_start, linear_end, device)

    def set_new_noise_schedule(self, schedule, n_timestep, linear_start, linear_end, device):
        """Sets a new beta schedule.

        Args:
            schedule: Defines the type of beta schedule. Possible types are const, linear, warmup10, warmup50, quad,
                    jsd and cosine.
            n_timestep: Number of diffusion timesteps.
            linear_start: Minimum value of the linear schedule.
            linear_end: Maximum value of the linear schedule.
            device: A torch.device object.
        """
        # Defining a partial fundtion that converts data into type of float32 and moves it onto the specified device.
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

        betas = make_beta_schedule(schedule=schedule, n_timestep=n_timestep,
                                   linear_start=linear_start, linear_end=linear_end)

        betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(np.append(1., alphas_cumprod))

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # Storing parameters into state dict of model.
        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))

        # Calculating constants for reverse conditional posterior distribution q(x_{t-1} | x_t, x_0).
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer("log_one_minus_alphas_cumprod", to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer("sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)  # Formula 7, arXiv:2006.11239.
        self.register_buffer("posterior_variance", to_torch(posterior_variance))

        # Clipping the minimum log value of posterior variance to be 1e-20 as posterior variance is 0 at timestep 0.
        self.register_buffer("posterior_log_variance_clipped", to_torch(np.log(np.maximum(posterior_variance, 1e-20))))

        # Calculating the coefficients of the mean q(x_{t-1} | x_t, x_0) [formula 7, arXiv:2006.11239].
        self.register_buffer("posterior_mean_coef1",
                             to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer("posterior_mean_coef2",
                             to_torch((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def forward(self, x_start, x_cond, noise=None):
        """

        """

        b = x_start.shape[0]  # Dimension of s_start is (B, C, H, W).

        # uniformly sample gaussianization time-step
        t = np.random.randint(1, self.T + 1)

        # get betas determining variance of noise added at each diffusion step
        betas = make_beta_schedule(schedule=self.noise_sched, n_timestep=self.T,
                                   linear_start=self.linear_start, linear_end=self.linear_end)

        # Sample gammas from piece-wise uniform distribution
        continuous_sqrt_alpha_cumprod = torch.FloatTensor(np.random.uniform(self.sqrt_alphas_cumprod_prev[t-1],
                                                                            self.sqrt_alphas_cumprod_prev[t],
                                                                            size=b)).to(x_start.device)
        if noise is None:
            noise = torch.randn_like(x_start)

        # First-step of diffusion process: corrupt high-res image with gaussian noise
        x_noisy = self.q_sample(x_start=x_start, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1),
                                noise=noise)

        # conditioning architecture predicts the Gaussian noise used to corrupt the HR image in the diffusion process.
        # here we condition on interpolated HR Image
        x_interp = torch.nn.functional.interpolate(x_cond, mode='bicubic', scale_factor=self.s)
        # import pdb; pdb.set_trace()
        # Estimation of Gaussian noise:
        noise_reconstructed = self.denoise_net(torch.cat([x_interp, x_noisy], dim=1), continuous_sqrt_alpha_cumprod)

        loss = self.loss(noise, noise_reconstructed) # to enforce x_recon to predict Gaussian Noise


        return None

    @staticmethod
    def q_sample(x_start: torch.Tensor, continuous_sqrt_alpha_cumprod: torch.Tensor,
                 noise: torch.Tensor = None) -> torch.Tensor:
        """Sampling from q(x_t | x_0) [formula 4, arXiv:2006.11239]. --> probably eqn. (3) is meant here which
           describes one diffusion step of the forward diffusion process after marginalizing out the intermediate
           steps.

        Args:
            x_start: Starting data point x_0 of size [B, C, H, W]. Often HR image.
            continuous_sqrt_alpha_cumprod: Square root of the product of alphas of size [B, 1, 1, 1].
            noise: Gaussian standard noise of the same size as x_start.

        Returns:
            Sampled noisy point of size [B, C, H, W].
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        return continuous_sqrt_alpha_cumprod * x_start + (1 - continuous_sqrt_alpha_cumprod**2).sqrt() * noise
