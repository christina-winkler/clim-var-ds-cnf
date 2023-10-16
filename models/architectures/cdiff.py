"""
Denoising Diffusion Probabilistic Model.

Conditioned on a specified denoising architecture a Denoising Diffusion Model is
defined and can be used for single image super-resolution.

Implementation is taken and adaption to works in: https://arxiv.org/pdf/2104.07636.pdf
"""

import torch
import torch.nn as nn

from models.architectures import RRDBNet


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
    def __init__(self, input_shape, bsz, s, nb, cond_channels,
                 noise_sched='cosine', T=2000):
        super().__init__()

        c,w,h = input_shape
        self.T = T # gaussianization steps

        self.denoising_arch = RRDBNet(c, cond_channels, s, input_shape, nb, gc=32)

    def forward(self, x_start):

        # uniformly sample gaussianization time-step
        t = np.random.randint(1, self.T + 1)

        # get betas determining variance of noise added at each diffusion step
        betas = make_beta_schedule(schedule=noise_sched, n_timestep=self.T,
                                    linear_start=linear_start, linear_end=linear_end)

        # Sample gammas from piece-wise uniform distribution


        return None
