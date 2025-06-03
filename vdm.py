"""
vdm.py - Continuous Time VDM Module

This module implements a continuous-time diffusion model (VDM) that jointly trains a 
parameterized noise schedule along with the noise prediction network. It includes two 
noise schedule options: a fixed linear schedule and a learned linear schedule.

The model expects input images with pixel values in [0, 1] (which will be converted to 
a discrete set of integers from 0 to vocab_size-1 and then rescaled to [-1, 1]). 
The diffusion loss is computed in bits-per-dimension (bpd).

Do NOT modify the core architecture. Use this module via an import in your model.py:
    from vdm import VDM as ContinuousVDM
"""

import numpy as np
import torch
from torch import allclose, argmax, autograd, exp, linspace, nn, sigmoid, sqrt
from torch.special import expm1
from tqdm import trange

from utils import maybe_unpack_batch, unsqueeze_right

# ------------------------------------------
# Continuous Time VDM Model Definition
# ------------------------------------------
class VDM(nn.Module):
    def __init__(self, model, cfg, image_shape):
        """
        Args:
            model (nn.Module): The base noise prediction network.
            cfg: Configuration object with attributes:
                - noise_schedule: One of "fixed_linear" or "learned_linear".
                - gamma_min: Minimum gamma value.
                - gamma_max: Maximum gamma value.
                - antithetic_time_sampling: Boolean.
            image_shape (tuple): Shape of the image, e.g. (1, 28, 28).
        """
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.image_shape = image_shape
        self.vocab_size = 256  # number of discrete levels per pixel
        if cfg.noise_schedule == "fixed_linear":
            self.gamma = FixedLinearSchedule(cfg.gamma_min, cfg.gamma_max)
        elif cfg.noise_schedule == "learned_linear":
            self.gamma = LearnedLinearSchedule(cfg.gamma_min, cfg.gamma_max)
        else:
            raise ValueError(f"Unknown noise schedule {cfg.noise_schedule}")

    @property
    def device(self):
        return next(self.model.parameters()).device

    @torch.no_grad()
    def sample_p_s_t(self, z, t, s, clip_samples):
        """
        Sample from p(z_s|z_t, x) using the ancestral sampling procedure.
        """
        gamma_t = self.gamma(t)
        gamma_s = self.gamma(s)
        c = -expm1(gamma_s - gamma_t)
        alpha_t = sqrt(sigmoid(-gamma_t))
        alpha_s = sqrt(sigmoid(-gamma_s))
        sigma_t = sqrt(sigmoid(gamma_t))
        sigma_s = sqrt(sigmoid(gamma_s))

        pred_noise = self.model(z, gamma_t)
        if clip_samples:
            x_start = (z - sigma_t * pred_noise) / alpha_t
            x_start.clamp_(-1.0, 1.0)
            mean = alpha_s * (z * (1 - c) / alpha_t + c * x_start)
        else:
            mean = alpha_s / alpha_t * (z - c * sigma_t * pred_noise)
        scale = sigma_s * sqrt(c)
        return mean + scale * torch.randn_like(z)

    @torch.no_grad()
    def sample(self, batch_size, n_sample_steps, clip_samples):
        """
        Ancestral sampling procedure to sample images.
        """
        z = torch.randn((batch_size, *self.image_shape), device=self.device)
        steps = linspace(1.0, 0.0, n_sample_steps + 1, device=self.device)
        for i in trange(n_sample_steps, desc="sampling"):
            z = self.sample_p_s_t(z, steps[i], steps[i + 1], clip_samples)
        # Decode z0 to image probabilities
        logprobs = self.log_probs_x_z0(z_0=z)  # (B, C, H, W, vocab_size)
        x = argmax(logprobs, dim=-1)  # (B, C, H, W)
        return x.float() / (self.vocab_size - 1)  # normalize to [0, 1]

    def sample_q_t_0(self, x, times, noise=None):
        """
        Samples from q(x_t | x_0) at the given time steps.
        """
        with torch.enable_grad():  # enables gradient computation
            gamma_t = self.gamma(times)
        gamma_t_padded = unsqueeze_right(gamma_t, x.ndim - gamma_t.ndim)
        mean = x * sqrt(sigmoid(-gamma_t_padded))  # alpha_t * x
        scale = sqrt(sigmoid(gamma_t_padded))
        if noise is None:
            noise = torch.randn_like(x)
        return mean + noise * scale, gamma_t

    def sample_times(self, batch_size):
        if self.cfg.antithetic_time_sampling:
            t0 = np.random.uniform(0, 1 / batch_size)
            times = torch.arange(t0, 1.0, 1.0 / batch_size, device=self.device)
        else:
            times = torch.rand(batch_size, device=self.device)
        return times

    def forward(self, batch, *, noise=None):
        """
        Computes the loss (in bits-per-dimension) and additional metrics.
        Expects batch to be a tuple (x, label) where x is in [0, 1] with shape (B, C, H, W).
        """
        x, label = maybe_unpack_batch(batch)
        assert x.shape[1:] == self.image_shape
        assert 0.0 <= x.min() and x.max() <= 1.0
        bpd_factor = 1 / (np.prod(x.shape[1:]) * np.log(2))

        # Convert image to discrete integers.
        img_int = torch.round(x * (self.vocab_size - 1)).long()
        assert (img_int >= 0).all() and (img_int <= self.vocab_size - 1).all()
        # Verify the discretization.
        assert allclose(img_int / (self.vocab_size - 1), x)

        # Rescale to [-1 + 1/vocab_size, 1 - 1/vocab_size]
        x = 2 * ((img_int + 0.5) / self.vocab_size) - 1

        # Sample x_t ~ q(x_t | x_0) at a random time.
        times = self.sample_times(x.shape[0]).requires_grad_(True)
        if noise is None:
            noise = torch.randn_like(x)
        x_t, gamma_t = self.sample_q_t_0(x=x, times=times, noise=noise)

        # Forward pass through the noise predictor network.
        model_out = self.model(x_t, gamma_t)

        # Diffusion loss.
        gamma_grad = autograd.grad(
            gamma_t,
            times,
            grad_outputs=torch.ones_like(gamma_t),
            create_graph=True,
            retain_graph=True,
        )[0]
        pred_loss = ((model_out - noise) ** 2).sum((1, 2, 3))
        diffusion_loss = 0.5 * pred_loss * gamma_grad * bpd_factor

        # Latent loss: KL divergence term.
        gamma_1 = self.gamma(torch.tensor([1.0], device=self.device))
        sigma_1_sq = sigmoid(gamma_1)
        mean1_sqr = (1 - sigma_1_sq) * (x ** 2)
        latent_loss = kl_std_normal(mean1_sqr, sigma_1_sq).sum((1, 2, 3)) * bpd_factor

        # Reconstruction loss.
        log_probs = self.log_probs_x_z0(x)  # (B, C, H, W, vocab_size)
        x_one_hot = torch.zeros((*x.shape, self.vocab_size), device=self.device)
        x_one_hot.scatter_(4, img_int.unsqueeze(-1), 1)
        log_probs = (x_one_hot * log_probs).sum(-1)
        recons_loss = -log_probs.sum((1, 2, 3)) * bpd_factor

        loss = diffusion_loss + latent_loss + recons_loss

        with torch.no_grad():
            gamma_0 = self.gamma(torch.tensor([0.0], device=self.device))
        metrics = {
            "bpd": loss.mean(),
            "diff_loss": diffusion_loss.mean(),
            "latent_loss": latent_loss.mean(),
            "loss_recon": recons_loss.mean(),
            "gamma_0": gamma_0.item(),
            "gamma_1": gamma_1.item(),
        }
        return loss.mean(), metrics

    def log_probs_x_z0(self, x=None, z_0=None):
        """
        Computes log p(x|z_0) for all possible pixel values, returning a tensor of shape 
        (B, C, H, W, vocab_size) containing the log probabilities.
        """
        gamma_0 = self.gamma(torch.tensor([0.0], device=self.device))
        if x is None and z_0 is not None:
            z_0_rescaled = z_0 / sqrt(sigmoid(-gamma_0))
        elif z_0 is None and x is not None:
            z_0_rescaled = x + exp(0.5 * gamma_0) * torch.randn_like(x)
        else:
            raise ValueError("Must provide either x or z_0, not both.")
        z_0_rescaled = z_0_rescaled.unsqueeze(-1)
        x_lim = 1 - 1 / self.vocab_size
        x_values = linspace(-x_lim, x_lim, self.vocab_size, device=self.device)
        logits = -0.5 * exp(-gamma_0) * (z_0_rescaled - x_values) ** 2
        log_probs = torch.log_softmax(logits, dim=-1)
        return log_probs


def kl_std_normal(mean_squared, var):
    return 0.5 * (var + mean_squared - torch.log(var.clamp(min=1e-15)) - 1.0)


# ------------------------------------------
# Noise Schedule Implementations
# ------------------------------------------
class FixedLinearSchedule(nn.Module):
    def __init__(self, gamma_min, gamma_max):
        super().__init__()
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

    def forward(self, t):
        return self.gamma_min + (self.gamma_max - self.gamma_min) * t


class LearnedLinearSchedule(nn.Module):
    def __init__(self, gamma_min, gamma_max):
        super().__init__()
        self.b = nn.Parameter(torch.tensor(gamma_min))
        self.w = nn.Parameter(torch.tensor(gamma_max - gamma_min))

    def forward(self, t):
        return self.b + self.w.abs() * t
