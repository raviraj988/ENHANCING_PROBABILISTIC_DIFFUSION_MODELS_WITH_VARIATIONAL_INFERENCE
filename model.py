#!/usr/bin/env python
"""
model.py - Contains all model definitions and related functions.
This module defines the functions for the diffusion process (gamma, sigma2, etc.),
the get_timestep_embedding(), ResNet, Encoder, Decoder, ScoreNet, VDM,
and the sampling/reconstruction functions.
A continuous‑time option is added that jointly trains the noise schedule with the loss.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from torch import allclose, autograd, linspace, argmax, exp, sqrt, sigmoid
from torch.special import expm1

# ---------------------------------------------------------------------
# Helper function, previously imported from utils but now defined here.
# ---------------------------------------------------------------------
def unsqueeze_right(x, num_dims):
    """
    Append 'num_dims' singleton dimensions to the right of x.
    If x has shape (..., ), the result is (..., 1, 1, ..., 1) with num_dims new dims.
    """
    return x.view(x.shape + (1,) * num_dims)

# ---------------------------------------------------------------------
# Forward process functions
# ---------------------------------------------------------------------
def gamma(ts, gamma_min=-6, gamma_max=6):
    """
    A simple linear interpolation from gamma_max to gamma_min, 
    used in discrete-time diffusion code.
    """
    return gamma_max + (gamma_min - gamma_max) * ts

def sigma2(gamma_val):
    return torch.sigmoid(-gamma_val)

def alpha(gamma_val):
    return torch.sqrt(1 - sigma2(gamma_val))

def variance_preserving_map(x, gamma_val, eps):
    """
    x_t = alpha(gamma_val)*x + sqrt(sigma2(gamma_val))*eps
    """
    a = alpha(gamma_val)
    var = sigma2(gamma_val)
    return a * x + torch.sqrt(var) * eps

# ---------------------------------------------------------------------
# Timestep embedding (for discrete-time ScoreNet)
# ---------------------------------------------------------------------
def get_timestep_embedding(timesteps, embedding_dim: int, dtype=torch.float32):
    timesteps = timesteps * 1000.0
    half_dim = embedding_dim // 2
    emb_factor = math.log(10000) / (half_dim - 1)
    emb = torch.exp(-emb_factor * torch.arange(half_dim, dtype=dtype, device=timesteps.device))
    emb = timesteps.unsqueeze(1).to(dtype) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb

# ---------------------------------------------------------------------
# Discrete‑time VDM classes: ResNet, Encoder, Decoder, ScoreNet, VDM
# ---------------------------------------------------------------------
class ResNet(nn.Module):
    def __init__(self, hidden_size, n_layers=1, middle_size=None, cond_dim=None):
        super().__init__()
        self.hidden_size = hidden_size
        if middle_size is None:
            middle_size = hidden_size * 4
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            block = nn.ModuleDict({
                'ln1': nn.LayerNorm(hidden_size),
                'fc1': nn.Linear(hidden_size, middle_size),
                'ln2': nn.LayerNorm(middle_size),
                'fc2': nn.Linear(middle_size, hidden_size)
            })
            nn.init.zeros_(block['fc2'].weight)
            nn.init.zeros_(block['fc2'].bias)
            self.layers.append(block)
        if cond_dim is None:
            cond_dim = hidden_size
        self.cond_fc = nn.Linear(cond_dim, middle_size, bias=False)

    def forward(self, x, cond=None):
        z = x
        for block in self.layers:
            h = F.gelu(block['ln1'](z))
            h = block['fc1'](h)
            if cond is not None:
                h = h + self.cond_fc(cond)
            h = F.gelu(block['ln2'](h))
            h = block['fc2'](h)
            z = z + h
        return z

class Encoder(nn.Module):
    def __init__(self, hidden_size=256, n_layers=3, z_dim=128, embedding_dim=32):
        super().__init__()
        self.fc_in = nn.Linear(28 * 28 * 1, hidden_size)
        self.resnet = ResNet(hidden_size, n_layers, middle_size=102, cond_dim=embedding_dim)
        self.fc_out = nn.Linear(hidden_size, z_dim)

    def forward(self, ims, cond=None):
        # Rescale to [-1,1]
        x = 2 * ims.float() - 1.0
        # Flatten from (B,H,W,C) -> (B,H*W*C)
        x = einops.rearrange(x, 'b h w c -> b (h w c)')
        x = self.fc_in(x)
        x = self.resnet(x, cond)
        return self.fc_out(x)

class Decoder(nn.Module):
    def __init__(self, hidden_size=512, n_layers=3, latent_dim=32):
        super().__init__()
        self.fc_in = nn.Linear(latent_dim, hidden_size)
        self.resnet = ResNet(hidden_size, n_layers, middle_size=1024, cond_dim=latent_dim)
        self.fc_out = nn.Linear(hidden_size, 28 * 28 * 1)

    def forward(self, z, cond=None):
        z = self.fc_in(z)
        z = self.resnet(z, cond)
        logits = self.fc_out(z).view(-1, 28, 28, 1)
        return torch.distributions.Independent(
            torch.distributions.Bernoulli(logits=logits), 3
        )

class ScoreNet(nn.Module):
    def __init__(self, embedding_dim=128, n_layers=10):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.fc_dense0 = nn.Linear(2 * embedding_dim, 4 * embedding_dim)
        self.fc_dense1 = nn.Linear(4 * embedding_dim, 4 * embedding_dim)
        self.fc_dense2 = nn.Linear(4 * embedding_dim, embedding_dim)
        self.fc_z = nn.Linear(embedding_dim, embedding_dim)
        self.resnet = ResNet(embedding_dim, n_layers, middle_size=embedding_dim*4)

    def forward(self, z, g_t, conditioning):
        B = z.size(0)
        if not isinstance(g_t, torch.Tensor):
            g_t = torch.tensor(g_t, device=z.device)
        if g_t.dim() == 0:
            g_t = g_t.repeat(B)
        temb = get_timestep_embedding(g_t, self.embedding_dim).to(z.device)
        cond_input = torch.cat([temb, conditioning], dim=1)
        cond = F.silu(self.fc_dense0(cond_input))
        cond = F.silu(self.fc_dense1(cond))
        cond = self.fc_dense2(cond)
        h = self.fc_z(z)
        h = self.resnet(h, cond)
        return z + h

class VDM(nn.Module):
    """
    Discrete-time VDM with a fixed number of timesteps (timesteps=1000).
    """
    def __init__(self, timesteps, gamma_min, gamma_max, embedding_dim, layers, classes):
        super().__init__()
        self.timesteps = timesteps
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.embedding_dim = embedding_dim
        self.layers = layers
        self.classes = classes
        
        self.gamma_fn = lambda t: gamma(t, gamma_min=self.gamma_min, gamma_max=self.gamma_max)
        self.score_model = ScoreNet(embedding_dim=embedding_dim, n_layers=layers)
        self.encoder = Encoder(z_dim=embedding_dim, embedding_dim=embedding_dim)
        self.decoder = Decoder(latent_dim=embedding_dim)
        self.embedding_vectors = nn.Embedding(self.classes, self.embedding_dim)
    
    def gammat(self, t):
        return self.gamma_fn(t)
    
    def recon_loss(self, x, f, cond):
        g_0 = self.gamma_fn(torch.tensor(0.0, device=x.device))
        eps_0 = torch.randn_like(f)
        z_0 = variance_preserving_map(f, g_0, eps_0)
        z_0_rescaled = z_0 / alpha(g_0)
        dist = self.decoder(z_0_rescaled, cond)
        loss_recon = -dist.log_prob(x.float())
        return loss_recon.mean()
    
    def latent_loss(self, f):
        g_1 = self.gamma_fn(torch.tensor(1.0, device=f.device))
        var_1 = sigma2(g_1)
        mean1_sqr = (1. - var_1) * (f ** 2)
        loss_klz = 0.5 * torch.sum(mean1_sqr + var_1 - torch.log(var_1) - 1., dim=-1)
        return loss_klz.mean()
    
    def diffusion_loss(self, t, f, cond):
        g_t = self.gamma_fn(t)
        eps = torch.randn_like(f)
        z_t = variance_preserving_map(f, g_t.unsqueeze(1), eps)
        eps_hat = self.score_model(z_t, g_t, cond)
        loss_diff_mse = torch.sum((eps - eps_hat) ** 2, dim=-1)
        T = self.timesteps
        s = t - (1. / T)
        g_s = self.gamma_fn(s)
        # Weighted by 0.5 * T * (expm1(g_s - g_t))
        loss_diff = 0.5 * T * (torch.expm1(g_s - g_t)) * loss_diff_mse
        return loss_diff.mean()
    
    def forward(self, images, conditioning):
        x = images
        B = x.size(0)
        cond = self.embedding_vectors(conditioning)
        f = self.encoder(x, cond)
        loss_recon = self.recon_loss(x, f, cond)
        loss_klz = self.latent_loss(f)
        # Time sampling
        if hasattr(self, "antithetic_time_sampling") and self.antithetic_time_sampling:
            t0 = torch.rand(1, device=x.device).item()
            t = (t0 + torch.arange(0, B, device=x.device).float() / B) % 1.0
        else:
            t = torch.rand(B, device=x.device)
        T = self.timesteps
        t = torch.ceil(t * T) / T
        loss_diff = self.diffusion_loss(t, f, cond)
        return loss_recon + loss_klz + loss_diff

    def embed(self, conditioning):
        return self.embedding_vectors(conditioning)
    
    def encode(self, ims, conditioning):
        cond = self.embedding_vectors(conditioning)
        return self.encoder(ims, cond)
    
    def decode(self, z0, conditioning):
        cond = self.embedding_vectors(conditioning)
        return self.decoder(z0, cond)
    
    def shortcut(self, ims, conditioning):
        cond = self.embedding_vectors(conditioning)
        f = self.encoder(ims, cond)
        eps_0 = torch.randn_like(f)
        g_0 = self.gamma_fn(torch.tensor(0.0, device=ims.device))
        z_0 = variance_preserving_map(f, g_0, eps_0)
        z_0_rescaled = z_0 / alpha(g_0)
        return self.decoder(z_0_rescaled, cond)
    
    def sample_step(self, z_t, conditioning, i, guidance_weight=0.0):
        eps = torch.randn_like(z_t)
        T = self.timesteps
        t = (T - i) / T
        s = (T - i - 1) / T
        g_s = self.gamma_fn(torch.tensor(s, device=z_t.device))
        g_t = self.gamma_fn(torch.tensor(t, device=z_t.device))
        cond = self.embedding_vectors(conditioning)
        B = z_t.size(0)
        ones = torch.ones(B, device=z_t.device, dtype=z_t.dtype)
        eps_hat_cond = self.score_model(z_t, g_t * ones, cond)
        eps_hat_uncond = self.score_model(z_t, g_t * ones, cond * 0.)
        eps_hat = (1. + guidance_weight)*eps_hat_cond - guidance_weight*eps_hat_uncond
        a = torch.sigmoid(g_s)
        b = torch.sigmoid(g_t)
        c = -torch.expm1(g_t - g_s)
        sigma_t = torch.sqrt(sigma2(g_t))
        z_s = torch.sqrt(a/b)*(z_t - sigma_t*c*eps_hat) + torch.sqrt((1.-a)*c)*eps
        return z_s

    def recon(self, z, t, conditioning):
        g_t = self.gamma_fn(t).unsqueeze(1)
        cond = self.embedding_vectors(conditioning)
        eps_hat = self.score_model(z, g_t, cond)
        sigmat = torch.sqrt(sigma2(g_t))
        alphat = torch.sqrt(1 - sigmat**2)
        xhat = (z - sigmat * eps_hat) / alphat
        return eps_hat, xhat

# ---------------------------------------------------------------------
# Sampling & ELBO for discrete-time
# ---------------------------------------------------------------------
def generate(vdm, shape, conditioning, guidance_weight=0.0, device='cpu'):
    zt = torch.randn(*shape, vdm.embedding_dim, device=device)
    for i in range(vdm.timesteps):
        zt = vdm.sample_step(zt, conditioning, i, guidance_weight)
    g0 = vdm.gamma_fn(torch.tensor(0.0, device=device))
    var0 = sigma2(g0)
    z0_rescaled = zt / torch.sqrt(1. - var0)
    return vdm.decode(z0_rescaled, conditioning)

def recon(vdm, t, ims, conditioning, device='cpu'):
    z_0 = vdm.encode(ims, conditioning)
    T = vdm.timesteps
    tn = torch.ceil(t * T)
    t_val = tn / T
    g_t = vdm.gamma_fn(t_val)
    eps = torch.randn_like(z_0)
    z_t = variance_preserving_map(z_0, g_t.unsqueeze(1), eps)
    for i in range(int(T - tn.item()), vdm.timesteps):
        z_t = vdm.sample_step(z_t, conditioning, i)
    g0 = vdm.gamma_fn(torch.tensor(0.0, device=device))
    var0 = sigma2(g0)
    z0_rescaled = z_t / torch.sqrt(1. - var0)
    return vdm.decode(z0_rescaled, conditioning)

def elbo(vdm, ims, conditioning):
    cond = vdm.embed(conditioning)
    f = vdm.encode(ims, conditioning)
    loss_recon = vdm.recon_loss(ims, f, cond)
    loss_klz = vdm.latent_loss(f)
    T = vdm.timesteps
    loss_diff_total = 0.0
    for i in range(T):
        t_i = torch.tensor([i / T], device=ims.device)
        loss_diff = vdm.diffusion_loss(t_i, f, cond)
        loss_diff_total += loss_diff / T
    return loss_recon + loss_klz + loss_diff_total

# ---------------------------------------------------------------------
# Continuous-time schedule and noise predictor
# ---------------------------------------------------------------------
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

class NoisePredictor(nn.Module):
    """
    For continuous-time model. Expects [B,C,H,W] images, plus a [B]-dim time 
    that we broadcast as a channel.
    """
    def __init__(self, image_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(image_channels + 1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, image_channels, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x, gamma):
        B, C, H, W = x.shape
        gamma_expanded = gamma.view(B, 1, 1, 1).expand(B, 1, H, W)
        x_input = torch.cat([x, gamma_expanded], dim=1)
        h = self.relu(self.conv1(x_input))
        h = self.relu(self.conv2(h))
        return self.conv3(h)

# ---------------------------------------------------------------------
# ContinuousVDM
# ---------------------------------------------------------------------
class ContinuousVDM(nn.Module):
    """
    Continuous time VDM model that trains a parameterized noise schedule 
    (fixed or learned) plus a noise predictor network.
    """
    def __init__(self, base_model, cfg, image_shape):
        super().__init__()
        self.model = base_model
        self.cfg = cfg
        self.image_shape = image_shape
        self.vocab_size = 256

        if cfg.noise_schedule == "fixed_linear":
            self.gamma = FixedLinearSchedule(cfg.gamma_min, cfg.gamma_max)
        elif cfg.noise_schedule == "learned_linear":
            self.gamma = LearnedLinearSchedule(cfg.gamma_min, cfg.gamma_max)
        else:
            raise ValueError(f"Unknown noise schedule {cfg.noise_schedule}")

    @torch.no_grad()
    def sample_p_s_t(self, z, t, s, clip_samples=False):
        """
        Single denoising step for continuous VDM.
        z: noisy sample at time t
        t: current time (scalar)
        s: next time (scalar, s < t)
        clip_samples: optional clipping
        """
        gamma_t = self.gamma(torch.tensor(t, device=self.device))
        gamma_s = self.gamma(torch.tensor(s, device=self.device))
        c = -expm1(gamma_s - gamma_t)

        alpha_t = sqrt(sigmoid(-gamma_t))
        alpha_s = sqrt(sigmoid(-gamma_s))
        sigma_t = sqrt(sigmoid(gamma_t))
        sigma_s = sqrt(sigmoid(gamma_s))

        pred_noise = self.model(z, torch.full((z.shape[0],), gamma_t.item(), device=self.device))

        if clip_samples:
            x_start = (z - sigma_t * pred_noise) / alpha_t
            x_start = torch.clamp(x_start, -1.0, 1.0)
            mean = alpha_s * ((z * (1 - c) / alpha_t) + c * x_start)
        else:
            mean = (alpha_s / alpha_t) * (z - c * sigma_t * pred_noise)

        scale = sigma_s * sqrt(c)
        noise = torch.randn_like(z)

        return mean + scale * noise


    @property
    def device(self):
        return next(self.model.parameters()).device

    def sample_times(self, batch_size):
        if self.cfg.antithetic_time_sampling:
            t0 = np.random.uniform(0, 1 / batch_size)
            times = torch.arange(t0, 1.0, 1.0 / batch_size, device=self.device)
        else:
            times = torch.rand(batch_size, device=self.device)
        return times

    def sample_q_t_0(self, x, times, noise=None):
        gamma_t = self.gamma(times)
        # unsqueeze to broadcast: x has shape (B,C,H,W), times shape (B,)
        gamma_t_padded = unsqueeze_right(gamma_t, x.ndim - gamma_t.ndim)
        mean = x * sqrt(sigmoid(-gamma_t_padded))
        scale = sqrt(sigmoid(gamma_t_padded))
        if noise is None:
            noise = torch.randn_like(x)
        return mean + noise * scale, gamma_t

    def forward(self, batch, *, noise=None):
        """
        Expects batch = (x, label) with x in [B,C,H,W] range [0,1].
        """
        x, label = batch
        assert x.shape[1:] == self.image_shape, "Input shape mismatch for continuous VDM!"
        bpd_factor = 1.0 / (np.prod(x.shape[1:]) * math.log(2))

        # Convert to discrete integers
        img_int = torch.round(x * (self.vocab_size - 1)).long()
        assert (img_int >= 0).all() and (img_int <= self.vocab_size - 1).all()
        # Check that x was discrete
        assert allclose(img_int / (self.vocab_size - 1), x), "x wasn't properly in discrete steps."

        # Rescale to [-1,1]
        x = 2.0 * ((img_int + 0.5) / self.vocab_size) - 1.0

        # Sample random times
        B = x.size(0)
        times = self.sample_times(B)
        if self.training:
            times = times.requires_grad_(True)


        # Noisy x_t
        if noise is None:
            noise = torch.randn_like(x)
        x_t, gamma_t = self.sample_q_t_0(x, times, noise)

        # Forward pass through base_model
        model_out = self.model(x_t, gamma_t)

        # diffusion_loss
        if self.training:
            gamma_grad = autograd.grad(
                gamma_t, times,
                grad_outputs=torch.ones_like(gamma_t),
                create_graph=True,
                retain_graph=True,
            )[0]
            pred_loss = ((model_out - noise)**2).sum(dim=(1,2,3))
            diffusion_loss = 0.5 * pred_loss * gamma_grad * bpd_factor
        else:
            diffusion_loss = torch.zeros(B, device=self.device)


        # latent_loss
        gamma_1 = self.gamma(torch.tensor([1.0], device=self.device))
        sigma_1_sq = sigmoid(gamma_1)
        mean1_sqr = (1.0 - sigma_1_sq)*(x**2)
        latent_loss = kl_std_normal(mean1_sqr, sigma_1_sq).sum(dim=(1,2,3)) * bpd_factor

        # recon_loss
        log_probs = self.log_probs_x_z0(x)
        # one-hot
        x_one_hot = torch.zeros((*x.shape, self.vocab_size), device=x.device)
        x_one_hot.scatter_(4, img_int.unsqueeze(-1), 1)
        log_probs = (x_one_hot*log_probs).sum(-1)
        recons_loss = -log_probs.sum(dim=(1,2,3)) * bpd_factor

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
        Returns p(x|z0) as logprobs of shape (B,C,H,W,vocab_size).
        Must pass either x or z_0 but not both.
        If x is passed, we sample z0 from q(z0|x).
        """
        gamma_0 = self.gamma(torch.tensor([0.0], device=self.device))
        if x is not None and z_0 is None:
            z_0_rescaled = x + exp(0.5*gamma_0)*torch.randn_like(x)
        elif z_0 is not None and x is None:
            z_0_rescaled = z_0 / sqrt(sigmoid(-gamma_0))
        else:
            raise ValueError("Must provide exactly one of x or z_0.")
        z_0_rescaled = z_0_rescaled.unsqueeze(-1)
        x_lim = 1.0 - 1.0/self.vocab_size
        x_values = linspace(-x_lim, x_lim, self.vocab_size, device=self.device)
        logits = -0.5*exp(-gamma_0)*(z_0_rescaled - x_values)**2
        return torch.log_softmax(logits, dim=-1)
    

 # ---------------------------------------------------------------------
# Additional KL helper
# ---------------------------------------------------------------------
def kl_std_normal(mean_squared, var):
    # 0.5*(var + mean_sq - log(var) -1)
    return 0.5*(var + mean_squared - torch.log(var.clamp(min=1e-15)) - 1.0)

# ---------------------------------------------------------------------
# get_model
# ---------------------------------------------------------------------
def get_model(args):
    """
    If --continuous_time, returns a ContinuousVDM with (fixed or learned) noise schedule.
    Otherwise, returns the discrete-time VDM.
    """
    if getattr(args, "continuous_time", False):
        noise_schedule_type = "learned_linear" if args.noise_schedule == "learned" else "fixed_linear"

        class CTConfig:
            pass
        cfg = CTConfig()
        cfg.noise_schedule = noise_schedule_type
        cfg.gamma_min = -5.0
        cfg.gamma_max = 1.0
        cfg.antithetic_time_sampling = True
        image_shape = (1, 28, 28)

        base_model = NoisePredictor(image_channels=1)
        return ContinuousVDM(base_model, cfg, image_shape)

    else:
        return VDM(
            timesteps=1000,
            gamma_min=-5.0,
            gamma_max=1.0,
            embedding_dim=32,
            layers=4,
            classes=11
        )
