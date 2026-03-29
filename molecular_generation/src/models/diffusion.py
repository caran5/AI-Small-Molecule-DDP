"""
Core diffusion model implementation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple
from .unet import SimpleUNet
from .embeddings import SinusoidalPositionalEmbedding


class NoiseScheduler:
    """Noise scheduling for diffusion process."""
    
    def __init__(self, 
                 num_timesteps: int = 1000,
                 beta_start: float = 0.0001,
                 beta_end: float = 0.02,
                 schedule: str = 'linear'):
        """
        Args:
            num_timesteps: Number of diffusion steps
            beta_start: Starting beta value
            beta_end: Ending beta value
            schedule: 'linear', 'quadratic', or 'cosine'
        """
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.schedule = schedule
        
        # Compute noise schedule
        if schedule == 'linear':
            betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule == 'quadratic':
            betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_timesteps) ** 2
        elif schedule == 'cosine':
            s = 0.008
            steps = torch.arange(num_timesteps + 1)
            alphas_cumprod = torch.cos(((steps / num_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, 0.0001, 0.9999)
        elif schedule == 'learned':
            # Learned schedule: start fast decay, then smooth out
            t = torch.linspace(0, 1, num_timesteps)
            # Smooth polynomial schedule: accelerates early, plateaus late
            betas = beta_start + (beta_end - beta_start) * (1 - (1 - t) ** 3)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
        
        # Precompute useful values
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        
        # Store schedule type for reference
        self.schedule_type = schedule
        
        # Precompute for q(x_t | x_0)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod))
        
        # Precompute for q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        posterior_log_variance_clipped = torch.log(torch.clamp(posterior_variance, min=1e-20))
        posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1 - alphas_cumprod)
        posterior_mean_coef2 = (1 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1 - alphas_cumprod)
        
        # Precompute for reparameterized sampling
        sqrt_recip_alphas_cumprod = torch.sqrt(1 / alphas_cumprod)
        sqrt_recip_m1_alphas_cumprod = torch.sqrt(1 / alphas_cumprod - 1)
        
        # Use PyTorch's native register_buffer (ensures GPU/device transfer)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped', posterior_log_variance_clipped)
        self.register_buffer('posterior_mean_coef1', posterior_mean_coef1)
        self.register_buffer('posterior_mean_coef2', posterior_mean_coef2)
        self.register_buffer('sqrt_recip_alphas_cumprod', sqrt_recip_alphas_cumprod)
        self.register_buffer('sqrt_recip_m1_alphas_cumprod', sqrt_recip_m1_alphas_cumprod)
    
    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Forward diffusion: q(x_t | x_0).
        
        Args:
            x_0: Clean data, shape (batch_size, n_atoms, features)
            t: Timesteps, shape (batch_size,)
            noise: Gaussian noise, shape (batch_size, n_atoms, features)
            
        Returns:
            Noisy data x_t, shape (batch_size, n_atoms, features)
        """
        device = x_0.device
        t_cpu = t.cpu()  # Move timesteps to CPU for indexing
        sqrt_alphas = self.sqrt_alphas_cumprod[t_cpu].to(device)  # (batch_size,)
        sqrt_one_minus_alphas = self.sqrt_one_minus_alphas_cumprod[t_cpu].to(device)
        
        # Reshape for broadcasting
        while sqrt_alphas.dim() < x_0.dim():
            sqrt_alphas = sqrt_alphas.unsqueeze(-1)
            sqrt_one_minus_alphas = sqrt_one_minus_alphas.unsqueeze(-1)
        
        return sqrt_alphas * x_0 + sqrt_one_minus_alphas * noise
    
    def get_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample random timesteps."""
        return torch.randint(0, self.num_timesteps, (batch_size,), device=device)
    
    def get_schedule_info(self) -> dict:
        """Get schedule information for logging/visualization."""
        return {
            'schedule': self.schedule,
            'num_timesteps': self.num_timesteps,
            'beta_start': self.beta_start,
            'beta_end': self.beta_end,
            'alphas_cumprod': self.alphas_cumprod.cpu().numpy(),
            'betas': self.betas.cpu().numpy()
        }


class DiffusionModel(nn.Module):
    """Molecular diffusion model."""
    
    def __init__(self,
                 in_channels: int = 5,
                 num_timesteps: int = 1000,
                 beta_start: float = 0.0001,
                 beta_end: float = 0.02,
                 schedule: str = 'linear',
                 max_atoms: int = 128,
                 unet_channels: list = None):
        """
        Args:
            in_channels: Input feature channels
            num_timesteps: Number of diffusion steps
            beta_start: Starting beta value
            beta_end: Ending beta value
            schedule: Noise schedule type
            max_atoms: Maximum atoms per molecule
            unet_channels: Channel dimensions for U-Net
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.num_timesteps = num_timesteps
        self.max_atoms = max_atoms
        
        # Noise scheduler
        self.noise_scheduler = NoiseScheduler(
            num_timesteps=num_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            schedule=schedule
        )
        
        # Time embedding
        self.time_embed = SinusoidalPositionalEmbedding(128)
        
        # U-Net
        self.unet = SimpleUNet(
            in_channels=in_channels,
            out_channels=in_channels,
            hidden_channels=unet_channels[0] if unet_channels else 64,
            time_dim=128,
            depth=2
        )
    
    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Predict noise/score from noisy data.
        
        Args:
            x_t: Noisy data, shape (batch_size, n_atoms, in_channels)
            t: Timesteps, shape (batch_size,)
            
        Returns:
            Predicted noise, shape (batch_size, n_atoms, in_channels)
        """
        # Embed timesteps
        time_emb = self.time_embed(t)
        
        # Predict noise
        noise_pred = self.unet(x_t, time_emb)
        
        return noise_pred
    
    def diffuse(self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add noise to data (forward process).
        
        Args:
            x_0: Clean data
            t: Timesteps
            noise: Optional pre-generated noise
            
        Returns:
            Noisy data and noise used
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        x_t = self.noise_scheduler.q_sample(x_0, t, noise)
        return x_t, noise
    
    @torch.no_grad()
    def sample(self, batch_size: int, device: torch.device, n_atoms: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate samples via reverse diffusion using proper DDPM reparameterization.
        
        Args:
            batch_size: Number of samples
            device: Device to generate on
            n_atoms: Optional tensor of shape (batch_size,) specifying atoms per sample.
                     If provided, atoms beyond n_atoms will be masked to zero.
            
        Returns:
            Generated samples, shape (batch_size, max_atoms, in_channels)
        """
        # Start from pure noise
        x_t = torch.randn(batch_size, self.max_atoms, self.in_channels, device=device)
        
        # Reverse diffusion
        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Predict noise
            noise_pred = self.forward(x_t, t_batch)
            
            # Get noise schedule values (ensure they're on correct device)
            sqrt_recip_at = self.noise_scheduler.sqrt_recip_alphas_cumprod[t].to(device)
            sqrt_recip_m1_at = self.noise_scheduler.sqrt_recip_m1_alphas_cumprod[t].to(device)
            
            if t > 0:
                # Compute posterior variance
                posterior_var = self.noise_scheduler.posterior_variance[t].to(device)
                sigma_t = posterior_var.sqrt()
                
                # Sample noise for this step
                z = torch.randn_like(x_t)
                
                # Reparameterized form (proper DDPM):
                # x_{t-1} = sqrt(1/alpha_t) * x_t - sqrt(1/alpha_t - 1) * noise_pred + sigma_t * z
                # This is equivalent to the posterior mean + noise, but numerically stable
                x_t = sqrt_recip_at * x_t - sqrt_recip_m1_at * noise_pred + sigma_t * z
            else:
                # Last step: deterministic, no noise added
                x_t = sqrt_recip_at * x_t - sqrt_recip_m1_at * noise_pred
        
        # Apply masking if n_atoms provided
        if n_atoms is not None:
            mask = torch.arange(self.max_atoms, device=device).unsqueeze(0) < n_atoms.unsqueeze(1)
            x_t = x_t * mask.unsqueeze(-1).float()
        
        return x_t
    
    def get_loss(self, x_0: torch.Tensor, noise: Optional[torch.Tensor] = None, n_atoms: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute diffusion loss with optional masking for variable-length molecules.
        
        Args:
            x_0: Clean data, shape (batch_size, max_atoms, in_channels)
            noise: Optional pre-generated noise
            n_atoms: Optional tensor of shape (batch_size,) specifying valid atoms per sample.
                     Loss will only be computed for valid atoms.
            
        Returns:
            Mean squared error loss
        """
        batch_size = x_0.shape[0]
        device = x_0.device
        
        # Sample random timesteps
        t = self.noise_scheduler.get_timesteps(batch_size, device)
        
        # Generate noise
        if noise is None:
            noise = torch.randn_like(x_0)
        
        # Add noise to data
        x_t, _ = self.diffuse(x_0, t, noise)
        
        # Predict noise
        noise_pred = self.forward(x_t, t)
        
        # Apply masking if n_atoms provided
        if n_atoms is not None:
            mask = torch.arange(x_0.shape[1], device=device).unsqueeze(0) < n_atoms.unsqueeze(1)
            mask = mask.unsqueeze(-1).float()  # (batch_size, max_atoms, 1)
            noise_pred = noise_pred * mask
            noise = noise * mask
        
        # MSE loss
        loss = torch.nn.functional.mse_loss(noise_pred, noise, reduction='mean')
        
        return loss
