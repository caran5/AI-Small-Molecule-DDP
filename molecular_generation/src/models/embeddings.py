"""
Time and property embeddings for diffusion models.
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional encoding for timesteps."""
    
    def __init__(self, dim: int):
        """
        Args:
            dim: Embedding dimension (must be even and >= 2)
        """
        super().__init__()
        self.dim = dim
        assert dim % 2 == 0, "Dimension must be even"
        assert dim >= 2, "Dimension must be at least 2"
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Timesteps, shape (batch_size,)
            
        Returns:
            Embeddings, shape (batch_size, dim)
        """
        device = t.device
        half_dim = self.dim // 2
        
        # Frequency scaling: avoid division by zero for small dims
        emb = math.log(10000) / max(1, half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        
        # Apply sin/cos
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        
        return emb


class TimeEmbedding(nn.Module):
    """Project timestep embedding to model dimension."""
    
    def __init__(self, time_dim: int, model_dim: int):
        """
        Args:
            time_dim: Input embedding dimension
            model_dim: Output model dimension
        """
        super().__init__()
        self.emb = nn.Sequential(
            nn.Linear(time_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim)
        )
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Time embeddings, shape (batch_size, time_dim)
            
        Returns:
            Projected embeddings, shape (batch_size, model_dim)
        """
        return self.emb(t)


class MolecularPropertyEmbedding(nn.Module):
    """Embed molecular properties like n_atoms."""
    
    def __init__(self, max_atoms: int, embedding_dim: int):
        """
        Args:
            max_atoms: Maximum number of atoms in molecules
            embedding_dim: Embedding dimension
        """
        super().__init__()
        self.max_atoms = max_atoms
        self.embedding = nn.Embedding(max_atoms + 1, embedding_dim)
    
    def forward(self, n_atoms: torch.Tensor) -> torch.Tensor:
        """
        Args:
            n_atoms: Number of atoms per molecule, shape (batch_size,)
            
        Returns:
            Embeddings, shape (batch_size, embedding_dim)
        """
        # Bounds check: ensure n_atoms doesn't exceed max_atoms
        assert (n_atoms <= self.max_atoms).all(), \
            f"n_atoms contains values > {self.max_atoms}: max={n_atoms.max().item()}"
        return self.embedding(n_atoms)


class ConditionalBatchNorm(nn.Module):
    """Batch normalization conditioned on timestep embeddings."""
    
    def __init__(self, num_features: int, time_dim: int):
        """
        Args:
            num_features: Number of input features
            time_dim: Timestep embedding dimension
        """
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, num_features),
            nn.SiLU(),
            nn.Linear(num_features, num_features * 2)
        )
        
        # Initialize gamma and beta scales for stability
        # Gamma initialized near 0 for multiplicative stability
        # Beta initialized near 0 for additive stability
        with torch.no_grad():
            self.time_mlp[-1].weight.zero_()
            self.time_mlp[-1].bias.zero_()
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features, shape (batch_size, num_features, ...)
            time_emb: Time embeddings, shape (batch_size, time_dim)
            
        Returns:
            Conditioned features
        """
        gamma, beta = self.time_mlp(time_emb).chunk(2, dim=1)
        
        # Reshape for broadcasting
        if x.dim() > 2:
            gamma = gamma.view(gamma.shape[0], gamma.shape[1], *([1] * (x.dim() - 2)))
            beta = beta.view(beta.shape[0], beta.shape[1], *([1] * (x.dim() - 2)))
        
        return self.bn(x) * (1 + gamma) + beta
