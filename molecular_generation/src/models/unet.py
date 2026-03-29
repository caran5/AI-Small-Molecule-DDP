"""
U-Net architecture for molecular diffusion models.
"""

import torch
import torch.nn as nn
from typing import List, Optional
from .embeddings import TimeEmbedding, ConditionalBatchNorm


class ResidualBlock(nn.Module):
    """Residual block with time conditioning."""
    
    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        """
        Args:
            in_channels: Input channels
            out_channels: Output channels
            time_dim: Timestep embedding dimension
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Main path
        # Use min to avoid GroupNorm crash with small channel counts
        num_groups_1 = min(8, in_channels)
        self.norm1 = nn.GroupNorm(num_groups_1, in_channels)
        self.conv1 = nn.Linear(in_channels, out_channels)
        
        # Time conditioning
        self.time_emb = nn.Sequential(
            nn.Linear(time_dim, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels)
        )
        
        num_groups_2 = min(8, out_channels)
        self.norm2 = nn.GroupNorm(num_groups_2, out_channels)
        self.conv2 = nn.Linear(out_channels, out_channels)
        
        # Skip connection
        if in_channels != out_channels:
            self.skip_proj = nn.Linear(in_channels, out_channels)
        else:
            self.skip_proj = nn.Identity()
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input, shape (batch_size, n_atoms, in_channels)
            time_emb: Time embeddings, shape (batch_size, time_dim)
            
        Returns:
            Output, shape (batch_size, n_atoms, out_channels)
        """
        h = self.norm1(x.transpose(1, 2)).transpose(1, 2)
        h = self.conv1(h)
        
        # Add time conditioning
        time_shift = self.time_emb(time_emb)  # (batch_size, out_channels)
        h = h + time_shift.unsqueeze(1)  # Broadcast over atoms
        
        h = nn.SiLU()(h)
        h = self.norm2(h.transpose(1, 2)).transpose(1, 2)
        h = self.conv2(h)
        
        # Skip connection
        return h + self.skip_proj(x)


class AttentionBlock(nn.Module):
    """Multi-head self-attention over atoms."""
    
    def __init__(self, channels: int, num_heads: int = 4):
        """
        Args:
            channels: Number of channels
            num_heads: Number of attention heads
        """
        super().__init__()
        self.num_heads = num_heads
        assert channels % num_heads == 0
        
        num_groups = min(8, channels)
        self.norm = nn.GroupNorm(num_groups, channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.out_proj = nn.Linear(channels, channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input, shape (batch_size, n_atoms, channels)
            
        Returns:
            Output, shape (batch_size, n_atoms, channels)
        """
        batch_size, n_atoms, channels = x.shape
        
        # Normalize
        h = self.norm(x.transpose(1, 2)).transpose(1, 2)
        
        # Project to Q, K, V
        qkv = self.qkv(h)  # (batch_size, n_atoms, 3*channels)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head attention
        head_dim = channels // self.num_heads
        q = q.reshape(batch_size, n_atoms, self.num_heads, head_dim).transpose(1, 2)
        k = k.reshape(batch_size, n_atoms, self.num_heads, head_dim).transpose(1, 2)
        v = v.reshape(batch_size, n_atoms, self.num_heads, head_dim).transpose(1, 2)
        
        # Attention
        scale = head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-1, -2)) * scale
        attn = torch.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn, v)  # (batch_size, num_heads, n_atoms, head_dim)
        
        # Reshape back
        out = out.transpose(1, 2).reshape(batch_size, n_atoms, channels)
        out = self.out_proj(out)
        
        return out + x  # Skip connection


class UNetBlock(nn.Module):
    """Single U-Net block with residual and attention."""
    
    def __init__(self, channels: int, time_dim: int, use_attention: bool = True):
        """
        Args:
            channels: Number of channels
            time_dim: Timestep embedding dimension
            use_attention: Whether to include attention
        """
        super().__init__()
        self.res1 = ResidualBlock(channels, channels, time_dim)
        self.attn = AttentionBlock(channels) if use_attention else nn.Identity()
        self.res2 = ResidualBlock(channels, channels, time_dim)
        # Register SiLU as module for consistency
        self.silu = nn.SiLU()
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input, shape (batch_size, n_atoms, channels)
            time_emb: Time embeddings, shape (batch_size, time_dim)
            
        Returns:
            Output, shape (batch_size, n_atoms, channels)
        """
        x = self.res1(x, time_emb)
        x = self.attn(x)
        x = self.res2(x, time_emb)
        return x


class AttentionGate(nn.Module):
    """Attention gate for encoder-decoder skip connections."""
    
    def __init__(self, channels: int):
        """Skip connection attention gate."""
        super().__init__()
        num_groups = min(8, channels)
        self.norm = nn.GroupNorm(num_groups, channels)
        # Use max(1, channels // 2) to handle edge case of channels=1
        hidden_dim = max(1, channels // 2)
        self.gate = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, skip: torch.Tensor, decoder: torch.Tensor) -> torch.Tensor:
        """Gate skip connection with attention."""
        normalized = self.norm(skip.transpose(1, 2)).transpose(1, 2)
        # Compute gate weights per atom
        gate_weights = self.gate(normalized)  # (batch, n_atoms, 1)
        return skip * gate_weights + decoder


class SimpleUNet(nn.Module):
    """Scaled U-Net for molecular diffusion with attention gates and increased capacity."""
    
    def __init__(self, 
                 in_channels: int = 5,
                 out_channels: int = 5,
                 hidden_channels: int = 128,
                 time_dim: int = 128,
                 depth: int = 3,
                 dropout_rate: float = 0.1):
        """
        Args:
            in_channels: Input feature channels
            out_channels: Output channels
            hidden_channels: Hidden layer dimension (128 for increased capacity)
            time_dim: Timestep embedding dimension
            depth: Number of U-Net blocks (increased to 3)
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.input_dropout = nn.Dropout(dropout_rate)
        
        # Time embedding
        from .embeddings import SinusoidalPositionalEmbedding
        self.time_embed = SinusoidalPositionalEmbedding(128)
        
        # Encoder blocks with dropout
        self.encoder_blocks = nn.ModuleList([
            UNetBlock(hidden_channels, time_dim, use_attention=(i > 0))
            for i in range(depth)
        ])
        self.encoder_dropout = nn.ModuleList([nn.Dropout(dropout_rate) for _ in range(depth)])
        
        # Skip connection attention gates
        self.attention_gates = nn.ModuleList([
            AttentionGate(hidden_channels) for _ in range(depth)
        ])
        
        # Decoder blocks with dropout
        self.decoder_blocks = nn.ModuleList([
            UNetBlock(hidden_channels, time_dim, use_attention=(i > 0))
            for i in range(depth)
        ])
        self.decoder_dropout = nn.ModuleList([nn.Dropout(dropout_rate) for _ in range(depth)])
        
        # Output projection
        num_groups = min(8, hidden_channels)
        self.output_norm = nn.GroupNorm(num_groups, hidden_channels)
        self.output_proj = nn.Linear(hidden_channels, out_channels)
        self.output_silu = nn.SiLU()
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features, shape (batch_size, n_atoms, in_channels)
            t: Timestep embeddings, shape (batch_size, time_dim) or (batch_size,)
            
        Returns:
            Output, shape (batch_size, n_atoms, out_channels)
        """
        # Create time embedding if needed
        if t.dim() == 1:
            # Assume t contains timestep indices
            t = self.time_embed(t)
        
        # Project input
        h = self.input_proj(x)  # (batch_size, n_atoms, hidden_channels)
        h = self.input_dropout(h)
        
        # Encoder with skip connections and dropout
        skip_connections = []
        for i, block in enumerate(self.encoder_blocks):
            h = block(h, t)
            h = self.encoder_dropout[i](h)
            skip_connections.append(h)
        
        # Decoder with attention-gated skip connections and dropout
        for i, block in enumerate(self.decoder_blocks):
            skip = skip_connections[-(i+1)]
            # Use attention gate to combine skip and decoder features
            h = self.attention_gates[i](skip, h)
            h = block(h, t)
            h = self.decoder_dropout[i](h)
        
        # Output projection
        h = self.output_norm(h.transpose(1, 2)).transpose(1, 2)
        h = self.output_silu(h)
        out = self.output_proj(h)
        
        return out


class ConditionalUNet(SimpleUNet):
    """U-Net that conditions on molecular properties for property steering."""
    
    def __init__(self, 
                 in_channels: int = 5,
                 out_channels: int = 5,
                 hidden_channels: int = 128,
                 time_dim: int = 128,
                 depth: int = 3,
                 n_properties: int = 5,
                 dropout_rate: float = 0.1):
        """
        Args:
            in_channels: Input feature channels
            out_channels: Output channels
            hidden_channels: Hidden layer dimension
            time_dim: Timestep embedding dimension
            depth: Number of U-Net blocks
            n_properties: Number of molecular properties to condition on
            dropout_rate: Dropout rate
        """
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            time_dim=time_dim,
            depth=depth,
            dropout_rate=dropout_rate
        )
        
        self.n_properties = n_properties
        
        # Property encoder network
        self.property_encoder = nn.Sequential(
            nn.Linear(n_properties, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_channels, hidden_channels)
        )
        
        # Fusion layer: combine time and property embeddings
        self.fusion = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_channels, hidden_channels)
        )
    
    def forward(self, 
                x: torch.Tensor, 
                t: torch.Tensor,
                properties: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input features, shape (batch_size, n_atoms, in_channels)
            t: Timestep embeddings, shape (batch_size, time_dim) or (batch_size,)
            properties: Molecular properties, shape (batch_size, n_properties), optional
            
        Returns:
            Output, shape (batch_size, n_atoms, out_channels)
        """
        # Create time embedding if needed
        if t.dim() == 1:
            t = self.time_embed(t)  # (batch_size, time_dim)
        
        # Fuse time and property embeddings
        if properties is not None:
            prop_embed = self.property_encoder(properties)  # (batch_size, hidden_channels)
            combined = torch.cat([t, prop_embed], dim=1)  # (batch_size, hidden_channels*2)
            time_embed = self.fusion(combined)  # (batch_size, hidden_channels)
        else:
            time_embed = t
        
        # Project input
        h = self.input_proj(x)  # (batch_size, n_atoms, hidden_channels)
        h = self.input_dropout(h)
        
        # Encoder with skip connections and dropout
        skip_connections = []
        for i, block in enumerate(self.encoder_blocks):
            h = block(h, time_embed)
            h = self.encoder_dropout[i](h)
            skip_connections.append(h)
        
        # Decoder with attention-gated skip connections and dropout
        for i, block in enumerate(self.decoder_blocks):
            skip = skip_connections[-(i+1)]
            h = self.attention_gates[i](skip, h)
            h = block(h, time_embed)
            h = self.decoder_dropout[i](h)
        
        # Output projection
        h = self.output_norm(h.transpose(1, 2)).transpose(1, 2)
        h = self.output_silu(h)
        out = self.output_proj(h)
        
        return out
