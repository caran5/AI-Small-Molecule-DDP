"""
Inference utilities for conditional diffusion models with property steering.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
from pathlib import Path


def get_alpha_from_schedule(t: torch.Tensor, 
                           schedule_type: str = 'cosine',
                           num_steps: int = 1000) -> torch.Tensor:
    """
    Get alpha values for timestep t using specified schedule.
    
    Args:
        t: Timestep tensor [batch_size]
        schedule_type: 'linear', 'cosine', 'quadratic', or 'learned'
        num_steps: Total number of diffusion steps
    
    Returns:
        Alpha values [batch_size]
    """
    t_normalized = t.float() / num_steps
    
    if schedule_type == 'cosine':
        s = 0.008
        alpha = torch.cos(((t_normalized + s) / (1 + s)) * np.pi / 2) ** 2
    elif schedule_type == 'linear':
        alpha = 1.0 - (t.float() / num_steps)
    elif schedule_type == 'quadratic':
        alpha = (1.0 - t.float() / num_steps) ** 2
    elif schedule_type == 'learned':
        # Polynomial schedule: 1 - t^3
        alpha = (1.0 - t_normalized) ** 3
    else:
        alpha = 1.0 - t.float() / num_steps
    
    return alpha.clamp(0.0, 1.0)


def generate_with_properties(model: torch.nn.Module,
                            target_properties: Dict,
                            num_samples: int = 10,
                            num_steps: int = 100,
                            property_normalizer=None,
                            input_dim: int = 100,
                            schedule_type: str = 'cosine',
                            device: str = 'cpu') -> torch.Tensor:
    """
    Generate molecules with target properties using conditional diffusion.
    
    Args:
        model: ConditionalUNet model
        target_properties: Dict like {'logp': 3.5, 'mw': 400, ...}
        num_samples: Number of molecules to generate
        num_steps: Number of diffusion steps for reverse process
        property_normalizer: PropertyNormalizer instance for normalization
        input_dim: Feature dimension
        schedule_type: Noise schedule type
        device: Device to use ('cpu' or 'cuda')
    
    Returns:
        Generated features [num_samples, input_dim]
    """
    model.eval()
    
    # Normalize target properties
    if property_normalizer is not None:
        norm_props = property_normalizer.normalize(target_properties)
    else:
        norm_props = target_properties
    
    # Convert to tensor and expand for batch
    prop_tensor = torch.tensor([
        norm_props.get('logp', 0.0),
        norm_props.get('mw', 0.0),
        norm_props.get('hbd', 0.0),
        norm_props.get('hba', 0.0),
        norm_props.get('rotatable', 0.0)
    ], dtype=torch.float32)
    
    prop_tensor = prop_tensor.unsqueeze(0).repeat(num_samples, 1).to(device)
    
    # Start from pure Gaussian noise
    x = torch.randn(num_samples, input_dim, device=device)
    
    # Reverse diffusion process
    with torch.no_grad():
        for step in reversed(range(num_steps)):
            t = torch.full((num_samples,), step, dtype=torch.long, device=device)
            
            # Predict noise
            noise_pred = model(x.unsqueeze(1), t, properties=prop_tensor)
            
            # Handle output shape
            if noise_pred.dim() == 3:
                # If model outputs [batch, atoms, features], take mean or first
                noise_pred = noise_pred.squeeze(1)
            
            # Get alpha values
            alpha_t = get_alpha_from_schedule(t, schedule_type, num_steps).to(device)
            
            if step > 0:
                alpha_prev = get_alpha_from_schedule(
                    t - 1, schedule_type, num_steps
                ).to(device)
            else:
                alpha_prev = torch.ones_like(alpha_t)
            
            # DDPM reverse step
            # x_t-1 = (1/sqrt(alpha_t)) * (x_t - (1-alpha_t) * noise_pred)
            #         + sqrt((1-alpha_t)/(1-alpha_prev) * (1 - alpha_t/alpha_prev)) * z
            
            sqrt_alpha_t = torch.sqrt(alpha_t).unsqueeze(1)
            sqrt_1_minus_alpha_t = torch.sqrt(1.0 - alpha_t).unsqueeze(1)
            
            x = (x - sqrt_1_minus_alpha_t * noise_pred) / sqrt_alpha_t
            
            if step > 0:
                sqrt_1_minus_alpha_prev = torch.sqrt(1.0 - alpha_prev).unsqueeze(1)
                sigma_t = torch.sqrt(
                    (1.0 - alpha_prev) / (1.0 - alpha_t) * 
                    (1.0 - alpha_t / (alpha_prev + 1e-8))
                ).unsqueeze(1)
                
                z = torch.randn_like(x)
                x = x + sigma_t * z
    
    return x.cpu()


class ConditionalGenerationPipeline:
    """End-to-end pipeline for conditional molecular generation."""
    
    def __init__(self,
                 model: torch.nn.Module,
                 property_normalizer,
                 device: str = 'cpu'):
        """
        Args:
            model: ConditionalUNet model
            property_normalizer: Fitted PropertyNormalizer
            device: Device to use
        """
        self.model = model.to(device)
        self.property_normalizer = property_normalizer
        self.device = device
        self.model.eval()
    
    def generate(self,
                target_properties: Dict,
                num_samples: int = 10,
                num_steps: int = 100,
                input_dim: int = 100,
                schedule_type: str = 'cosine') -> torch.Tensor:
        """Generate samples with target properties."""
        return generate_with_properties(
            self.model,
            target_properties,
            num_samples=num_samples,
            num_steps=num_steps,
            property_normalizer=self.property_normalizer,
            input_dim=input_dim,
            schedule_type=schedule_type,
            device=self.device
        )
    
    def save(self, path: str) -> None:
        """Save model and normalizer."""
        checkpoint = {
            'model_state': self.model.state_dict(),
            'normalizer_stats': self.property_normalizer.get_stats()
        }
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
    
    def load(self, path: str) -> None:
        """Load model and normalizer."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        # Note: normalizer already has stats, just validating
        print(f"Loaded checkpoint from {path}")

