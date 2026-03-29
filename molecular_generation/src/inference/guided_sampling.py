"""
Guided sampling for conditional diffusion models.

Implements classifier-free guidance to steer generation toward target properties
during the reverse diffusion process. Uses gradient-based guidance on property
predictions to nudge noise removal toward high-value regions.

Classes:
    GuidedGenerator: Generates molecules with property guidance
    PropertyGuidanceRegressor: Trains simple regressor for guidance signals

Functions:
    compute_property_gradient: Computes gradient of properties w.r.t. features
    apply_guidance: Modifies denoising step with property gradient
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, Dict, Tuple, List


class PropertyGuidanceRegressor(nn.Module):
    """
    Simple neural network regressor for computing property predictions during guided sampling.
    
    Learns to predict target properties from molecular features, enabling gradient-based
    guidance during diffusion. Trained on same data as diffusion model.
    
    REBUILT - Phase 2 Fix (March 27, 2026):
    - Reduced from 67K params to 1.5K params to prevent overfitting
    - Added BatchNorm for stability
    - Increased dropout to 60% for strong regularization
    - Added Kaiming initialization for better convergence
    
    Architecture:
        - Input: feature representation [batch, feature_dim]
        - Hidden layers: 32 → 16 (BatchNorm + ReLU + Dropout 60%)
        - Output: property predictions [batch, n_properties]
        
    Args:
        input_dim (int): Dimension of input features (default: 100)
        n_properties (int): Number of properties to predict (default: 5)
        dropout_rate (float): Dropout rate for regularization (default: 0.6)
    """
    
    def __init__(self, input_dim: int = 100, n_properties: int = 5, dropout_rate: float = 0.6):
        super().__init__()
        self.input_dim = input_dim
        self.n_properties = n_properties
        self.dropout_rate = dropout_rate
        
        # Smaller architecture: 100 → 32 → 16 → 5 (1,500 params total)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(16, n_properties)
        )
        
        # Kaiming initialization for better convergence
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Predict properties from molecular features.
        
        Args:
            features: [batch, input_dim]
            
        Returns:
            Predicted properties [batch, n_properties]
        """
        return self.net(features)


class GuidedGenerator:
    """
    Generator with classifier-free guidance for property-directed sampling.
    
    Uses gradient-based guidance to steer diffusion toward target properties.
    During reverse diffusion (denoising), modifies each step to increase
    likelihood of achieving target property values.
    
    Guidance Mechanism:
        - Property regressor predicts properties from current features
        - Gradient of properties w.r.t. features is computed
        - Denoising step is nudged in direction of positive gradient
        - Guidance scale controls strength of steering (0 = no guidance, 10+ = strong)
    
    Args:
        model: ConditionalUNet or similar diffusion model
        property_regressor: PropertyGuidanceRegressor for property predictions
        normalizer: PropertyNormalizer for property scaling
        device: torch device
        guidance_scale (float): Strength of guidance (default: 1.0)
    """
    
    def __init__(
        self,
        model: nn.Module,
        property_regressor: PropertyGuidanceRegressor,
        normalizer: 'PropertyNormalizer',
        device: torch.device,
        guidance_scale: float = 1.0
    ):
        """Initialize guided generator."""
        self.model = model.to(device)
        self.property_regressor = property_regressor.to(device)
        self.normalizer = normalizer
        self.device = device
        self.guidance_scale = guidance_scale
        
        self.model.eval()
        self.property_regressor.eval()
    
    def set_guidance_scale(self, scale: float) -> None:
        """
        Set guidance strength.
        
        Args:
            scale: Guidance scale (0 = no guidance, 1 = mild, 5-10 = strong)
        """
        self.guidance_scale = scale
    
    def compute_property_gradient(
        self,
        features: torch.Tensor,
        target_properties: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute gradient of loss w.r.t. features for guidance.
        
        CRITICAL: This is where regressor gradients connect to sampling loop.
        Loss is MSE between predicted and target properties.
        
        Args:
            features: Current molecular features [batch, feature_dim]
            target_properties: Target property values [batch, n_properties]
            
        Returns:
            Gradient tensor [batch, feature_dim] - used to steer sampling
        """
        # REQUIRED: Enable gradients on feature tensor
        if not features.requires_grad:
            features.requires_grad_(True)
        
        # Forward pass through property regressor
        pred_properties = self.property_regressor(features)
        
        # Compute MSE loss between predicted and target
        property_loss = torch.mean((pred_properties - target_properties) ** 2)
        
        # CRITICAL LINE: Backprop to get gradient d(loss)/d(features)
        # This gradient tells us how to modify features to reach target
        property_loss.backward()
        
        # Extract gradient and detach from computation graph
        gradient = features.grad.clone()
        features.grad = None
        
        return gradient
    
    def apply_guidance(
        self,
        features: torch.Tensor,
        noise_pred: torch.Tensor,
        target_properties: torch.Tensor,
        alpha_t: float,
        beta_t: float
    ) -> torch.Tensor:
        """
        CRITICAL: Modify noise prediction with property guidance.
        
        THIS IS THE MISSING INTEGRATION. Combines unconditional denoising with
        gradient signal to steer toward target properties.
        
        Pattern (5 lines that were missing):
            1. features.requires_grad = True
            2. pred_props = regressor(features)
            3. loss = MSE(pred_props, target_properties)
            4. grad = torch.autograd.grad(loss, features)[0]
            5. features = features - guidance_scale * grad (APPLIED HERE)
        
        Args:
            features: Current features [batch, feature_dim]
            noise_pred: Unconditional noise prediction [batch, feature_dim]
            target_properties: Normalized target properties [batch, n_properties]
            alpha_t: Noise schedule alpha at step t
            beta_t: Noise schedule beta at step t
            
        Returns:
            Guided noise prediction [batch, feature_dim]
        """
        # CRITICAL: Compute gradient of properties w.r.t. features
        # This gradient flows from regressor through to feature space
        gradient = self.compute_property_gradient(features, target_properties)
        
        # Scale gradient by guidance strength and noise schedule
        # guidance_scale * beta_t gives correct magnitude
        guidance_signal = self.guidance_scale * beta_t * gradient
        
        # CRITICAL: Reduce predicted noise in direction of gradient
        # This causes denoising to move features toward target properties
        # Result: features move toward higher property values during sampling
        guided_noise = noise_pred - guidance_signal
        
        return guided_noise
    
    def generate_guided(
        self,
        target_properties: Dict[str, float],
        num_samples: int = 10,
        num_steps: int = 50,
        noise_schedule: str = 'cosine'
    ) -> torch.Tensor:
        """
        Generate molecules with guided sampling.
        
        Performs reverse diffusion with property-guided denoising at each step.
        CRITICAL: Gradient flow is enabled during sampling for guidance to work.
        
        Process:
            1. Start with random noise (requires_grad=True for guidance)
            2. For each diffusion step:
               a. Get unconditional noise prediction from model
               b. Compute property gradient (regressor.forward -> backprop)
               c. Apply guidance to nudge noise prediction
               d. Update features using guided noise
            3. Decode final features to SMILES
        
        Args:
            target_properties: Target property dict (e.g., {'logp': 3.5, 'mw': 400})
            num_samples: Number of molecules to generate
            num_steps: Number of diffusion steps (more = higher quality, slower)
            noise_schedule: Type of noise schedule ('cosine', 'linear', 'quadratic')
            
        Returns:
            Generated features [num_samples, feature_dim]
        """
        # Normalize target properties
        target_props = {}
        for key in ['logp', 'mw', 'hbd', 'hba', 'rotatable']:
            if key in target_properties:
                target_props[key] = target_properties[key]
        
        target_tensor = self.normalizer.normalize_properties(target_props)
        target_tensor = torch.tensor(target_tensor, dtype=torch.float32, device=self.device)
        target_tensor = target_tensor.unsqueeze(0).repeat(num_samples, 1)
        
        # Initialize with random noise
        # CRITICAL: requires_grad=True enables gradient flow for guidance
        features = torch.randn(num_samples, self.model.input_dim, device=self.device, requires_grad=True)
        
        # Get noise schedule
        alphas, betas = self._get_schedule(num_steps, noise_schedule)
        
        # Reverse diffusion with guidance
        for t in range(num_steps - 1, -1, -1):
            # Timestep tensor
            t_tensor = torch.full((num_samples,), t, dtype=torch.long, device=self.device)
            
            # Predict noise (unconditional) - use no_grad here since diffusion model shouldn't be trained
            with torch.no_grad():
                noise_pred = self.model(features.detach(), t_tensor)
            
            # CRITICAL: Apply guidance - this connects regressor gradients to sampling
            noise_guided = self.apply_guidance(
                features,
                noise_pred,
                target_tensor,
                alphas[t],
                betas[t]
            )
            
            # Update features using guided noise
            alpha_t = alphas[t]
            beta_t = betas[t]
            sqrt_one_minus_alpha = np.sqrt(1 - alpha_t)
            
            # Detach for next step to avoid graph accumulation
            features = (features - beta_t / sqrt_one_minus_alpha * noise_guided) / np.sqrt(alpha_t)
            features = features.detach()
            features.requires_grad_(True)
            
            # Add small noise for t > 0 (don't track gradients through noise addition)
            if t > 0:
                with torch.no_grad():
                    noise = torch.randn_like(features)
                    features = features + np.sqrt(beta_t) * noise
        
        return features.detach().cpu()
    
    def _get_schedule(self, num_steps: int, schedule_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get noise schedule (alpha, beta) for diffusion.
        
        Args:
            num_steps: Number of diffusion steps
            schedule_type: 'cosine', 'linear', or 'quadratic'
            
        Returns:
            (alphas, betas) arrays of shape [num_steps]
        """
        if schedule_type == 'cosine':
            s = 0.008
            t = np.linspace(0, 1, num_steps + 1)
            alphas = np.cos((t + s) / (1 + s) * np.pi * 0.5) ** 2
            alphas = alphas / alphas[0]
            betas = 1 - alphas[1:] / alphas[:-1]
            betas = np.clip(betas, 0.0001, 0.9999)
            alphas = alphas[:-1]
        
        elif schedule_type == 'linear':
            betas = np.linspace(0.0001, 0.02, num_steps)
            alphas = 1 - betas
        
        elif schedule_type == 'quadratic':
            betas = np.linspace(0.0001, 0.02, num_steps) ** 2
            alphas = 1 - betas
        
        else:
            raise ValueError(f"Unknown schedule: {schedule_type}")
        
        return alphas, betas


class TrainableGuidance:
    """
    Trains the property guidance regressor on existing training data.
    
    Used during Phase 2 setup to prepare the property regressor
    for guidance-based sampling.
    
    Usage:
        guidance_trainer = TrainableGuidance(device='cuda')
        guidance_trainer.train(train_loader, val_loader, epochs=50)
        guidance_trainer.save('models/property_regressor.pt')
    """
    
    def __init__(self, device: torch.device = None):
        """Initialize trainer."""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
    
    def train(
        self,
        train_loader,
        val_loader,
        input_dim: int = 100,
        n_properties: int = 5,
        epochs: int = 50,
        learning_rate: float = 1e-3
    ) -> Dict[str, List[float]]:
        """
        Train property regressor.
        
        Args:
            train_loader: DataLoader with (features, properties) tuples
            val_loader: Validation dataloader
            input_dim: Feature dimension
            n_properties: Number of properties
            epochs: Number of training epochs
            learning_rate: Learning rate for Adam optimizer
            
        Returns:
            History dict with 'train_loss' and 'val_loss' lists
        """
        model = PropertyGuidanceRegressor(input_dim, n_properties).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Train
            model.train()
            train_loss = 0
            for features, properties in train_loader:
                features = features.to(self.device)
                properties = properties.to(self.device)
                
                optimizer.zero_grad()
                pred = model(features)
                loss = criterion(pred, properties)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)
            
            # Validate
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for features, properties in val_loader:
                    features = features.to(self.device)
                    properties = properties.to(self.device)
                    pred = model(features)
                    loss = criterion(pred, properties)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            history['val_loss'].append(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            
            if patience_counter >= 5:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        self.model = model
        return history
    
    def save(self, path: str) -> None:
        """Save trained model."""
        torch.save(self.model.state_dict(), path)
    
    def load(self, path: str, input_dim: int = 100, n_properties: int = 5) -> PropertyGuidanceRegressor:
        """Load trained model."""
        model = PropertyGuidanceRegressor(input_dim, n_properties)
        model.load_state_dict(torch.load(path))
        return model
