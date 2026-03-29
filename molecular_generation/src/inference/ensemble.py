"""
Ensemble model inference and training.
Provides uncertainty quantification through model disagreement.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from src.models.unet import ConditionalUNet
from src.inference.generate import generate_with_properties


class EnsembleModel:
    """Ensemble of conditional models for uncertainty quantification."""
    
    def __init__(self, checkpoint_paths: List[str], device: str = 'cpu'):
        """
        Load multiple models into ensemble.
        
        Args:
            checkpoint_paths: List of paths to model checkpoints
            device: Device to use ('cpu' or 'cuda')
        """
        self.models = []
        self.device = device
        self.n_models = len(checkpoint_paths)
        
        print(f"Loading {self.n_models} models for ensemble...")
        
        for i, path in enumerate(checkpoint_paths):
            model = ConditionalUNet(
                in_channels=100,
                out_channels=100,
                hidden_channels=128,
                depth=3,
                n_properties=5
            )
            
            state_dict = torch.load(path, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            
            self.models.append(model)
            print(f"  ✓ Loaded model {i+1}/{self.n_models}")
    
    def generate(self,
                target_properties: Dict,
                num_samples: int = 10,
                num_steps: int = 100,
                input_dim: int = 100,
                property_normalizer=None,
                schedule_type: str = 'cosine') -> Dict:
        """
        Generate samples from all ensemble models.
        
        Args:
            target_properties: Target properties dict
            num_samples: Number of samples per model
            num_steps: Diffusion steps
            input_dim: Feature dimension
            property_normalizer: Property normalizer
            schedule_type: Noise schedule
        
        Returns:
            Dict with 'mean', 'std', and 'all' tensors
        """
        all_samples = []
        
        print(f"Generating {num_samples} samples from {self.n_models} models...")
        
        for i, model in enumerate(self.models):
            samples = generate_with_properties(
                model,
                target_properties,
                num_samples=num_samples,
                num_steps=num_steps,
                property_normalizer=property_normalizer,
                input_dim=input_dim,
                schedule_type=schedule_type,
                device=self.device
            )
            all_samples.append(samples)
            print(f"  ✓ Model {i+1}/{self.n_models} generated")
        
        # Stack samples: [n_models, num_samples, input_dim]
        all_samples = torch.stack(all_samples)
        
        # Compute statistics
        samples_mean = all_samples.mean(dim=0)
        samples_std = all_samples.std(dim=0)
        
        return {
            'mean': samples_mean,
            'std': samples_std,
            'all': all_samples,
            'n_models': self.n_models
        }
    
    def filter_by_confidence(self,
                            results: Dict,
                            threshold: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Filter samples by ensemble uncertainty (std).
        Lower std = higher confidence.
        
        Args:
            results: Output from generate()
            threshold: Maximum allowed std (per dimension)
        
        Returns:
            (filtered_samples, confidence, mask)
        """
        # Max std across dimensions for each sample
        max_std = results['std'].max(dim=1)[0]
        
        # Create mask for confident samples
        mask = max_std < threshold
        
        filtered = results['mean'][mask]
        confidence = results['std'][mask]
        
        print(f"Filtered {mask.sum().item()}/{len(mask)} samples "
              f"(confidence threshold={threshold})")
        
        return filtered, confidence, mask


def train_ensemble(
    train_loader,
    val_loader,
    n_models: int = 3,
    input_dim: int = 100,
    hidden_dim: int = 128,
    depth: int = 3,
    n_properties: int = 5,
    epochs: int = 20,
    lr: float = 1e-3,
    early_stopping_patience: int = 5,
    device: str = 'cpu',
    save_dir: str = 'checkpoints/ensemble/'
) -> Tuple[List[str], List[Dict]]:
    """
    Train multiple independent models for ensemble.
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        n_models: Number of models to train
        input_dim: Feature dimension
        hidden_dim: Hidden dimension
        depth: Network depth
        n_properties: Number of properties
        epochs: Training epochs
        lr: Learning rate
        early_stopping_patience: Early stopping patience
        device: Device to use
        save_dir: Directory for saving models
    
    Returns:
        (checkpoint_paths, metrics_list)
    """
    from scripts.train_conditional import ConditionalTrainer
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    checkpoints = []
    metrics_all = []
    
    print("\n" + "="*70)
    print(f"Training Ensemble: {n_models} Independent Models")
    print("="*70)
    
    for model_idx in range(n_models):
        print(f"\n{'─'*70}")
        print(f"Training Model {model_idx + 1}/{n_models}")
        print(f"{'─'*70}")
        
        # Different seed for each model
        seed = 42 + model_idx
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Create fresh model
        model = ConditionalUNet(
            in_channels=input_dim,
            out_channels=input_dim,
            hidden_channels=hidden_dim,
            depth=depth,
            n_properties=n_properties,
            dropout_rate=0.1
        )
        
        # Train
        trainer = ConditionalTrainer(
            model,
            device=device,
            lr=lr,
            schedule_type='cosine',
            save_dir=str(save_path / f"model_{model_idx}")
        )
        
        metrics = trainer.train(
            train_loader,
            val_loader,
            epochs=epochs,
            early_stopping_patience=early_stopping_patience
        )
        
        # Save
        checkpoint_path = save_path / f"model_{model_idx}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        checkpoints.append(str(checkpoint_path))
        metrics_all.append(metrics)
        
        print(f"✓ Saved to {checkpoint_path}")
        print(f"  Best val loss: {metrics['best_val_loss']:.4f}")
        print(f"  Final epoch: {metrics['final_epoch']}")
    
    print("\n" + "="*70)
    print(f"✓ Ensemble Training Complete")
    print("="*70)
    
    return checkpoints, metrics_all


if __name__ == '__main__':
    print("Ensemble module loaded successfully.")
    print("Use train_ensemble() to train or EnsembleModel() to load ensemble.")

