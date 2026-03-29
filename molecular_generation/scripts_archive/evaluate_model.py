#!/usr/bin/env python3
"""
Comprehensive Model Evaluation Script

Evaluates the trained diffusion model on:
1. Generation validity & diversity
2. Property metrics
3. Model introspection
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List
import json
from datetime import datetime

# Project imports
from src.models.unet import SimpleUNet
from src.models.diffusion import NoiseScheduler
from src.inference.decoder import MolecularDecoder
from src.eval.metrics import (
    chemical_validity,
    diversity_metric,
    property_fidelity,
)


class ModelEvaluator:
    """Comprehensive evaluation of diffusion model."""
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.device = device
        self.model_path = model_path
        
        # Load model
        print("Loading model...")
        self.model = self._load_model(model_path)
        self.noise_scheduler = NoiseScheduler()
        self.decoder = MolecularDecoder()
        
    def _load_model(self, model_path: str) -> SimpleUNet:
        """Load trained model checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Determine model dimensions from checkpoint
        sample_key = list(checkpoint.keys())[0]
        
        model = SimpleUNet(
            in_channels=5,
            out_channels=5,
            hidden_channels=128,
            time_dim=128,
            depth=2,
            dropout_rate=0.1
        ).to(self.device)
        
        # Handle checkpoint format with 'unet.' prefix
        if isinstance(checkpoint, dict):
            # If checkpoint has 'unet.' prefix, strip it
            state_dict = {}
            for k, v in checkpoint.items():
                if k.startswith('unet.'):
                    state_dict[k[5:]] = v
                else:
                    state_dict[k] = v
            
            # Try loading
            try:
                model.load_state_dict(state_dict)
            except RuntimeError as e:
                print(f"Warning: Could not load full state dict: {e}")
                # Try partial loading
                model_state = model.state_dict()
                for k in state_dict.keys():
                    if k in model_state:
                        model_state[k] = state_dict[k]
                model.load_state_dict(model_state)
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        print(f"✓ Model loaded from {model_path}")
        return model
    
    def evaluate_unconditional_generation(self, num_samples: int = 50) -> Dict:
        """Evaluate unconditional generation."""
        print(f"\n{'='*60}")
        print("UNCONDITIONAL GENERATION EVALUATION")
        print(f"{'='*60}")
        print(f"Generating {num_samples} molecules...")
        
        with torch.no_grad():
            # Sample random noise (batch, features)
            x = torch.randn(num_samples, 5, device=self.device)
            
            # Generate features - forward pass through model
            # Model expects (batch, atoms, features)
            t = torch.full((num_samples,), 0, dtype=torch.long, device=self.device)
            generated = self.model(x.unsqueeze(1), t)
            
            # Squeeze back to (batch, features)
            if generated.dim() == 3:
                generated = generated.squeeze(1)
            
            # Decode to SMILES
            smiles_list = self.decoder.decode_batch(generated.cpu().numpy())
        
        # Calculate metrics
        validity = chemical_validity(smiles_list)
        print(f"✓ Validity: {validity['validity']*100:.1f}% ({validity['valid_count']}/{validity['total_count']})")
        
        # Filter valid molecules for diversity calculation
        valid_smiles = [s for s in smiles_list if s is not None]
        
        diversity = 0.0
        unique_count = 0
        if len(valid_smiles) > 1:
            try:
                # Calculate diversity using Tanimoto similarity
                from rdkit import Chem
                from rdkit.Chem import AllChem
                fps = []
                for s in valid_smiles:
                    try:
                        mol = Chem.MolFromSmiles(s)
                        if mol:
                            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                            fps.append(fp)
                    except:
                        pass
                
                if len(fps) > 1:
                    similarities = []
                    for i in range(len(fps)):
                        for j in range(i+1, len(fps)):
                            intersection = sum(fps[i] & fps[j])
                            union = sum(fps[i] | fps[j])
                            if union > 0:
                                tanimoto = 1 - (intersection / union)
                                similarities.append(tanimoto)
                    
                    diversity = np.mean(similarities) if similarities else 0.0
                    print(f"✓ Diversity (Tanimoto distance): {diversity:.3f}")
            except Exception as e:
                print(f"⚠ Could not compute diversity: {e}")
        
        unique_count = len(set(valid_smiles))
        print(f"✓ Unique molecules: {unique_count}/{num_samples}")
        
        results = {
            'num_samples': num_samples,
            'validity': validity['validity'],
            'valid_count': validity['valid_count'],
            'diversity': diversity,
            'unique_smiles': unique_count
        }
        
        return results
    
    def evaluate_property_statistics(self, num_samples: int = 50) -> Dict:
        """Evaluate property distribution in generated molecules."""
        print(f"\n{'='*60}")
        print("PROPERTY STATISTICS EVALUATION")
        print(f"{'='*60}")
        print(f"Analyzing properties of {num_samples} molecules...")
        
        with torch.no_grad():
            x = torch.randn(num_samples, 5, device=self.device)
            t = torch.full((num_samples,), 0, dtype=torch.long, device=self.device)
            generated = self.model(x, t)
            smiles_list = self.decoder.decode_batch(generated.cpu().numpy())
        
        validity = chemical_validity(smiles_list)
        valid_smiles = [s for s in smiles_list if s is not None]
        
        if not valid_smiles:
            print("✗ No valid molecules generated")
            return {'status': 'no_valid_molecules'}
        
        print(f"✓ Valid molecules: {len(valid_smiles)}/{num_samples}")
        
        # Compute property statistics
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, Crippen, Lipinski
            
            properties = {
                'logp': [],
                'mw': [],
                'hbd': [],
                'hba': [],
                'rotatable': []
            }
            
            for smiles in valid_smiles:
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        properties['logp'].append(float(Crippen.MolLogP(mol)))
                        properties['mw'].append(float(Descriptors.MolWt(mol)))
                        properties['hbd'].append(float(Lipinski.NumHDonors(mol)))
                        properties['hba'].append(float(Lipinski.NumHAcceptors(mol)))
                        properties['rotatable'].append(float(Descriptors.NumRotatableBonds(mol)))
                except:
                    pass
            
            # Print statistics
            results = {'status': 'success'}
            for prop, values in properties.items():
                if values:
                    results[prop] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values))
                    }
                    print(f"\n{prop.upper()}:")
                    print(f"  Mean: {results[prop]['mean']:.2f} ± {results[prop]['std']:.2f}")
                    print(f"  Range: [{results[prop]['min']:.2f}, {results[prop]['max']:.2f}]")
            
            return results
        except Exception as e:
            print(f"⚠ Error computing properties: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def evaluate_generation_speed(self, num_samples: int = 10) -> Dict:
        """Evaluate generation speed."""
        print(f"\n{'='*60}")
        print("GENERATION SPEED EVALUATION")
        print(f"{'='*60}")
        print(f"Testing generation speed with {num_samples} samples...")
        
        import time
        
        with torch.no_grad():
            x = torch.randn(num_samples, 5, device=self.device)
            t = torch.full((num_samples,), 0, dtype=torch.long, device=self.device)
            
            start = time.time()
            generated = self.model(x, t)
            forward_time = time.time() - start
            
            start = time.time()
            smiles_list = self.decoder.decode_batch(generated.cpu().numpy())
            decode_time = time.time() - start
        
        total_time = forward_time + decode_time
        speed_per_sample = total_time / num_samples
        
        print(f"✓ Forward pass: {forward_time:.3f}s ({forward_time/num_samples:.3f}s/sample)")
        print(f"✓ Decoding: {decode_time:.3f}s ({decode_time/num_samples:.3f}s/sample)")
        print(f"✓ Total: {total_time:.3f}s ({speed_per_sample:.3f}s/sample)")
        
        results = {
            'num_samples': num_samples,
            'forward_time': forward_time,
            'decode_time': decode_time,
            'total_time': total_time,
            'ms_per_sample': speed_per_sample * 1000
        }
        
        return results
    
    def get_model_info(self) -> Dict:
        """Get model architecture info."""
        print(f"\n{'='*60}")
        print("MODEL INFORMATION")
        print(f"{'='*60}")
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"✓ Total parameters: {total_params:,}")
        print(f"✓ Trainable parameters: {trainable_params:,}")
        print(f"✓ Model device: {self.device}")
        print(f"✓ Model dtype: {next(self.model.parameters()).dtype}")
        
        results = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_class': self.model.__class__.__name__,
            'device': str(self.device)
        }
        
        return results
    
    def run_full_evaluation(self) -> Dict:
        """Run all evaluation tests."""
        print("\n" + "="*60)
        print("DIFFUSION MODEL COMPREHENSIVE EVALUATION")
        print("="*60)
        print(f"Model: {self.model_path}")
        print(f"Device: {self.device}")
        print(f"Timestamp: {datetime.now()}")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'model_path': self.model_path,
            'device': self.device,
            'evaluations': {}
        }
        
        # Run evaluations
        results['evaluations']['model_info'] = self.get_model_info()
        results['evaluations']['unconditional'] = self.evaluate_unconditional_generation(num_samples=50)
        results['evaluations']['properties'] = self.evaluate_property_statistics(num_samples=50)
        results['evaluations']['speed'] = self.evaluate_generation_speed(num_samples=10)
        
        # Summary
        print(f"\n{'='*60}")
        print("EVALUATION COMPLETE")
        print(f"{'='*60}")
        
        # Save results
        output_file = Path('evaluation_results.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n✓ Results saved to {output_file}")
        
        return results


def main():
    """Run evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate diffusion model')
    parser.add_argument('--model', type=str, default='improved_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device (cpu or cuda)')
    
    args = parser.parse_args()
    
    evaluator = ModelEvaluator(model_path=args.model, device=args.device)
    results = evaluator.run_full_evaluation()
    
    return results


if __name__ == '__main__':
    main()
