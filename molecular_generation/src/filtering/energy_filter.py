"""
Energy-based filtering for 3D molecular conformations.

Generates 3D conformations from SMILES and filters out strained/implausible
molecules using MMFF94 force field. Identifies molecules with unfavorable
geometry, steric clashes, or high-energy conformations.

Classes:
    ConformationFilter: Generates 3D structures and computes energies
    EnergyResults: Stores energy filtering results

Functions:
    compute_mmff94_energy: Calculate MMFF94 energy for a molecule
    filter_by_energy: Filter molecules by energy threshold
    identify_strained: Identify strained/implausible molecules
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import warnings


class EnergyResults:
    """
    Container for energy filtering results.
    
    Stores energy values, validity flags, and filtering decisions for
    a batch of molecules.
    
    Attributes:
        smiles_list (List[str]): Input SMILES
        energies (np.ndarray): MMFF94 energies [batch]
        valid_3d (np.ndarray): Successfully generated 3D [batch] (bool)
        passes_filter (np.ndarray): Passes energy filter [batch] (bool)
        strain_indicators (np.ndarray): Strain scores [batch]
        summary (Dict): Filtering statistics
    """
    
    def __init__(self):
        """Initialize empty results container."""
        self.smiles_list = []
        self.energies = None
        self.valid_3d = None
        self.passes_filter = None
        self.strain_indicators = None
        self.summary = {}
    
    def add_molecule(self, smiles: str, energy: float, valid: bool, passes: bool, strain: float):
        """Add result for one molecule."""
        self.smiles_list.append(smiles)
        if self.energies is None:
            self.energies = []
            self.valid_3d = []
            self.passes_filter = []
            self.strain_indicators = []
        
        self.energies.append(energy)
        self.valid_3d.append(valid)
        self.passes_filter.append(passes)
        self.strain_indicators.append(strain)
    
    def finalize(self):
        """Convert lists to numpy arrays."""
        if self.energies is not None:
            self.energies = np.array(self.energies)
            self.valid_3d = np.array(self.valid_3d)
            self.passes_filter = np.array(self.passes_filter)
            self.strain_indicators = np.array(self.strain_indicators)
            
            # Compute summary statistics
            self.summary = {
                'total_molecules': len(self.smiles_list),
                'valid_3d': int(np.sum(self.valid_3d)),
                'passes_filter': int(np.sum(self.passes_filter)),
                'mean_energy': float(np.mean(self.energies[self.valid_3d])) if np.any(self.valid_3d) else np.nan,
                'median_energy': float(np.median(self.energies[self.valid_3d])) if np.any(self.valid_3d) else np.nan,
                'max_energy': float(np.max(self.energies[self.valid_3d])) if np.any(self.valid_3d) else np.nan,
                'min_energy': float(np.min(self.energies[self.valid_3d])) if np.any(self.valid_3d) else np.nan,
                'mean_strain': float(np.mean(self.strain_indicators[self.valid_3d])) if np.any(self.valid_3d) else np.nan,
            }


class ConformationFilter:
    """
    Filter molecules based on 3D conformation energy.
    
    Generates 3D coordinates from 2D SMILES using RDKit distance geometry,
    optimizes geometry with MMFF94 force field, and identifies strained
    molecules by energy thresholding.
    
    Strategy:
        1. Parse SMILES and add hydrogens
        2. Generate 3D coordinates (distance geometry)
        3. Optimize with MMFF94 force field
        4. Compute energy and strain indicators
        5. Filter by energy threshold
    
    Energy Thresholds:
        - Healthy: <50 kcal/mol (relaxed geometry)
        - Warning: 50-100 kcal/mol (moderate strain)
        - High strain: >100 kcal/mol (implausible/strained)
        - Reject: molecules failing 3D generation or optimization
    
    Strain Indicators:
        - MMFF94 energy (main)
        - Steric clashes (close non-bonded atoms)
        - Unfavorable angles/torsions
    
    Args:
        energy_threshold (float): Max acceptable energy (kcal/mol), default 100
        num_conformers (int): Number of conformations to try, default 5
        use_random_coords (bool): Use random vs distance geometry, default False
    """
    
    def __init__(
        self,
        energy_threshold: float = 100.0,
        num_conformers: int = 5,
        use_random_coords: bool = False
    ):
        """Initialize filter."""
        self.energy_threshold = energy_threshold
        self.num_conformers = num_conformers
        self.use_random_coords = use_random_coords
    
    def set_energy_threshold(self, threshold: float) -> None:
        """Update energy threshold."""
        self.energy_threshold = threshold
    
    def _generate_3d_conformation(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        """
        Generate 3D conformation for molecule.
        
        Uses distance geometry (recommended) or random coordinates to
        initialize 3D positions, then optimizes with MMFF94.
        
        Args:
            mol: RDKit molecule with hydrogens
            
        Returns:
            Molecule with 3D coordinates, or None if generation fails
        """
        try:
            # Add hydrogens
            mol = Chem.AddHs(mol)
            
            # Try multiple conformations and keep best energy one
            best_energy = float('inf')
            best_conf_id = None
            
            for _ in range(self.num_conformers):
                try:
                    if self.use_random_coords:
                        # Random coordinates
                        conf = Chem.Conformer(mol.GetNumAtoms())
                        for i in range(mol.GetNumAtoms()):
                            conf.SetAtomPosition(i, (np.random.random(), np.random.random(), np.random.random()))
                        conf_id = mol.AddConformer(conf, assignId=True)
                    else:
                        # Distance geometry (more realistic)
                        conf_id = AllChem.EmbedMolecule(
                            mol,
                            randomSeed=42,
                            useRandomCoords=False,
                            clearConfs=False
                        )
                    
                    if conf_id < 0:
                        continue
                    
                    # Optimize geometry
                    props = Chem.MMFFGetMoleculeProperties(mol)
                    if props is None:
                        continue
                    
                    ff = Chem.MMFFGetMoleculeForceField(mol, props, confId=conf_id)
                    if ff is None:
                        continue
                    
                    minimization_result = ff.Minimize()
                    energy = ff.CalcEnergy()
                    
                    # Keep best conformer
                    if energy < best_energy:
                        best_energy = energy
                        best_conf_id = conf_id
                
                except Exception:
                    continue
            
            if best_conf_id is None:
                return None
            
            # Remove other conformers, keep only best
            conf_ids_to_remove = [i for i in range(mol.GetNumConformers()) if i != best_conf_id]
            for conf_id in sorted(conf_ids_to_remove, reverse=True):
                mol.RemoveConformer(conf_id)
            
            return mol
        
        except Exception as e:
            warnings.warn(f"Failed to generate 3D conformation: {e}")
            return None
    
    def _compute_strain_indicators(self, mol: Chem.Mol) -> Dict[str, float]:
        """
        Compute strain indicators for molecule.
        
        Checks for:
        - High MMFF94 energy
        - Steric clashes (atoms <2.5 Å apart)
        - Unfavorable geometry
        
        Args:
            mol: Molecule with 3D coordinates
            
        Returns:
            Dict with strain metrics
        """
        indicators = {}
        
        try:
            if mol.GetNumConformers() == 0:
                return {'strain_score': np.inf, 'has_clashes': True, 'energy': np.inf}
            
            conf = mol.GetConformer()
            
            # Get MMFF94 energy
            props = Chem.MMFFGetMoleculeProperties(mol)
            if props is not None:
                ff = Chem.MMFFGetMoleculeForceField(mol, props)
                if ff is not None:
                    energy = ff.CalcEnergy()
                    indicators['energy'] = energy
                else:
                    indicators['energy'] = np.inf
            else:
                indicators['energy'] = np.inf
            
            # Check for steric clashes
            clash_count = 0
            clash_threshold = 2.5  # Angstroms
            
            for i in range(mol.GetNumAtoms()):
                for j in range(i + 1, mol.GetNumAtoms()):
                    bond = mol.GetBondBetweenAtoms(i, j)
                    
                    # Skip bonded atoms
                    if bond is not None:
                        continue
                    
                    # Skip 1-3 pairs (angles)
                    neighbors_i = [a.GetIdx() for a in mol.GetAtomWithIdx(i).GetNeighbors()]
                    if j in neighbors_i:
                        continue
                    
                    # Compute distance
                    pos_i = conf.GetAtomPosition(i)
                    pos_j = conf.GetAtomPosition(j)
                    dist = pos_i.Distance(pos_j)
                    
                    # Check for clash
                    if dist < clash_threshold:
                        clash_count += 1
            
            indicators['clash_count'] = clash_count
            indicators['has_clashes'] = clash_count > 0
            
            # Strain score: combination of energy and clashes
            energy_score = min(indicators['energy'] / 100, 1.0)  # Normalize to 0-1
            clash_score = min(clash_count / 3, 1.0)  # Each clash adds ~1/3
            indicators['strain_score'] = energy_score + clash_score
            
        except Exception as e:
            warnings.warn(f"Failed to compute strain indicators: {e}")
            indicators['strain_score'] = np.inf
            indicators['energy'] = np.inf
            indicators['has_clashes'] = True
        
        return indicators
    
    def filter_smiles(
        self,
        smiles_list: List[str],
        verbose: bool = False
    ) -> Tuple[List[str], EnergyResults]:
        """
        Filter SMILES by 3D conformation energy.
        
        Process each SMILES:
            1. Parse SMILES
            2. Generate 3D conformation
            3. Compute energy with MMFF94
            4. Check strain indicators
            5. Filter by threshold
        
        Args:
            smiles_list: List of SMILES strings
            verbose: Print progress
            
        Returns:
            (filtered_smiles, results) where results contains energies and filtering decisions
        """
        results = EnergyResults()
        filtered_smiles = []
        
        for idx, smiles in enumerate(smiles_list):
            try:
                # Parse SMILES
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    results.add_molecule(smiles, energy=np.inf, valid=False, passes=False, strain=np.inf)
                    if verbose:
                        print(f"[{idx+1}/{len(smiles_list)}] Invalid SMILES: {smiles}")
                    continue
                
                # Generate 3D conformation
                mol_3d = self._generate_3d_conformation(mol)
                if mol_3d is None:
                    results.add_molecule(smiles, energy=np.inf, valid=False, passes=False, strain=np.inf)
                    if verbose:
                        print(f"[{idx+1}/{len(smiles_list)}] Failed 3D generation: {smiles}")
                    continue
                
                # Compute strain indicators
                indicators = self._compute_strain_indicators(mol_3d)
                energy = indicators.get('energy', np.inf)
                strain = indicators.get('strain_score', np.inf)
                
                # Check if passes filter
                passes_filter = (energy < self.energy_threshold) and (energy != np.inf)
                
                results.add_molecule(smiles, energy=energy, valid=True, passes=passes_filter, strain=strain)
                
                if passes_filter:
                    filtered_smiles.append(smiles)
                    status = "✓ PASS"
                else:
                    status = "✗ FAIL"
                
                if verbose:
                    print(f"[{idx+1}/{len(smiles_list)}] {status} | Energy: {energy:7.2f} | Strain: {strain:5.2f} | {smiles}")
            
            except Exception as e:
                results.add_molecule(smiles, energy=np.inf, valid=False, passes=False, strain=np.inf)
                if verbose:
                    print(f"[{idx+1}/{len(smiles_list)}] Error: {e}")
        
        results.finalize()
        return filtered_smiles, results
    
    def filter_with_batch_stats(
        self,
        smiles_list: List[str],
        use_percentile: bool = False,
        percentile: float = 75.0,
        verbose: bool = False
    ) -> Tuple[List[str], EnergyResults]:
        """
        Filter using statistics from batch (percentile-based).
        
        Instead of fixed threshold, compute threshold as percentile of
        observed energies. Useful for adaptive filtering.
        
        Args:
            smiles_list: List of SMILES
            use_percentile: Use percentile instead of fixed threshold
            percentile: Percentile for threshold (default 75%, top 25% quality)
            verbose: Print progress
            
        Returns:
            (filtered_smiles, results)
        """
        if not use_percentile:
            return self.filter_smiles(smiles_list, verbose)
        
        results = EnergyResults()
        
        # First pass: compute all energies
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    results.add_molecule(smiles, energy=np.inf, valid=False, passes=False, strain=np.inf)
                    continue
                
                mol_3d = self._generate_3d_conformation(mol)
                if mol_3d is None:
                    results.add_molecule(smiles, energy=np.inf, valid=False, passes=False, strain=np.inf)
                    continue
                
                indicators = self._compute_strain_indicators(mol_3d)
                energy = indicators.get('energy', np.inf)
                strain = indicators.get('strain_score', np.inf)
                
                results.add_molecule(smiles, energy=energy, valid=True, passes=False, strain=strain)
            
            except Exception:
                results.add_molecule(smiles, energy=np.inf, valid=False, passes=False, strain=np.inf)
        
        results.finalize()
        
        # Compute percentile threshold
        valid_energies = results.energies[results.valid_3d]
        if len(valid_energies) > 0:
            threshold = np.percentile(valid_energies, percentile)
        else:
            threshold = self.energy_threshold
        
        # Second pass: filter by threshold
        filtered_smiles = []
        for idx, (smiles, energy, valid) in enumerate(zip(
            results.smiles_list,
            results.energies,
            results.valid_3d
        )):
            passes = valid and (energy < threshold)
            results.passes_filter[idx] = passes
            if passes:
                filtered_smiles.append(smiles)
            
            if verbose:
                status = "✓ PASS" if passes else "✗ FAIL"
                print(f"[{idx+1}/{len(smiles_list)}] {status} | Energy: {energy:7.2f} | {smiles}")
        
        return filtered_smiles, results
    
    def get_filtered_with_energies(
        self,
        smiles_list: List[str]
    ) -> List[Tuple[str, float, float]]:
        """
        Get filtered SMILES with energy and strain values.
        
        Returns list of (smiles, energy, strain) tuples for passed molecules,
        sorted by energy (best first).
        
        Args:
            smiles_list: List of SMILES
            
        Returns:
            List of (smiles, energy, strain) sorted by energy
        """
        filtered_smiles, results = self.filter_smiles(smiles_list)
        
        # Get energy and strain for filtered molecules
        filtered_with_energy = []
        for smiles in filtered_smiles:
            idx = results.smiles_list.index(smiles)
            energy = results.energies[idx]
            strain = results.strain_indicators[idx]
            filtered_with_energy.append((smiles, energy, strain))
        
        # Sort by energy (best first)
        filtered_with_energy.sort(key=lambda x: x[1])
        
        return filtered_with_energy
