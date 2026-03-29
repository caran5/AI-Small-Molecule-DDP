"""
Decoder: Convert generated molecular features back to structures.
"""

import torch
import numpy as np
from typing import Tuple, List, Dict
from rdkit import Chem
from rdkit.Chem import AllChem


class MolecularDecoder:
    """Convert normalized features back to molecular structures."""

    # Atomic number to element symbol
    ATOMIC_NUMS = {0: 'X', 1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 16: 'S', 17: 'Cl'}
    COMMON_ATOMS = [6, 7, 8, 9, 16, 17]  # C, N, O, F, S, Cl
    COVALENT_RADII = {6: 0.77, 7: 0.75, 8: 0.73, 9: 0.71, 16: 1.03, 17: 0.99}  # Angstroms

    @staticmethod
    def denormalize_atomic_number(norm_val: float) -> int:
        """Convert normalized atomic number [0, 1] back to actual element.

        Preprocessing encodes as atoms / 118.0, so we invert by multiplying by 118
        and snapping to the nearest common atom.
        """
        raw = norm_val * 118.0
        # Snap to nearest element in COMMON_ATOMS
        return min(MolecularDecoder.COMMON_ATOMS,
                   key=lambda a: abs(a - raw))

    @staticmethod
    def denormalize_coordinates(
        coords: np.ndarray,
        coord_range: float = 10.0
    ) -> np.ndarray:
        """
        Denormalize coordinates from [-1, 1] range.

        Preprocessing encodes as positions / 10.0, so default coord_range is 10.0.

        Args:
            coords: Normalized coordinates in [-1, 1]
            coord_range: Physical coordinate range (Angstroms). Default 10.0 inverts preprocessing.

        Returns:
            Denormalized coordinates
        """
        return coords * coord_range

    @staticmethod
    def features_to_atoms(
        features: torch.Tensor,
        threshold: float = 0.01
    ) -> Tuple[List[int], np.ndarray]:
        """
        Extract atoms and coordinates from feature tensor.

        Args:
            features: Tensor of shape (n_atoms, 5)
                      Features: [atomic_num, x, y, z, dist_from_com]
            threshold: Threshold for filtering out padding atoms

        Returns:
            (atomic_numbers, coordinates in Angstroms)
        """
        features_np = features.cpu().numpy() if isinstance(features, torch.Tensor) else features

        atomic_nums = []
        coordinates = []

        for atom_features in features_np:
            atom_num_norm = atom_features[0]

            # Skip padding (zero atomic numbers)
            if abs(atom_num_norm) < threshold:
                continue

            # Denormalize atomic number
            atomic_num = MolecularDecoder.denormalize_atomic_number(atom_num_norm)

            # Denormalize coordinates (features 1:4 are x, y, z)
            xyz = MolecularDecoder.denormalize_coordinates(atom_features[1:4])

            atomic_nums.append(atomic_num)
            coordinates.append(xyz)

        return atomic_nums, np.array(coordinates) if coordinates else np.zeros((0, 3))

    @staticmethod
    def infer_bonds_from_coords(
        atomic_nums: List[int],
        coordinates: np.ndarray,
        tolerance: float = 0.4
    ) -> List[Tuple[int, int]]:
        """
        Infer bonds from atomic coordinates using covalent radii.

        Args:
            atomic_nums: List of atomic numbers
            coordinates: Array of shape (n_atoms, 3) with coordinates in Angstroms
            tolerance: Tolerance for bond distance threshold (Angstroms)

        Returns:
            List of (i, j) tuples representing bonds (i < j)
        """
        bonds = []
        for i in range(len(atomic_nums)):
            for j in range(i + 1, len(atomic_nums)):
                ri = MolecularDecoder.COVALENT_RADII.get(atomic_nums[i], 0.77)
                rj = MolecularDecoder.COVALENT_RADII.get(atomic_nums[j], 0.77)
                threshold = ri + rj + tolerance
                dist = np.linalg.norm(coordinates[i] - coordinates[j])
                if dist < threshold:
                    bonds.append((i, j))
        return bonds

    @staticmethod
    def build_rdkit_mol(
        atomic_nums: List[int],
        coordinates: np.ndarray,
        bonds: List[Tuple[int, int]]
    ) -> Tuple:
        """
        Build RDKit Mol object from atoms and bonds.

        Args:
            atomic_nums: List of atomic numbers
            coordinates: Array of shape (n_atoms, 3) with coordinates
            bonds: List of (i, j) tuples

        Returns:
            (Chem.Mol, smiles_str) or (None, None) if sanitization fails
        """
        from rdkit import Chem

        rw = Chem.RWMol()
        for anum in atomic_nums:
            rw.AddAtom(Chem.Atom(int(anum)))

        for i, j in bonds:
            rw.AddBond(i, j, Chem.BondType.SINGLE)

        try:
            Chem.SanitizeMol(rw)
            mol = rw.GetMol()

            # Set coordinates
            conf = Chem.Conformer(mol.GetNumAtoms())
            for idx, coord in enumerate(coordinates):
                conf.SetAtomPosition(idx, (float(coord[0]), float(coord[1]), float(coord[2])))
            mol.AddConformer(conf, assignId=True)

            smiles = Chem.MolToSmiles(mol)
            return mol, smiles
        except Exception as e:
            # Sanitization failed or other RDKit error
            return None, None

    @staticmethod
    def create_molecule_from_atoms(
        atomic_nums: List[int],
        coordinates: np.ndarray
    ) -> Dict:
        """
        Create molecular representation from atoms and coordinates.

        Args:
            atomic_nums: List of atomic numbers
            coordinates: Array of shape (n_atoms, 3) with coordinates

        Returns:
            Dict with 'atoms', 'coords', 'formula', 'smiles', 'mol', 'valid'
        """
        result = {
            'atoms': atomic_nums,
            'coordinates': coordinates,
            'n_atoms': len(atomic_nums),
        }

        # Try to create molecular formula
        from collections import Counter
        atom_counts = Counter(atomic_nums)
        formula = ''.join([
            f"{MolecularDecoder.ATOMIC_NUMS.get(num, 'X')}{count if count > 1 else ''}"
            for num, count in sorted(atom_counts.items())
        ])
        result['formula'] = formula

        # Infer bonds and build RDKit Mol
        if len(atomic_nums) > 0:
            bonds = MolecularDecoder.infer_bonds_from_coords(atomic_nums, coordinates)
            mol, smiles = MolecularDecoder.build_rdkit_mol(atomic_nums, coordinates, bonds)
            result['mol'] = mol
            result['smiles'] = smiles
            result['valid'] = mol is not None
        else:
            result['mol'] = None
            result['smiles'] = None
            result['valid'] = False

        return result

    @staticmethod
    def features_to_molecule_dict(
        features: torch.Tensor,
        threshold: float = 0.01
    ) -> Dict:
        """
        Convert feature tensor to molecular structure dict.

        Args:
            features: Tensor of shape (n_atoms, 5)
            threshold: Threshold for filtering atoms

        Returns:
            Dict with molecular structure info including SMILES and validity
        """
        atomic_nums, coordinates = MolecularDecoder.features_to_atoms(
            features,
            threshold=threshold
        )

        if len(atomic_nums) == 0:
            return {
                'atoms': [],
                'coordinates': np.zeros((0, 3)),
                'n_atoms': 0,
                'formula': 'Empty',
                'smiles': None,
                'mol': None,
                'valid': False
            }

        mol_dict = MolecularDecoder.create_molecule_from_atoms(atomic_nums, coordinates)
        return mol_dict


class SMILESGenerator:
    """Generate SMILES strings from molecules (requires more sophisticated approach)."""

    @staticmethod
    def estimate_smiles(mol_dict: Dict) -> str:
        """
        Simple SMILES estimation based on atom composition.
        Real SMILES generation would require graph inference.
        """
        formula = mol_dict.get('formula', '')
        n_atoms = mol_dict.get('n_atoms', 0)

        # Very basic heuristic - this is a placeholder
        if n_atoms == 0:
            return ''

        # For real SMILES, you'd need to:
        # 1. Infer connectivity from atomic coordinates
        # 2. Infer bond orders
        # 3. Determine stereochemistry
        # This is complex and beyond scope of basic decoder

        return f"{{Estimated: {formula}}}"
