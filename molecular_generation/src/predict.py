"""
LogP Prediction Module
Predicts how oily a chemical is (LogP) from molecular structure
Uses multiple approaches: RDKit descriptors, atom-based calculation, and correction model
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen, Descriptors3D
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
import json
import os


# Atom-based LogP contributions (empirical)
ATOM_LOGP_CONTRIB = {
    'C': 0.5,      # Carbon
    'H': 0.1,      # Hydrogen (implicit)
    'N': -0.7,     # Nitrogen (polar)
    'O': -1.0,     # Oxygen (very polar)
    'S': 0.5,      # Sulfur (lipophilic)
    'Cl': 0.8,     # Chlorine (lipophilic)
    'Br': 1.0,     # Bromine (very lipophilic)
    'F': -0.3,     # Fluorine (slightly hydrophilic)
    'I': 1.3,      # Iodine (very lipophilic)
    'P': 0.2,      # Phosphorus
}

class LogPPredictor:
    """Predicts LogP for molecules using multiple methods"""
    
    # Known drug LogP values for correction model training
    KNOWN_LOGP_VALUES = {
        "CC(=O)Oc1ccccc1C(=O)O": 1.31,  # Aspirin
        "CCO": -0.07,  # Ethanol
        "c1ccccc1O": 1.46,  # Phenol
        "c1ccccc1": 2.13,  # Benzene
        "CC(=O)Nc1ccc(cc1)O": 0.46,  # Acetaminophen
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O": 3.97,  # Ibuprofen
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C": 0.16,  # Caffeine
    }
    
    def __init__(self):
        """Initialize with model parameters from Phase 4 Path 2"""
        self.model_params = {
            "morgan_radius": 1,
            "morgan_nbits": 2048,
            "pca_components": 200,
            "descriptors": [
                "MolWt", "TPSA", "NumRotatableBonds", "NumHDonors", "NumHAcceptors",
                "NumAromaticRings", "RingCount", "NumAliphaticRings", "NumHeavyAtoms",
                "NumSaturatedRings", "MolLogP", "ExactMolWt", "FractionCsp3",
                "NumBridgeheadAtoms", "NumSpiro", "HallKierAlpha", "LabuteASA",
                "LabeuteSA", "PEOE_VSA1", "SlogP_VSA1"
            ]
        }
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.model_params["pca_components"], random_state=42)
        self.model = GradientBoostingRegressor(
            n_estimators=200, 
            learning_rate=0.1, 
            max_depth=5, 
            random_state=42
        )
        self.correction_model = Ridge(alpha=1.0)
        self._model_trained = False
        self._correction_model_trained = False
        self._train_correction_model()
    
    def extract_descriptors(self, mol):
        """Extract RDKit descriptors from molecule"""
        if mol is None:
            return None
        
        descriptors = []
        for desc_name in self.model_params["descriptors"]:
            try:
                if hasattr(Descriptors, desc_name):
                    desc_func = getattr(Descriptors, desc_name)
                    value = desc_func(mol)
                else:
                    value = 0.0
            except:
                value = 0.0
            descriptors.append(value)
        
        return np.array(descriptors)
    
    def extract_morgan(self, mol):
        """Extract Morgan fingerprint from molecule"""
        if mol is None:
            return None
        
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, 
            radius=self.model_params["morgan_radius"],
            nBits=self.model_params["morgan_nbits"]
        )
        return np.array(fp, dtype=float)
    
    def predict(self, smiles):
        """
        Predict LogP for a chemical
        
        Args:
            smiles (str): SMILES string of chemical
            
        Returns:
            dict: {
                "smiles": input SMILES,
                "logp": predicted value,
                "interpretation": human-readable explanation,
                "hydrophilicity": classification (hydrophilic/balanced/hydrophobic)
            }
        """
        # Parse SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {
                "error": f"Invalid SMILES: {smiles}",
                "smiles": smiles
            }
        
        # Extract features
        morgan = self.extract_morgan(mol)
        descriptors = self.extract_descriptors(mol)
        
        if morgan is None or descriptors is None:
            return {"error": "Failed to extract features", "smiles": smiles}
        
        # Use ensemble prediction (combines RDKit, correction model, and atom-based)
        logp_pred = self._ensemble_logp_prediction(mol, descriptors)
        
        # Interpret
        if logp_pred < -1:
            hydro = "Very Hydrophilic (water-loving)"
        elif logp_pred < 0:
            hydro = "Hydrophilic (water-loving)"
        elif logp_pred < 2:
            hydro = "Balanced (good absorption)"
            reason = "✅ Ideal for drugs - good balance of water/oil solubility"
        elif logp_pred < 4:
            hydro = "Hydrophobic (lipid-loving)"
            reason = "⚠️ May have poor water solubility"
        else:
            hydro = "Very Hydrophobic (very lipid-loving)"
            reason = "❌ Poor water solubility - may not work as drug"
        
        return {
            "smiles": smiles,
            "logp": round(float(logp_pred), 2),
            "hydrophilicity": hydro,
            "interpretation": f"LogP = {logp_pred:.2f}: {hydro}. {reason if logp_pred >= 2 else ''}",
            "formula_weight": round(Descriptors.MolWt(mol), 2),
            "h_donors": int(Descriptors.NumHDonors(mol)),
            "h_acceptors": int(Descriptors.NumHAcceptors(mol)),
            "rotatable_bonds": int(Descriptors.NumRotatableBonds(mol))
        }
    
    def _estimate_logp(self, descriptors, morgan):
        """Estimate LogP from RDKit MolLogP descriptor"""
        # descriptors[10] is MolLogP (Crippen's LogP calculation)
        logp_value = descriptors[10] if len(descriptors) > 10 else 0.5
        
        # MolLogP is already accurate, so just return it
        # RDKit's Crippen method is well-validated
        return float(logp_value)
    
    def _train_correction_model(self):
        """Train a simple correction model on known LogP values"""
        if len(self.KNOWN_LOGP_VALUES) < 2:
            return
        
        X_train = []
        y_train = []
        
        for smiles, expected_logp in self.KNOWN_LOGP_VALUES.items():
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            
            # Extract RDKit LogP prediction
            rdkit_logp = Descriptors.MolLogP(mol)
            
            # Additional features for correction
            features = [
                rdkit_logp,  # Base prediction
                Descriptors.TPSA(mol) / 100.0,  # Polarity
                Descriptors.NumHDonors(mol),  # H-donors
                Descriptors.NumHAcceptors(mol),  # H-acceptors
                Descriptors.RingCount(mol),  # Rings
            ]
            
            X_train.append(features)
            y_train.append(expected_logp)
        
        if len(X_train) > 1:
            self.correction_model.fit(np.array(X_train), np.array(y_train))
            self._correction_model_trained = True
    
    def _calculate_atom_based_logp(self, mol):
        """Calculate LogP based on atom contributions"""
        if mol is None:
            return 0.0
        
        logp = 0.0
        aromatic_bonus = 0.0
        
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            
            # Base atom contribution
            if symbol in ATOM_LOGP_CONTRIB:
                logp += ATOM_LOGP_CONTRIB[symbol]
            else:
                logp += 0.2  # Default for unknown atoms
            
            # Aromatic rings boost lipophilicity
            if atom.GetIsAromatic():
                aromatic_bonus += 0.1
        
        # Add aromatic bonus
        logp += aromatic_bonus
        
        # Normalize by number of atoms
        n_atoms = mol.GetNumAtoms()
        if n_atoms > 0:
            logp = logp / (n_atoms ** 0.5)
        
        return float(logp)
    
    def _ensemble_logp_prediction(self, mol, descriptors):
        """Ensemble prediction using multiple methods"""
        if mol is None:
            return 0.0
        
        # Method 1: RDKit MolLogP (most reliable for drug-like molecules)
        rdkit_logp = Descriptors.MolLogP(mol)
        
        # Method 2: Atom-based calculation (good for simple molecules)
        atom_logp = self._calculate_atom_based_logp(mol)
        
        # Method 3: Correction model (learns from known values)
        corrected_logp = rdkit_logp
        if self._correction_model_trained:
            features = np.array([[
                rdkit_logp,
                Descriptors.TPSA(mol) / 100.0,
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.RingCount(mol),
            ]])
            corrected_logp = self.correction_model.predict(features)[0]
        
        # Ensemble: weight the predictions
        # RDKit is most reliable (50%), correction model (30%), atom-based (20%)
        ensemble_logp = (
            0.50 * rdkit_logp +
            0.30 * corrected_logp +
            0.20 * atom_logp
        )
        
        return float(ensemble_logp)
    
    def batch_predict(self, smiles_list):
        """Predict LogP for multiple chemicals"""
        return [self.predict(smi) for smi in smiles_list]


# Singleton instance
_predictor = None

def get_predictor():
    """Get or create predictor instance"""
    global _predictor
    if _predictor is None:
        _predictor = LogPPredictor()
    return _predictor


def predict_logp(smiles):
    """
    Quick prediction function
    
    Usage:
        result = predict_logp("CC(=O)Oc1ccccc1C(=O)O")  # Aspirin
        print(result["logp"])  # ~1.19
    """
    predictor = get_predictor()
    return predictor.predict(smiles)
