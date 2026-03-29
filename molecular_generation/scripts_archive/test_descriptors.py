#!/usr/bin/env python3
import sys
sys.path.insert(0, 'src')
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen

smiles = "Cc1cc(-n2ncc(=O)[nH]c2=O)ccc1C(=O)c1ccccc1Cl"
mol = Chem.MolFromSmiles(smiles)

descriptors = [
    ('MolWt', Descriptors.MolWt),
    ('FractionCSP3', Descriptors.FractionCSP3),
    ('RingCount', Descriptors.RingCount),
    ('BertzCT', Descriptors.BertzCT),
    ('Chi0', Descriptors.Chi0),
    ('HallKierAlpha', Descriptors.HallKierAlpha),
    ('Kappa1', Descriptors.Kappa1),
    ('Kappa2', Descriptors.Kappa2),
    ('Kappa3', Descriptors.Kappa3),
    ('MolLogP', Crippen.MolLogP),
    ('LabuteASA', Descriptors.LabuteASA),
    ('NumSaturatedRings', Descriptors.NumSaturatedRings),
    ('NumAliphaticRings', Descriptors.NumAliphaticRings),
    ('NumRotatableBonds', Descriptors.NumRotatableBonds),
    ('NOCount', Descriptors.NOCount),
    ('NumHBD', Descriptors.NumHBD),
    ('NumHBA', Descriptors.NumHBA),
    ('NumHeteroatoms', Descriptors.NumHeteroatoms),
    ('NumAromaticRings', Descriptors.NumAromaticRings),
    ('NumAromaticHeterocycles', Descriptors.NumAromaticHeterocycles),
    ('TopoLogP', Descriptors.TopoLogP),
    ('PEOE_VSA1', Descriptors.PEOE_VSA1),
    ('SlogP_VSA1', Descriptors.SlogP_VSA1),
    ('MLOGP', Descriptors.MLOGP),
]

for name, func in descriptors:
    try:
        val = func(mol)
        print(f"✓ {name}: {val}")
    except Exception as e:
        print(f"✗ {name}: {type(e).__name__}: {e}")
