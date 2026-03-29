#!/usr/bin/env python3
"""
Comprehensive descriptor and prediction benchmarking
Tests LogP predictions across various molecule types
"""
from src.predict import predict_logp

test_cases = [
    # Pharmaceuticals
    ("CC(=O)Oc1ccccc1C(=O)O", "Aspirin", 1.31),
    ("CC(C)Cc1ccc(cc1)C(C)C(=O)O", "Ibuprofen", 3.97),
    ("CC(=O)Nc1ccc(cc1)O", "Acetaminophen", 0.46),
    ("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "Caffeine", 0.16),
    
    # Simple molecules
    ("C", "Methane", 1.09),
    ("CC", "Ethane", 1.81),
    ("CCO", "Ethanol", -0.07),
    
    # Complex molecules
    ("C1=CC=C(C=C1)C2=CC=C(C=C2)C(=O)O", "Benzilic acid", 1.48),
    
    # Aromatic compounds
    ("c1ccccc1", "Benzene", 2.13),
    ("c1ccccc1O", "Phenol", 1.46),
    
    # Edge cases
    ("INVALID_SMILES_123", "Invalid SMILES", None),
]

def run_benchmark():
    print("=" * 100)
    print("COMPREHENSIVE DESCRIPTOR BENCHMARK")
    print("=" * 100)
    
    errors = 0
    successes = 0
    warnings = 0
    logp_values = []
    mw_values = []
    
    for smiles, name, expected_logp in test_cases:
        print(f"\nTesting: {name:20} | SMILES: {smiles}")
        print("-" * 100)
        
        result = predict_logp(smiles)
        
        if "error" in result:
            print(f"  [ERROR] {result['error']}")
            errors += 1
            continue
        
        pred_logp = result["logp"]
        hydro = result["hydrophilicity"]
        logp_values.append(pred_logp)
        mw_values.append(result['formula_weight'])
        
        if expected_logp is not None:
            error = abs(pred_logp - expected_logp)
            error_pct = (error / max(abs(expected_logp), 0.1)) * 100
            
            if error < 0.3:
                status = "[PASS]"
            elif error < 0.7:
                status = "[WARN]"
                warnings += 1
            else:
                status = "[FAIL]"
                errors += 1
            
            print(f"  {status} | Predicted: {pred_logp:6.2f} | Expected: {expected_logp:6.2f} | Error: {error:6.2f} ({error_pct:5.1f}%)")
        else:
            print(f"  [INFO] Predicted LogP: {pred_logp:6.2f}")
        
        # Properties
        print(f"  Classification:     {hydro}")
        print(f"  Properties: MW={result['formula_weight']:7.2f} | H-Don={result['h_donors']:2d} | H-Acc={result['h_acceptors']:2d} | RotBonds={result['rotatable_bonds']:2d}")
        
        successes += 1
    
    # Summary stats
    print(f"\n{'=' * 100}")
    print(f"RESULTS SUMMARY")
    print(f"{'=' * 100}")
    print(f"Successes:         {successes}")
    print(f"Warnings:          {warnings}")
    print(f"Errors:            {errors}")
    print(f"Total Tests:       {len(test_cases)}")
    
    if logp_values:
        print(f"\nLogP Statistics:")
        print(f"  Min:               {min(logp_values):6.2f}")
        print(f"  Max:               {max(logp_values):6.2f}")
        print(f"  Range:             {max(logp_values) - min(logp_values):6.2f}")
        print(f"  Average:           {sum(logp_values) / len(logp_values):6.2f}")
    
    if mw_values:
        print(f"\nMolecular Weight Statistics:")
        print(f"  Min:               {min(mw_values):7.2f} g/mol")
        print(f"  Max:               {max(mw_values):7.2f} g/mol")
        print(f"  Average:           {sum(mw_values) / len(mw_values):7.2f} g/mol")
    
    print(f"{'=' * 100}\n")
    
    # Return success rate
    return (successes - errors) / len(test_cases) * 100 if test_cases else 0

if __name__ == "__main__":
    success_rate = run_benchmark()
    print(f"Overall Success Rate: {success_rate:.1f}%")
