#!/usr/bin/env python3
"""
Batch Molecular Screening Example

Demonstrates efficient processing of multiple molecules
"""

import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import SMILESPreprocessor, MolecularFingerprints


def screen_molecules():
    """Screen a batch of molecules"""
    print("=" * 70)
    print("Batch Molecular Screening Example")
    print("=" * 70)

    # Sample molecules
    molecules = [
        ("CC(C)Cc1ccc(cc1)C(C)C(O)=O", "Ibuprofen"),
        ("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "Caffeine"),
        ("CC(=O)Oc1ccccc1C(=O)O", "Aspirin"),
        ("CCO", "Ethanol"),
        ("c1ccccc1", "Benzene"),
    ]

    preprocessor = SMILESPreprocessor()
    fp_gen = MolecularFingerprints()

    results = []
    print(f"\nScreening {len(molecules)} molecules...\n")

    for smiles, name in tqdm(molecules, desc="Processing"):
        try:
            clean_smiles = preprocessor.preprocess(smiles)
            props = preprocessor.get_molecular_properties(clean_smiles)
            lipinski = preprocessor.check_lipinski(clean_smiles)
            fp = fp_gen.generate_fingerprint(clean_smiles)

            results.append({
                'name': name,
                'smiles': clean_smiles,
                'mw': props['molecular_weight'],
                'logp': props['logp'],
                'drug_like': lipinski['overall']
            })
        except Exception as e:
            print(f"Error processing {name}: {e}")

    # Summary
    df = pd.DataFrame(results)
    print(f"\n{'=' * 70}")
    print(f"Successfully processed: {len(df)} molecules")
    print(f"Drug-like molecules: {df['drug_like'].sum()}")
    print(f"Average MW: {df['mw'].mean():.2f} g/mol")

    # Save results
    Path("results").mkdir(exist_ok=True)
    df.to_csv("results/screening_results.csv", index=False)
    print(f"Results saved to: results/screening_results.csv")
    print("=" * 70)


if __name__ == "__main__":
    screen_molecules()
