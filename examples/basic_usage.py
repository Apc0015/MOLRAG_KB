#!/usr/bin/env python3
"""
Basic usage example for MolRAG

Demonstrates:
1. Initializing MolRAG
2. Predicting molecular properties
3. Interpreting results
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.molrag import MolRAG


def main():
    """Main example"""

    print("=" * 60)
    print("MolRAG Basic Usage Example")
    print("=" * 60)

    # Example molecule: Ibuprofen
    ibuprofen_smiles = "CC(C)Cc1ccc(cc1)C(C)C(O)=O"
    property_query = "Is this molecule toxic?"

    print(f"\nQuery Molecule: {ibuprofen_smiles}")
    print(f"Property Question: {property_query}\n")

    # Initialize MolRAG (without auto_init for this example)
    print("Initializing MolRAG system...")
    molrag = MolRAG(
        config_path=Path("config/models.yaml"),
        kg_config_path=Path("config/knowledge_graphs.yaml"),
        auto_init=False  # Manual initialization for demo
    )

    # Initialize only preprocessing (databases not required for this example)
    print("Initializing preprocessing...")
    molrag.initialize_preprocessing()

    print("\nNote: Full prediction requires databases to be set up.")
    print("Run 'python scripts/setup_databases.py' first.\n")

    # Demonstrate preprocessing
    print("=" * 60)
    print("SMILES Preprocessing")
    print("=" * 60)

    processed_smiles = molrag.preprocessor.preprocess(ibuprofen_smiles)
    print(f"Original:   {ibuprofen_smiles}")
    print(f"Processed:  {processed_smiles}\n")

    # Calculate molecular properties
    properties = molrag.preprocessor.get_molecular_properties(ibuprofen_smiles)
    print("Molecular Properties:")
    for prop, value in properties.items():
        print(f"  {prop}: {value}")

    # Check Lipinski's Rule of Five
    passes_ro5 = molrag.preprocessor.passes_lipinski_rule_of_five(ibuprofen_smiles)
    print(f"\nPasses Lipinski's Rule of Five: {passes_ro5}")

    # Generate fingerprint
    print("\n" + "=" * 60)
    print("Molecular Fingerprint Generation")
    print("=" * 60)

    fingerprint = molrag.fp_generator.generate_fingerprint(ibuprofen_smiles)
    if fingerprint:
        print(f"Fingerprint Type: Morgan ECFP4")
        print(f"Fingerprint Size: {len(fingerprint)} bits")
        print(f"On Bits: {fingerprint.GetNumOnBits()}")

    print("\n" + "=" * 60)
    print("Full Prediction Example (requires setup)")
    print("=" * 60)
    print("""
To run full prediction:

1. Set up databases:
   python scripts/setup_databases.py

2. Load knowledge graphs:
   python scripts/load_knowledge_graphs.py --kg primekg

3. Run prediction:
   from src.molrag import MolRAG

   molrag = MolRAG(auto_init=True)

   result = molrag.predict(
       smiles="CC(C)Cc1ccc(cc1)C(C)C(O)=O",
       query="Is this molecule toxic?",
       cot_strategy="sim_cot",
       top_k=10
   )

   print(f"Prediction: {result.prediction}")
   print(f"Confidence: {result.confidence:.2f}")
   print(f"Reasoning: {result.reasoning}")
    """)

    print("=" * 60)


if __name__ == "__main__":
    main()
