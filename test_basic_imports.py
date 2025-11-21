#!/usr/bin/env python3
"""
Quick test to verify basic MolRAG imports work
"""

print("="*70)
print("MOLRAG BASIC IMPORT TEST")
print("="*70)

# Test 1: Basic data processing
print("\n1. Testing data processing modules...")
try:
    from src.data import SMILESPreprocessor, MolecularFingerprints
    print("   ✓ SMILESPreprocessor and MolecularFingerprints imported")
except Exception as e:
    print(f"   ✗ Error: {e}")
    exit(1)

# Test 2: Process a simple SMILES
print("\n2. Testing SMILES preprocessing...")
try:
    preprocessor = SMILESPreprocessor()
    smiles = "CC(C)Cc1ccc(cc1)C(C)C(O)=O"  # Ibuprofen
    canonical = preprocessor.preprocess(smiles)
    print(f"   ✓ Original: {smiles}")
    print(f"   ✓ Canonical: {canonical}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    exit(1)

# Test 3: Calculate properties
print("\n3. Testing property calculation...")
try:
    props = preprocessor.get_molecular_properties(canonical)
    print(f"   ✓ Molecular Weight: {props['molecular_weight']:.2f}")
    print(f"   ✓ LogP: {props['logp']:.2f}")
    print(f"   ✓ H-Bond Donors: {props['num_h_donors']}")
    print(f"   ✓ H-Bond Acceptors: {props['num_h_acceptors']}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    exit(1)

# Test 4: Generate fingerprint
print("\n4. Testing fingerprint generation...")
try:
    fp_gen = MolecularFingerprints(fingerprint_type="morgan", n_bits=2048)
    fp = fp_gen.generate_fingerprint(canonical)
    print(f"   ✓ Fingerprint generated: {fp.GetNumOnBits()} bits set out of 2048")
except Exception as e:
    print(f"   ✗ Error: {e}")
    exit(1)

# Test 5: Gradio import
print("\n5. Testing Gradio import...")
try:
    import gradio as gr
    print(f"   ✓ Gradio {gr.__version__} imported")
except Exception as e:
    print(f"   ✗ Error: {e}")
    exit(1)

print("\n" + "="*70)
print("✅ ALL BASIC TESTS PASSED!")
print("="*70)
print("\nYou can now run: python app.py")
print("="*70)
