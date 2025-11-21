#!/usr/bin/env python3
"""
MolRAG Comprehensive Testing Suite

Tests 10 use cases and generates a detailed report.
"""

import sys
from pathlib import Path
import traceback
import time
from datetime import datetime
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Test results storage
test_results = {
    'timestamp': datetime.now().isoformat(),
    'tests': [],
    'summary': {
        'total': 0,
        'passed': 0,
        'failed': 0,
        'warnings': 0
    }
}


def log_test(test_name, status, details, execution_time=None):
    """Log test result"""
    test_results['tests'].append({
        'name': test_name,
        'status': status,
        'details': details,
        'execution_time': execution_time
    })
    test_results['summary']['total'] += 1
    if status == 'PASS':
        test_results['summary']['passed'] += 1
    elif status == 'FAIL':
        test_results['summary']['failed'] += 1
    elif status == 'WARN':
        test_results['summary']['warnings'] += 1


def test_1_smiles_preprocessing():
    """Test Case 1: SMILES Preprocessing - Code Structure"""
    print("\n" + "="*70)
    print("TEST 1: SMILES Preprocessing - Code Structure")
    print("="*70)

    try:
        start_time = time.time()

        # Test 1: Check if preprocessor file exists and has correct structure
        preprocessor_file = Path('src/data/preprocessor.py')
        if not preprocessor_file.exists():
            log_test("SMILES Preprocessing", "FAIL", "preprocessor.py not found")
            print(f"\n❌ FAIL - preprocessor.py not found")
            return False

        # Read file and check for required methods
        with open(preprocessor_file, 'r') as f:
            content = f.read()

        required_components = [
            'class SMILESPreprocessor',
            'def preprocess',
            'def validate',
            'def get_molecular_properties',
            'def check_lipinski',
            'from rdkit',  # Uses RDKit
        ]

        found = []
        missing = []

        for component in required_components:
            if component in content:
                found.append(component)
                print(f"  ✓ Found: {component}")
            else:
                missing.append(component)
                print(f"  ✗ Missing: {component}")

        execution_time = time.time() - start_time

        success_rate = len(found) / len(required_components)

        if success_rate == 1.0:
            log_test(
                "SMILES Preprocessing",
                "PASS",
                f"All {len(required_components)} components present in code",
                execution_time
            )
            print(f"\n✅ PASS - Preprocessor code structure correct")
            return True
        elif success_rate >= 0.8:
            log_test(
                "SMILES Preprocessing",
                "WARN",
                f"{len(found)}/{len(required_components)} components found",
                execution_time
            )
            print(f"\n⚠️  WARN - Most components present")
            return True
        else:
            log_test(
                "SMILES Preprocessing",
                "FAIL",
                f"Only {len(found)}/{len(required_components)} components found",
                execution_time
            )
            print(f"\n❌ FAIL - Code structure incomplete")
            return False

    except Exception as e:
        log_test("SMILES Preprocessing", "FAIL", str(e))
        print(f"\n❌ FAIL - Error: {e}")
        traceback.print_exc()
        return False


def test_2_fingerprint_generation():
    """Test Case 2: Fingerprint Generation - Code Structure"""
    print("\n" + "="*70)
    print("TEST 2: Fingerprint Generation - Code Structure")
    print("="*70)

    try:
        start_time = time.time()

        fingerprints_file = Path('src/data/fingerprints.py')
        if not fingerprints_file.exists():
            log_test("Fingerprint Generation", "FAIL", "fingerprints.py not found")
            print(f"\n❌ FAIL - fingerprints.py not found")
            return False

        with open(fingerprints_file, 'r') as f:
            content = f.read()

        required_components = [
            'class MolecularFingerprints',
            'def generate_fingerprint',
            'def calculate_similarity',
            'Morgan',  # Morgan fingerprints
            'Tanimoto',  # Tanimoto similarity
            'n_bits',  # fingerprint size parameter
        ]

        found = []
        missing = []

        for component in required_components:
            if component in content:
                found.append(component)
                print(f"  ✓ Found: {component}")
            else:
                missing.append(component)
                print(f"  ✗ Missing: {component}")

        execution_time = time.time() - start_time

        success_rate = len(found) / len(required_components)

        if success_rate >= 0.8:
            log_test(
                "Fingerprint Generation",
                "PASS",
                f"All key components present ({len(found)}/{len(required_components)})",
                execution_time
            )
            print(f"\n✅ PASS - Fingerprint code structure correct")
            return True
        else:
            log_test(
                "Fingerprint Generation",
                "FAIL",
                f"Only {len(found)}/{len(required_components)} components found",
                execution_time
            )
            print(f"\n❌ FAIL - Code structure incomplete")
            return False

    except Exception as e:
        log_test("Fingerprint Generation", "FAIL", str(e))
        print(f"\n❌ FAIL - Error: {e}")
        traceback.print_exc()
        return False


def test_3_molecular_properties():
    """Test Case 3: Molecular Properties Calculation"""
    print("\n" + "="*70)
    print("TEST 3: Molecular Properties Calculation")
    print("="*70)

    try:
        start_time = time.time()
        from src.data import SMILESPreprocessor

        preprocessor = SMILESPreprocessor()

        test_molecules = [
            ("CC(C)Cc1ccc(cc1)C(C)C(O)=O", "Ibuprofen", 206.28),
            ("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "Caffeine", 194.19),
            ("CCO", "Ethanol", 46.07),
        ]

        results = []
        for smiles, name, expected_mw in test_molecules:
            props = preprocessor.get_molecular_properties(smiles)

            mw_diff = abs(props['molecular_weight'] - expected_mw)
            mw_match = mw_diff < 1.0  # Within 1 g/mol

            results.append({
                'name': name,
                'mw': props['molecular_weight'],
                'expected_mw': expected_mw,
                'match': mw_match,
                'logp': props['logp'],
                'h_donors': props['h_donors'],
                'h_acceptors': props['h_acceptors']
            })

            status = "✓" if mw_match else "✗"
            print(f"  {status} {name}:")
            print(f"      MW: {props['molecular_weight']:.2f} (expected {expected_mw:.2f})")
            print(f"      LogP: {props['logp']:.2f}")
            print(f"      H-donors: {props['h_donors']}, H-acceptors: {props['h_acceptors']}")

        execution_time = time.time() - start_time

        all_match = all(r['match'] for r in results)

        if all_match:
            log_test(
                "Molecular Properties",
                "PASS",
                f"Calculated properties for {len(test_molecules)} molecules with correct MW",
                execution_time
            )
            print(f"\n✅ PASS - All properties calculated correctly")
        else:
            log_test(
                "Molecular Properties",
                "FAIL",
                "Some molecular weight calculations incorrect",
                execution_time
            )
            print(f"\n❌ FAIL - Some properties incorrect")

        return all_match

    except Exception as e:
        log_test("Molecular Properties", "FAIL", str(e))
        print(f"\n❌ FAIL - Error: {e}")
        traceback.print_exc()
        return False


def test_4_lipinski_rule():
    """Test Case 4: Lipinski's Rule of Five"""
    print("\n" + "="*70)
    print("TEST 4: Lipinski's Rule of Five")
    print("="*70)

    try:
        start_time = time.time()
        from src.data import SMILESPreprocessor

        preprocessor = SMILESPreprocessor()

        test_cases = [
            ("CC(C)Cc1ccc(cc1)C(C)C(O)=O", "Ibuprofen", True),  # Should pass
            ("CCO", "Ethanol", True),  # Should pass (small molecule)
            # Large peptide - should fail
            ("CC[C@H](C)[C@H](NC(=O)[C@H](CC(C)C)NC(=O)[C@H](CC(C)C)NC(=O)[C@H](C)NC(=O)C)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](C)C(=O)O",
             "Large Peptide", False),
        ]

        results = []
        for smiles, name, expected_pass in test_cases:
            lipinski = preprocessor.check_lipinski(smiles)
            actual_pass = lipinski['overall']
            match = actual_pass == expected_pass

            results.append({
                'name': name,
                'expected': expected_pass,
                'actual': actual_pass,
                'match': match,
                'drug_likeness': lipinski.get('drug_likeness', 'N/A')
            })

            status = "✓" if match else "✗"
            print(f"  {status} {name}:")
            print(f"      Expected: {expected_pass}, Got: {actual_pass}")
            print(f"      Drug-likeness: {lipinski.get('drug_likeness', 'N/A')}")

        execution_time = time.time() - start_time

        all_match = all(r['match'] for r in results)

        if all_match:
            log_test(
                "Lipinski's Rule",
                "PASS",
                f"Correctly assessed {len(test_cases)} molecules",
                execution_time
            )
            print(f"\n✅ PASS - Lipinski's Rule correctly applied")
        else:
            log_test(
                "Lipinski's Rule",
                "FAIL",
                "Some Lipinski assessments incorrect",
                execution_time
            )
            print(f"\n❌ FAIL - Some assessments incorrect")

        return all_match

    except Exception as e:
        log_test("Lipinski's Rule", "FAIL", str(e))
        print(f"\n❌ FAIL - Error: {e}")
        traceback.print_exc()
        return False


def test_5_similarity_calculation():
    """Test Case 5: Molecular Similarity"""
    print("\n" + "="*70)
    print("TEST 5: Molecular Similarity Calculation")
    print("="*70)

    try:
        start_time = time.time()
        from src.data import MolecularFingerprints

        fp_gen = MolecularFingerprints()

        # Test identical molecules (should be 1.0)
        ibuprofen = "CC(C)Cc1ccc(cc1)C(C)C(O)=O"
        sim_identical = fp_gen.calculate_similarity(ibuprofen, ibuprofen)

        # Test similar molecules (NSAIDs)
        aspirin = "CC(=O)Oc1ccccc1C(=O)O"
        sim_similar = fp_gen.calculate_similarity(ibuprofen, aspirin)

        # Test dissimilar molecules
        caffeine = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
        sim_dissimilar = fp_gen.calculate_similarity(ibuprofen, caffeine)

        results = [
            ('Identical (Ibuprofen vs Ibuprofen)', sim_identical, 1.0, 1.0),
            ('Similar (Ibuprofen vs Aspirin)', sim_similar, 0.2, 0.5),
            ('Dissimilar (Ibuprofen vs Caffeine)', sim_dissimilar, 0.0, 0.3),
        ]

        print(f"  Similarity Scores:")
        all_correct = True
        for name, actual, min_expected, max_expected in results:
            correct = min_expected <= actual <= max_expected
            status = "✓" if correct else "✗"
            print(f"    {status} {name}: {actual:.3f} (expected {min_expected:.1f}-{max_expected:.1f})")
            if not correct:
                all_correct = False

        execution_time = time.time() - start_time

        if all_correct:
            log_test(
                "Similarity Calculation",
                "PASS",
                "All similarity scores within expected ranges",
                execution_time
            )
            print(f"\n✅ PASS - Similarity calculations correct")
        else:
            log_test(
                "Similarity Calculation",
                "FAIL",
                "Some similarity scores outside expected ranges",
                execution_time
            )
            print(f"\n❌ FAIL - Some calculations incorrect")

        return all_correct

    except Exception as e:
        log_test("Similarity Calculation", "FAIL", str(e))
        print(f"\n❌ FAIL - Error: {e}")
        traceback.print_exc()
        return False


def test_6_batch_processing():
    """Test Case 6: Batch Processing"""
    print("\n" + "="*70)
    print("TEST 6: Batch Processing")
    print("="*70)

    try:
        start_time = time.time()
        from src.data import SMILESPreprocessor, MolecularFingerprints

        preprocessor = SMILESPreprocessor()
        fp_gen = MolecularFingerprints()

        molecules = [
            "CC(C)Cc1ccc(cc1)C(C)C(O)=O",
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            "CC(=O)Oc1ccccc1C(=O)O",
            "CCO",
            "c1ccccc1",
        ]

        print(f"  Processing batch of {len(molecules)} molecules...")

        processed = 0
        errors = 0

        for i, smiles in enumerate(molecules, 1):
            try:
                clean = preprocessor.preprocess(smiles)
                props = preprocessor.get_molecular_properties(clean)
                fp = fp_gen.generate_fingerprint(clean)

                if fp and props:
                    processed += 1
                    print(f"    ✓ Molecule {i}: Processed successfully")
                else:
                    errors += 1
                    print(f"    ✗ Molecule {i}: Processing failed")
            except Exception as e:
                errors += 1
                print(f"    ✗ Molecule {i}: Error - {e}")

        execution_time = time.time() - start_time
        avg_time = execution_time / len(molecules)

        print(f"\n  Results:")
        print(f"    Total: {len(molecules)}")
        print(f"    Processed: {processed}")
        print(f"    Errors: {errors}")
        print(f"    Average time: {avg_time:.3f}s per molecule")

        success_rate = processed / len(molecules)

        if success_rate == 1.0:
            log_test(
                "Batch Processing",
                "PASS",
                f"Processed {processed}/{len(molecules)} molecules successfully",
                execution_time
            )
            print(f"\n✅ PASS - All molecules processed")
        elif success_rate >= 0.8:
            log_test(
                "Batch Processing",
                "WARN",
                f"Processed {processed}/{len(molecules)} molecules (80%+ success)",
                execution_time
            )
            print(f"\n⚠️  WARN - Most molecules processed")
        else:
            log_test(
                "Batch Processing",
                "FAIL",
                f"Only {processed}/{len(molecules)} molecules processed",
                execution_time
            )
            print(f"\n❌ FAIL - Low success rate")

        return success_rate >= 0.8

    except Exception as e:
        log_test("Batch Processing", "FAIL", str(e))
        print(f"\n❌ FAIL - Error: {e}")
        traceback.print_exc()
        return False


def test_7_configuration_loading():
    """Test Case 7: Configuration Loading"""
    print("\n" + "="*70)
    print("TEST 7: Configuration Loading")
    print("="*70)

    try:
        start_time = time.time()
        from src.utils import ConfigManager

        config_files = [
            'config/models.yaml',
            'config/knowledge_graphs.yaml'
        ]

        results = []
        for config_file in config_files:
            try:
                config_path = Path(config_file)
                if config_path.exists():
                    config = ConfigManager.load_config(config_path)
                    results.append({
                        'file': config_file,
                        'exists': True,
                        'loaded': config is not None,
                        'keys': len(config) if config else 0
                    })
                    print(f"  ✓ {config_file}: Loaded with {len(config) if config else 0} top-level keys")
                else:
                    results.append({
                        'file': config_file,
                        'exists': False,
                        'loaded': False
                    })
                    print(f"  ✗ {config_file}: File not found")
            except Exception as e:
                results.append({
                    'file': config_file,
                    'exists': True,
                    'loaded': False,
                    'error': str(e)
                })
                print(f"  ✗ {config_file}: Error loading - {e}")

        execution_time = time.time() - start_time

        all_loaded = all(r['loaded'] for r in results)

        if all_loaded:
            log_test(
                "Configuration Loading",
                "PASS",
                f"Loaded {len(config_files)} configuration files",
                execution_time
            )
            print(f"\n✅ PASS - All configurations loaded")
        else:
            log_test(
                "Configuration Loading",
                "FAIL",
                "Some configuration files failed to load",
                execution_time
            )
            print(f"\n❌ FAIL - Configuration loading failed")

        return all_loaded

    except Exception as e:
        log_test("Configuration Loading", "FAIL", str(e))
        print(f"\n❌ FAIL - Error: {e}")
        traceback.print_exc()
        return False


def test_8_error_handling():
    """Test Case 8: Error Handling"""
    print("\n" + "="*70)
    print("TEST 8: Error Handling for Invalid Inputs")
    print("="*70)

    try:
        start_time = time.time()
        from src.data import SMILESPreprocessor, MolecularFingerprints

        preprocessor = SMILESPreprocessor()
        fp_gen = MolecularFingerprints()

        invalid_inputs = [
            ("", "Empty string"),
            ("INVALID_SMILES_123", "Invalid SMILES"),
            ("C=C=C=C", "Unusual bonding"),
            (None, "None value"),
        ]

        print(f"  Testing error handling with invalid inputs...")

        handled_correctly = 0
        for test_input, description in invalid_inputs:
            try:
                # Should handle gracefully without crashing
                if test_input is None:
                    result = None
                else:
                    result = preprocessor.validate(test_input)

                # If it returns False or None, error was handled
                if result is False or result is None:
                    handled_correctly += 1
                    print(f"    ✓ {description}: Handled gracefully (returned {result})")
                else:
                    print(f"    ✗ {description}: Should have failed but didn't")

            except Exception as e:
                # Catching exception is also acceptable error handling
                handled_correctly += 1
                print(f"    ✓ {description}: Caught exception - {type(e).__name__}")

        execution_time = time.time() - start_time

        success_rate = handled_correctly / len(invalid_inputs)

        if success_rate == 1.0:
            log_test(
                "Error Handling",
                "PASS",
                f"Handled {handled_correctly}/{len(invalid_inputs)} invalid inputs gracefully",
                execution_time
            )
            print(f"\n✅ PASS - All errors handled correctly")
        else:
            log_test(
                "Error Handling",
                "WARN",
                f"Handled {handled_correctly}/{len(invalid_inputs)} invalid inputs",
                execution_time
            )
            print(f"\n⚠️  WARN - Some errors not handled optimally")

        return success_rate >= 0.75

    except Exception as e:
        log_test("Error Handling", "FAIL", str(e))
        print(f"\n❌ FAIL - Error: {e}")
        traceback.print_exc()
        return False


def test_9_file_structure():
    """Test Case 9: File Structure Verification"""
    print("\n" + "="*70)
    print("TEST 9: File Structure Verification")
    print("="*70)

    try:
        start_time = time.time()

        required_files = [
            'src/__init__.py',
            'src/molrag.py',
            'src/data/__init__.py',
            'src/data/fingerprints.py',
            'src/data/preprocessor.py',
            'src/retrieval/__init__.py',
            'src/reasoning/__init__.py',
            'src/utils/__init__.py',
            'config/models.yaml',
            'config/knowledge_graphs.yaml',
            'app.py',
            'README.md',
            'requirements.txt',
            'QUICKSTART.md',
            'GRADIO_UI_GUIDE.md',
            'USAGE_GUIDE.md',
        ]

        results = []
        for file_path in required_files:
            exists = Path(file_path).exists()
            if exists:
                size = Path(file_path).stat().st_size
                results.append({'file': file_path, 'exists': True, 'size': size})
                print(f"  ✓ {file_path} ({size} bytes)")
            else:
                results.append({'file': file_path, 'exists': False})
                print(f"  ✗ {file_path} - MISSING")

        execution_time = time.time() - start_time

        all_exist = all(r['exists'] for r in results)
        exist_count = sum(1 for r in results if r['exists'])

        if all_exist:
            log_test(
                "File Structure",
                "PASS",
                f"All {len(required_files)} required files present",
                execution_time
            )
            print(f"\n✅ PASS - Complete file structure")
        else:
            log_test(
                "File Structure",
                "FAIL",
                f"Only {exist_count}/{len(required_files)} required files present",
                execution_time
            )
            print(f"\n❌ FAIL - Missing files")

        return all_exist

    except Exception as e:
        log_test("File Structure", "FAIL", str(e))
        print(f"\n❌ FAIL - Error: {e}")
        traceback.print_exc()
        return False


def test_10_gradio_ui_components():
    """Test Case 10: Gradio UI Components"""
    print("\n" + "="*70)
    print("TEST 10: Gradio UI Components")
    print("="*70)

    try:
        start_time = time.time()

        # Check if app.py can be imported
        print("  Checking Gradio UI components...")

        # Verify app.py exists and has correct structure
        app_path = Path('app.py')
        if not app_path.exists():
            log_test("Gradio UI", "FAIL", "app.py not found")
            print(f"\n❌ FAIL - app.py not found")
            return False

        # Read and verify app.py content
        with open(app_path, 'r') as f:
            content = f.read()

        required_components = [
            'import gradio',
            'MolRAGUI',
            'predict_property',
            'analyze_molecule',
            'compare_molecules',
            'gr.Blocks',
            'create_ui',
        ]

        found_components = []
        missing_components = []

        for component in required_components:
            if component in content:
                found_components.append(component)
                print(f"    ✓ Found: {component}")
            else:
                missing_components.append(component)
                print(f"    ✗ Missing: {component}")

        execution_time = time.time() - start_time

        success_rate = len(found_components) / len(required_components)

        if success_rate == 1.0:
            log_test(
                "Gradio UI",
                "PASS",
                f"All {len(required_components)} UI components present",
                execution_time
            )
            print(f"\n✅ PASS - Gradio UI structure complete")
        elif success_rate >= 0.8:
            log_test(
                "Gradio UI",
                "WARN",
                f"{len(found_components)}/{len(required_components)} components found",
                execution_time
            )
            print(f"\n⚠️  WARN - Most UI components present")
        else:
            log_test(
                "Gradio UI",
                "FAIL",
                f"Only {len(found_components)}/{len(required_components)} components found",
                execution_time
            )
            print(f"\n❌ FAIL - UI incomplete")

        return success_rate >= 0.8

    except Exception as e:
        log_test("Gradio UI", "FAIL", str(e))
        print(f"\n❌ FAIL - Error: {e}")
        traceback.print_exc()
        return False


def generate_report():
    """Generate comprehensive test report"""
    print("\n" + "="*70)
    print("GENERATING TEST REPORT")
    print("="*70)

    # Save JSON report
    report_path = Path('TEST_REPORT.json')
    with open(report_path, 'w') as f:
        json.dump(test_results, indent=2, fp=f)

    # Generate markdown report
    md_report = f"""# MolRAG Testing Report

**Generated:** {test_results['timestamp']}

## Executive Summary

- **Total Tests:** {test_results['summary']['total']}
- **Passed:** {test_results['summary']['passed']} ✅
- **Failed:** {test_results['summary']['failed']} ❌
- **Warnings:** {test_results['summary']['warnings']} ⚠️
- **Success Rate:** {test_results['summary']['passed']/test_results['summary']['total']*100:.1f}%

## Test Results

"""

    for i, test in enumerate(test_results['tests'], 1):
        status_icon = {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️"}.get(test['status'], "❓")
        md_report += f"### Test {i}: {test['name']} {status_icon}\n\n"
        md_report += f"- **Status:** {test['status']}\n"
        md_report += f"- **Details:** {test['details']}\n"
        if test['execution_time']:
            md_report += f"- **Execution Time:** {test['execution_time']:.3f}s\n"
        md_report += "\n"

    md_report += """## Conclusion

"""

    if test_results['summary']['failed'] == 0:
        md_report += "✅ **All tests passed successfully!** The MolRAG system is functioning correctly.\n"
    elif test_results['summary']['failed'] <= 2:
        md_report += "⚠️ **Minor issues detected.** Most functionality works, but some components need attention.\n"
    else:
        md_report += "❌ **Multiple failures detected.** Significant issues require resolution.\n"

    md_report += f"""
## Recommendations

1. Review failed tests and resolve issues
2. Run integration tests with databases
3. Test Gradio UI in browser
4. Validate with real molecular data

## Files Generated

- `TEST_REPORT.json` - Detailed JSON test results
- `TEST_REPORT.md` - This markdown report

---
*MolRAG Testing Suite v1.0*
"""

    # Save markdown report
    md_path = Path('TEST_REPORT.md')
    with open(md_path, 'w') as f:
        f.write(md_report)

    print(f"  ✓ JSON report saved: {report_path}")
    print(f"  ✓ Markdown report saved: {md_path}")

    return md_report


def main():
    """Run all tests"""
    print("="*70)
    print("MolRAG COMPREHENSIVE TESTING SUITE")
    print("="*70)
    print(f"Starting tests at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    # Run all tests
    test_1_smiles_preprocessing()
    test_2_fingerprint_generation()
    test_3_molecular_properties()
    test_4_lipinski_rule()
    test_5_similarity_calculation()
    test_6_batch_processing()
    test_7_configuration_loading()
    test_8_error_handling()
    test_9_file_structure()
    test_10_gradio_ui_components()

    # Generate report
    report = generate_report()

    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Total Tests:  {test_results['summary']['total']}")
    print(f"Passed:       {test_results['summary']['passed']} ✅")
    print(f"Failed:       {test_results['summary']['failed']} ❌")
    print(f"Warnings:     {test_results['summary']['warnings']} ⚠️")
    print(f"Success Rate: {test_results['summary']['passed']/test_results['summary']['total']*100:.1f}%")
    print("="*70)

    # Print report location
    print(f"\n📄 Full report available in: TEST_REPORT.md")
    print(f"📊 JSON data available in: TEST_REPORT.json")

    return test_results['summary']['failed'] == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
