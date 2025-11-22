#!/usr/bin/env python3
"""
Download real molecular databases for MolRAG

This script downloads:
1. PrimeKG (4M+ relationships) - Biomedical knowledge graph
2. ChEMBL (2M+ compounds) - Bioactivity database
3. DrugBank (optional, requires API key)
4. Reactome pathways

Data sources are automatically downloaded and converted to MolRAG format.
"""

import sys
import requests
import gzip
import zipfile
from pathlib import Path
from typing import Optional
import pandas as pd
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Data directory
DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Create directories
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def download_file(url: str, dest: Path, desc: str = "Downloading") -> bool:
    """Download file with progress bar"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(dest, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        logger.info(f"✓ Downloaded: {dest}")
        return True

    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False


def download_primekg() -> bool:
    """
    Download PrimeKG - Precision Medicine Knowledge Graph

    Source: Harvard Medical School
    Size: ~4M relationships, ~130K nodes
    URL: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IXA7BM
    """
    logger.info("=" * 60)
    logger.info("DOWNLOADING PRIMEKG")
    logger.info("=" * 60)

    # PrimeKG full dataset
    primekg_url = "https://dataverse.harvard.edu/api/access/datafile/6180620"
    primekg_file = RAW_DIR / "kg_giant_csv.zip"

    logger.info("Downloading PrimeKG (this may take 10-15 minutes, ~500MB)...")

    if primekg_file.exists():
        logger.info(f"File already exists: {primekg_file}")
        response = input("Re-download? (y/N): ")
        if response.lower() != 'y':
            logger.info("Skipping download")
            return True

    # Download
    if not download_file(primekg_url, primekg_file, "PrimeKG"):
        logger.error("Failed to download PrimeKG")
        return False

    # Extract
    logger.info("Extracting PrimeKG...")
    try:
        with zipfile.ZipFile(primekg_file, 'r') as zip_ref:
            zip_ref.extractall(RAW_DIR / "primekg")
        logger.info("✓ Extracted PrimeKG")
    except Exception as e:
        logger.error(f"Failed to extract: {e}")
        return False

    # Process to MolRAG format
    logger.info("Processing PrimeKG to MolRAG format...")
    try:
        # Load PrimeKG edges
        primekg_edges = RAW_DIR / "primekg" / "kg_giant.csv"

        if not primekg_edges.exists():
            # Try alternative name
            primekg_edges = list((RAW_DIR / "primekg").glob("*.csv"))[0]

        df = pd.read_csv(primekg_edges)
        logger.info(f"Loaded {len(df)} relationships")

        # Convert to MolRAG format
        # PrimeKG format: x_index, x_id, x_type, x_name, x_source, relation, y_index, y_id, y_type, y_name, y_source

        # Extract unique nodes
        logger.info("Extracting nodes...")
        x_nodes = df[['x_id', 'x_type', 'x_name', 'x_source']].rename(columns={
            'x_id': 'node_id',
            'x_type': 'node_type',
            'x_name': 'node_name',
            'x_source': 'node_source'
        })

        y_nodes = df[['y_id', 'y_type', 'y_name', 'y_source']].rename(columns={
            'y_id': 'node_id',
            'y_type': 'node_type',
            'y_name': 'node_name',
            'y_source': 'node_source'
        })

        all_nodes = pd.concat([x_nodes, y_nodes]).drop_duplicates(subset=['node_id'])

        # Extract edges
        logger.info("Extracting edges...")
        edges = df[['x_id', 'y_id', 'relation', 'x_type', 'y_type']].rename(columns={
            'x_id': 'source',
            'y_id': 'target',
            'x_type': 'source_type',
            'y_type': 'target_type'
        })

        # Save
        nodes_file = DATA_DIR / "kg_nodes.csv"
        edges_file = DATA_DIR / "kg_edges.csv"

        all_nodes.to_csv(nodes_file, index=False)
        edges.to_csv(edges_file, index=False)

        logger.info(f"✓ Saved {len(all_nodes)} nodes to {nodes_file}")
        logger.info(f"✓ Saved {len(edges)} edges to {edges_file}")

        # Extract molecules specifically
        molecule_nodes = all_nodes[all_nodes['node_type'] == 'drug']
        if len(molecule_nodes) > 0:
            molecules_file = DATA_DIR / "molecules.csv"
            molecule_nodes.to_csv(molecules_file, index=False)
            logger.info(f"✓ Saved {len(molecule_nodes)} molecules to {molecules_file}")

        return True

    except Exception as e:
        logger.error(f"Failed to process PrimeKG: {e}")
        import traceback
        traceback.print_exc()
        return False


def download_chembl() -> bool:
    """
    Download ChEMBL - Bioactivity Database

    Source: EMBL-EBI
    Size: ~2M compounds, 15M+ bioactivities
    """
    logger.info("=" * 60)
    logger.info("DOWNLOADING CHEMBL")
    logger.info("=" * 60)

    logger.info("ChEMBL download requires manual steps:")
    logger.info("1. Visit: https://www.ebi.ac.uk/chembl/")
    logger.info("2. Download: ChEMBL database (SQLite or CSV)")
    logger.info("3. Place files in: data/raw/chembl/")
    logger.info("")
    logger.info("Alternatively, use ChEMBL web services:")
    logger.info("  https://www.ebi.ac.uk/chembl/api/data/molecule")
    logger.info("")

    # Try to download a subset via API
    logger.info("Downloading ChEMBL sample (1000 molecules)...")

    try:
        # Get top 1000 molecules
        chembl_api = "https://www.ebi.ac.uk/chembl/api/data/molecule.json"
        params = {"limit": 1000, "format": "json"}

        response = requests.get(chembl_api, params=params)
        response.raise_for_status()

        data = response.json()
        molecules = data.get('molecules', [])

        if molecules:
            # Convert to DataFrame
            mol_data = []
            for mol in molecules:
                mol_data.append({
                    'chembl_id': mol.get('molecule_chembl_id', ''),
                    'canonical_smiles': mol.get('molecule_structures', {}).get('canonical_smiles', ''),
                    'pref_name': mol.get('pref_name', ''),
                    'max_phase': mol.get('max_phase', 0),
                    'molecular_weight': mol.get('molecule_properties', {}).get('full_mwt', 0)
                })

            df = pd.DataFrame(mol_data)

            # Save
            chembl_file = DATA_DIR / "chembl_molecules.csv"
            df.to_csv(chembl_file, index=False)

            logger.info(f"✓ Saved {len(df)} ChEMBL molecules to {chembl_file}")
            logger.info("Note: This is a small subset. For full ChEMBL, download manually.")

            return True

    except Exception as e:
        logger.error(f"Failed to download ChEMBL: {e}")
        return False


def download_drugbank() -> bool:
    """
    Download DrugBank (requires account and API key)

    Source: DrugBank.ca
    Note: Requires free academic account
    """
    logger.info("=" * 60)
    logger.info("DRUGBANK DOWNLOAD")
    logger.info("=" * 60)

    logger.info("DrugBank requires a free academic account:")
    logger.info("1. Register at: https://go.drugbank.com/")
    logger.info("2. Download: 'All drugs' CSV file")
    logger.info("3. Place in: data/raw/drugbank/")
    logger.info("")
    logger.info("Skipping automated download (requires authentication)")

    return False


def download_all() -> dict:
    """Download all available datasets"""

    results = {
        'primekg': False,
        'chembl': False,
        'drugbank': False
    }

    logger.info("\n")
    logger.info("=" * 60)
    logger.info("MOLRAG DATA DOWNLOAD")
    logger.info("=" * 60)
    logger.info("")
    logger.info("This will download real molecular databases:")
    logger.info("  - PrimeKG: ~500MB, 4M+ relationships")
    logger.info("  - ChEMBL: Sample of 1K molecules (full requires manual download)")
    logger.info("  - DrugBank: Requires manual download")
    logger.info("")

    response = input("Continue? (y/N): ")
    if response.lower() != 'y':
        logger.info("Cancelled")
        return results

    # Download PrimeKG
    logger.info("\n1/3: PrimeKG")
    results['primekg'] = download_primekg()

    # Download ChEMBL
    logger.info("\n2/3: ChEMBL")
    results['chembl'] = download_chembl()

    # DrugBank (manual)
    logger.info("\n3/3: DrugBank")
    results['drugbank'] = download_drugbank()

    # Summary
    logger.info("\n")
    logger.info("=" * 60)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("=" * 60)
    logger.info(f"PrimeKG:  {'✅ SUCCESS' if results['primekg'] else '❌ FAILED'}")
    logger.info(f"ChEMBL:   {'✅ SUCCESS' if results['chembl'] else '⚠️  MANUAL'}")
    logger.info(f"DrugBank: {'✅ SUCCESS' if results['drugbank'] else '⚠️  MANUAL'}")
    logger.info("=" * 60)

    if results['primekg']:
        logger.info("\n✨ PrimeKG is ready!")
        logger.info("\nNext steps:")
        logger.info("  1. Load into databases: python scripts/load_knowledge_graphs.py")
        logger.info("  2. Start UI: python app.py")

    return results


def main():
    """Main entry point"""

    import argparse
    parser = argparse.ArgumentParser(description="Download molecular databases")
    parser.add_argument('--dataset', choices=['primekg', 'chembl', 'drugbank', 'all'],
                       default='all', help='Dataset to download')
    args = parser.parse_args()

    if args.dataset == 'primekg':
        download_primekg()
    elif args.dataset == 'chembl':
        download_chembl()
    elif args.dataset == 'drugbank':
        download_drugbank()
    else:
        download_all()


if __name__ == "__main__":
    main()
