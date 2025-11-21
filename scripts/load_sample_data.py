#!/usr/bin/env python3
"""
Load sample molecular data for demo purposes

This script loads a small sample dataset to demonstrate MolRAG functionality
without requiring full database setup of ChEMBL/DrugBank.
"""

import sys
from pathlib import Path
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import Config, Neo4jConnector, QdrantConnector
from src.data import MolecularFingerprints, SMILESPreprocessor
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Sample molecules with known properties
SAMPLE_MOLECULES = [
    {
        "smiles": "CC(C)Cc1ccc(cc1)C(C)C(O)=O",
        "name": "Ibuprofen",
        "properties": {
            "toxic": False,
            "water_soluble": False,
            "anti_inflammatory": True,
            "molecular_weight": 206.28,
            "logP": 3.97
        },
        "description": "Nonsteroidal anti-inflammatory drug (NSAID)"
    },
    {
        "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "name": "Aspirin",
        "properties": {
            "toxic": False,
            "water_soluble": False,
            "anti_inflammatory": True,
            "molecular_weight": 180.16,
            "logP": 1.19
        },
        "description": "Analgesic, antipyretic, and anti-inflammatory drug"
    },
    {
        "smiles": "CCO",
        "name": "Ethanol",
        "properties": {
            "toxic": True,
            "water_soluble": True,
            "anti_inflammatory": False,
            "molecular_weight": 46.07,
            "logP": -0.31
        },
        "description": "Common alcohol, can be toxic in high doses"
    },
    {
        "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "name": "Caffeine",
        "properties": {
            "toxic": False,
            "water_soluble": True,
            "anti_inflammatory": False,
            "molecular_weight": 194.19,
            "logP": -0.07
        },
        "description": "Central nervous system stimulant"
    },
    {
        "smiles": "CN1C2CCC1CC(C2)OC(=O)C(CO)c3ccccc3",
        "name": "Atropine",
        "properties": {
            "toxic": True,
            "water_soluble": True,
            "anti_inflammatory": False,
            "molecular_weight": 289.37,
            "logP": 1.83
        },
        "description": "Anticholinergic medication, can be toxic"
    },
    {
        "smiles": "CC(C)NCC(COc1ccc(COCCOC(C)C)cc1)O",
        "name": "Metoprolol",
        "properties": {
            "toxic": False,
            "water_soluble": True,
            "anti_inflammatory": False,
            "molecular_weight": 267.36,
            "logP": 1.88
        },
        "description": "Beta-blocker for cardiovascular conditions"
    },
    {
        "smiles": "CC(C)(C)NCC(O)COc1ccccc1CCOC",
        "name": "Atenolol",
        "properties": {
            "toxic": False,
            "water_soluble": True,
            "anti_inflammatory": False,
            "molecular_weight": 266.34,
            "logP": 0.16
        },
        "description": "Beta-blocker medication"
    },
    {
        "smiles": "CC1=C(C(=O)N(N1C)c2ccccc2)N(C)CS(=O)(=O)O",
        "name": "Metamizole",
        "properties": {
            "toxic": False,
            "water_soluble": True,
            "anti_inflammatory": True,
            "molecular_weight": 333.38,
            "logP": 0.86
        },
        "description": "Analgesic and antipyretic medication"
    },
    {
        "smiles": "c1ccc2c(c1)ccc3c2ccc4c3cccc4",
        "name": "Anthracene",
        "properties": {
            "toxic": True,
            "water_soluble": False,
            "anti_inflammatory": False,
            "molecular_weight": 178.23,
            "logP": 4.45
        },
        "description": "Polycyclic aromatic hydrocarbon, potential carcinogen"
    },
    {
        "smiles": "CC(=O)Oc1ccccc1C(=O)O",
        "name": "Acetylsalicylic acid",
        "properties": {
            "toxic": False,
            "water_soluble": False,
            "anti_inflammatory": True,
            "molecular_weight": 180.16,
            "logP": 1.19
        },
        "description": "Anti-inflammatory, analgesic (Aspirin)"
    },
]


def load_to_neo4j(config: Config):
    """Load sample data to Neo4j"""
    logger.info("Loading sample data to Neo4j...")

    neo4j = Neo4jConnector(
        uri=config.neo4j_uri,
        user=config.neo4j_user,
        password=config.neo4j_password,
        database=config.neo4j_database
    )

    # Create molecule nodes with properties
    for mol in SAMPLE_MOLECULES:
        query = """
        MERGE (m:Molecule {smiles: $smiles})
        SET m.name = $name,
            m.description = $description,
            m.toxic = $toxic,
            m.water_soluble = $water_soluble,
            m.anti_inflammatory = $anti_inflammatory,
            m.molecular_weight = $molecular_weight,
            m.logP = $logP
        RETURN m
        """
        neo4j.execute_query(query, {
            "smiles": mol["smiles"],
            "name": mol["name"],
            "description": mol["description"],
            "toxic": mol["properties"]["toxic"],
            "water_soluble": mol["properties"]["water_soluble"],
            "anti_inflammatory": mol["properties"]["anti_inflammatory"],
            "molecular_weight": mol["properties"]["molecular_weight"],
            "logP": mol["properties"]["logP"]
        })

    logger.info(f"Loaded {len(SAMPLE_MOLECULES)} molecules to Neo4j")
    neo4j.close()


def load_to_qdrant(config: Config):
    """Load sample fingerprints to Qdrant"""
    logger.info("Loading sample fingerprints to Qdrant...")

    qdrant = QdrantConnector(
        url=config.qdrant_url,
        api_key=config.qdrant_api_key,
        collection_name="molecular_fingerprints",
        vector_size=2048
    )

    # Generate fingerprints
    preprocessor = SMILESPreprocessor()
    fp_generator = MolecularFingerprints(
        fingerprint_type="morgan",
        radius=2,
        n_bits=2048
    )

    vectors = []
    for i, mol in enumerate(SAMPLE_MOLECULES):
        # Preprocess SMILES
        processed = preprocessor.preprocess(mol["smiles"])
        if not processed:
            logger.warning(f"Failed to process: {mol['name']}")
            continue

        # Generate fingerprint
        fp = fp_generator.generate_fingerprint(processed)
        if fp is None:
            logger.warning(f"Failed to generate fingerprint: {mol['name']}")
            continue

        # Convert to vector
        fp_array = fp_generator.fingerprint_to_array(fp)

        # Add to batch
        vectors.append({
            "id": i,
            "vector": fp_array.tolist(),
            "payload": {
                "smiles": mol["smiles"],
                "name": mol["name"],
                "description": mol["description"],
                **mol["properties"]
            }
        })

    # Upsert to Qdrant
    if vectors:
        qdrant.upsert_vectors(vectors)
        logger.info(f"Loaded {len(vectors)} fingerprints to Qdrant")


def main():
    """Main function"""
    logger.info("Loading sample molecular data...")

    try:
        config = Config()

        # Load to databases
        load_to_neo4j(config)
        load_to_qdrant(config)

        logger.info("Sample data loaded successfully!")
        logger.info("\nLoaded molecules:")
        for mol in SAMPLE_MOLECULES:
            logger.info(f"  - {mol['name']}: {mol['smiles']}")

        logger.info("\nYou can now:")
        logger.info("  1. Start the UI: python app.py")
        logger.info("  2. Try queries like:")
        logger.info("     - 'Is CCO toxic?' (Ethanol)")
        logger.info("     - 'Is CC(C)Cc1ccc(cc1)C(C)C(O)=O anti-inflammatory?' (Ibuprofen)")

    except Exception as e:
        logger.error(f"Failed to load sample data: {e}")
        logger.error("Make sure databases are running: docker-compose up -d")
        sys.exit(1)


if __name__ == "__main__":
    main()
