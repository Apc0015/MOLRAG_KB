#!/usr/bin/env python3
"""
Setup databases for MolRAG

Initializes:
- Neo4j with indexes
- Qdrant collections
- Redis cache
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import Config, Neo4jConnector, QdrantConnector, RedisConnector
from src.utils.logger import get_logger

logger = get_logger(__name__)


def setup_neo4j(config: Config):
    """Setup Neo4j database"""
    logger.info("Setting up Neo4j...")

    neo4j = Neo4jConnector(
        uri=config.neo4j_uri,
        user=config.neo4j_user,
        password=config.neo4j_password,
        database=config.neo4j_database
    )

    # Create indexes
    indexes = ['molecular_weight', 'smiles', 'target_id', 'disease_id', 'pathway_id']
    neo4j.create_indexes(indexes)

    logger.info("Neo4j setup complete")
    neo4j.close()


def setup_qdrant(config: Config):
    """Setup Qdrant vector database"""
    logger.info("Setting up Qdrant...")

    qdrant = QdrantConnector(
        url=config.qdrant_url,
        api_key=config.qdrant_api_key,
        collection_name="molecular_fingerprints",
        vector_size=2048
    )

    # Create collection with HNSW indexing
    qdrant.create_collection(hnsw_m=16, hnsw_ef_construct=200)

    logger.info("Qdrant setup complete")


def setup_redis(config: Config):
    """Setup Redis cache"""
    logger.info("Setting up Redis...")

    redis = RedisConnector(
        host=config.redis_host,
        port=config.redis_port,
        password=config.redis_password
    )

    # Test connection
    redis.client.ping()

    logger.info("Redis setup complete")


def main():
    """Main setup function"""
    logger.info("Starting database setup...")

    # Load configuration
    config = Config()

    try:
        # Setup databases
        setup_neo4j(config)
        setup_qdrant(config)
        setup_redis(config)

        logger.info("All databases setup successfully!")
        logger.info("\nNext steps:")
        logger.info("1. Load knowledge graphs: python scripts/load_knowledge_graphs.py")
        logger.info("2. Index molecules: python scripts/index_molecules.py")

    except Exception as e:
        logger.error(f"Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
