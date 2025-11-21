"""Database connectors for MolRAG"""

from typing import Any, Dict, List, Optional
from contextlib import contextmanager

import redis
from neo4j import GraphDatabase, Session
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from .logger import get_logger

logger = get_logger(__name__)


class Neo4jConnector:
    """
    Neo4j database connector for knowledge graph storage and traversal

    Based on blueprint specifications for graph database operations
    """

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str = "neo4j"
    ):
        """
        Initialize Neo4j connector

        Args:
            uri: Neo4j connection URI (e.g., bolt://localhost:7687)
            user: Username
            password: Password
            database: Database name
        """
        self.uri = uri
        self.user = user
        self.database = database

        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            logger.info(f"Connected to Neo4j at {uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    @contextmanager
    def session(self) -> Session:
        """Context manager for Neo4j sessions"""
        session = self.driver.session(database=self.database)
        try:
            yield session
        finally:
            session.close()

    def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute Cypher query

        Args:
            query: Cypher query string
            parameters: Query parameters

        Returns:
            List of result records as dictionaries
        """
        with self.session() as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]

    def create_indexes(self, indexes: List[str]) -> None:
        """
        Create indexes on specified properties for performance optimization

        Args:
            indexes: List of property names to index
        """
        for index in indexes:
            query = f"CREATE INDEX IF NOT EXISTS FOR (n:Molecule) ON (n.{index})"
            try:
                self.execute_query(query)
                logger.info(f"Created index on Molecule.{index}")
            except Exception as e:
                logger.warning(f"Failed to create index on {index}: {e}")

    def find_similar_molecules(
        self,
        smiles: str,
        similarity_threshold: float = 0.6,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Find similar molecules using Tanimoto similarity

        Note: Requires RDKit plugin for Neo4j for substructure search

        Args:
            smiles: Query molecule SMILES
            similarity_threshold: Minimum similarity score
            limit: Maximum number of results

        Returns:
            List of similar molecules with properties
        """
        query = """
        MATCH (m:Molecule)
        WHERE m.smiles <> $smiles
        WITH m, gds.similarity.cosine(m.fingerprint, $fingerprint) AS similarity
        WHERE similarity >= $threshold
        RETURN m.smiles AS smiles,
               m.properties AS properties,
               similarity
        ORDER BY similarity DESC
        LIMIT $limit
        """

        # Note: Fingerprint should be pre-computed and stored
        # This is a placeholder - actual implementation needs fingerprint calculation
        parameters = {
            "smiles": smiles,
            "fingerprint": [],  # Placeholder
            "threshold": similarity_threshold,
            "limit": limit
        }

        return self.execute_query(query, parameters)

    def traverse_graph(
        self,
        smiles: str,
        relationship_types: List[str],
        max_hops: int = 2,
        limit: int = 40
    ) -> List[Dict[str, Any]]:
        """
        Traverse knowledge graph using metapath reasoning

        Example: molecule→binds→protein→participates_in→pathway

        Args:
            smiles: Query molecule SMILES
            relationship_types: List of relationship types to follow
            max_hops: Maximum number of hops (1-2 recommended)
            limit: Maximum number of results

        Returns:
            List of related molecules with pathway information
        """
        # Build relationship pattern
        rel_pattern = "|".join(relationship_types)

        query = f"""
        MATCH path = (m1:Molecule {{smiles: $smiles}})
                     -[r:{rel_pattern}*1..{max_hops}]-
                     (m2:Molecule)
        WHERE m1 <> m2
        RETURN m2.smiles AS smiles,
               m2.properties AS properties,
               [rel IN relationships(path) | type(rel)] AS pathway,
               length(path) AS hops
        ORDER BY hops ASC
        LIMIT $limit
        """

        parameters = {
            "smiles": smiles,
            "limit": limit
        }

        return self.execute_query(query, parameters)

    def get_molecule_pathways(
        self,
        smiles: str
    ) -> List[Dict[str, Any]]:
        """
        Get biological pathways for a molecule

        Args:
            smiles: Molecule SMILES

        Returns:
            List of pathways with relationships
        """
        query = """
        MATCH (m:Molecule {smiles: $smiles})
              -[:BINDS|TARGETS]->
              (p:Protein)
              -[:PARTICIPATES_IN]->
              (pathway:Pathway)
        RETURN pathway.name AS pathway_name,
               pathway.id AS pathway_id,
               collect(p.name) AS proteins
        """

        return self.execute_query(query, {"smiles": smiles})

    def close(self):
        """Close database connection"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")


class QdrantConnector:
    """
    Qdrant vector database connector for molecular fingerprint storage

    Based on blueprint: HNSW indexing with sub-ms ANN search
    """

    def __init__(
        self,
        url: str,
        api_key: Optional[str] = None,
        collection_name: str = "molecular_fingerprints",
        vector_size: int = 2048,
        distance: Distance = Distance.COSINE
    ):
        """
        Initialize Qdrant connector

        Args:
            url: Qdrant server URL
            api_key: API key (optional)
            collection_name: Name of the collection
            vector_size: Dimension of vectors (2048 for Morgan ECFP4)
            distance: Distance metric (COSINE, DOT, EUCLID)
        """
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.distance = distance

        try:
            self.client = QdrantClient(url=url, api_key=api_key)
            logger.info(f"Connected to Qdrant at {url}")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise

    def create_collection(
        self,
        hnsw_m: int = 16,
        hnsw_ef_construct: int = 200
    ) -> None:
        """
        Create collection with HNSW indexing

        Based on blueprint specs: M=16, ef_construct=200

        Args:
            hnsw_m: Number of connections per layer
            hnsw_ef_construct: Size of dynamic candidate list for construction
        """
        from qdrant_client.models import HnswConfigDiff

        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=self.distance
                ),
                hnsw_config=HnswConfigDiff(
                    m=hnsw_m,
                    ef_construct=hnsw_ef_construct
                )
            )
            logger.info(f"Created collection: {self.collection_name}")
        except Exception as e:
            logger.warning(f"Collection may already exist: {e}")

    def insert_vectors(
        self,
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
        ids: Optional[List[int]] = None
    ) -> None:
        """
        Insert molecular fingerprints into vector database

        Args:
            vectors: List of fingerprint vectors
            payloads: List of metadata (smiles, properties, etc.)
            ids: Optional list of point IDs
        """
        if ids is None:
            ids = list(range(len(vectors)))

        points = [
            PointStruct(
                id=point_id,
                vector=vector,
                payload=payload
            )
            for point_id, vector, payload in zip(ids, vectors, payloads)
        ]

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        logger.info(f"Inserted {len(vectors)} vectors into {self.collection_name}")

    def search(
        self,
        query_vector: List[float],
        top_k: int = 50,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar molecules using ANN

        Args:
            query_vector: Query fingerprint vector
            top_k: Number of results to return
            score_threshold: Minimum similarity score

        Returns:
            List of similar molecules with scores and metadata
        """
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            score_threshold=score_threshold
        )

        return [
            {
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload
            }
            for hit in results
        ]


class RedisConnector:
    """
    Redis connector for caching embeddings and LLM responses

    Based on blueprint: TTL=7 days for embedding cache
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        password: Optional[str] = None,
        db: int = 0
    ):
        """
        Initialize Redis connector

        Args:
            host: Redis host
            port: Redis port
            password: Redis password (optional)
            db: Database number
        """
        try:
            self.client = redis.Redis(
                host=host,
                port=port,
                password=password,
                db=db,
                decode_responses=False  # We'll handle encoding
            )
            self.client.ping()
            logger.info(f"Connected to Redis at {host}:{port}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set key-value pair with optional TTL

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (default: 7 days = 604800)

        Returns:
            True if successful
        """
        if ttl is None:
            ttl = 7 * 24 * 60 * 60  # 7 days in seconds

        import pickle
        serialized_value = pickle.dumps(value)

        return self.client.setex(key, ttl, serialized_value)

    def get(self, key: str) -> Optional[Any]:
        """
        Get value by key

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        value = self.client.get(key)
        if value is None:
            return None

        import pickle
        return pickle.loads(value)

    def delete(self, key: str) -> bool:
        """
        Delete key

        Args:
            key: Cache key

        Returns:
            True if key was deleted
        """
        return bool(self.client.delete(key))

    def exists(self, key: str) -> bool:
        """
        Check if key exists

        Args:
            key: Cache key

        Returns:
            True if key exists
        """
        return bool(self.client.exists(key))

    def clear_all(self) -> None:
        """Clear all keys in current database"""
        self.client.flushdb()
        logger.warning("Cleared all Redis cache")
