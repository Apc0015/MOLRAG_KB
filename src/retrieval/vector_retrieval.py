"""
Vector-based retrieval using molecular fingerprints

Based on blueprint specifications:
- Morgan fingerprints (ECFP4, 2048-bit)
- Qdrant vector database with HNSW indexing
- Top-50 results
- Tanimoto similarity
- Sub-millisecond ANN search
"""

from typing import List, Optional, Tuple
import numpy as np

from ..data.fingerprints import MolecularFingerprints
from ..data.models import RetrievalResult
from ..utils.database import QdrantConnector
from ..utils.logger import get_logger

logger = get_logger(__name__)


class VectorRetriever:
    """
    Retrieve molecules using fingerprint-based vector similarity

    Pipeline:
    1. Generate Morgan fingerprint for query molecule
    2. Search Qdrant vector database using HNSW
    3. Return top-K similar molecules with Tanimoto scores
    """

    def __init__(
        self,
        qdrant_connector: QdrantConnector,
        fingerprint_generator: MolecularFingerprints,
        top_k: int = 50
    ):
        """
        Initialize vector retriever

        Args:
            qdrant_connector: Qdrant database connector
            fingerprint_generator: Molecular fingerprint generator
            top_k: Number of results to return (default: 50)
        """
        self.qdrant = qdrant_connector
        self.fp_generator = fingerprint_generator
        self.top_k = top_k

        logger.info(f"Initialized VectorRetriever with top_k={top_k}")

    def retrieve(
        self,
        query_smiles: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve similar molecules using vector similarity

        Args:
            query_smiles: Query molecule SMILES
            top_k: Number of results (overrides default)
            score_threshold: Minimum similarity threshold

        Returns:
            List of retrieval results sorted by descending similarity
        """
        if top_k is None:
            top_k = self.top_k

        logger.info(f"Retrieving top-{top_k} similar molecules for: {query_smiles}")

        # Generate query fingerprint
        query_fp = self.fp_generator.generate_fingerprint(query_smiles)
        if query_fp is None:
            logger.error(f"Failed to generate fingerprint for: {query_smiles}")
            return []

        # Convert to list for Qdrant
        query_vector = self.fp_generator.fingerprint_to_list(query_fp)

        # Search vector database
        try:
            search_results = self.qdrant.search(
                query_vector=query_vector,
                top_k=top_k,
                score_threshold=score_threshold
            )
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

        # Convert to RetrievalResult objects
        results = []
        for hit in search_results:
            payload = hit['payload']

            result = RetrievalResult(
                smiles=payload.get('smiles', ''),
                score=hit['score'],
                source='vector',
                properties=payload.get('properties', {}),
                tanimoto_score=hit['score']  # Qdrant score is Tanimoto similarity
            )
            results.append(result)

        logger.info(f"Retrieved {len(results)} molecules via vector search")
        return results

    def retrieve_with_fingerprint(
        self,
        query_fingerprint: np.ndarray,
        top_k: Optional[int] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve using pre-computed fingerprint

        Args:
            query_fingerprint: Pre-computed fingerprint array
            top_k: Number of results

        Returns:
            List of retrieval results
        """
        if top_k is None:
            top_k = self.top_k

        query_vector = query_fingerprint.tolist()

        try:
            search_results = self.qdrant.search(
                query_vector=query_vector,
                top_k=top_k
            )
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

        results = []
        for hit in search_results:
            payload = hit['payload']
            result = RetrievalResult(
                smiles=payload.get('smiles', ''),
                score=hit['score'],
                source='vector',
                properties=payload.get('properties', {}),
                tanimoto_score=hit['score']
            )
            results.append(result)

        return results

    def bulk_retrieve(
        self,
        query_smiles_list: List[str],
        top_k: Optional[int] = None
    ) -> List[List[RetrievalResult]]:
        """
        Retrieve for multiple queries in batch

        Args:
            query_smiles_list: List of query SMILES
            top_k: Number of results per query

        Returns:
            List of result lists (one per query)
        """
        all_results = []

        for smiles in query_smiles_list:
            results = self.retrieve(smiles, top_k=top_k)
            all_results.append(results)

        return all_results
