"""
GNN-based retrieval using knowledge-aware embeddings

Based on blueprint specifications:
- KPGT pre-trained embeddings
- ElementKG knowledge integration
- Top-30 results
- Cosine similarity in embedding space
- 2-7% improvement over fingerprints
"""

from typing import List, Optional
import numpy as np

from ..data.gnn_embeddings import GNNEmbedder
from ..data.models import RetrievalResult
from ..utils.logger import get_logger

logger = get_logger(__name__)


class GNNRetriever:
    """
    Retrieve molecules using GNN embeddings

    Pipeline:
    1. Generate GNN embedding for query molecule
    2. Search embedding space for similar molecules
    3. Return top-K with GNN similarity scores
    """

    def __init__(
        self,
        gnn_embedder: GNNEmbedder,
        embedding_database: dict,  # {smiles: embedding} mapping
        top_k: int = 30
    ):
        """
        Initialize GNN retriever

        Args:
            gnn_embedder: GNN embedding generator
            embedding_database: Pre-computed embeddings database
            top_k: Number of results (default: 30)
        """
        self.gnn_embedder = gnn_embedder
        self.embedding_db = embedding_database
        self.top_k = top_k

        logger.info(
            f"Initialized GNNRetriever with {len(embedding_database)} "
            f"pre-computed embeddings, top_k={top_k}"
        )

    def retrieve(
        self,
        query_smiles: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve similar molecules using GNN embeddings

        Args:
            query_smiles: Query molecule SMILES
            top_k: Number of results
            score_threshold: Minimum similarity threshold

        Returns:
            List of retrieval results sorted by GNN similarity
        """
        if top_k is None:
            top_k = self.top_k

        logger.info(f"GNN retrieval for: {query_smiles}")

        # Generate query embedding
        query_embedding = self.gnn_embedder.generate_embedding(query_smiles)
        if query_embedding is None:
            logger.error(f"Failed to generate GNN embedding for: {query_smiles}")
            return []

        # Search embedding database
        similarities = []
        for smiles, embedding in self.embedding_db.items():
            if smiles == query_smiles:
                continue  # Skip self

            if embedding is None:
                continue

            sim = self.gnn_embedder.calculate_similarity(
                query_embedding,
                embedding,
                metric='cosine'
            )

            if score_threshold is not None and sim < score_threshold:
                continue

            similarities.append((smiles, sim, embedding))

        # Sort by descending similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        similarities = similarities[:top_k]

        # Convert to RetrievalResult
        results = []
        for smiles, sim, _ in similarities:
            result = RetrievalResult(
                smiles=smiles,
                score=sim,
                source='gnn',
                properties={},  # Properties fetched separately if needed
                gnn_score=sim
            )
            results.append(result)

        logger.info(f"Retrieved {len(results)} molecules via GNN search")
        return results

    def retrieve_with_embedding(
        self,
        query_embedding: np.ndarray,
        top_k: Optional[int] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve using pre-computed embedding

        Args:
            query_embedding: Pre-computed GNN embedding
            top_k: Number of results

        Returns:
            List of retrieval results
        """
        if top_k is None:
            top_k = self.top_k

        similarities = []
        for smiles, embedding in self.embedding_db.items():
            if embedding is None:
                continue

            sim = self.gnn_embedder.calculate_similarity(
                query_embedding,
                embedding,
                metric='cosine'
            )
            similarities.append((smiles, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        similarities = similarities[:top_k]

        results = []
        for smiles, sim in similarities:
            result = RetrievalResult(
                smiles=smiles,
                score=sim,
                source='gnn',
                properties={},
                gnn_score=sim
            )
            results.append(result)

        return results

    def add_to_database(
        self,
        smiles: str,
        embedding: Optional[np.ndarray] = None
    ) -> bool:
        """
        Add molecule embedding to database

        Args:
            smiles: Molecule SMILES
            embedding: Pre-computed embedding (or will be generated)

        Returns:
            True if successful
        """
        if embedding is None:
            embedding = self.gnn_embedder.generate_embedding(smiles)

        if embedding is None:
            logger.warning(f"Failed to add {smiles} to database")
            return False

        self.embedding_db[smiles] = embedding
        return True

    def bulk_add(
        self,
        smiles_list: List[str],
        batch_size: int = 32
    ) -> int:
        """
        Add multiple molecules to database in batch

        Args:
            smiles_list: List of SMILES
            batch_size: Batch size for embedding generation

        Returns:
            Number of successfully added molecules
        """
        embeddings = self.gnn_embedder.batch_generate(
            smiles_list,
            batch_size=batch_size
        )

        added = 0
        for smiles, embedding in zip(smiles_list, embeddings):
            if embedding is not None:
                self.embedding_db[smiles] = embedding
                added += 1

        logger.info(f"Added {added}/{len(smiles_list)} molecules to GNN database")
        return added
