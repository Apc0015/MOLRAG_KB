"""
Hybrid re-ranking system for triple retrieval

Based on blueprint formula:
Combined_Score = 0.4 × Tanimoto + 0.3 × PathRelevance + 0.3 × GNN_Similarity

Combines results from:
1. Vector retrieval (fingerprint similarity)
2. Graph retrieval (knowledge graph traversal)
3. GNN retrieval (embedding similarity)
"""

from typing import List, Dict, Optional
from collections import defaultdict

from ..data.models import RetrievalResult
from ..utils.logger import get_logger

logger = get_logger(__name__)


class HybridReranker:
    """
    Re-rank molecules using combined scores from multiple retrieval sources

    Pipeline:
    1. Merge results from vector, graph, and GNN retrieval
    2. Normalize individual scores
    3. Calculate weighted combined score
    4. Select top-K final results
    """

    def __init__(
        self,
        vector_weight: float = 0.4,
        graph_weight: float = 0.3,
        gnn_weight: float = 0.3,
        final_top_k: int = 10,
        normalize_scores: bool = True
    ):
        """
        Initialize hybrid re-ranker

        Args:
            vector_weight: Weight for vector (fingerprint) scores
            graph_weight: Weight for graph (pathway) scores
            gnn_weight: Weight for GNN embedding scores
            final_top_k: Number of final results
            normalize_scores: Whether to normalize scores before combining
        """
        self.vector_weight = vector_weight
        self.graph_weight = graph_weight
        self.gnn_weight = gnn_weight
        self.final_top_k = final_top_k
        self.normalize_scores = normalize_scores

        # Validate weights sum to 1.0
        total_weight = vector_weight + graph_weight + gnn_weight
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(
                f"Weights sum to {total_weight}, normalizing to 1.0"
            )
            self.vector_weight /= total_weight
            self.graph_weight /= total_weight
            self.gnn_weight /= total_weight

        logger.info(
            f"Initialized HybridReranker: "
            f"vector={self.vector_weight:.2f}, "
            f"graph={self.graph_weight:.2f}, "
            f"gnn={self.gnn_weight:.2f}, "
            f"top_k={final_top_k}"
        )

    def rerank(
        self,
        vector_results: List[RetrievalResult],
        graph_results: List[RetrievalResult],
        gnn_results: List[RetrievalResult],
        top_k: Optional[int] = None
    ) -> List[RetrievalResult]:
        """
        Re-rank results using hybrid scoring

        Args:
            vector_results: Results from vector retrieval
            graph_results: Results from graph retrieval
            gnn_results: Results from GNN retrieval
            top_k: Number of final results (overrides default)

        Returns:
            Re-ranked results with combined scores
        """
        if top_k is None:
            top_k = self.final_top_k

        logger.info(
            f"Re-ranking: {len(vector_results)} vector + "
            f"{len(graph_results)} graph + {len(gnn_results)} GNN results"
        )

        # Merge all results by SMILES
        merged = self._merge_results(vector_results, graph_results, gnn_results)

        # Normalize scores if requested
        if self.normalize_scores:
            merged = self._normalize_merged_scores(merged)

        # Calculate combined scores
        scored_results = []
        for smiles, data in merged.items():
            combined_score = self._calculate_combined_score(data)

            # Create result with combined score
            result = data.get('result')  # Get original result object
            if result is None:
                # Create new result if none exists
                result = RetrievalResult(
                    smiles=smiles,
                    score=combined_score,
                    source='hybrid',
                    properties={},
                    tanimoto_score=data.get('tanimoto'),
                    path_relevance_score=data.get('path_relevance'),
                    gnn_score=data.get('gnn')
                )
            else:
                # Update existing result
                result.score = combined_score
                result.source = 'hybrid'
                result.tanimoto_score = data.get('tanimoto')
                result.path_relevance_score = data.get('path_relevance')
                result.gnn_score = data.get('gnn')

            scored_results.append(result)

        # Sort by combined score (descending)
        scored_results.sort(key=lambda x: x.score, reverse=True)

        # Select top-K
        final_results = scored_results[:top_k]

        logger.info(
            f"Re-ranking complete: {len(final_results)} final results "
            f"(avg score: {sum(r.score for r in final_results) / len(final_results):.3f})"
        )

        return final_results

    def _merge_results(
        self,
        vector_results: List[RetrievalResult],
        graph_results: List[RetrievalResult],
        gnn_results: List[RetrievalResult]
    ) -> Dict[str, dict]:
        """
        Merge results from all sources by SMILES

        Returns:
            Dictionary mapping SMILES to score data
        """
        merged = defaultdict(lambda: {
            'tanimoto': None,
            'path_relevance': None,
            'gnn': None,
            'result': None
        })

        # Add vector results (Tanimoto scores)
        for result in vector_results:
            smiles = result.smiles
            merged[smiles]['tanimoto'] = result.tanimoto_score or result.score
            if merged[smiles]['result'] is None:
                merged[smiles]['result'] = result

        # Add graph results (path relevance)
        for result in graph_results:
            smiles = result.smiles
            merged[smiles]['path_relevance'] = result.path_relevance_score or result.score
            if merged[smiles]['result'] is None:
                merged[smiles]['result'] = result
            # Preserve pathway information
            if result.pathway:
                merged[smiles]['result'].pathway = result.pathway
                merged[smiles]['result'].hops = result.hops

        # Add GNN results
        for result in gnn_results:
            smiles = result.smiles
            merged[smiles]['gnn'] = result.gnn_score or result.score
            if merged[smiles]['result'] is None:
                merged[smiles]['result'] = result

        return dict(merged)

    def _normalize_merged_scores(
        self,
        merged: Dict[str, dict]
    ) -> Dict[str, dict]:
        """
        Normalize scores within each source to [0, 1] range

        Uses min-max normalization
        """
        # Collect all scores by source
        tanimoto_scores = [d['tanimoto'] for d in merged.values() if d['tanimoto'] is not None]
        path_scores = [d['path_relevance'] for d in merged.values() if d['path_relevance'] is not None]
        gnn_scores = [d['gnn'] for d in merged.values() if d['gnn'] is not None]

        # Calculate min/max for normalization
        tanimoto_min, tanimoto_max = (min(tanimoto_scores), max(tanimoto_scores)) if tanimoto_scores else (0, 1)
        path_min, path_max = (min(path_scores), max(path_scores)) if path_scores else (0, 1)
        gnn_min, gnn_max = (min(gnn_scores), max(gnn_scores)) if gnn_scores else (0, 1)

        # Normalize each molecule's scores
        for smiles, data in merged.items():
            if data['tanimoto'] is not None:
                data['tanimoto'] = self._normalize_score(
                    data['tanimoto'], tanimoto_min, tanimoto_max
                )

            if data['path_relevance'] is not None:
                data['path_relevance'] = self._normalize_score(
                    data['path_relevance'], path_min, path_max
                )

            if data['gnn'] is not None:
                data['gnn'] = self._normalize_score(
                    data['gnn'], gnn_min, gnn_max
                )

        return merged

    def _normalize_score(
        self,
        score: float,
        min_val: float,
        max_val: float
    ) -> float:
        """Min-max normalization"""
        if max_val - min_val < 1e-8:
            return 0.5  # All scores are the same
        return (score - min_val) / (max_val - min_val)

    def _calculate_combined_score(self, data: dict) -> float:
        """
        Calculate weighted combined score

        Formula: 0.4 × Tanimoto + 0.3 × PathRelevance + 0.3 × GNN

        If a score is missing, redistribute its weight proportionally
        """
        score = 0.0
        total_weight = 0.0

        if data['tanimoto'] is not None:
            score += self.vector_weight * data['tanimoto']
            total_weight += self.vector_weight

        if data['path_relevance'] is not None:
            score += self.graph_weight * data['path_relevance']
            total_weight += self.graph_weight

        if data['gnn'] is not None:
            score += self.gnn_weight * data['gnn']
            total_weight += self.gnn_weight

        # Normalize by actual weight used
        if total_weight > 0:
            score = score / total_weight

        return score
