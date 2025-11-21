"""
Triple Retriever: Orchestrates all three retrieval methods

Combines:
1. Vector retrieval (top-50)
2. Graph retrieval (top-40)
3. GNN retrieval (top-30)
4. Hybrid re-ranking → final top-10
"""

from typing import List, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor

from ..data.models import RetrievalResult
from .vector_retrieval import VectorRetriever
from .graph_retrieval import GraphRetriever
from .gnn_retrieval import GNNRetriever
from .reranker import HybridReranker
from ..utils.logger import get_logger

logger = get_logger(__name__)


class TripleRetriever:
    """
    Orchestrate triple retrieval system

    Executes vector, graph, and GNN retrieval in parallel,
    then applies hybrid re-ranking
    """

    def __init__(
        self,
        vector_retriever: VectorRetriever,
        graph_retriever: GraphRetriever,
        gnn_retriever: GNNRetriever,
        reranker: HybridReranker,
        parallel_execution: bool = True
    ):
        """
        Initialize triple retriever

        Args:
            vector_retriever: Vector retrieval module
            graph_retriever: Graph retrieval module
            gnn_retriever: GNN retrieval module
            reranker: Hybrid re-ranker
            parallel_execution: Execute retrievals in parallel
        """
        self.vector_retriever = vector_retriever
        self.graph_retriever = graph_retriever
        self.gnn_retriever = gnn_retriever
        self.reranker = reranker
        self.parallel_execution = parallel_execution

        logger.info("Initialized TripleRetriever with all three retrieval methods")

    def retrieve(
        self,
        query_smiles: str,
        enable_vector: bool = True,
        enable_graph: bool = True,
        enable_gnn: bool = True,
        final_top_k: Optional[int] = None
    ) -> List[RetrievalResult]:
        """
        Perform triple retrieval

        Args:
            query_smiles: Query molecule SMILES
            enable_vector: Enable vector retrieval
            enable_graph: Enable graph retrieval
            enable_gnn: Enable GNN retrieval
            final_top_k: Number of final results

        Returns:
            Re-ranked retrieval results
        """
        logger.info(f"Triple retrieval for: {query_smiles}")
        logger.info(
            f"Enabled: vector={enable_vector}, "
            f"graph={enable_graph}, gnn={enable_gnn}"
        )

        # Execute retrievals
        if self.parallel_execution:
            vector_results, graph_results, gnn_results = \
                self._retrieve_parallel(
                    query_smiles, enable_vector, enable_graph, enable_gnn
                )
        else:
            vector_results, graph_results, gnn_results = \
                self._retrieve_sequential(
                    query_smiles, enable_vector, enable_graph, enable_gnn
                )

        # Re-rank combined results
        final_results = self.reranker.rerank(
            vector_results=vector_results,
            graph_results=graph_results,
            gnn_results=gnn_results,
            top_k=final_top_k
        )

        logger.info(f"Triple retrieval complete: {len(final_results)} final results")
        return final_results

    def _retrieve_sequential(
        self,
        query_smiles: str,
        enable_vector: bool,
        enable_graph: bool,
        enable_gnn: bool
    ) -> tuple:
        """Execute retrievals sequentially"""
        vector_results = []
        graph_results = []
        gnn_results = []

        if enable_vector:
            try:
                vector_results = self.vector_retriever.retrieve(query_smiles)
            except Exception as e:
                logger.error(f"Vector retrieval failed: {e}")

        if enable_graph:
            try:
                graph_results = self.graph_retriever.retrieve(query_smiles)
            except Exception as e:
                logger.error(f"Graph retrieval failed: {e}")

        if enable_gnn:
            try:
                gnn_results = self.gnn_retriever.retrieve(query_smiles)
            except Exception as e:
                logger.error(f"GNN retrieval failed: {e}")

        return vector_results, graph_results, gnn_results

    def _retrieve_parallel(
        self,
        query_smiles: str,
        enable_vector: bool,
        enable_graph: bool,
        enable_gnn: bool
    ) -> tuple:
        """
        Execute retrievals in parallel using ThreadPoolExecutor

        This provides ~3x latency reduction per blueprint specifications
        """
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {}

            if enable_vector:
                futures['vector'] = executor.submit(
                    self.vector_retriever.retrieve, query_smiles
                )

            if enable_graph:
                futures['graph'] = executor.submit(
                    self.graph_retriever.retrieve, query_smiles
                )

            if enable_gnn:
                futures['gnn'] = executor.submit(
                    self.gnn_retriever.retrieve, query_smiles
                )

            # Collect results
            vector_results = []
            graph_results = []
            gnn_results = []

            if 'vector' in futures:
                try:
                    vector_results = futures['vector'].result(timeout=30)
                except Exception as e:
                    logger.error(f"Vector retrieval failed: {e}")

            if 'graph' in futures:
                try:
                    graph_results = futures['graph'].result(timeout=30)
                except Exception as e:
                    logger.error(f"Graph retrieval failed: {e}")

            if 'gnn' in futures:
                try:
                    gnn_results = futures['gnn'].result(timeout=30)
                except Exception as e:
                    logger.error(f"GNN retrieval failed: {e}")

        return vector_results, graph_results, gnn_results

    def explain_retrieval(
        self,
        results: List[RetrievalResult]
    ) -> dict:
        """
        Provide explanation of retrieval results

        Args:
            results: Retrieval results

        Returns:
            Explanation dictionary with statistics
        """
        explanation = {
            'total_results': len(results),
            'sources': {},
            'score_distribution': {},
            'pathway_info': []
        }

        # Count by source
        for result in results:
            source = result.source
            explanation['sources'][source] = \
                explanation['sources'].get(source, 0) + 1

        # Score statistics
        if results:
            scores = [r.score for r in results]
            explanation['score_distribution'] = {
                'min': min(scores),
                'max': max(scores),
                'mean': sum(scores) / len(scores),
                'top_score': scores[0] if scores else 0
            }

        # Pathway information (from graph retrieval)
        for result in results:
            if result.pathway:
                explanation['pathway_info'].append({
                    'smiles': result.smiles,
                    'pathway': result.pathway,
                    'hops': result.hops
                })

        return explanation
