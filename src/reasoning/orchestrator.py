"""
Multi-agent orchestrator for MolRAG

Coordinates:
1. Planning Agent → Query classification
2. Triple Retrieval → Vector + Graph + GNN
3. CoT Reasoning → Generate reasoning chain
4. Synthesis Agent → Final prediction
"""

from typing import Dict, Any, Optional
from ..data.models import PredictionResult, RetrievalResult
from .agents import (
    PlanningAgent,
    GraphRetrievalAgent,
    VectorRetrievalAgent,
    GNNPredictionAgent,
    SynthesisAgent
)
from .cot_strategies import get_cot_strategy
from ..retrieval import TripleRetriever
from ..utils.logger import get_logger

logger = get_logger(__name__)


class MultiAgentOrchestrator:
    """
    Orchestrate multi-agent reasoning pipeline

    Pipeline:
    1. Planning: Classify query and select strategy
    2. Retrieval: Execute triple retrieval (parallel)
    3. CoT Reasoning: Generate reasoning chain
    4. Synthesis: LLM integration of all sources
    """

    def __init__(
        self,
        triple_retriever: TripleRetriever,
        synthesis_model: str = "gpt-4",
        api_key: Optional[str] = None,
        enable_path_cot: bool = True
    ):
        """
        Initialize multi-agent orchestrator

        Args:
            triple_retriever: Triple retrieval system
            synthesis_model: LLM model for synthesis (gpt-4, claude-3)
            api_key: API key for LLM
            enable_path_cot: Enable Path-CoT reasoning
        """
        self.triple_retriever = triple_retriever
        self.enable_path_cot = enable_path_cot

        # Initialize agents
        self.planning_agent = PlanningAgent()

        self.graph_agent = GraphRetrievalAgent(
            triple_retriever.graph_retriever
        )

        self.vector_agent = VectorRetrievalAgent(
            triple_retriever.vector_retriever
        )

        self.gnn_agent = GNNPredictionAgent(
            triple_retriever.gnn_retriever
        )

        self.synthesis_agent = SynthesisAgent(
            model=synthesis_model,
            api_key=api_key
        )

        logger.info(
            f"Initialized MultiAgentOrchestrator with {synthesis_model}"
        )

    def reason(
        self,
        query_smiles: str,
        property_query: str,
        cot_strategy: Optional[str] = None,
        final_top_k: int = 10
    ) -> PredictionResult:
        """
        Execute complete reasoning pipeline

        Args:
            query_smiles: Query molecule SMILES
            property_query: Property question
            cot_strategy: CoT strategy (overrides planning)
            final_top_k: Number of final retrieved molecules

        Returns:
            Complete prediction result
        """
        logger.info(f"Starting reasoning pipeline for: {query_smiles}")

        import time
        start_time = time.time()

        # Step 1: Planning
        logger.info("Step 1: Planning")
        strategy = self.planning_agent.execute(
            query_smiles=query_smiles,
            property_query=property_query
        )

        # Override CoT strategy if specified
        if cot_strategy:
            strategy['cot_strategy'] = cot_strategy

        logger.info(f"Strategy selected: {strategy}")

        # Step 2: Triple Retrieval
        logger.info("Step 2: Triple Retrieval")
        retrieval_results = self.triple_retriever.retrieve(
            query_smiles=query_smiles,
            enable_vector=strategy['enable_vector'],
            enable_graph=strategy['enable_graph'],
            enable_gnn=strategy['enable_gnn'],
            final_top_k=final_top_k
        )

        logger.info(f"Retrieved {len(retrieval_results)} molecules")

        # Step 3: CoT Reasoning
        logger.info(f"Step 3: CoT Reasoning ({strategy['cot_strategy']})")
        cot_strategy_obj = get_cot_strategy(strategy['cot_strategy'])

        cot_reasoning = cot_strategy_obj.generate_reasoning(
            query_smiles=query_smiles,
            property_query=property_query,
            retrieval_results=retrieval_results
        )

        logger.info(f"Generated CoT reasoning ({len(cot_reasoning)} chars)")

        # Step 4: Synthesis
        logger.info("Step 4: Synthesis")
        synthesis_result = self.synthesis_agent.execute(
            query_smiles=query_smiles,
            property_query=property_query,
            retrieval_results=retrieval_results,
            cot_reasoning=cot_reasoning
        )

        # Extract pathways
        pathways = []
        for result in retrieval_results:
            if result.pathway:
                pathways.append({
                    'smiles': result.smiles,
                    'pathway': result.pathway,
                    'hops': result.hops,
                    'relevance': result.path_relevance_score
                })

        # Build final result
        execution_time = time.time() - start_time

        prediction_result = PredictionResult(
            query_smiles=query_smiles,
            property_query=property_query,
            prediction=synthesis_result.get('prediction', 'Unknown'),
            confidence=synthesis_result.get('confidence', 0.5),
            reasoning=cot_reasoning,
            cot_strategy=strategy['cot_strategy'],
            retrieved_molecules=retrieval_results,
            pathways=pathways,
            citations=synthesis_result.get('citations', []),
            metadata={
                'execution_time_seconds': execution_time,
                'num_retrieved': len(retrieval_results),
                'strategy': strategy,
                'model': self.synthesis_agent.model
            }
        )

        logger.info(
            f"Reasoning complete: {prediction_result.prediction} "
            f"(confidence: {prediction_result.confidence:.2f}, "
            f"time: {execution_time:.2f}s)"
        )

        return prediction_result

    def batch_reason(
        self,
        queries: list,
        final_top_k: int = 10
    ) -> list:
        """
        Process multiple queries in batch

        Args:
            queries: List of (smiles, property_query) tuples
            final_top_k: Number of results per query

        Returns:
            List of prediction results
        """
        results = []

        for smiles, property_query in queries:
            try:
                result = self.reason(
                    query_smiles=smiles,
                    property_query=property_query,
                    final_top_k=final_top_k
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {smiles}: {e}")
                # Add error result
                results.append(None)

        return results
