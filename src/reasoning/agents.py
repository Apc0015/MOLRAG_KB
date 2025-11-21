"""
Multi-agent architecture following CLADD pattern

Agents:
1. Planning Agent - Query classification
2. Graph Retrieval Agent - KG queries
3. Vector Retrieval Agent - Fingerprint search
4. GNN Prediction Agent - GNN embeddings
5. Synthesis Agent - Multi-source integration (GPT-4/Claude)
"""

from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod

from ..data.models import RetrievalResult
from ..utils.logger import get_logger

logger = get_logger(__name__)


class BaseAgent(ABC):
    """Base class for all agents"""

    def __init__(self, name: str):
        self.name = name
        logger.info(f"Initialized {name} agent")

    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute agent task"""
        pass


class PlanningAgent(BaseAgent):
    """
    Classifies queries and selects retrieval strategy

    Query types:
    - structural: Based on molecular structure
    - mechanistic: Based on biological mechanisms
    - multi_property: Multiple properties
    """

    def __init__(self):
        super().__init__("PlanningAgent")

    def execute(
        self,
        query_smiles: str,
        property_query: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Classify query and determine strategy

        Args:
            query_smiles: Molecule SMILES
            property_query: Property question

        Returns:
            Strategy dictionary
        """
        logger.info(f"Planning query: {property_query}")

        # Simple keyword-based classification
        # In production, use LLM for better classification
        property_lower = property_query.lower()

        query_type = "structural"  # Default
        if any(word in property_lower for word in ['pathway', 'mechanism', 'target', 'binds']):
            query_type = "mechanistic"
        elif any(word in property_lower for word in ['and', 'multiple', 'various']):
            query_type = "multi_property"

        # Determine retrieval strategy
        strategy = {
            'query_type': query_type,
            'enable_vector': True,
            'enable_graph': query_type in ['mechanistic', 'multi_property'],
            'enable_gnn': True,
            'cot_strategy': 'sim_cot'  # Default to best performer
        }

        if query_type == 'mechanistic':
            strategy['cot_strategy'] = 'path_cot'  # Use pathway reasoning

        logger.info(f"Query classified as: {query_type}, strategy: {strategy}")
        return strategy


class GraphRetrievalAgent(BaseAgent):
    """
    Executes knowledge graph queries
    Handles anchoring for novel molecules
    """

    def __init__(self, graph_retriever):
        super().__init__("GraphRetrievalAgent")
        self.graph_retriever = graph_retriever

    def execute(
        self,
        query_smiles: str,
        max_hops: int = 2,
        top_k: int = 40,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute graph retrieval

        Args:
            query_smiles: Query molecule
            max_hops: Maximum graph traversal hops
            top_k: Number of results

        Returns:
            Graph retrieval results
        """
        logger.info(f"Graph agent retrieving for: {query_smiles}")

        try:
            results = self.graph_retriever.retrieve(
                query_smiles=query_smiles,
                max_hops=max_hops,
                top_k=top_k
            )

            return {
                'success': True,
                'results': results,
                'num_results': len(results)
            }

        except Exception as e:
            logger.error(f"Graph retrieval failed: {e}")
            return {
                'success': False,
                'results': [],
                'error': str(e)
            }


class VectorRetrievalAgent(BaseAgent):
    """
    Executes fingerprint-based vector search
    Configurable top-k
    """

    def __init__(self, vector_retriever):
        super().__init__("VectorRetrievalAgent")
        self.vector_retriever = vector_retriever

    def execute(
        self,
        query_smiles: str,
        top_k: int = 50,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute vector retrieval

        Args:
            query_smiles: Query molecule
            top_k: Number of results

        Returns:
            Vector retrieval results
        """
        logger.info(f"Vector agent retrieving for: {query_smiles}")

        try:
            results = self.vector_retriever.retrieve(
                query_smiles=query_smiles,
                top_k=top_k
            )

            return {
                'success': True,
                'results': results,
                'num_results': len(results)
            }

        except Exception as e:
            logger.error(f"Vector retrieval failed: {e}")
            return {
                'success': False,
                'results': [],
                'error': str(e)
            }


class GNNPredictionAgent(BaseAgent):
    """
    Applies pre-trained GNN models to contexts
    Knowledge-aware embeddings
    """

    def __init__(self, gnn_retriever):
        super().__init__("GNNPredictionAgent")
        self.gnn_retriever = gnn_retriever

    def execute(
        self,
        query_smiles: str,
        top_k: int = 30,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute GNN retrieval

        Args:
            query_smiles: Query molecule
            top_k: Number of results

        Returns:
            GNN retrieval results
        """
        logger.info(f"GNN agent retrieving for: {query_smiles}")

        try:
            results = self.gnn_retriever.retrieve(
                query_smiles=query_smiles,
                top_k=top_k
            )

            return {
                'success': True,
                'results': results,
                'num_results': len(results)
            }

        except Exception as e:
            logger.error(f"GNN retrieval failed: {e}")
            return {
                'success': False,
                'results': [],
                'error': str(e)
            }


class SynthesisAgent(BaseAgent):
    """
    Integrates information from all sources using LLM
    Uses GPT-4 or Claude for final synthesis
    """

    def __init__(
        self,
        model: str = "gpt-4",
        api_key: Optional[str] = None
    ):
        super().__init__("SynthesisAgent")
        self.model = model
        self.api_key = api_key

        # Import LLM client
        try:
            if "gpt" in model:
                from openai import OpenAI
                self.client = OpenAI(api_key=api_key)
            elif "claude" in model:
                from anthropic import Anthropic
                self.client = Anthropic(api_key=api_key)
            else:
                raise ValueError(f"Unknown model: {model}")
        except ImportError as e:
            logger.warning(f"LLM client not available: {e}")
            self.client = None

    def execute(
        self,
        query_smiles: str,
        property_query: str,
        retrieval_results: List[RetrievalResult],
        cot_reasoning: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Synthesize final prediction from all sources

        Args:
            query_smiles: Query molecule
            property_query: Property question
            retrieval_results: Retrieved similar molecules
            cot_reasoning: Chain-of-thought reasoning

        Returns:
            Final prediction with reasoning
        """
        logger.info("Synthesis agent integrating information...")

        if self.client is None:
            logger.warning("LLM client not available, returning mock response")
            return self._mock_synthesis(
                query_smiles, property_query, retrieval_results
            )

        # Build synthesis prompt
        prompt = self._build_synthesis_prompt(
            query_smiles,
            property_query,
            retrieval_results,
            cot_reasoning
        )

        # Call LLM
        try:
            response = self._call_llm(prompt)

            return {
                'success': True,
                'prediction': response.get('prediction', 'Unknown'),
                'confidence': response.get('confidence', 0.5),
                'reasoning': response.get('reasoning', ''),
                'citations': response.get('citations', [])
            }

        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _build_synthesis_prompt(
        self,
        query_smiles: str,
        property_query: str,
        results: List[RetrievalResult],
        cot_reasoning: str
    ) -> str:
        """Build prompt for LLM synthesis"""
        prompt = f"""You are a molecular property prediction expert.

Query Molecule: {query_smiles}
Property Question: {property_query}

Retrieved Similar Molecules:
"""

        for i, result in enumerate(results[:5], 1):  # Top 5
            prompt += f"\n{i}. SMILES: {result.smiles}\n"
            prompt += f"   Similarity: {result.score:.3f}\n"
            prompt += f"   Properties: {result.properties}\n"
            if result.pathway:
                prompt += f"   Pathway: {' → '.join(result.pathway)}\n"

        prompt += f"\nChain-of-Thought Reasoning:\n{cot_reasoning}\n"

        prompt += """
Based on the retrieved molecules and reasoning, predict the property.

Respond in JSON format:
{
    "prediction": "Yes/No or numerical value",
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation",
    "citations": []
}
"""

        return prompt

    def _call_llm(self, prompt: str) -> Dict[str, Any]:
        """Call LLM API"""
        if "gpt" in self.model:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1024
            )
            content = response.choices[0].message.content

        elif "claude" in self.model:
            response = self.client.messages.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1024
            )
            content = response.content[0].text

        # Parse JSON response
        import json
        try:
            return json.loads(content)
        except:
            # Fallback parsing
            return {
                'prediction': 'Unknown',
                'confidence': 0.5,
                'reasoning': content,
                'citations': []
            }

    def _mock_synthesis(
        self,
        query_smiles: str,
        property_query: str,
        results: List[RetrievalResult]
    ) -> Dict[str, Any]:
        """Mock synthesis for testing without LLM"""
        # Simple heuristic: majority vote from top results
        if results:
            avg_score = sum(r.score for r in results[:5]) / min(5, len(results))
            prediction = "Yes" if avg_score > 0.6 else "No"
            confidence = avg_score
        else:
            prediction = "Unknown"
            confidence = 0.5

        return {
            'success': True,
            'prediction': prediction,
            'confidence': confidence,
            'reasoning': f"Based on {len(results)} similar molecules with average similarity {confidence:.2f}",
            'citations': []
        }
