"""
Graph-based retrieval using knowledge graph traversal

Based on blueprint specifications:
- Neo4j Cypher queries
- Metapath reasoning (molecule→protein→pathway)
- 1-2 hop traversal
- Relationship types: BINDS, TARGETS, PARTICIPATES_IN
- Top-40 results
- Path relevance scoring
"""

from typing import List, Optional, Dict, Any
from ..data.models import RetrievalResult
from ..utils.database import Neo4jConnector
from ..utils.logger import get_logger

logger = get_logger(__name__)


class GraphRetriever:
    """
    Retrieve molecules using knowledge graph traversal

    Pipeline:
    1. Find query molecule in KG
    2. Traverse relationships (1-2 hops)
    3. Find mechanistically related molecules
    4. Calculate path relevance scores
    """

    def __init__(
        self,
        neo4j_connector: Neo4jConnector,
        relationship_types: Optional[List[str]] = None,
        max_hops: int = 2,
        top_k: int = 40
    ):
        """
        Initialize graph retriever

        Args:
            neo4j_connector: Neo4j database connector
            relationship_types: Relationship types to traverse
            max_hops: Maximum path length (1-2 recommended)
            top_k: Number of results to return
        """
        self.neo4j = neo4j_connector
        self.max_hops = max_hops
        self.top_k = top_k

        if relationship_types is None:
            self.relationship_types = [
                'BINDS', 'TARGETS', 'INHIBITS', 'ACTIVATES',
                'PARTICIPATES_IN', 'TREATS', 'ASSOCIATES_WITH'
            ]
        else:
            self.relationship_types = relationship_types

        logger.info(
            f"Initialized GraphRetriever with max_hops={max_hops}, "
            f"top_k={top_k}, relationships={self.relationship_types}"
        )

    def retrieve(
        self,
        query_smiles: str,
        top_k: Optional[int] = None,
        max_hops: Optional[int] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve molecules via graph traversal

        Args:
            query_smiles: Query molecule SMILES
            top_k: Number of results
            max_hops: Maximum hops

        Returns:
            List of retrieval results with pathway information
        """
        if top_k is None:
            top_k = self.top_k
        if max_hops is None:
            max_hops = self.max_hops

        logger.info(f"Graph retrieval for: {query_smiles}")

        # Try direct traversal first
        results = self._traverse_metapaths(query_smiles, max_hops, top_k)

        # If no results, try anchoring (for novel molecules)
        if not results:
            logger.info("No direct results, trying anchoring strategy")
            results = self._anchor_and_traverse(query_smiles, max_hops, top_k)

        logger.info(f"Retrieved {len(results)} molecules via graph traversal")
        return results

    def _traverse_metapaths(
        self,
        query_smiles: str,
        max_hops: int,
        limit: int
    ) -> List[RetrievalResult]:
        """
        Traverse knowledge graph using metapaths

        Example path: Molecule→BINDS→Protein→PARTICIPATES_IN→Pathway→[reverse]→Molecule
        """
        rel_pattern = '|'.join(self.relationship_types)

        query = f"""
        MATCH (m1:Molecule {{smiles: $smiles}})
        MATCH path = (m1)-[r:{rel_pattern}*1..{max_hops}]-(m2:Molecule)
        WHERE m1 <> m2
        WITH m2, path,
             [rel IN relationships(path) | type(rel)] AS pathway,
             length(path) AS hops
        RETURN DISTINCT
               m2.smiles AS smiles,
               m2.properties AS properties,
               pathway,
               hops,
               count(path) AS path_count
        ORDER BY path_count DESC, hops ASC
        LIMIT $limit
        """

        try:
            neo4j_results = self.neo4j.execute_query(
                query,
                {'smiles': query_smiles, 'limit': limit}
            )
        except Exception as e:
            logger.error(f"Graph traversal failed: {e}")
            return []

        # Convert to RetrievalResult
        results = []
        for record in neo4j_results:
            # Calculate path relevance score
            hops = record['hops']
            path_count = record['path_count']
            path_score = self._calculate_path_relevance(
                pathway=record['pathway'],
                hops=hops,
                path_count=path_count
            )

            result = RetrievalResult(
                smiles=record['smiles'],
                score=path_score,
                source='graph',
                properties=record.get('properties', {}),
                pathway=record['pathway'],
                hops=hops,
                path_relevance_score=path_score
            )
            results.append(result)

        return results

    def _anchor_and_traverse(
        self,
        query_smiles: str,
        max_hops: int,
        limit: int
    ) -> List[RetrievalResult]:
        """
        Anchoring strategy for novel molecules not in KG

        Strategy:
        1. Find structurally similar molecules in KG (using fingerprints)
        2. Traverse from anchor molecules
        3. Aggregate results
        """
        # This requires integration with vector retrieval
        # Placeholder: return empty for now
        logger.warning("Anchoring not yet implemented - requires vector integration")
        return []

    def _calculate_path_relevance(
        self,
        pathway: List[str],
        hops: int,
        path_count: int,
        edge_confidence: float = 0.8
    ) -> float:
        """
        Calculate path relevance score

        Based on blueprint formula:
        - Edge confidence weight: 0.6
        - Hop distance penalty: 0.2 per hop
        - Path frequency boost

        Args:
            pathway: List of relationship types in path
            hops: Number of hops
            path_count: Number of paths between nodes
            edge_confidence: Confidence in edge quality

        Returns:
            Path relevance score (0-1)
        """
        # Base score from edge confidence
        base_score = edge_confidence * 0.6

        # Penalty for longer paths (0.2 per hop)
        hop_penalty = 0.2 * hops
        hop_score = max(0, 1.0 - hop_penalty)

        # Boost for relationship types
        relationship_boost = 0.0
        high_value_rels = ['BINDS', 'TARGETS', 'TREATS']
        for rel in pathway:
            if rel in high_value_rels:
                relationship_boost += 0.1

        # Frequency boost (log scale)
        import math
        frequency_boost = min(0.2, math.log(1 + path_count) * 0.05)

        # Combined score
        score = base_score * hop_score + relationship_boost + frequency_boost
        score = min(1.0, score)  # Cap at 1.0

        return score

    def get_molecule_pathways(
        self,
        smiles: str
    ) -> List[Dict[str, Any]]:
        """
        Get biological pathways for a molecule

        Args:
            smiles: Molecule SMILES

        Returns:
            List of pathway dictionaries
        """
        query = """
        MATCH (m:Molecule {smiles: $smiles})
              -[:BINDS|:TARGETS]->
              (p:Protein)
              -[:PARTICIPATES_IN]->
              (pathway:Pathway)
        RETURN pathway.name AS pathway_name,
               pathway.id AS pathway_id,
               pathway.description AS description,
               collect(DISTINCT p.name) AS proteins
        """

        try:
            results = self.neo4j.execute_query(query, {'smiles': smiles})
            return results
        except Exception as e:
            logger.error(f"Failed to get pathways: {e}")
            return []

    def explain_connection(
        self,
        smiles1: str,
        smiles2: str,
        max_hops: int = 3
    ) -> List[List[str]]:
        """
        Find and explain connections between two molecules

        Args:
            smiles1: First molecule
            smiles2: Second molecule
            max_hops: Maximum path length

        Returns:
            List of paths (each path is a list of relationship types)
        """
        rel_pattern = '|'.join(self.relationship_types)

        query = f"""
        MATCH (m1:Molecule {{smiles: $smiles1}})
        MATCH (m2:Molecule {{smiles: $smiles2}})
        MATCH path = shortestPath((m1)-[r:{rel_pattern}*1..{max_hops}]-(m2))
        RETURN [rel IN relationships(path) | type(rel)] AS pathway,
               [node IN nodes(path) | labels(node)[0] + ': ' + coalesce(node.name, node.id)] AS entities
        LIMIT 5
        """

        try:
            results = self.neo4j.execute_query(
                query,
                {'smiles1': smiles1, 'smiles2': smiles2}
            )
            return [r['pathway'] for r in results]
        except Exception as e:
            logger.error(f"Failed to explain connection: {e}")
            return []
