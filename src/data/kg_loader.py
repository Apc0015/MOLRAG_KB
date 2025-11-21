"""
Knowledge Graph ETL (Extract, Transform, Load) pipelines

Based on blueprint specifications for loading:
- PrimeKG (4M relationships)
- DrugBank (10K drugs)
- ChEMBL (2M+ bioactivities)
- Hetionet (2.25M relationships)
- SPOKE (53M edges)
- Reactome (pathways)
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
import csv
import json
from dataclasses import dataclass

import pandas as pd
from tqdm import tqdm

from ..utils.logger import get_logger
from ..utils.database import Neo4jConnector

logger = get_logger(__name__)


@dataclass
class KGStats:
    """Statistics about loaded knowledge graph"""
    nodes: int = 0
    relationships: int = 0
    node_types: Dict[str, int] = None
    relationship_types: Dict[str, int] = None

    def __post_init__(self):
        if self.node_types is None:
            self.node_types = {}
        if self.relationship_types is None:
            self.relationship_types = {}


class KnowledgeGraphLoader:
    """
    Load and transform knowledge graphs into Neo4j

    Supports multiple KG formats and sources
    """

    def __init__(self, neo4j_connector: Neo4jConnector):
        """
        Initialize KG loader

        Args:
            neo4j_connector: Neo4j database connector
        """
        self.neo4j = neo4j_connector
        self.stats = KGStats()

    def load_primekg(
        self,
        nodes_file: Path,
        edges_file: Path,
        batch_size: int = 1000
    ) -> KGStats:
        """
        Load PrimeKG knowledge graph

        PrimeKG format:
        - Nodes: node_id, node_type, node_name, node_source
        - Edges: source, relation, target, source_type, target_type

        Args:
            nodes_file: Path to nodes CSV
            edges_file: Path to edges CSV
            batch_size: Batch size for bulk loading

        Returns:
            Loading statistics
        """
        logger.info("Loading PrimeKG...")

        # Load nodes
        logger.info(f"Loading nodes from {nodes_file}")
        nodes_df = pd.read_csv(nodes_file)

        node_batches = []
        for idx, row in tqdm(nodes_df.iterrows(), total=len(nodes_df), desc="Nodes"):
            node = {
                'id': str(row['node_id']),
                'type': row['node_type'],
                'name': row.get('node_name', ''),
                'source': row.get('node_source', '')
            }
            node_batches.append(node)

            if len(node_batches) >= batch_size:
                self._batch_create_nodes(node_batches)
                node_batches = []

        if node_batches:
            self._batch_create_nodes(node_batches)

        # Load edges
        logger.info(f"Loading edges from {edges_file}")
        edges_df = pd.read_csv(edges_file)

        edge_batches = []
        for idx, row in tqdm(edges_df.iterrows(), total=len(edges_df), desc="Edges"):
            edge = {
                'source': str(row['source']),
                'target': str(row['target']),
                'relation': row['relation'],
                'source_type': row.get('source_type', ''),
                'target_type': row.get('target_type', '')
            }
            edge_batches.append(edge)

            if len(edge_batches) >= batch_size:
                self._batch_create_relationships(edge_batches)
                edge_batches = []

        if edge_batches:
            self._batch_create_relationships(edge_batches)

        logger.info("PrimeKG loaded successfully")
        return self.stats

    def load_drugbank(
        self,
        drugbank_file: Path,
        batch_size: int = 1000
    ) -> KGStats:
        """
        Load DrugBank data

        Args:
            drugbank_file: Path to DrugBank XML or CSV
            batch_size: Batch size for bulk loading

        Returns:
            Loading statistics
        """
        logger.info("Loading DrugBank...")

        # Assuming CSV format with columns: drug_id, name, smiles, targets, indications
        df = pd.read_csv(drugbank_file)

        drug_batches = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="DrugBank"):
            drug = {
                'id': row['drug_id'],
                'name': row['name'],
                'smiles': row.get('smiles', ''),
                'type': 'Drug'
            }
            drug_batches.append(drug)

            if len(drug_batches) >= batch_size:
                self._batch_create_nodes(drug_batches)
                drug_batches = []

        if drug_batches:
            self._batch_create_nodes(drug_batches)

        logger.info("DrugBank loaded successfully")
        return self.stats

    def load_chembl(
        self,
        activities_file: Path,
        molecules_file: Path,
        targets_file: Path,
        batch_size: int = 1000
    ) -> KGStats:
        """
        Load ChEMBL bioactivity data

        Args:
            activities_file: Bioactivity data
            molecules_file: Molecule data
            targets_file: Target protein data
            batch_size: Batch size

        Returns:
            Loading statistics
        """
        logger.info("Loading ChEMBL...")

        # Load molecules
        molecules_df = pd.read_csv(molecules_file)
        mol_batches = []

        for idx, row in tqdm(molecules_df.iterrows(), total=len(molecules_df), desc="Molecules"):
            mol = {
                'id': row['chembl_id'],
                'smiles': row['canonical_smiles'],
                'name': row.get('pref_name', ''),
                'type': 'Molecule'
            }
            mol_batches.append(mol)

            if len(mol_batches) >= batch_size:
                self._batch_create_nodes(mol_batches)
                mol_batches = []

        if mol_batches:
            self._batch_create_nodes(mol_batches)

        # Load activities (relationships)
        activities_df = pd.read_csv(activities_file)
        activity_batches = []

        for idx, row in tqdm(activities_df.iterrows(), total=len(activities_df), desc="Activities"):
            activity = {
                'source': row['molecule_chembl_id'],
                'target': row['target_chembl_id'],
                'relation': 'HAS_ACTIVITY',
                'properties': {
                    'activity_type': row.get('standard_type', ''),
                    'activity_value': row.get('standard_value', ''),
                    'activity_units': row.get('standard_units', '')
                }
            }
            activity_batches.append(activity)

            if len(activity_batches) >= batch_size:
                self._batch_create_relationships(activity_batches)
                activity_batches = []

        if activity_batches:
            self._batch_create_relationships(activity_batches)

        logger.info("ChEMBL loaded successfully")
        return self.stats

    def _batch_create_nodes(self, nodes: List[Dict[str, Any]]) -> None:
        """
        Batch create nodes in Neo4j

        Args:
            nodes: List of node dictionaries
        """
        if not nodes:
            return

        # Group by node type for efficient creation
        nodes_by_type = {}
        for node in nodes:
            node_type = node.get('type', 'Node')
            if node_type not in nodes_by_type:
                nodes_by_type[node_type] = []
            nodes_by_type[node_type].append(node)

        for node_type, type_nodes in nodes_by_type.items():
            query = f"""
            UNWIND $nodes AS node
            MERGE (n:{node_type} {{id: node.id}})
            SET n += node
            """

            try:
                self.neo4j.execute_query(query, {'nodes': type_nodes})
                self.stats.nodes += len(type_nodes)
                self.stats.node_types[node_type] = \
                    self.stats.node_types.get(node_type, 0) + len(type_nodes)
            except Exception as e:
                logger.error(f"Failed to create {node_type} nodes: {e}")

    def _batch_create_relationships(self, edges: List[Dict[str, Any]]) -> None:
        """
        Batch create relationships in Neo4j

        Args:
            edges: List of edge dictionaries
        """
        if not edges:
            return

        # Group by relationship type
        edges_by_type = {}
        for edge in edges:
            rel_type = edge.get('relation', 'RELATED_TO').upper().replace(' ', '_')
            if rel_type not in edges_by_type:
                edges_by_type[rel_type] = []
            edges_by_type[rel_type].append(edge)

        for rel_type, type_edges in edges_by_type.items():
            query = f"""
            UNWIND $edges AS edge
            MATCH (source {{id: edge.source}})
            MATCH (target {{id: edge.target}})
            MERGE (source)-[r:{rel_type}]->(target)
            """

            # Add properties if present
            if type_edges[0].get('properties'):
                query += "\nSET r += edge.properties"

            try:
                self.neo4j.execute_query(query, {'edges': type_edges})
                self.stats.relationships += len(type_edges)
                self.stats.relationship_types[rel_type] = \
                    self.stats.relationship_types.get(rel_type, 0) + len(type_edges)
            except Exception as e:
                logger.error(f"Failed to create {rel_type} relationships: {e}")

    def create_indexes(self, indexes: List[str]) -> None:
        """
        Create indexes for performance

        Args:
            indexes: List of property names to index
        """
        logger.info("Creating indexes...")

        # Node type indexes
        node_types = ['Molecule', 'Protein', 'Disease', 'Pathway', 'Drug']

        for node_type in node_types:
            for prop in indexes:
                try:
                    self.neo4j.create_indexes([prop])
                    logger.info(f"Created index on {node_type}.{prop}")
                except Exception as e:
                    logger.warning(f"Failed to create index: {e}")

    def get_stats(self) -> KGStats:
        """Get loading statistics"""
        # Query actual counts from Neo4j
        try:
            node_count_query = "MATCH (n) RETURN count(n) as count"
            result = self.neo4j.execute_query(node_count_query)
            self.stats.nodes = result[0]['count'] if result else 0

            rel_count_query = "MATCH ()-[r]->() RETURN count(r) as count"
            result = self.neo4j.execute_query(rel_count_query)
            self.stats.relationships = result[0]['count'] if result else 0

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")

        return self.stats

    def validate_kg(self) -> Dict[str, Any]:
        """
        Validate knowledge graph integrity

        Returns:
            Validation results
        """
        logger.info("Validating knowledge graph...")

        validation = {
            'total_nodes': 0,
            'total_relationships': 0,
            'orphan_nodes': 0,
            'node_types': {},
            'relationship_types': {},
            'errors': []
        }

        try:
            # Check node types
            query = "MATCH (n) RETURN labels(n) as types, count(n) as count"
            results = self.neo4j.execute_query(query)

            for result in results:
                types = result['types']
                count = result['count']
                validation['node_types'][str(types)] = count
                validation['total_nodes'] += count

            # Check relationship types
            query = "MATCH ()-[r]->() RETURN type(r) as type, count(r) as count"
            results = self.neo4j.execute_query(query)

            for result in results:
                rel_type = result['type']
                count = result['count']
                validation['relationship_types'][rel_type] = count
                validation['total_relationships'] += count

            # Check orphan nodes
            query = "MATCH (n) WHERE NOT (n)-[]-() RETURN count(n) as count"
            results = self.neo4j.execute_query(query)
            validation['orphan_nodes'] = results[0]['count'] if results else 0

            logger.info(f"Validation complete: {validation['total_nodes']} nodes, "
                       f"{validation['total_relationships']} relationships")

        except Exception as e:
            validation['errors'].append(str(e))
            logger.error(f"Validation failed: {e}")

        return validation
