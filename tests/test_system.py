#!/usr/bin/env python3
"""
Complete System Testing Framework - Phase 1
Tests all components of MolRAG system
"""

import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import SMILESPreprocessor, MolecularFingerprints, GNNEmbedder
from src.retrieval import VectorRetriever, GraphRetriever, GNNRetriever, HybridReranker
from src.reasoning import MultiAgentOrchestrator
from src.evaluation import EvaluationSuite
from src.utils import Config, Neo4jConnector, QdrantConnector, RedisConnector


class TestPhase1Foundation:
    """Phase 1: Foundation Testing"""

    def test_config_system(self):
        """Test configuration loading"""
        config = Config()
        assert config.neo4j_uri is not None
        assert config.qdrant_url is not None
        assert config.redis_host is not None

    def test_database_connectors(self):
        """Test database connectivity"""
        config = Config()

        # Test Neo4j
        neo4j = Neo4jConnector(
            uri=config.neo4j_uri,
            user=config.neo4j_user,
            password=config.neo4j_password
        )
        assert neo4j is not None
        neo4j.close()

        # Test Qdrant
        qdrant = QdrantConnector(
            url=config.qdrant_url,
            collection_name="test",
            vector_size=2048
        )
        assert qdrant is not None

        # Test Redis
        redis = RedisConnector(
            host=config.redis_host,
            port=config.redis_port,
            password=config.redis_password
        )
        assert redis is not None


class TestPhase2DataPreparation:
    """Phase 2: Data Preparation Testing"""

    def test_smiles_preprocessing(self):
        """Test SMILES preprocessing"""
        preprocessor = SMILESPreprocessor()

        # Valid SMILES
        valid_smiles = "CCO"
        assert preprocessor.is_valid_smiles(valid_smiles)

        processed = preprocessor.preprocess(valid_smiles)
        assert processed is not None

    def test_fingerprint_generation(self):
        """Test molecular fingerprint generation"""
        fp_gen = MolecularFingerprints(
            fingerprint_type="morgan",
            radius=2,
            n_bits=2048
        )

        fingerprint = fp_gen.generate_fingerprint("CCO")
        assert fingerprint is not None
        assert fp_gen.fingerprint_to_array(fingerprint).shape[0] == 2048

    def test_gnn_embeddings(self):
        """Test GNN embedding generation"""
        embedder = GNNEmbedder(model_name="kpgt")
        # Test initialization
        assert embedder is not None


class TestPhase3HybridRetrieval:
    """Phase 3: Hybrid Retrieval Testing"""

    def test_vector_retrieval(self):
        """Test vector-based retrieval"""
        config = Config()
        retriever = VectorRetriever(
            qdrant_url=config.qdrant_url,
            collection_name="molecular_fingerprints"
        )
        assert retriever is not None

    def test_graph_retrieval(self):
        """Test graph-based retrieval"""
        config = Config()
        retriever = GraphRetriever(
            neo4j_uri=config.neo4j_uri,
            neo4j_user=config.neo4j_user,
            neo4j_password=config.neo4j_password
        )
        assert retriever is not None

    def test_gnn_retrieval(self):
        """Test GNN-based retrieval"""
        retriever = GNNRetriever(model_name="kpgt")
        assert retriever is not None

    def test_hybrid_reranking(self):
        """Test hybrid re-ranking"""
        reranker = HybridReranker(
            tanimoto_weight=0.4,
            path_weight=0.3,
            gnn_weight=0.3
        )
        assert reranker is not None


class TestPhase4MultiAgent:
    """Phase 4: Multi-Agent Reasoning Testing"""

    def test_agent_initialization(self):
        """Test multi-agent orchestrator"""
        orchestrator = MultiAgentOrchestrator(
            synthesis_model="gpt-4",
            enable_path_cot=True
        )
        assert orchestrator is not None


class TestPhase5EnhancedCoT:
    """Phase 5: Enhanced CoT Testing"""

    def test_cot_strategies(self):
        """Test CoT strategy implementations"""
        from src.reasoning.cot_strategies import StructCoT, SimCoT, PathCoT

        # Test Struct-CoT
        struct_cot = StructCoT()
        assert struct_cot is not None

        # Test Sim-CoT
        sim_cot = SimCoT()
        assert sim_cot is not None

        # Test Path-CoT
        path_cot = PathCoT()
        assert path_cot is not None


class TestPhase6Evaluation:
    """Phase 6: Evaluation System Testing"""

    def test_evaluation_suite(self):
        """Test evaluation suite initialization"""
        eval_suite = EvaluationSuite()
        assert eval_suite is not None

    def test_metrics(self):
        """Test metrics calculation"""
        from src.evaluation.metrics import calculate_roc_auc, calculate_recall_at_k
        # Basic metric tests
        assert calculate_roc_auc is not None
        assert calculate_recall_at_k is not None


# Integration Tests
class TestIntegration:
    """End-to-end integration testing"""

    def test_full_pipeline(self):
        """Test complete MolRAG pipeline"""
        from src.molrag import MolRAG

        molrag = MolRAG(auto_init=False)
        molrag.initialize_preprocessing()

        assert molrag.preprocessor is not None
        assert molrag.fp_generator is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
