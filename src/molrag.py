"""
Main MolRAG class - Complete system orchestration

Training-Free Molecular Property Prediction with LLMs and Knowledge Graphs
"""

from typing import Optional, List
from pathlib import Path

from .data import MolecularFingerprints, SMILESPreprocessor, GNNEmbedder
from .data.models import PredictionResult, FingerprintConfig, RetrievalConfig
from .data.kg_loader import KnowledgeGraphLoader
from .utils import Config, Neo4jConnector, QdrantConnector, RedisConnector
from .utils.logger import get_logger
from .retrieval import (
    VectorRetriever,
    GraphRetriever,
    GNNRetriever,
    HybridReranker,
    TripleRetriever
)
from .reasoning import MultiAgentOrchestrator
from .evaluation import EvaluationSuite

logger = get_logger(__name__)


class MolRAG:
    """
    Complete MolRAG system

    Integrates:
    - Triple retrieval (vector + graph + GNN)
    - Multi-agent reasoning (CLADD pattern)
    - Chain-of-Thought strategies
    - LLM synthesis
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        kg_config_path: Optional[Path] = None,
        auto_init: bool = False
    ):
        """
        Initialize MolRAG system

        Args:
            config_path: Path to models.yaml configuration
            kg_config_path: Path to knowledge_graphs.yaml configuration
            auto_init: Automatically initialize all components
        """
        logger.info("Initializing MolRAG system...")

        # Load configuration
        self.config = Config(
            models_config_path=config_path or Path("config/models.yaml"),
            kg_config_path=kg_config_path or Path("config/knowledge_graphs.yaml")
        )

        # Components (initialized later)
        self.preprocessor = None
        self.fp_generator = None
        self.gnn_embedder = None
        self.triple_retriever = None
        self.orchestrator = None
        self.evaluation = None

        # Database connectors
        self.neo4j = None
        self.qdrant = None
        self.redis = None

        if auto_init:
            self.initialize_all()

        logger.info("MolRAG initialization complete")

    def initialize_all(self):
        """Initialize all components"""
        logger.info("Initializing all components...")

        self.initialize_databases()
        self.initialize_preprocessing()
        self.initialize_retrieval()
        self.initialize_reasoning()
        self.initialize_evaluation()

        logger.info("All components initialized")

    def initialize_databases(self):
        """Initialize database connections"""
        logger.info("Connecting to databases...")

        try:
            # Neo4j
            self.neo4j = Neo4jConnector(
                uri=self.config.neo4j_uri,
                user=self.config.neo4j_user,
                password=self.config.neo4j_password,
                database=self.config.neo4j_database
            )

            # Qdrant
            self.qdrant = QdrantConnector(
                url=self.config.qdrant_url,
                api_key=self.config.qdrant_api_key
            )

            # Redis
            self.redis = RedisConnector(
                host=self.config.redis_host,
                port=self.config.redis_port,
                password=self.config.redis_password
            )

            logger.info("Database connections established")

        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise

    def initialize_preprocessing(self):
        """Initialize preprocessing components"""
        logger.info("Initializing preprocessing...")

        # SMILES preprocessor
        self.preprocessor = SMILESPreprocessor(
            canonicalize=True,
            remove_salts=True,
            neutralize_charges=False
        )

        # Fingerprint generator
        self.fp_generator = MolecularFingerprints(
            fingerprint_type="morgan",
            radius=2,
            n_bits=2048,
            use_chirality=True
        )

        # GNN embedder (if model available)
        try:
            self.gnn_embedder = GNNEmbedder(
                model_name="kpgt",
                model_path=self.config.kpgt_model_path,
                device=self.config.device
            )
        except Exception as e:
            logger.warning(f"GNN embedder initialization failed: {e}")
            self.gnn_embedder = None

        logger.info("Preprocessing initialized")

    def initialize_retrieval(self):
        """Initialize triple retrieval system"""
        logger.info("Initializing retrieval system...")

        # Vector retriever
        vector_retriever = VectorRetriever(
            qdrant_connector=self.qdrant,
            fingerprint_generator=self.fp_generator,
            top_k=50
        )

        # Graph retriever
        graph_retriever = GraphRetriever(
            neo4j_connector=self.neo4j,
            max_hops=2,
            top_k=40
        )

        # GNN retriever (if available)
        gnn_embedding_db = {}  # Load pre-computed embeddings
        gnn_retriever = GNNRetriever(
            gnn_embedder=self.gnn_embedder,
            embedding_database=gnn_embedding_db,
            top_k=30
        )

        # Hybrid re-ranker
        reranker = HybridReranker(
            vector_weight=0.4,
            graph_weight=0.3,
            gnn_weight=0.3,
            final_top_k=10
        )

        # Triple retriever
        self.triple_retriever = TripleRetriever(
            vector_retriever=vector_retriever,
            graph_retriever=graph_retriever,
            gnn_retriever=gnn_retriever,
            reranker=reranker,
            parallel_execution=True
        )

        logger.info("Retrieval system initialized")

    def initialize_reasoning(self):
        """Initialize multi-agent reasoning"""
        logger.info("Initializing reasoning system...")

        self.orchestrator = MultiAgentOrchestrator(
            triple_retriever=self.triple_retriever,
            synthesis_model="gpt-4",
            api_key=self.config.openai_api_key
        )

        logger.info("Reasoning system initialized")

    def initialize_evaluation(self):
        """Initialize evaluation suite"""
        self.evaluation = EvaluationSuite()
        logger.info("Evaluation suite initialized")

    def predict(
        self,
        smiles: str,
        query: str,
        cot_strategy: str = "sim_cot",
        top_k: int = 10,
        preprocess: bool = True
    ) -> PredictionResult:
        """
        Predict molecular property

        Args:
            smiles: Molecule SMILES string
            query: Property query (e.g., "Is this molecule toxic?")
            cot_strategy: CoT strategy (struct_cot, sim_cot, path_cot)
            top_k: Number of retrieved molecules
            preprocess: Preprocess SMILES before prediction

        Returns:
            Prediction result with reasoning
        """
        logger.info(f"Predicting property for: {smiles}")

        # Preprocess SMILES
        if preprocess and self.preprocessor:
            smiles = self.preprocessor.preprocess(smiles)
            if smiles is None:
                raise ValueError("Invalid SMILES string")

        # Execute reasoning pipeline
        result = self.orchestrator.reason(
            query_smiles=smiles,
            property_query=query,
            cot_strategy=cot_strategy,
            final_top_k=top_k
        )

        return result

    def batch_predict(
        self,
        molecules: List[tuple],
        cot_strategy: str = "sim_cot",
        top_k: int = 10
    ) -> List[PredictionResult]:
        """
        Batch prediction for multiple molecules

        Args:
            molecules: List of (smiles, query) tuples
            cot_strategy: CoT strategy
            top_k: Number of retrieved molecules

        Returns:
            List of prediction results
        """
        results = []

        for smiles, query in molecules:
            try:
                result = self.predict(
                    smiles=smiles,
                    query=query,
                    cot_strategy=cot_strategy,
                    top_k=top_k
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Prediction failed for {smiles}: {e}")
                results.append(None)

        return results

    def load_knowledge_graph(
        self,
        kg_name: str,
        data_path: Path
    ):
        """
        Load knowledge graph into Neo4j

        Args:
            kg_name: Knowledge graph name (primekg, drugbank, chembl)
            data_path: Path to data files
        """
        logger.info(f"Loading {kg_name} knowledge graph...")

        kg_loader = KnowledgeGraphLoader(self.neo4j)

        if kg_name == "primekg":
            nodes_file = data_path / "primekg_nodes.csv"
            edges_file = data_path / "primekg_edges.csv"
            stats = kg_loader.load_primekg(nodes_file, edges_file)

        elif kg_name == "drugbank":
            drugbank_file = data_path / "drugbank.csv"
            stats = kg_loader.load_drugbank(drugbank_file)

        elif kg_name == "chembl":
            activities_file = data_path / "chembl_activities.csv"
            molecules_file = data_path / "chembl_molecules.csv"
            targets_file = data_path / "chembl_targets.csv"
            stats = kg_loader.load_chembl(
                activities_file, molecules_file, targets_file
            )

        else:
            raise ValueError(f"Unknown knowledge graph: {kg_name}")

        # Create indexes
        kg_loader.create_indexes(['molecular_weight', 'smiles', 'target_id'])

        logger.info(f"Loaded {kg_name}: {stats.nodes} nodes, {stats.relationships} relationships")
        return stats

    def close(self):
        """Close all connections"""
        logger.info("Closing MolRAG connections...")

        if self.neo4j:
            self.neo4j.close()
        if self.qdrant:
            pass  # Qdrant client auto-closes
        if self.redis:
            pass  # Redis client auto-closes

        logger.info("Connections closed")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
