"""
GNN (Graph Neural Network) embedding generation for molecules

Based on blueprint specifications:
- KPGT (Knowledge-guided Pre-trained Graph Transformer)
- Pre-trained on 2M ChEMBL molecules
- ElementKG knowledge integration
- Expected 2-7% improvement over fingerprints
"""

from typing import List, Optional, Dict, Any
import numpy as np
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    from torch_geometric.data import Data
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Data = None

from ..utils.logger import get_logger

logger = get_logger(__name__)


class GNNEmbedder:
    """
    Generate molecular embeddings using pre-trained GNN models

    Supports:
    - KPGT (Knowledge-guided Pre-trained Graph Transformer)
    - KANO (Knowledge-aware Neural Operator)
    - Custom GNN models
    """

    def __init__(
        self,
        model_name: str = "kpgt",
        model_path: Optional[Path] = None,
        device: str = "cuda",
        embedding_dim: int = 512
    ):
        """
        Initialize GNN embedder

        Args:
            model_name: Model type (kpgt, kano, custom)
            model_path: Path to pre-trained model weights
            device: Device for inference (cuda, cpu)
            embedding_dim: Dimension of output embeddings
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch and PyTorch Geometric are required for GNN embeddings. "
                "Install with: pip install torch torch-geometric"
            )

        self.model_name = model_name.lower()
        self.model_path = model_path
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.embedding_dim = embedding_dim

        self.model = None

        if model_path and model_path.exists():
            self.load_model(model_path)
        else:
            logger.warning(
                f"Model path not found: {model_path}. "
                "Using mock embeddings. Download KPGT model from: "
                "https://github.com/lihan97/KPGT"
            )

    def load_model(self, model_path: Path) -> None:
        """
        Load pre-trained GNN model

        Args:
            model_path: Path to model checkpoint
        """
        try:
            checkpoint = torch.load(model_path, map_location=self.device)

            if self.model_name == "kpgt":
                self.model = self._build_kpgt_model()
            elif self.model_name == "kano":
                self.model = self._build_kano_model()
            else:
                raise ValueError(f"Unknown model: {self.model_name}")

            self.model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
            self.model.to(self.device)
            self.model.eval()

            logger.info(f"Loaded {self.model_name} model from {model_path}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None

    def _build_kpgt_model(self) -> nn.Module:
        """
        Build KPGT model architecture

        Note: This is a placeholder. Actual KPGT implementation requires
        the full model code from https://github.com/lihan97/KPGT
        """
        logger.warning("Using placeholder KPGT model. Implement actual architecture.")

        # Placeholder model
        class PlaceholderKPGT(nn.Module):
            def __init__(self, embedding_dim):
                super().__init__()
                self.embedding_dim = embedding_dim

            def forward(self, data):
                # Placeholder: return random embeddings
                batch_size = data.num_graphs if hasattr(data, 'num_graphs') else 1
                return torch.randn(batch_size, self.embedding_dim)

        return PlaceholderKPGT(self.embedding_dim)

    def _build_kano_model(self) -> nn.Module:
        """Build KANO model architecture"""
        logger.warning("Using placeholder KANO model.")
        return self._build_kpgt_model()  # Use same placeholder

    def smiles_to_graph(self, smiles: str) -> Optional[Data]:
        """
        Convert SMILES to PyTorch Geometric graph

        Args:
            smiles: SMILES string

        Returns:
            PyG Data object or None if conversion fails
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            # Add hydrogens for accurate representation
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=42)

            # Node features (atom features)
            node_features = []
            for atom in mol.GetAtoms():
                features = [
                    atom.GetAtomicNum(),
                    atom.GetDegree(),
                    atom.GetFormalCharge(),
                    atom.GetHybridization(),
                    atom.GetIsAromatic(),
                    atom.GetTotalNumHs()
                ]
                node_features.append(features)

            x = torch.tensor(node_features, dtype=torch.float)

            # Edge indices
            edge_indices = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                edge_indices.append([i, j])
                edge_indices.append([j, i])  # Undirected graph

            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

            # Create PyG Data object
            data = Data(x=x, edge_index=edge_index)

            return data

        except Exception as e:
            logger.error(f"Failed to convert SMILES to graph: {e}")
            return None

    def generate_embedding(self, smiles: str) -> Optional[np.ndarray]:
        """
        Generate embedding for a single molecule

        Args:
            smiles: SMILES string

        Returns:
            Embedding vector (numpy array) or None if generation fails
        """
        if self.model is None:
            # Return mock embedding if model not loaded
            return self._generate_mock_embedding(smiles)

        graph = self.smiles_to_graph(smiles)
        if graph is None:
            return None

        try:
            graph = graph.to(self.device)

            with torch.no_grad():
                embedding = self.model(graph)

            embedding = embedding.cpu().numpy()

            # Normalize embedding
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

            return embedding.flatten()

        except Exception as e:
            logger.error(f"Failed to generate embedding for '{smiles}': {e}")
            return None

    def _generate_mock_embedding(self, smiles: str) -> np.ndarray:
        """
        Generate deterministic mock embedding based on SMILES
        (Used when model is not available)
        """
        # Use hash of SMILES as seed for reproducibility
        import hashlib
        seed = int(hashlib.md5(smiles.encode()).hexdigest()[:8], 16)
        np.random.seed(seed)

        embedding = np.random.randn(self.embedding_dim)
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

        return embedding

    def batch_generate(
        self,
        smiles_list: List[str],
        batch_size: int = 32
    ) -> List[Optional[np.ndarray]]:
        """
        Generate embeddings for multiple molecules in batches

        Args:
            smiles_list: List of SMILES strings
            batch_size: Batch size for inference

        Returns:
            List of embedding arrays
        """
        embeddings = []

        for i in range(0, len(smiles_list), batch_size):
            batch = smiles_list[i:i + batch_size]

            for smiles in batch:
                emb = self.generate_embedding(smiles)
                embeddings.append(emb)

        logger.info(
            f"Generated {sum(e is not None for e in embeddings)}/{len(smiles_list)} "
            "GNN embeddings"
        )

        return embeddings

    def calculate_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        metric: str = "cosine"
    ) -> float:
        """
        Calculate similarity between two embeddings

        Args:
            embedding1: First embedding
            embedding2: Second embedding
            metric: Similarity metric (cosine, euclidean, dot)

        Returns:
            Similarity score
        """
        if metric == "cosine":
            # Cosine similarity
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            return dot_product / (norm1 * norm2 + 1e-8)

        elif metric == "euclidean":
            # Euclidean distance (converted to similarity)
            distance = np.linalg.norm(embedding1 - embedding2)
            return 1.0 / (1.0 + distance)

        elif metric == "dot":
            # Dot product
            return np.dot(embedding1, embedding2)

        else:
            raise ValueError(f"Unknown metric: {metric}")

    def find_most_similar(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: List[np.ndarray],
        top_k: int = 30,
        metric: str = "cosine"
    ) -> List[tuple]:
        """
        Find most similar embeddings

        Args:
            query_embedding: Query embedding
            candidate_embeddings: List of candidate embeddings
            top_k: Number of results
            metric: Similarity metric

        Returns:
            List of (index, similarity_score) tuples
        """
        similarities = []

        for idx, candidate_emb in enumerate(candidate_embeddings):
            if candidate_emb is None:
                continue

            sim = self.calculate_similarity(query_embedding, candidate_emb, metric)
            similarities.append((idx, sim))

        # Sort by descending similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]
