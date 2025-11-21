"""Data models for MolRAG"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Molecule(BaseModel):
    """Molecular structure and properties"""

    smiles: str = Field(..., description="SMILES string representation")
    name: Optional[str] = Field(None, description="Molecule name")
    properties: Dict[str, Any] = Field(
        default_factory=dict,
        description="Molecular properties (e.g., toxicity, solubility)"
    )
    fingerprint: Optional[List[int]] = Field(None, description="Molecular fingerprint")
    embedding: Optional[List[float]] = Field(None, description="GNN embedding")

    class Config:
        arbitrary_types_allowed = True


class RetrievalResult(BaseModel):
    """Result from retrieval system"""

    smiles: str = Field(..., description="Retrieved molecule SMILES")
    score: float = Field(..., description="Similarity/relevance score")
    source: str = Field(..., description="Retrieval source (vector/graph/gnn)")
    properties: Dict[str, Any] = Field(
        default_factory=dict,
        description="Molecular properties"
    )
    pathway: Optional[List[str]] = Field(
        None,
        description="Biological pathway (for graph retrieval)"
    )
    hops: Optional[int] = Field(
        None,
        description="Number of hops in graph traversal"
    )

    # Individual similarity scores
    tanimoto_score: Optional[float] = Field(None, description="Tanimoto similarity")
    path_relevance_score: Optional[float] = Field(
        None,
        description="Graph path relevance"
    )
    gnn_score: Optional[float] = Field(None, description="GNN embedding similarity")


class PredictionResult(BaseModel):
    """Final prediction result from MolRAG"""

    query_smiles: str = Field(..., description="Query molecule SMILES")
    property_query: str = Field(..., description="Property question")

    # Prediction
    prediction: str = Field(..., description="Predicted value (Yes/No or numeric)")
    confidence: float = Field(..., description="Confidence score (0-1)")

    # Reasoning
    reasoning: str = Field(..., description="Chain-of-Thought reasoning trace")
    cot_strategy: str = Field(..., description="CoT strategy used")

    # Supporting evidence
    retrieved_molecules: List[RetrievalResult] = Field(
        default_factory=list,
        description="Retrieved similar molecules"
    )
    pathways: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Biological pathways involved"
    )
    citations: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Literature citations"
    )

    # Metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (execution time, model used, etc.)"
    )


@dataclass
class FingerprintConfig:
    """Configuration for molecular fingerprint generation"""

    fingerprint_type: str = "morgan"  # morgan, rdkit, maccs
    radius: int = 2
    n_bits: int = 2048
    use_features: bool = False
    use_chirality: bool = True


@dataclass
class RetrievalConfig:
    """Configuration for retrieval system"""

    # Retrieval strategy
    enable_vector: bool = True
    enable_graph: bool = True
    enable_gnn: bool = True

    # Top-K values
    vector_top_k: int = 50
    graph_top_k: int = 40
    gnn_top_k: int = 30
    final_top_k: int = 10

    # Weights for re-ranking
    vector_weight: float = 0.4
    graph_weight: float = 0.3
    gnn_weight: float = 0.3

    # Graph traversal
    max_hops: int = 2
    relationship_types: List[str] = field(
        default_factory=lambda: ["BINDS", "TARGETS", "PARTICIPATES_IN"]
    )

    # Thresholds
    similarity_threshold: Optional[float] = None
    score_threshold: Optional[float] = None


@dataclass
class CoTConfig:
    """Configuration for Chain-of-Thought reasoning"""

    strategy: str = "sim_cot"  # struct_cot, sim_cot, path_cot
    num_examples: int = 4  # Few-shot learning
    max_tokens: int = 2048
    temperature: float = 0.1
