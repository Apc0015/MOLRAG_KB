# 📖 MolRAG Complete Usage Guide

## 📋 Table of Contents

1. [Introduction](#introduction)
2. [Installation & Setup](#installation--setup)
3. [Using the Gradio UI](#using-the-gradio-ui)
4. [Using the Python API](#using-the-python-api)
5. [Data Processing](#data-processing)
6. [Triple Retrieval System](#triple-retrieval-system)
7. [Multi-Agent Reasoning](#multi-agent-reasoning)
8. [Chain-of-Thought Strategies](#chain-of-thought-strategies)
9. [Evaluation & Metrics](#evaluation--metrics)
10. [Configuration](#configuration)
11. [Knowledge Graph Management](#knowledge-graph-management)
12. [Advanced Usage](#advanced-usage)
13. [Batch Processing](#batch-processing)
14. [API Reference](#api-reference)
15. [Examples](#examples)
16. [Troubleshooting](#troubleshooting)

---

## 🎯 Introduction

MolRAG is a **training-free molecular property prediction system** that combines:
- **Retrieval-Augmented Generation (RAG)** with Large Language Models
- **Knowledge Graph reasoning** for mechanistic understanding
- **Multi-agent architecture** for specialized tasks
- **Chain-of-Thought prompting** for interpretable predictions

**Key Benefits:**
- ✅ No training required - deploy immediately
- ✅ Achieves 72-96% of supervised method accuracy
- ✅ 20-45% improvement over direct LLM predictions
- ✅ Interpretable pathway-level explanations
- ✅ Continuous knowledge updates without retraining

---

## 📦 Installation & Setup

### Prerequisites

- **Python 3.9+**
- **Docker** (for databases)
- **8GB+ RAM** (16GB recommended)
- **CUDA GPU** (optional, for GNN models)

### Step-by-Step Installation

#### 1. Clone Repository

```bash
git clone https://github.com/Apc0015/MOLRAG_KB.git
cd MOLRAG_KB
git checkout claude/review-project-codebase-01XHMakVV7QgEdagpdU78VWB
```

#### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

#### 3. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install all dependencies
pip install -r requirements.txt
```

**Dependencies include:**
- RDKit (molecular chemistry)
- PyTorch + PyTorch Geometric (GNN models)
- Neo4j, Qdrant, Redis clients
- LlamaIndex, LangChain (LLM orchestration)
- OpenAI, Anthropic (API clients)
- Gradio (web UI)

#### 4. Verify Installation

```bash
python verify_installation.py
```

**Expected output:**
```
✓ Git Branch: correct
✓ File Structure: 6,646 lines of code
⚠ Dependencies: install with pip install -r requirements.txt
```

#### 5. Set Up Databases

**Start Docker containers:**

```bash
# Neo4j (Knowledge Graph)
docker run -d --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/molrag123 \
  -v $PWD/data/neo4j:/data \
  neo4j:latest

# Qdrant (Vector Database)
docker run -d --name qdrant \
  -p 6333:6333 \
  -v $PWD/data/qdrant:/qdrant/storage \
  qdrant/qdrant:latest

# Redis (Cache)
docker run -d --name redis \
  -p 6379:6379 \
  -v $PWD/data/redis:/data \
  redis:latest
```

**Verify databases are running:**

```bash
# Check containers
docker ps

# Test Neo4j
curl http://localhost:7474

# Test Qdrant
curl http://localhost:6333/health

# Test Redis
redis-cli ping  # Should return: PONG
```

#### 6. Configure Environment

```bash
# Create .env file
cp .env.example .env
```

**Edit `.env` with your settings:**

```bash
# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=molrag123

# Qdrant Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_API_KEY=  # Optional

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=  # Optional
REDIS_DB=0

# LLM API Keys
OPENAI_API_KEY=sk-...  # Get from https://platform.openai.com/
ANTHROPIC_API_KEY=sk-ant-...  # Get from https://console.anthropic.com/

# Model Configuration
LLM_MODEL=gpt-4  # or gpt-4-turbo, claude-3-opus-20240229
EMBEDDING_MODEL=text-embedding-3-small
GNN_MODEL=kpgt  # or kano

# Retrieval Settings
RETRIEVAL_TOP_K=10
VECTOR_WEIGHT=0.4
GRAPH_WEIGHT=0.3
GNN_WEIGHT=0.3

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/molrag.log
```

#### 7. Initialize Databases

```bash
# Initialize database schemas and indices
python scripts/setup_databases.py
```

This will:
- ✅ Create Neo4j graph schema
- ✅ Create Qdrant collections
- ✅ Initialize Redis keys
- ✅ Set up indices for fast retrieval

#### 8. Load Knowledge Graphs (Optional but Recommended)

**Download knowledge graphs:**

- **PrimeKG** (4M relationships): https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IXA7BM
- **DrugBank** (10K drugs): https://go.drugbank.com/releases/latest
- **ChEMBL** (2M+ bioactivities): https://www.ebi.ac.uk/chembl/

**Load into MolRAG:**

```bash
# Load PrimeKG (recommended to start)
python scripts/load_knowledge_graphs.py \
  --kg primekg \
  --data-path data/primekg/primekg.csv

# Load DrugBank
python scripts/load_knowledge_graphs.py \
  --kg drugbank \
  --data-path data/drugbank/drugbank.xml

# Load ChEMBL
python scripts/load_knowledge_graphs.py \
  --kg chembl \
  --data-path data/chembl/chembl_molecules.sdf
```

---

## 🎨 Using the Gradio UI

### Launch the UI

```bash
python app.py
```

Access at: **http://localhost:7860**

### UI Features

For complete Gradio UI documentation, see **[GRADIO_UI_GUIDE.md](GRADIO_UI_GUIDE.md)**.

**Quick overview:**

| Tab | Purpose | Example |
|-----|---------|---------|
| **🎯 Property Prediction** | Predict molecular properties | "Is this molecule toxic?" |
| **🔬 Molecular Analysis** | Analyze structure and properties | Lipinski's Rule, fingerprints |
| **⚖️ Molecule Comparison** | Compare two molecules | Tanimoto similarity |
| **ℹ️ About** | System information | Architecture, metrics, citation |

---

## 🐍 Using the Python API

### Basic Prediction

```python
from src.molrag import MolRAG

# Initialize MolRAG
molrag = MolRAG(
    config_path="config/models.yaml",
    kg_config_path="config/knowledge_graphs.yaml",
    auto_init=True  # Automatically initialize all components
)

# Predict molecular property
result = molrag.predict(
    smiles="CC(C)Cc1ccc(cc1)C(C)C(O)=O",  # Ibuprofen
    query="Is this molecule toxic?",
    cot_strategy="sim_cot",  # Options: struct_cot, sim_cot, path_cot
    top_k=10,
    preprocess=True  # Automatically canonicalize and clean SMILES
)

# Access results
print(f"Prediction: {result.prediction}")
print(f"Confidence: {result.confidence:.2%}")
print(f"Reasoning: {result.reasoning}")
print(f"Execution Time: {result.metadata['execution_time']:.2f}s")

# Retrieved molecules
for mol in result.retrieved_molecules:
    print(f"  - {mol.smiles} (score: {mol.score:.3f})")

# Pathways
for pathway in result.pathways:
    print(f"  - {pathway.name}: {pathway.description}")
```

### Batch Prediction

```python
# Predict multiple molecules
smiles_list = [
    "CC(C)Cc1ccc(cc1)C(C)C(O)=O",  # Ibuprofen
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
    "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
]

query = "Is this molecule toxic?"

results = molrag.batch_predict(
    smiles_list=smiles_list,
    query=query,
    cot_strategy="sim_cot",
    top_k=10,
    parallel=True,  # Process in parallel for speed
    max_workers=4
)

# Process results
for smiles, result in zip(smiles_list, results):
    print(f"{smiles}: {result.prediction} ({result.confidence:.0%})")
```

### Custom Configuration

```python
from src.molrag import MolRAG
from pathlib import Path

molrag = MolRAG(
    config_path=Path("config/custom_models.yaml"),
    kg_config_path=Path("config/custom_kg.yaml"),
    auto_init=False  # Manual initialization
)

# Initialize specific components
molrag.initialize_preprocessor()
molrag.initialize_retrieval()
molrag.initialize_reasoning(llm_model="gpt-4-turbo")

# Now use normally
result = molrag.predict(...)
```

---

## 🔬 Data Processing

### SMILES Preprocessing

```python
from src.data import SMILESPreprocessor

preprocessor = SMILESPreprocessor(
    canonicalize=True,  # Canonicalize SMILES
    remove_salts=True,  # Remove salt fragments
    neutralize=True,    # Neutralize charges
    remove_isotopes=True,  # Remove isotope information
    validate=True       # Validate molecule
)

# Preprocess SMILES
smiles = "CC(C)Cc1ccc(cc1)C(C)C(O)=O.[Na]"
clean_smiles = preprocessor.preprocess(smiles)
print(clean_smiles)  # CC(C)Cc1ccc(cc1)C(C)C(O)=O (salt removed)

# Get molecular properties
props = preprocessor.get_molecular_properties(clean_smiles)
print(props)
# {
#   'molecular_weight': 206.28,
#   'logp': 3.97,
#   'h_donors': 1,
#   'h_acceptors': 2,
#   'rotatable_bonds': 4,
#   'tpsa': 37.3,
#   'num_atoms': 15,
#   'num_heavy_atoms': 15,
#   'num_aromatic_rings': 1,
#   'num_rings': 1
# }

# Check Lipinski's Rule of Five
lipinski = preprocessor.check_lipinski(clean_smiles)
print(lipinski)
# {
#   'molecular_weight': {'value': 206.28, 'pass': True, 'limit': 500},
#   'logp': {'value': 3.97, 'pass': True, 'limit': 5},
#   'h_donors': {'value': 1, 'pass': True, 'limit': 5},
#   'h_acceptors': {'value': 2, 'pass': True, 'limit': 10},
#   'overall': True,
#   'drug_likeness': 'EXCELLENT'
# }
```

### Molecular Fingerprints

```python
from src.data import MolecularFingerprints

# Initialize fingerprint generator
fp_gen = MolecularFingerprints(
    fingerprint_type="morgan",  # Options: morgan, rdkit, atom_pair, topological
    n_bits=2048,
    radius=2,  # ECFP4 (radius 2)
    use_features=False,
    use_chirality=False
)

# Generate fingerprint
smiles = "CC(C)Cc1ccc(cc1)C(C)C(O)=O"
fingerprint = fp_gen.generate_fingerprint(smiles)
print(f"Fingerprint: {fingerprint.GetNumOnBits()} bits set out of {fp_gen.n_bits}")

# Calculate similarity
smiles1 = "CC(C)Cc1ccc(cc1)C(C)C(O)=O"  # Ibuprofen
smiles2 = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin

tanimoto = fp_gen.calculate_similarity(smiles1, smiles2, metric="tanimoto")
dice = fp_gen.calculate_similarity(smiles1, smiles2, metric="dice")
print(f"Tanimoto: {tanimoto:.3f}, Dice: {dice:.3f}")

# Batch fingerprint generation
smiles_list = ["CC(C)Cc1ccc(cc1)C(C)C(O)=O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"]
fingerprints = fp_gen.generate_batch(smiles_list, parallel=True)
```

### GNN Embeddings

```python
from src.data import GNNEmbedding

# Initialize GNN model
gnn = GNNEmbedding(
    model_name="kpgt",  # Options: kpgt, kano
    device="cuda",  # or "cpu"
    batch_size=32
)

# Generate embedding for single molecule
smiles = "CC(C)Cc1ccc(cc1)C(C)C(O)=O"
embedding = gnn.generate_embedding(smiles)
print(f"Embedding shape: {embedding.shape}")  # (768,) for KPGT

# Calculate similarity using embeddings
smiles1 = "CC(C)Cc1ccc(cc1)C(C)C(O)=O"
smiles2 = "CC(=O)Oc1ccccc1C(=O)O"
similarity = gnn.calculate_similarity(smiles1, smiles2)
print(f"GNN Similarity: {similarity:.3f}")

# Batch embedding generation
smiles_list = ["CCO", "CC(C)O", "CCCO"]
embeddings = gnn.generate_batch_embeddings(smiles_list)
print(f"Batch embeddings shape: {embeddings.shape}")  # (3, 768)
```

---

## 🔍 Triple Retrieval System

MolRAG uses three complementary retrieval methods:

1. **Vector Retrieval** (fingerprints) - Structural similarity
2. **Graph Retrieval** (knowledge graphs) - Mechanistic reasoning
3. **GNN Retrieval** (embeddings) - Knowledge-aware similarity

### Using TripleRetriever

```python
from src.retrieval import TripleRetriever

# Initialize retriever
retriever = TripleRetriever(
    config_path="config/models.yaml",
    vector_weight=0.4,  # Weight for fingerprint similarity
    graph_weight=0.3,   # Weight for graph path relevance
    gnn_weight=0.3      # Weight for GNN similarity
)

# Retrieve similar molecules
query_smiles = "CC(C)Cc1ccc(cc1)C(C)C(O)=O"
results = retriever.retrieve(
    query_smiles=query_smiles,
    top_k=10,
    enable_vector=True,
    enable_graph=True,
    enable_gnn=True,
    rerank=True  # Apply hybrid re-ranking
)

# Process results
for rank, mol in enumerate(results, 1):
    print(f"{rank}. {mol.smiles}")
    print(f"   Combined Score: {mol.score:.3f}")
    print(f"   Tanimoto: {mol.vector_score:.3f}")
    print(f"   Path Relevance: {mol.graph_score:.3f}")
    print(f"   GNN Similarity: {mol.gnn_score:.3f}")
    print(f"   Properties: {mol.properties}")
    print()
```

### Vector Retrieval Only

```python
from src.retrieval import VectorRetriever

retriever = VectorRetriever(
    qdrant_host="localhost",
    qdrant_port=6333,
    collection_name="molecular_fingerprints"
)

# Search by similarity
query_smiles = "CC(C)Cc1ccc(cc1)C(C)C(O)=O"
results = retriever.search(
    query_smiles=query_smiles,
    top_k=50,
    score_threshold=0.5  # Only return if Tanimoto > 0.5
)

for mol in results:
    print(f"{mol.smiles}: {mol.tanimoto_similarity:.3f}")
```

### Graph Retrieval Only

```python
from src.retrieval import GraphRetriever

retriever = GraphRetriever(
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="molrag123"
)

# Retrieve using knowledge graph traversal
query_smiles = "CC(C)Cc1ccc(cc1)C(C)C(O)=O"
results = retriever.retrieve(
    query_smiles=query_smiles,
    top_k=40,
    max_hops=2,  # 1-2 hop traversal
    relationship_types=[
        "binds", "targets", "inhibits",
        "treats", "causes", "participates_in"
    ]
)

for mol in results:
    print(f"{mol.smiles}: {mol.path_relevance:.3f}")
    print(f"  Pathways: {[p.name for p in mol.pathways]}")
    print(f"  Relationships: {mol.relationships}")
```

### GNN Retrieval Only

```python
from src.retrieval import GNNRetriever

retriever = GNNRetriever(
    gnn_model="kpgt",
    qdrant_host="localhost",
    collection_name="gnn_embeddings"
)

# Retrieve using GNN embeddings
query_smiles = "CC(C)Cc1ccc(cc1)C(C)C(O)=O"
results = retriever.retrieve(
    query_smiles=query_smiles,
    top_k=30
)

for mol in results:
    print(f"{mol.smiles}: {mol.gnn_similarity:.3f}")
```

### Hybrid Re-ranking

```python
from src.retrieval import HybridReranker

reranker = HybridReranker(
    vector_weight=0.4,
    graph_weight=0.3,
    gnn_weight=0.3,
    min_score=0.3  # Minimum combined score
)

# Combine results from all three retrievers
reranked = reranker.rerank(
    query_smiles=query_smiles,
    vector_results=vector_results,
    graph_results=graph_results,
    gnn_results=gnn_results,
    final_top_k=10
)

# Formula: Combined_Score = 0.4 × Tanimoto + 0.3 × PathRelevance + 0.3 × GNN
```

---

## 🤖 Multi-Agent Reasoning

MolRAG uses a **CLADD (Closed-Loop, Agent-Driven Design)** architecture with 5 specialized agents:

### Using Multi-Agent Orchestrator

```python
from src.reasoning import MultiAgentOrchestrator

orchestrator = MultiAgentOrchestrator(
    config_path="config/models.yaml",
    synthesis_model="gpt-4",  # or claude-3-opus-20240229
    enable_planning=True,
    enable_graph_agent=True,
    enable_vector_agent=True,
    enable_gnn_agent=True
)

# Run multi-agent reasoning
result = orchestrator.reason(
    query_smiles="CC(C)Cc1ccc(cc1)C(C)C(O)=O",
    property_query="Is this molecule toxic?",
    cot_strategy="sim_cot",
    final_top_k=10
)

# Access agent outputs
print("Planning Agent:", result.planning_output)
print("Graph Agent:", result.graph_reasoning)
print("Vector Agent:", result.vector_reasoning)
print("GNN Agent:", result.gnn_reasoning)
print("Synthesis:", result.final_prediction)
```

### Individual Agents

#### Planning Agent

```python
from src.reasoning.agents import PlanningAgent

agent = PlanningAgent(llm_model="gpt-4")

plan = agent.execute(
    query_smiles="CC(C)Cc1ccc(cc1)C(C)C(O)=O",
    property_query="Is this molecule toxic?"
)

print(plan)
# {
#   'property_type': 'toxicity',
#   'reasoning_strategy': 'sim_cot',
#   'retrieval_focus': 'similar_molecules',
#   'expected_pathways': ['metabolic', 'hepatic']
# }
```

#### Graph Retrieval Agent

```python
from src.reasoning.agents import GraphRetrievalAgent

agent = GraphRetrievalAgent(
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="molrag123"
)

reasoning = agent.execute(
    query_smiles="CC(C)Cc1ccc(cc1)C(C)C(O)=O",
    property_query="Is this molecule toxic?",
    planning_output=plan
)

print(reasoning)
# {
#   'pathways': [...],
#   'mechanisms': [...],
#   'evidence': '...',
#   'confidence': 0.85
# }
```

#### Vector Retrieval Agent

```python
from src.reasoning.agents import VectorRetrievalAgent

agent = VectorRetrievalAgent(
    qdrant_host="localhost",
    qdrant_port=6333
)

reasoning = agent.execute(
    query_smiles="CC(C)Cc1ccc(cc1)C(C)C(O)=O",
    property_query="Is this molecule toxic?",
    planning_output=plan
)
```

#### GNN Prediction Agent

```python
from src.reasoning.agents import GNNPredictionAgent

agent = GNNPredictionAgent(
    gnn_model="kpgt",
    device="cuda"
)

reasoning = agent.execute(
    query_smiles="CC(C)Cc1ccc(cc1)C(C)C(O)=O",
    property_query="Is this molecule toxic?",
    planning_output=plan
)
```

#### Synthesis Agent

```python
from src.reasoning.agents import SynthesisAgent

agent = SynthesisAgent(llm_model="gpt-4")

final_prediction = agent.execute(
    query_smiles="CC(C)Cc1ccc(cc1)C(C)C(O)=O",
    property_query="Is this molecule toxic?",
    retrieval_results=results,
    cot_reasoning=cot_output
)

print(final_prediction)
# {
#   'prediction': 'LOW TOXICITY',
#   'confidence': 0.82,
#   'reasoning': '...',
#   'citations': [...]
# }
```

---

## 💡 Chain-of-Thought Strategies

MolRAG implements three CoT strategies:

| Strategy | Best For | Performance |
|----------|----------|-------------|
| **Sim-CoT** | General property prediction | ⭐⭐⭐⭐⭐ (Best on 6/7 datasets) |
| **Struct-CoT** | Structure-dependent properties | ⭐⭐⭐⭐ |
| **Path-CoT** | Mechanism-based queries | ⭐⭐⭐⭐ (New approach) |

### Sim-CoT (Similarity-Based)

```python
from src.reasoning import SimCoT

cot = SimCoT(
    retrieval_results=results,
    property_query="Is this molecule toxic?"
)

reasoning = cot.generate()
print(reasoning)
```

**Principle:** Property Continuity - molecules with similar structures have similar properties.

### Struct-CoT (Structure-Based)

```python
from src.reasoning import StructCoT

cot = StructCoT(
    query_smiles="CC(C)Cc1ccc(cc1)C(C)C(O)=O",
    property_query="Is this molecule toxic?"
)

reasoning = cot.generate()
```

**Principle:** Analyze functional groups, structural features, and their known effects.

### Path-CoT (Pathway-Based)

```python
from src.reasoning import PathCoT

cot = PathCoT(
    query_smiles="CC(C)Cc1ccc(cc1)C(C)C(O)=O",
    property_query="Is this molecule toxic?",
    pathways=pathways,
    mechanisms=mechanisms
)

reasoning = cot.generate()
```

**Principle:** Trace biological pathways and mechanistic understanding.

---

## 📊 Evaluation & Metrics

### Running Evaluations

```bash
# Evaluate on BACE dataset
python scripts/evaluate.py \
  --dataset BACE \
  --cot-strategy sim_cot \
  --top-k 10 \
  --output results/bace_eval.json

# Evaluate on custom dataset
python scripts/evaluate.py \
  --dataset custom \
  --data-path data/my_molecules.csv \
  --property-column "toxicity" \
  --metrics roc_auc aupr recall@10 \
  --output results/custom_eval.json
```

### Using Metrics API

```python
from src.evaluation import RetrievalMetrics, PredictionMetrics

# Retrieval metrics
retrieval_metrics = RetrievalMetrics()

metrics = retrieval_metrics.calculate(
    retrieved_smiles=[...],
    ground_truth_smiles=[...],
    k_values=[5, 10, 20]
)

print(f"Recall@10: {metrics['recall@10']:.3f}")
print(f"Precision@10: {metrics['precision@10']:.3f}")
print(f"MRR: {metrics['mrr']:.3f}")

# Prediction metrics
prediction_metrics = PredictionMetrics()

metrics = prediction_metrics.calculate(
    predictions=[0.8, 0.6, 0.9, ...],
    ground_truth=[1, 0, 1, ...],
    task="classification"  # or "regression"
)

print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
print(f"AUPR: {metrics['aupr']:.3f}")
print(f"Accuracy: {metrics['accuracy']:.3f}")
```

---

## ⚙️ Configuration

### Models Configuration (`config/models.yaml`)

```yaml
# Vector Database
vector_db:
  type: qdrant  # or milvus
  host: localhost
  port: 6333
  collection: molecular_fingerprints
  index_type: HNSW
  distance_metric: cosine

# Fingerprints
fingerprints:
  type: morgan
  radius: 2
  n_bits: 2048

# GNN Models
gnn:
  model: kpgt  # or kano
  device: cuda
  batch_size: 32

# LLM Configuration
llm:
  synthesis_model: gpt-4
  planning_model: gpt-4-turbo
  temperature: 0.7
  max_tokens: 2000

# Retrieval
retrieval:
  triple_retrieval:
    vector:
      enabled: true
      top_k: 50
      weight: 0.4
    graph:
      enabled: true
      top_k: 40
      weight: 0.3
    gnn:
      enabled: true
      top_k: 30
      weight: 0.3

# Chain-of-Thought
cot:
  default_strategy: sim_cot
  strategies:
    - struct_cot
    - sim_cot
    - path_cot
```

### Knowledge Graph Configuration (`config/knowledge_graphs.yaml`)

```yaml
knowledge_graphs:
  primekg:
    enabled: true
    priority: 1
    source: "Harvard Dataverse"
    nodes: 129375
    relationships: 4050249
    relationship_types:
      - binds
      - targets
      - inhibits
      - treats
      - causes
      # ... 24 more types

  drugbank:
    enabled: true
    priority: 2
    nodes: 10000
    focus: "Drug-target interactions"

  chembl:
    enabled: true
    priority: 3
    focus: "Bioactivity data"
```

---

## 🗄️ Knowledge Graph Management

### Loading Knowledge Graphs

```python
from src.molrag import MolRAG
from pathlib import Path

molrag = MolRAG(auto_init=True)

# Load PrimeKG
stats = molrag.load_knowledge_graph(
    kg_name="primekg",
    data_path=Path("data/primekg/primekg.csv"),
    batch_size=1000,
    parallel=True
)

print(f"Loaded: {stats.nodes} nodes, {stats.relationships} relationships")
print(f"Time: {stats.loading_time:.2f}s")
```

### Querying Knowledge Graphs

```python
from src.utils.database import Neo4jConnector

neo4j = Neo4jConnector(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="molrag123"
)

# Find pathways for a molecule
query = """
MATCH (m:Molecule {smiles: $smiles})-[r:binds]->(p:Protein)-[:participates_in]->(pw:Pathway)
RETURN m, p, pw
LIMIT 10
"""

results = neo4j.run_query(query, parameters={"smiles": "CC(C)Cc1ccc(cc1)C(C)C(O)=O"})

for record in results:
    print(f"Protein: {record['p']['name']}")
    print(f"Pathway: {record['pw']['name']}")
```

### Updating Knowledge Graphs

```python
# Add new molecules
neo4j.run_query("""
CREATE (m:Molecule {
    smiles: $smiles,
    name: $name,
    molecular_weight: $mw
})
""", parameters={
    "smiles": "CCO",
    "name": "Ethanol",
    "mw": 46.07
})

# Add relationships
neo4j.run_query("""
MATCH (m:Molecule {smiles: $smiles1})
MATCH (n:Molecule {smiles: $smiles2})
CREATE (m)-[:similar_to {tanimoto: $similarity}]->(n)
""", parameters={
    "smiles1": "CCO",
    "smiles2": "CC(C)O",
    "similarity": 0.75
})
```

---

## 🚀 Advanced Usage

### Custom Retrieval Pipeline

```python
from src.retrieval import VectorRetriever, GraphRetriever, GNNRetriever, HybridReranker

# Initialize retrievers with custom settings
vector_ret = VectorRetriever(...)
graph_ret = GraphRetriever(...)
gnn_ret = GNNRetriever(...)

# Retrieve from each source
query_smiles = "CC(C)Cc1ccc(cc1)C(C)C(O)=O"
vector_results = vector_ret.search(query_smiles, top_k=100)
graph_results = graph_ret.retrieve(query_smiles, top_k=80, max_hops=3)
gnn_results = gnn_ret.retrieve(query_smiles, top_k=60)

# Custom re-ranking with different weights
reranker = HybridReranker(
    vector_weight=0.5,  # Emphasize structural similarity
    graph_weight=0.2,
    gnn_weight=0.3
)

final_results = reranker.rerank(
    query_smiles=query_smiles,
    vector_results=vector_results,
    graph_results=graph_results,
    gnn_results=gnn_results,
    final_top_k=20
)
```

### Custom Chain-of-Thought

```python
from src.reasoning import BaseCoT

class CustomCoT(BaseCoT):
    """Custom CoT strategy for specific domain"""

    def generate(self, query_smiles, property_query, retrieval_results):
        # Your custom reasoning logic
        reasoning_steps = []

        # Step 1: Analyze structural features
        reasoning_steps.append(self._analyze_structure(query_smiles))

        # Step 2: Compare with similar molecules
        reasoning_steps.append(self._compare_similar(retrieval_results))

        # Step 3: Apply domain-specific rules
        reasoning_steps.append(self._apply_domain_rules(query_smiles))

        return "\n".join(reasoning_steps)

# Use custom CoT
custom_cot = CustomCoT()
reasoning = custom_cot.generate(...)
```

### Ensemble Predictions

```python
from src.molrag import MolRAG

molrag = MolRAG(auto_init=True)

# Get predictions from all CoT strategies
strategies = ["struct_cot", "sim_cot", "path_cot"]
predictions = []

for strategy in strategies:
    result = molrag.predict(
        smiles="CC(C)Cc1ccc(cc1)C(C)C(O)=O",
        query="Is this molecule toxic?",
        cot_strategy=strategy,
        top_k=10
    )
    predictions.append({
        'strategy': strategy,
        'prediction': result.prediction,
        'confidence': result.confidence
    })

# Ensemble: weighted average
weights = {'struct_cot': 0.3, 'sim_cot': 0.5, 'path_cot': 0.2}
ensemble_confidence = sum(
    pred['confidence'] * weights[pred['strategy']]
    for pred in predictions
)

print(f"Ensemble Confidence: {ensemble_confidence:.2%}")
```

---

## 📦 Batch Processing

### Process Large Datasets

```python
import pandas as pd
from src.molrag import MolRAG
from tqdm import tqdm

# Load dataset
df = pd.read_csv("data/molecules.csv")
# Expected columns: smiles, property_query

molrag = MolRAG(auto_init=True)

# Batch predict with progress bar
results = []
for _, row in tqdm(df.iterrows(), total=len(df)):
    try:
        result = molrag.predict(
            smiles=row['smiles'],
            query=row['property_query'],
            cot_strategy="sim_cot",
            top_k=10
        )
        results.append({
            'smiles': row['smiles'],
            'prediction': result.prediction,
            'confidence': result.confidence,
            'execution_time': result.metadata['execution_time']
        })
    except Exception as e:
        print(f"Error processing {row['smiles']}: {e}")
        results.append({
            'smiles': row['smiles'],
            'prediction': None,
            'confidence': 0.0,
            'error': str(e)
        })

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv("results/predictions.csv", index=False)
```

### Parallel Batch Processing

```python
from src.molrag import MolRAG
from concurrent.futures import ProcessPoolExecutor
import pandas as pd

def predict_molecule(args):
    smiles, query = args
    molrag = MolRAG(auto_init=True)
    result = molrag.predict(smiles=smiles, query=query, top_k=10)
    return {
        'smiles': smiles,
        'prediction': result.prediction,
        'confidence': result.confidence
    }

# Load data
df = pd.read_csv("data/molecules.csv")
args_list = [(row['smiles'], row['property_query']) for _, row in df.iterrows()]

# Parallel processing
with ProcessPoolExecutor(max_workers=8) as executor:
    results = list(executor.map(predict_molecule, args_list))

# Save
results_df = pd.DataFrame(results)
results_df.to_csv("results/predictions_parallel.csv", index=False)
```

---

## 📚 API Reference

### MolRAG Class

```python
class MolRAG:
    def __init__(
        self,
        config_path: Path = "config/models.yaml",
        kg_config_path: Path = "config/knowledge_graphs.yaml",
        auto_init: bool = True
    )

    def predict(
        self,
        smiles: str,
        query: str,
        cot_strategy: str = "sim_cot",
        top_k: int = 10,
        preprocess: bool = True
    ) -> PredictionResult

    def batch_predict(
        self,
        smiles_list: List[str],
        query: str,
        cot_strategy: str = "sim_cot",
        top_k: int = 10,
        parallel: bool = True,
        max_workers: int = 4
    ) -> List[PredictionResult]

    def load_knowledge_graph(
        self,
        kg_name: str,
        data_path: Path,
        batch_size: int = 1000,
        parallel: bool = True
    ) -> KGLoadStats
```

### PredictionResult

```python
@dataclass
class PredictionResult:
    prediction: str  # The predicted answer
    confidence: float  # Confidence score (0-1)
    reasoning: str  # Chain-of-thought reasoning
    retrieved_molecules: List[MoleculeResult]  # Retrieved similar molecules
    pathways: List[Pathway]  # Relevant biological pathways
    citations: List[str]  # Literature citations
    metadata: Dict[str, Any]  # Execution metadata
```

---

## 💡 Examples

### Example 1: Drug Safety Screening

```python
from src.molrag import MolRAG

molrag = MolRAG(auto_init=True)

# Screen molecule for toxicity
result = molrag.predict(
    smiles="Cc1ccc(cc1)S(=O)(=O)N",
    query="Is this molecule hepatotoxic?",
    cot_strategy="path_cot",  # Use pathway-based reasoning
    top_k=20
)

print(f"Hepatotoxicity: {result.prediction}")
print(f"Confidence: {result.confidence:.0%}")
print(f"\nKey Pathways:")
for pathway in result.pathways[:5]:
    print(f"  - {pathway.name}")
```

### Example 2: Blood-Brain Barrier Permeability

```python
# Predict BBB permeability
result = molrag.predict(
    smiles="CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
    query="Can this molecule cross the blood-brain barrier?",
    cot_strategy="struct_cot",  # Structure-based reasoning
    top_k=15
)

print(f"BBB Permeability: {result.prediction}")
print(f"Confidence: {result.confidence:.0%}")

# Check molecular properties
from src.data import SMILESPreprocessor
preprocessor = SMILESPreprocessor()
props = preprocessor.get_molecular_properties(result.query_smiles)

print(f"\nKey Properties:")
print(f"  MW: {props['molecular_weight']:.2f} (< 400 for BBB)")
print(f"  LogP: {props['logp']:.2f} (1-4 optimal for BBB)")
print(f"  TPSA: {props['tpsa']:.2f} (< 90 for BBB)")
```

### Example 3: Drug Repurposing

```python
# Test if existing drug has new application
result = molrag.predict(
    smiles="CC(C)Cc1ccc(cc1)C(C)C(O)=O",  # Ibuprofen
    query="Does this molecule have neuroprotective effects?",
    cot_strategy="sim_cot",
    top_k=10
)

print(f"Neuroprotective: {result.prediction}")
print(f"Confidence: {result.confidence:.0%}")
print(f"\nReasoning:\n{result.reasoning}")

# Compare with known neuroprotective agents
from src.retrieval import TripleRetriever
retriever = TripleRetriever()

known_neuroprotective = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"  # Caffeine
similar_mols = retriever.retrieve(
    query_smiles=known_neuroprotective,
    top_k=20
)

# Check if ibuprofen is among similar molecules
ibuprofen_similar = any(
    mol.smiles == "CC(C)Cc1ccc(cc1)C(C)C(O)=O"
    for mol in similar_mols
)
print(f"Similar to known neuroprotective agents: {ibuprofen_similar}")
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. Import Errors

**Error:** `ModuleNotFoundError: No module named 'src'`

**Solution:**
```bash
# Ensure you're in the project root
cd MOLRAG_KB

# Add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or install in development mode
pip install -e .
```

#### 2. Database Connection Errors

**Error:** `neo4j.exceptions.ServiceUnavailable`

**Solution:**
```bash
# Check if Neo4j is running
docker ps | grep neo4j

# Restart Neo4j
docker restart neo4j

# Check logs
docker logs neo4j
```

#### 3. GPU/CUDA Errors

**Error:** `RuntimeError: CUDA out of memory`

**Solution:**
```python
# Reduce batch size
gnn = GNNEmbedding(model_name="kpgt", batch_size=8)

# Or use CPU
gnn = GNNEmbedding(model_name="kpgt", device="cpu")
```

#### 4. API Rate Limits

**Error:** `openai.error.RateLimitError`

**Solution:**
```python
import time

# Add retry logic
max_retries = 3
for attempt in range(max_retries):
    try:
        result = molrag.predict(...)
        break
    except Exception as e:
        if "rate_limit" in str(e).lower() and attempt < max_retries - 1:
            time.sleep(2 ** attempt)  # Exponential backoff
        else:
            raise
```

---

## 📞 Support

- **Documentation:** See `README.md`, `QUICKSTART.md`, `GRADIO_UI_GUIDE.md`
- **Issues:** https://github.com/Apc0015/MOLRAG_KB/issues
- **Discussions:** https://github.com/Apc0015/MOLRAG_KB/discussions

---

**🎉 You're now ready to use MolRAG for molecular property prediction!**

For Gradio UI-specific instructions, see **[GRADIO_UI_GUIDE.md](GRADIO_UI_GUIDE.md)**.
