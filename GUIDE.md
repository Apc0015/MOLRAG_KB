# MolRAG: Molecular Property Prediction System

**Complete Guide - Setup, Usage, and Testing**

---

## 📖 Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Installation](#installation)
4. [Running the UI](#running-the-ui)
5. [Testing](#testing)
6. [Usage Examples](#usage-examples)
7. [Configuration](#configuration)
8. [Troubleshooting](#troubleshooting)

---

## Overview

MolRAG is a **training-free molecular property prediction system** that combines:
- **Retrieval-Augmented Generation (RAG)** with knowledge graphs
- **Triple Retrieval**: Fingerprints + Knowledge Graphs + GNN embeddings
- **Multi-Agent Reasoning** with Chain-of-Thought strategies
- **Zero training required** - deploy immediately

### Key Features
- 🚀 No training needed - works out of the box
- 📊 72-96% accuracy of supervised methods
- 🔍 Explainable predictions with pathway-level reasoning
- 🎨 Interactive Gradio web interface
- 💡 Two modes: Demo (no databases) and Full (with databases)

### Performance
- **BACE Dataset**: 72.25% (+20.39% over baseline)
- **CYP450**: 72.29% (+21.22% over baseline)
- **PubMedQA**: 86.3% (+28.4% over baseline)

---

## Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/Apc0015/MOLRAG_KB.git
cd MOLRAG_KB

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your settings (optional for demo mode)
```

### 3. Launch UI

```bash
python app.py
```

Access at **http://localhost:7860**

---

## Installation

### Prerequisites
- Python 3.9+
- 4GB RAM minimum
- Optional: Neo4j, Qdrant, Redis (for full features)

### Dependencies

The main packages include:
- **RDKit**: Molecular chemistry toolkit
- **Gradio**: Web interface
- **PyTorch**: Deep learning (optional, for GNN)
- **Pandas, NumPy**: Data processing
- **Neo4j, Qdrant, Redis**: Databases (optional)

### Install Steps

```bash
# 1. Basic installation
pip install -r requirements.txt

# 2. PyTorch Geometric (optional, for GNN features)
pip install torch-scatter torch-sparse torch-geometric -f https://data.pyg.org/whl/torch-2.8.0+cpu.html
```

---

## Running the UI

### Demo Mode (No Databases)

Works immediately with basic features:

```bash
python app.py
```

**Available features:**
- ✅ SMILES validation and preprocessing
- ✅ Molecular property calculation
- ✅ Fingerprint generation  
- ✅ Molecule similarity comparison
- ✅ Lipinski's Rule of Five checking

### Full Mode (With Databases)

For complete predictions with knowledge graphs:

**1. Start databases:**
```bash
# Neo4j
docker run -d --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:latest

# Qdrant
docker run -d --name qdrant \
  -p 6333:6333 \
  qdrant/qdrant:latest

# Redis
docker run -d --name redis \
  -p 6379:6379 \
  redis:latest
```

**2. Configure .env:**
```bash
NEO4J_PASSWORD=your_password
OPENAI_API_KEY=your_key  # or ANTHROPIC_API_KEY
```

**3. Setup databases:**
```bash
python scripts/setup_databases.py
```

**4. Load knowledge graphs (optional):**
```bash
python scripts/load_knowledge_graphs.py --kg primekg
```

---

## Testing

### Quick Test
```bash
python test_basic_imports.py
```

Expected output:
```
✓ SMILESPreprocessor and MolecularFingerprints imported
✓ SMILES preprocessing working
✓ Molecular properties calculated
✓ Fingerprint generated
✓ Gradio imported
✅ ALL BASIC TESTS PASSED!
```

### Full Structure Test
```bash
python test_structure.py
```

Validates all 10 components:
1. ✅ SMILES Preprocessor
2. ✅ Molecular Fingerprints
3. ✅ Triple Retrieval System
4. ✅ Multi-Agent System
5. ✅ CoT Strategies
6. ✅ Configuration System
7. ✅ Gradio UI
8. ✅ Documentation
9. ✅ Examples
10. ✅ File Integrity

### Comprehensive Test
```bash
python test_molrag.py
```

---

## Usage Examples

### Tab 1: Property Prediction

**Test Case: Ibuprofen**
- SMILES: `CC(C)Cc1ccc(cc1)C(C)C(O)=O`
- Query: `Is this molecule toxic?`
- CoT Strategy: `sim_cot`
- Top-K: `10`

**Test Case: Aspirin**
- SMILES: `CC(=O)Oc1ccccc1C(=O)O`
- Query: `What are the therapeutic properties?`
- CoT Strategy: `path_cot`

**Test Case: Caffeine**
- SMILES: `CN1C=NC2=C1C(=O)N(C(=O)N2C)C`
- Query: `Does this cross the blood-brain barrier?`
- CoT Strategy: `struct_cot`

### Tab 2: Molecular Analysis

**Simple Molecule - Ethanol:**
- SMILES: `CCO`
- Click "Analyze Molecule"
- View: MW, LogP, H-bonds, TPSA, Lipinski check

**Drug Molecule - Morphine:**
- SMILES: `CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C=C4)O`
- Check drug-likeness properties

### Tab 3: Molecule Comparison

**Similar Molecules:**
- Molecule 1: `CCO` (Ethanol)
- Molecule 2: `CCCO` (Propanol)
- Expected: High similarity (>0.8)

**Different Molecules:**
- Molecule 1: `CCO` (Ethanol)
- Molecule 2: `c1ccccc1` (Benzene)
- Expected: Low similarity (<0.3)

### Python API

```python
from src.data import SMILESPreprocessor, MolecularFingerprints

# Preprocess SMILES
preprocessor = SMILESPreprocessor(canonicalize=True)
canonical = preprocessor.preprocess("CC(C)Cc1ccc(cc1)C(C)C(O)=O")
print(canonical)  # CC(C)Cc1ccc(C(C)C(=O)O)cc1

# Get properties
props = preprocessor.get_molecular_properties(canonical)
print(f"MW: {props['molecular_weight']:.2f}")
print(f"LogP: {props['logp']:.2f}")
print(f"Lipinski: {preprocessor.passes_lipinski_rule_of_five(canonical)}")

# Generate fingerprint
fp_gen = MolecularFingerprints(fingerprint_type="morgan", n_bits=2048)
fingerprint = fp_gen.generate_fingerprint(canonical)
print(f"Bits set: {fingerprint.GetNumOnBits()}")

# Calculate similarity
smiles2 = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
similarity = fp_gen.calculate_similarity(canonical, smiles2)
print(f"Tanimoto similarity: {similarity:.3f}")
```

### Full MolRAG System (with databases)

```python
from src.molrag import MolRAG

# Initialize
molrag = MolRAG(auto_init=True)

# Predict
result = molrag.predict(
    smiles="CC(C)Cc1ccc(cc1)C(C)C(O)=O",
    query="Is this molecule toxic?",
    cot_strategy="sim_cot",
    top_k=10
)

print(f"Prediction: {result.prediction}")
print(f"Confidence: {result.confidence:.2%}")
print(f"Reasoning: {result.reasoning}")
```

---

## Configuration

### .env File

```bash
# API Keys (optional for demo mode)
OPENAI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
OPENROUTER_API_KEY=your_key

# Neo4j (optional)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j

# Qdrant (optional)
QDRANT_URL=http://localhost:6333

# Redis (optional)
REDIS_HOST=localhost
REDIS_PORT=6379

# Settings
LOG_LEVEL=INFO
DEVICE=cpu  # or cuda
```

### Model Configuration

Edit `config/models.yaml`:

```yaml
retrieval:
  vector_weight: 0.4
  graph_weight: 0.3
  gnn_weight: 0.3
  top_k: 10

llm:
  provider: openai  # or anthropic, openrouter
  model: gpt-4
  temperature: 0.7
```

---

## Troubleshooting

### Issue: "Module not found" errors

**Solution:**
```bash
cd MOLRAG_KB
source venv/bin/activate
pip install -r requirements.txt
```

### Issue: RDKit errors

**Solution:**
```bash
pip uninstall rdkit rdkit-pypi
pip install rdkit
```

### Issue: App won't start

**Solution:**
```bash
# Check if port 7860 is in use
lsof -i :7860
kill -9 <PID>

# Restart app
python app.py
```

### Issue: "Database not connected" in UI

**Expected** - This is demo mode. Basic features work without databases.

To enable full features:
1. Start databases (see "Full Mode" section)
2. Configure .env file
3. Run `python scripts/setup_databases.py`
4. Restart app

### Issue: Pydantic validation errors

**Solution:**
```bash
# Edit .env file - remove invalid lines
# Valid format:
KEY=value
# Invalid format:
KEY=value # comment
```

---

## Project Structure

```
MOLRAG_KB/
├── app.py                      # Gradio web UI
├── requirements.txt            # Python dependencies
├── .env                        # Configuration (copy from .env.example)
├── GUIDE.md                    # This file
│
├── config/
│   ├── models.yaml             # Model settings
│   └── knowledge_graphs.yaml   # KG configuration
│
├── src/
│   ├── data/                   # Data processing
│   │   ├── preprocessor.py     # SMILES preprocessing
│   │   ├── fingerprints.py     # Fingerprint generation
│   │   ├── gnn_embeddings.py   # GNN embeddings
│   │   └── kg_loader.py        # Knowledge graph loader
│   │
│   ├── retrieval/              # Triple retrieval
│   │   ├── vector_retrieval.py # Fingerprint search
│   │   ├── graph_retrieval.py  # Neo4j search
│   │   ├── gnn_retrieval.py    # GNN search
│   │   └── reranker.py         # Hybrid ranking
│   │
│   ├── reasoning/              # Multi-agent system
│   │   ├── agents.py           # Agent implementations
│   │   ├── cot_strategies.py   # Chain-of-Thought
│   │   └── orchestrator.py     # Agent coordination
│   │
│   ├── evaluation/             # Metrics
│   │   └── metrics.py
│   │
│   └── utils/                  # Utilities
│       ├── config.py           # Config management
│       ├── database.py         # DB connectors
│       └── logger.py           # Logging
│
├── scripts/
│   ├── setup_databases.py      # Initialize databases
│   └── load_knowledge_graphs.py # Load KG data
│
├── examples/
│   ├── basic_usage.py          # Simple examples
│   ├── advanced_prediction.py  # Advanced usage
│   └── batch_screening.py      # Batch processing
│
└── tests/
    ├── test_basic_imports.py   # Quick test
    ├── test_structure.py       # Structure validation
    └── test_molrag.py          # Comprehensive tests
```

---

## Citation

```bibtex
@article{krotkov2025nanostructured,
  title={Nanostructured Material Design via RAG},
  author={Krotkov, Nikita A and others},
  journal={Journal of Chemical Information and Modeling},
  year={2025}
}

@inproceedings{xian2025molrag,
  title={MolRAG: Unlocking LLMs for Molecular Property Prediction},
  author={Xian et al.},
  booktitle={ACL 2025},
  year={2025}
}
```

---

## Support

- **GitHub**: https://github.com/Apc0015/MOLRAG_KB
- **Issues**: https://github.com/Apc0015/MOLRAG_KB/issues

---

**Built with ❤️ for molecular AI research**
