# 🎨 MolRAG Gradio UI - Complete Usage Guide

## 📋 Table of Contents
- [Quick Start](#quick-start)
- [UI Overview](#ui-overview)
- [Tab 1: Property Prediction](#tab-1-property-prediction)
- [Tab 2: Molecular Analysis](#tab-2-molecular-analysis)
- [Tab 3: Molecule Comparison](#tab-3-molecule-comparison)
- [Tab 4: About](#tab-4-about)
- [Demo Mode vs Full Mode](#demo-mode-vs-full-mode)
- [Example Workflows](#example-workflows)
- [Troubleshooting](#troubleshooting)

---

## 🚀 Quick Start

### Launch the UI

```bash
# Make sure you're in the MOLRAG_KB directory
cd MOLRAG_KB

# Activate virtual environment (if you created one)
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Launch Gradio
python app.py
```

**Expected output:**
```
Starting MolRAG UI...
Running on local URL:  http://127.0.0.1:7860
```

### Access the UI

Open your web browser and navigate to:
```
http://localhost:7860
```

Or use the public URL if shown (for sharing).

---

## 🖥️ UI Overview

The Gradio UI has **4 main tabs**:

| Tab | Purpose | Works Without Databases? |
|-----|---------|--------------------------|
| **🎯 Property Prediction** | Predict molecular properties | Partially (demo mode) |
| **🔬 Molecular Analysis** | Analyze single molecules | ✅ Yes |
| **⚖️ Molecule Comparison** | Compare two molecules | ✅ Yes |
| **ℹ️ About** | System information | ✅ Yes |

---

## 🎯 Tab 1: Property Prediction

**Purpose:** Predict molecular properties using the full MolRAG pipeline with retrieval and reasoning.

### How to Use

#### Step 1: Enter SMILES String

In the **"Molecule SMILES"** textbox, enter a valid SMILES string.

**Examples:**
```
CC(C)Cc1ccc(cc1)C(C)C(O)=O          # Ibuprofen
CN1C=NC2=C1C(=O)N(C(=O)N2C)C        # Caffeine
CC(=O)Oc1ccccc1C(=O)O               # Aspirin
c1ccccc1                             # Benzene
CCO                                  # Ethanol
```

**Tips:**
- SMILES will be automatically validated
- Invalid SMILES will show an error message
- SMILES will be automatically preprocessed (canonicalized, salts removed)

#### Step 2: Enter Property Query

In the **"Property Query"** textbox, describe what you want to know.

**Example Queries:**
```
Is this molecule toxic?
Does this compound have anti-inflammatory properties?
Can this molecule cross the blood-brain barrier?
What is the predicted solubility of this compound?
Is this drug likely to cause liver toxicity?
Does this molecule bind to protein kinases?
```

**Tips:**
- Be specific about the property you want to predict
- Use clear, natural language questions
- Medical/biological terminology is understood

#### Step 3: Select Chain-of-Thought Strategy

Choose from the dropdown menu:

| Strategy | Best For | Performance |
|----------|----------|-------------|
| **Sim-CoT** | General use, property prediction | ⭐⭐⭐⭐⭐ (Best on 6/7 datasets) |
| **Struct-CoT** | Structure-based reasoning | ⭐⭐⭐⭐ |
| **Path-CoT** | Pathway-based analysis | ⭐⭐⭐⭐ (New approach) |

**Recommendation:** Start with **Sim-CoT** for best results.

**What each strategy does:**
- **Sim-CoT**: Focuses on similar molecules' properties (Property Continuity Principle)
- **Struct-CoT**: Analyzes structural features and functional groups
- **Path-CoT**: Traces biological pathways and mechanisms

#### Step 4: Adjust Top-K Retrieval (Optional)

Use the **slider** to control how many similar molecules to retrieve:

| Value | Speed | Quality | Use When |
|-------|-------|---------|----------|
| **5** | ⚡ Fast | Basic | Quick checks |
| **10** | ⚡ Fast | Good | **Recommended default** |
| **20** | 🔄 Medium | Better | Important predictions |
| **30** | 🐌 Slow | Best | Critical decisions |

**Recommendation:** Use **10** for balance of speed and quality.

#### Step 5: Click "Predict Property"

The system will:
1. ✅ Validate and preprocess SMILES
2. 🔍 Retrieve similar molecules (vector + graph + GNN)
3. 📊 Re-rank results using hybrid scoring
4. 🧠 Apply Chain-of-Thought reasoning
5. 💡 Generate prediction with explanation

### Understanding the Output

#### Result Text
```
Prediction: YES (or NO, or numerical value)
Confidence: 87.5%
Query: Is this molecule toxic?
```

**What to look for:**
- **Prediction:** The answer to your query
- **Confidence:** How certain the system is (70%+ is reliable)
- **Query:** Confirmation of what was asked

#### Reasoning Text
```
Chain-of-Thought Reasoning:
=========================

1. Retrieved Similar Molecules:
   - Molecule A (similarity: 0.92)
   - Molecule B (similarity: 0.88)
   ...

2. Property Analysis:
   Known toxicity profiles from similar compounds...

3. Pathway Evidence:
   - Binds to cytochrome P450
   - Metabolized via hepatic pathway
   ...

4. Conclusion:
   Based on structural similarity and pathway analysis...
```

**How to interpret:**
- **Similar Molecules:** Which known molecules were used for reasoning
- **Property Analysis:** What's known about those molecules
- **Pathway Evidence:** Biological mechanisms involved
- **Conclusion:** Final reasoning summary

#### Status Text
```
✓ Successfully retrieved 10 similar molecules
✓ Found 5 relevant pathways
✓ Reasoning completed with Sim-CoT strategy
Execution time: 2.4 seconds
```

**Status indicators:**
- ✓ Success (green)
- ⚠ Warning (yellow)
- ✗ Error (red)

### Demo Mode vs Full Mode

**Demo Mode** (databases not configured):
```
⚠️ Demo Mode: Using cached example data
This is a simplified prediction without database retrieval.
```

**What works:**
- ✅ SMILES validation
- ✅ Molecular properties
- ✅ Basic structural analysis
- ❌ Real retrieval (uses cached examples)
- ❌ Pathway analysis (limited)

**Full Mode** (databases configured):
```
✓ Using Neo4j for knowledge graph retrieval
✓ Using Qdrant for vector search
✓ Using Redis for caching
```

**What works:**
- ✅ Everything in demo mode
- ✅ Real-time retrieval from knowledge graphs
- ✅ Full pathway analysis
- ✅ Up-to-date similarity search
- ✅ Multi-agent reasoning

### Example Use Cases

#### Use Case 1: Drug Safety Screening
```
SMILES: CC(C)Cc1ccc(cc1)C(C)C(O)=O
Query: Is this molecule toxic?
Strategy: Sim-CoT
Top-K: 10

Expected Output:
Prediction: LOW TOXICITY
Confidence: 82%
Reasoning: Similar to ibuprofen, known safety profile...
```

#### Use Case 2: Drug Repurposing
```
SMILES: CN1C=NC2=C1C(=O)N(C(=O)N2C)C
Query: Does this compound have anti-inflammatory properties?
Strategy: Path-CoT
Top-K: 20

Expected Output:
Prediction: MODERATE ACTIVITY
Confidence: 68%
Reasoning: Caffeine has weak anti-inflammatory effects via adenosine receptor antagonism...
```

#### Use Case 3: BBB Permeability
```
SMILES: CC(=O)Oc1ccccc1C(=O)O
Query: Can this molecule cross the blood-brain barrier?
Strategy: Struct-CoT
Top-K: 15

Expected Output:
Prediction: YES
Confidence: 75%
Reasoning: Molecular weight < 400 Da, logP favorable, similar to known CNS drugs...
```

---

## 🔬 Tab 2: Molecular Analysis

**Purpose:** Analyze a single molecule's properties, structure, and drug-likeness.

### How to Use

#### Step 1: Enter SMILES String

Same as Property Prediction tab.

**Examples:**
```
CC(C)Cc1ccc(cc1)C(C)C(O)=O          # Ibuprofen
CN1C=NC2=C1C(=O)N(C(=O)N2C)C        # Caffeine
```

#### Step 2: Click "Analyze Molecule"

The system calculates:
- ✅ Basic molecular properties
- ✅ Lipinski's Rule of Five compliance
- ✅ Molecular fingerprint statistics
- ✅ Drug-likeness assessment

### Understanding the Output

#### Basic Properties Section
```
Molecular Properties:
====================
Molecular Weight: 206.28 g/mol
LogP (Lipophilicity): 3.97
H-Bond Donors: 1
H-Bond Acceptors: 2
Rotatable Bonds: 4
TPSA (Topological Polar Surface Area): 37.3 Ų
```

**What each means:**

| Property | Description | Ideal Range |
|----------|-------------|-------------|
| **Molecular Weight** | Mass of the molecule | 150-500 g/mol |
| **LogP** | Lipophilicity (fat solubility) | 0-5 |
| **H-Bond Donors** | Can donate hydrogen bonds | 0-5 |
| **H-Bond Acceptors** | Can accept hydrogen bonds | 0-10 |
| **Rotatable Bonds** | Flexibility | 0-10 |
| **TPSA** | Polar surface area | 20-140 Ų |

#### Lipinski's Rule of Five
```
Lipinski's Rule of Five:
=======================
✓ Molecular Weight ≤ 500 Da: PASS (206.28)
✓ LogP ≤ 5: PASS (3.97)
✓ H-Bond Donors ≤ 5: PASS (1)
✓ H-Bond Acceptors ≤ 10: PASS (2)

Overall: PASS (4/4 rules)
Drug-likeness: EXCELLENT
```

**What it means:**
- **PASS:** Molecule is likely to be orally bioavailable
- **FAIL:** Molecule may have poor drug-like properties
- **Drug-likeness ratings:**
  - EXCELLENT: All rules pass
  - GOOD: 3/4 rules pass
  - MODERATE: 2/4 rules pass
  - POOR: ≤1 rules pass

#### Fingerprint Information
```
Molecular Fingerprint:
=====================
Type: Morgan ECFP4
Radius: 2
Bits: 2048
Bits Set: 87 (4.2%)
```

**What it means:**
- **Type:** Algorithm used (Morgan ECFP4 is industry standard)
- **Radius:** How far to look from each atom (2 = up to 2 bonds away)
- **Bits Set:** How many unique structural features (more = more complex)

### Example Analyses

#### Example 1: Ibuprofen (Drug-Like Molecule)
```
Input: CC(C)Cc1ccc(cc1)C(C)C(O)=O

Output:
✓ Molecular Weight: 206 g/mol (PASS)
✓ LogP: 3.97 (PASS)
✓ H-Bond Donors: 1 (PASS)
✓ H-Bond Acceptors: 2 (PASS)
Drug-likeness: EXCELLENT
```

#### Example 2: Small Molecule (Ethanol)
```
Input: CCO

Output:
✓ Molecular Weight: 46 g/mol (PASS - but very small)
✓ LogP: -0.07 (PASS)
✓ H-Bond Donors: 1 (PASS)
✓ H-Bond Acceptors: 1 (PASS)
Drug-likeness: EXCELLENT (but may be too simple for drug development)
```

#### Example 3: Large Peptide (Fails Lipinski)
```
Input: CC[C@H](C)[C@H](NC(=O)[C@H](CC(C)C)NC(=O)[C@H](CC(C)C)NC(=O)[C@H](C)NC(=O)C)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](C)C(=O)O

Output:
✗ Molecular Weight: 697 g/mol (FAIL - exceeds 500)
✓ LogP: 4.2 (PASS)
✗ H-Bond Donors: 7 (FAIL - exceeds 5)
✓ H-Bond Acceptors: 9 (PASS)
Drug-likeness: MODERATE (2/4 rules pass)
Note: May require special delivery methods
```

---

## ⚖️ Tab 3: Molecule Comparison

**Purpose:** Compare two molecules and calculate their structural similarity.

### How to Use

#### Step 1: Enter First Molecule

In **"First Molecule SMILES"** textbox:
```
CC(C)Cc1ccc(cc1)C(C)C(O)=O          # Ibuprofen
```

#### Step 2: Enter Second Molecule

In **"Second Molecule SMILES"** textbox:
```
CC(=O)Oc1ccccc1C(=O)O               # Aspirin
```

#### Step 3: Click "Compare Molecules"

The system calculates:
- ✅ Tanimoto similarity (fingerprint-based)
- ✅ Visual similarity rating
- ✅ Property comparison table
- ✅ Similarity interpretation

### Understanding the Output

#### Similarity Score
```
Tanimoto Similarity: 0.342

Similarity Rating: MODERATE
```

**Similarity Scale:**

| Score | Rating | Meaning |
|-------|--------|---------|
| **0.90 - 1.00** | Very High | Nearly identical structures |
| **0.70 - 0.89** | High | Same chemical family, similar properties |
| **0.50 - 0.69** | Moderate-High | Related structures |
| **0.30 - 0.49** | Moderate | Some shared features |
| **0.10 - 0.29** | Low | Different structures |
| **0.00 - 0.09** | Very Low | Completely different |

#### Property Comparison Table
```
Property Comparison:
===================
| Property           | Molecule 1 | Molecule 2 | Difference |
|--------------------|------------|------------|------------|
| Molecular Weight   | 206.28     | 180.16     | +26.12     |
| LogP               | 3.97       | 1.19       | +2.78      |
| H-Bond Donors      | 1          | 1          | 0          |
| H-Bond Acceptors   | 2          | 4          | -2         |
| Rotatable Bonds    | 4          | 3          | +1         |
| TPSA               | 37.3       | 63.6       | -26.3      |
```

**How to interpret:**
- **Similar values:** Molecules likely have similar behavior
- **Large differences:** Different physicochemical properties
- **Positive difference:** Molecule 1 has more/higher
- **Negative difference:** Molecule 2 has more/higher

#### Interpretation Text
```
Interpretation:
==============
These molecules show MODERATE similarity (34.2%).

Key Observations:
- Both contain aromatic rings
- Both have carboxylic acid groups
- Different side chains (isobutyl vs. acetyl)
- Similar molecular weight range
- Different lipophilicity (LogP difference: 2.78)

Possible Implications:
- May target similar biological pathways
- Different pharmacokinetic profiles expected
- Similar oral bioavailability (both pass Lipinski)
```

### Example Comparisons

#### Example 1: Ibuprofen vs. Aspirin (Related NSAIDs)
```
Molecule 1: CC(C)Cc1ccc(cc1)C(C)C(O)=O          (Ibuprofen)
Molecule 2: CC(=O)Oc1ccccc1C(=O)O               (Aspirin)

Similarity: 0.34 (MODERATE)
Reason: Both are NSAIDs with aromatic rings and carboxylic acids
```

#### Example 2: Ibuprofen vs. Ibuprofen (Identical)
```
Molecule 1: CC(C)Cc1ccc(cc1)C(C)C(O)=O
Molecule 2: CC(C)Cc1ccc(cc1)C(C)C(O)=O

Similarity: 1.00 (VERY HIGH)
Reason: Identical molecules
```

#### Example 3: Ibuprofen vs. Caffeine (Unrelated)
```
Molecule 1: CC(C)Cc1ccc(cc1)C(C)C(O)=O          (Ibuprofen)
Molecule 2: CN1C=NC2=C1C(=O)N(C(=O)N2C)C        (Caffeine)

Similarity: 0.12 (LOW)
Reason: Different chemical classes, different targets
```

#### Example 4: Aspirin vs. Salicylic Acid (Very Similar)
```
Molecule 1: CC(=O)Oc1ccccc1C(=O)O               (Aspirin)
Molecule 2: OC(=O)c1ccccc1O                     (Salicylic Acid)

Similarity: 0.78 (HIGH)
Reason: Aspirin is acetylated salicylic acid
```

---

## ℹ️ Tab 4: About

**Purpose:** Learn about MolRAG system, architecture, and metrics.

### What's Included

#### System Information
- MolRAG version
- Python version
- Key dependencies status
- Database connection status

#### Architecture Overview
- Visual diagram of MolRAG pipeline
- Component descriptions
- Data flow explanation

#### Performance Metrics
- Benchmark results
- Comparison with baselines
- Published accuracy scores

#### Citation Information
- How to cite MolRAG
- Related papers
- Links to resources

---

## 🔄 Demo Mode vs Full Mode

### What is Demo Mode?

When databases (Neo4j, Qdrant, Redis) are not configured, the UI runs in **Demo Mode**.

**Demo Mode Features:**
```
⚠️ MolRAG Demo Mode

Available Features:
✅ SMILES validation and preprocessing
✅ Molecular property calculation
✅ Fingerprint generation
✅ Lipinski's Rule checking
✅ Molecule similarity comparison
✅ Basic structural analysis

Limited Features:
⚠️ Property prediction (uses cached examples)
⚠️ Retrieval (simulated with mock data)
⚠️ Pathway analysis (basic only)

To enable full features, set up databases:
- Neo4j (Knowledge Graph)
- Qdrant (Vector Database)
- Redis (Caching)

See QUICKSTART.md for setup instructions.
```

### How to Enable Full Mode

#### Step 1: Start Databases
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

#### Step 2: Configure Environment
```bash
# Create .env file
cp .env.example .env

# Edit .env with your settings
nano .env  # or your preferred editor
```

Required settings:
```
# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# LLM API Keys (for full reasoning)
OPENAI_API_KEY=sk-...
# OR
ANTHROPIC_API_KEY=sk-ant-...
```

#### Step 3: Initialize Databases
```bash
python scripts/setup_databases.py
```

#### Step 4: Restart UI
```bash
python app.py
```

You should now see:
```
✓ Neo4j connected
✓ Qdrant connected
✓ Redis connected
✓ Full mode enabled
```

---

## 📚 Example Workflows

### Workflow 1: Drug Safety Screening Pipeline

**Scenario:** Screen a new compound for potential toxicity.

1. **Go to Tab 1 (Property Prediction)**
2. Enter SMILES: `Cc1ccc(cc1)S(=O)(=O)N`
3. Query: "Is this molecule hepatotoxic?"
4. Strategy: Sim-CoT
5. Top-K: 20 (for thorough analysis)
6. Click "Predict Property"
7. **Review Results:**
   - Check confidence (should be >70%)
   - Read pathway evidence
   - Note similar molecules with known toxicity
8. **Go to Tab 2 (Molecular Analysis)**
9. Enter same SMILES
10. Click "Analyze Molecule"
11. **Check Lipinski's Rule:**
    - PASS = good oral bioavailability
    - FAIL = may need special formulation
12. **Document findings** for safety report

### Workflow 2: Lead Optimization

**Scenario:** Compare your lead compound to an existing drug.

1. **Go to Tab 3 (Molecule Comparison)**
2. Enter your lead compound in "First Molecule"
3. Enter reference drug in "Second Molecule"
4. Click "Compare Molecules"
5. **Analyze Results:**
   - Similarity > 0.7: Very similar, expect similar activity
   - Similarity 0.3-0.7: Related, but different properties
   - Similarity < 0.3: Different class, novel mechanism
6. **Check property differences:**
   - LogP difference: Affects membrane permeability
   - MW difference: Affects bioavailability
   - TPSA difference: Affects BBB penetration
7. **Go to Tab 1 for each molecule**
8. Query same property for both
9. Compare predictions and confidence
10. **Decide on modifications** to improve lead

### Workflow 3: Drug Repurposing Discovery

**Scenario:** Find new uses for an existing drug.

1. **Go to Tab 2 (Molecular Analysis)**
2. Enter drug SMILES
3. Click "Analyze Molecule"
4. Note key properties (especially LogP, MW, Lipinski)
5. **Go to Tab 1 (Property Prediction)**
6. Try multiple queries:
   - "Does this bind to GPCRs?"
   - "Can this inhibit kinases?"
   - "Does this have anti-inflammatory effects?"
   - "Can this cross the blood-brain barrier?"
7. **Analyze each prediction:**
   - High confidence (>80%): Strong candidate
   - Moderate confidence (60-80%): Worth investigating
   - Low confidence (<60%): Uncertain, need validation
8. **Check pathway evidence** for unexpected mechanisms
9. **Go to Tab 3 (Comparison)**
10. Compare with drugs that treat target indication
11. **Prioritize repurposing candidates** based on:
    - Prediction confidence
    - Pathway overlap
    - Structural similarity to known drugs

### Workflow 4: Batch Property Screening

**Scenario:** Screen 50 compounds for BBB permeability.

1. Create a CSV file with SMILES:
```csv
smiles,compound_id
CC(C)Cc1ccc(cc1)C(C)C(O)=O,COMP001
CN1C=NC2=C1C(=O)N(C(=O)N2C)C,COMP002
...
```

2. **For each compound:**
   - **Tab 1:** Enter SMILES, query "Can this cross the blood-brain barrier?"
   - **Tab 2:** Check Lipinski and MW < 400 Da
   - **Record results** in spreadsheet

3. **Alternative: Use Python API** (recommended for batches):
```python
from src.molrag import MolRAG
import pandas as pd

molrag = MolRAG(auto_init=True)
df = pd.read_csv("compounds.csv")

results = []
for _, row in df.iterrows():
    result = molrag.predict(
        smiles=row['smiles'],
        query="Can this cross the blood-brain barrier?",
        cot_strategy="sim_cot",
        top_k=10
    )
    results.append({
        'compound_id': row['compound_id'],
        'prediction': result.prediction,
        'confidence': result.confidence
    })

results_df = pd.DataFrame(results)
results_df.to_csv("bbb_screening_results.csv", index=False)
```

---

## 🔧 Troubleshooting

### Issue 1: UI Won't Start

**Error:**
```
ModuleNotFoundError: No module named 'gradio'
```

**Solution:**
```bash
pip install gradio
# Or install all dependencies:
pip install -r requirements.txt
```

---

**Error:**
```
Address already in use: 7860
```

**Solution:**
```bash
# Kill existing process
lsof -ti:7860 | xargs kill -9

# Or use different port
python app.py --server-port 7861
```

### Issue 2: SMILES Validation Fails

**Error:**
```
❌ Invalid SMILES: Could not parse molecule
```

**Solutions:**
1. **Check for typos** - Common mistakes:
   - Missing closing parentheses: `CC(C` → `CC(C)`
   - Wrong bonds: `C=C-C≡C` → `C=CC#C`
   - Invalid atoms: `Xx` → Use valid element symbols

2. **Canonicalize SMILES** using RDKit:
```python
from rdkit import Chem
mol = Chem.MolFromSmiles("your_smiles_here")
if mol:
    canonical = Chem.MolToSmiles(mol)
    print(canonical)
```

3. **Use standard SMILES format:**
   - No spaces
   - Use standard bond symbols: `-` (single), `=` (double), `#` (triple)
   - Use brackets for charged atoms: `[O-]`, `[NH4+]`

### Issue 3: Low Confidence Predictions

**Issue:**
```
Confidence: 42%
⚠️ Low confidence prediction
```

**Possible Causes:**
1. **Novel molecule** - No similar compounds in database
2. **Ambiguous query** - Question is too vague
3. **Insufficient retrieval** - Top-K too low
4. **Wrong CoT strategy** - Try different strategy

**Solutions:**
1. **Increase Top-K** to 20 or 30
2. **Try different CoT strategy:**
   - Sim-CoT for general predictions
   - Struct-CoT for structure-dependent properties
   - Path-CoT for mechanism-based questions
3. **Rephrase query** to be more specific:
   - Vague: "Is this good?"
   - Specific: "Does this molecule inhibit CYP3A4?"
4. **Check in Tab 2** if molecule is drug-like (Lipinski)

### Issue 4: Demo Mode Instead of Full Mode

**Issue:**
```
⚠️ Running in demo mode
```

**Causes:**
- Databases not running
- Connection settings incorrect
- Databases not initialized

**Solutions:**

1. **Check databases are running:**
```bash
# Neo4j
docker ps | grep neo4j
# Should show running container

# Qdrant
curl http://localhost:6333/health
# Should return: {"status":"ok"}

# Redis
redis-cli ping
# Should return: PONG
```

2. **Check .env file exists and is configured:**
```bash
ls -la .env
cat .env | grep NEO4J_URI
```

3. **Initialize databases:**
```bash
python scripts/setup_databases.py
```

4. **Restart UI:**
```bash
python app.py
```

### Issue 5: Slow Predictions

**Issue:**
```
Execution time: 45.2 seconds
```

**Solutions:**

1. **Reduce Top-K:**
   - Change from 30 → 10 (3x faster)

2. **Enable Redis caching:**
```bash
# Start Redis if not running
docker start redis

# In .env
REDIS_ENABLED=true
```

3. **Use faster CoT strategy:**
   - Struct-CoT is faster than Path-CoT

4. **Optimize database indices:**
```bash
python scripts/optimize_databases.py
```

### Issue 6: Property Comparison Shows Error

**Issue:**
```
❌ Error: Could not calculate similarity
```

**Causes:**
- One or both SMILES are invalid
- Fingerprint generation failed

**Solutions:**
1. **Validate each SMILES separately** in Tab 2
2. **Check for special characters** or unusual atoms
3. **Try simpler molecules** to test functionality

---

## 📞 Getting Help

### Documentation
- **Quick Start:** See `QUICKSTART.md`
- **Full Usage:** See `USAGE_GUIDE.md`
- **Project README:** See `README.md`
- **Pull Instructions:** See `PULL_INSTRUCTIONS.md`

### Commands
```bash
# Verify installation
python verify_installation.py

# Check database status
python scripts/check_databases.py

# Run tests
pytest tests/

# View logs
tail -f logs/molrag.log
```

### Support Channels
- **GitHub Issues:** Report bugs and request features
- **Discussions:** Ask questions and share ideas
- **Documentation:** Check docs/ directory

---

## 🎉 Success Checklist

When everything is working, you should see:

**At Startup:**
```
✓ Gradio UI starting on port 7860
✓ Neo4j connected (or Demo Mode for features that don't need it)
✓ Qdrant connected (or Demo Mode)
✓ Redis connected (or caching disabled)
✓ MolRAG system initialized
```

**In the UI:**
```
Tab 1: ✅ Predictions complete with confidence >70%
Tab 2: ✅ All molecular properties calculated
Tab 3: ✅ Similarity scores computed correctly
Tab 4: ✅ System information displayed
```

**In the Terminal:**
```
No errors in logs
Fast response times (<5 seconds for predictions)
Successful database queries logged
```

---

## 🚀 Next Steps

1. **Start with Tab 2** (Molecular Analysis) - No databases needed
2. **Try example molecules** - Use provided SMILES
3. **Move to Tab 1** (Property Prediction) - Works in demo mode
4. **Set up databases** - For full functionality (see QUICKSTART.md)
5. **Explore workflows** - Try the example workflows above
6. **Use Python API** - For batch processing (see USAGE_GUIDE.md)

---

**🎨 Enjoy using MolRAG's Gradio UI!**

For complete project documentation, see `README.md` and `USAGE_GUIDE.md`.
