"""
Gradio Web UI for MolRAG

Provides interactive interface for:
- Molecular property prediction
- SMILES validation and preprocessing
- Retrieval visualization
- Reasoning chain explanation
"""

import sys
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import gradio as gr
except ImportError:
    print("Installing gradio...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gradio"])
    import gradio as gr

from src.data import SMILESPreprocessor, MolecularFingerprints
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MolRAGUI:
    """Gradio UI for MolRAG system"""

    def __init__(self):
        """Initialize UI components"""
        self.preprocessor = SMILESPreprocessor(
            canonicalize=True,
            remove_salts=True,
            neutralize_charges=False
        )

        self.fp_generator = MolecularFingerprints(
            fingerprint_type="morgan",
            radius=2,
            n_bits=2048,
            use_chirality=True
        )

        # Full MolRAG system (lazy loaded)
        self.molrag = None

        logger.info("MolRAG UI initialized")

    def validate_smiles(self, smiles: str) -> Tuple[bool, str, str]:
        """
        Validate and preprocess SMILES

        Returns:
            (is_valid, processed_smiles, message)
        """
        if not smiles or not smiles.strip():
            return False, "", "❌ Please enter a SMILES string"

        smiles = smiles.strip()

        # Validate
        if not self.preprocessor.is_valid_smiles(smiles):
            return False, "", f"❌ Invalid SMILES: {smiles}"

        # Preprocess
        processed = self.preprocessor.preprocess(smiles)
        if processed is None:
            return False, "", f"❌ Failed to preprocess SMILES"

        return True, processed, f"✅ Valid SMILES"

    def get_molecular_properties(self, smiles: str) -> str:
        """Get molecular properties table"""
        if not smiles or not smiles.strip():
            return "No SMILES provided"

        is_valid, processed, msg = self.validate_smiles(smiles)
        if not is_valid:
            return msg

        props = self.preprocessor.get_molecular_properties(processed)
        if not props:
            return "❌ Failed to calculate properties"

        # Format as table
        output = "## Molecular Properties\n\n"
        output += "| Property | Value |\n"
        output += "|----------|-------|\n"

        for prop, value in props.items():
            if isinstance(value, float):
                output += f"| {prop} | {value:.2f} |\n"
            else:
                output += f"| {prop} | {value} |\n"

        # Lipinski's Rule of Five
        passes_ro5 = self.preprocessor.passes_lipinski_rule_of_five(processed)
        output += f"\n**Lipinski's Rule of Five**: {'✅ PASS' if passes_ro5 else '❌ FAIL'}\n"

        return output

    def generate_fingerprint_info(self, smiles: str) -> str:
        """Generate fingerprint information"""
        if not smiles or not smiles.strip():
            return "No SMILES provided"

        is_valid, processed, msg = self.validate_smiles(smiles)
        if not is_valid:
            return msg

        fp = self.fp_generator.generate_fingerprint(processed)
        if fp is None:
            return "❌ Failed to generate fingerprint"

        output = "## Molecular Fingerprint\n\n"
        output += f"**Type**: Morgan (ECFP4)\n"
        output += f"**Radius**: 2\n"
        output += f"**Size**: {len(fp)} bits\n"
        output += f"**On Bits**: {fp.GetNumOnBits()}\n"
        output += f"**Density**: {fp.GetNumOnBits() / len(fp) * 100:.2f}%\n"

        return output

    def analyze_molecule(self, smiles: str) -> Tuple[str, str]:
        """
        Analyze a molecule - wrapper function for testing
        Returns: (properties_text, fingerprint_text)
        """
        props = self.get_molecular_properties(smiles)
        fp_info = self.generate_fingerprint_info(smiles)
        return props, fp_info

    def compare_molecules(self, smiles1: str, smiles2: str) -> str:
        """Compare two molecules"""
        if not smiles1 or not smiles2:
            return "Please provide both SMILES strings"

        # Validate both
        is_valid1, proc1, msg1 = self.validate_smiles(smiles1)
        is_valid2, proc2, msg2 = self.validate_smiles(smiles2)

        if not is_valid1:
            return f"Molecule 1: {msg1}"
        if not is_valid2:
            return f"Molecule 2: {msg2}"

        # Calculate similarity
        similarity = self.fp_generator.compare_smiles(proc1, proc2, metric="tanimoto")

        if similarity is None:
            return "❌ Failed to calculate similarity"

        output = "## Molecular Similarity\n\n"
        output += f"**Molecule 1**: `{proc1}`\n"
        output += f"**Molecule 2**: `{proc2}`\n"
        output += f"\n**Tanimoto Similarity**: {similarity:.4f}\n\n"

        if similarity > 0.85:
            output += "🟢 **Very Similar** - Molecules are highly similar\n"
        elif similarity > 0.70:
            output += "🟡 **Similar** - Molecules share significant structural features\n"
        elif similarity > 0.50:
            output += "🟠 **Moderately Similar** - Some structural similarity\n"
        else:
            output += "🔴 **Dissimilar** - Molecules are quite different\n"

        return output

    def predict_property(
        self,
        smiles: str,
        property_query: str,
        cot_strategy: str,
        top_k: int,
        progress=gr.Progress()
    ) -> Tuple[str, str, str]:
        """
        Predict molecular property (full MolRAG)

        Returns:
            (result_text, reasoning_text, status_text)
        """
        if not smiles or not property_query:
            return "❌ Please provide both SMILES and property query", "", ""

        # Validate SMILES
        is_valid, processed, msg = self.validate_smiles(smiles)
        if not is_valid:
            return msg, "", ""

        # Check if full system is available
        if self.molrag is None:
            progress(0.3, desc="Initializing MolRAG system...")
            try:
                from src.molrag import MolRAG
                self.molrag = MolRAG()
                self.molrag.initialize_preprocessing()
                logger.info("MolRAG system initialized")
            except Exception as e:
                logger.error(f"Failed to initialize MolRAG: {e}")
                return self._mock_prediction(processed, property_query, cot_strategy)

        # Check if databases are available
        try:
            progress(0.5, desc="Executing prediction...")
            result = self.molrag.predict(
                smiles=processed,
                query=property_query,
                cot_strategy=cot_strategy,
                top_k=top_k,
                preprocess=False  # Already preprocessed
            )

            progress(1.0, desc="Complete!")

            # Format result
            result_text = f"""# Prediction Result

**Query Molecule**: `{processed}`
**Property Question**: {property_query}

## 🎯 Prediction
**Result**: {result.prediction}
**Confidence**: {result.confidence:.2%}

## 📊 Retrieved Molecules
- **Total Retrieved**: {len(result.retrieved_molecules)}
- **Top Similarity**: {max([r.score for r in result.retrieved_molecules], default=0):.4f}

## 🔬 Biological Pathways
- **Pathways Found**: {len(result.pathways)}

## ⏱️ Performance
- **Execution Time**: {result.metadata.get('execution_time_seconds', 0):.2f}s
- **Strategy Used**: {result.cot_strategy}
- **Model**: {result.metadata.get('model', 'N/A')}
"""

            reasoning_text = f"# Reasoning Chain\n\n{result.reasoning}"

            # Status with molecules
            status_text = "## Retrieved Molecules\n\n"
            for i, mol in enumerate(result.retrieved_molecules[:5], 1):
                status_text += f"**{i}. SMILES**: `{mol.smiles}`\n"
                status_text += f"   - **Score**: {mol.score:.4f}\n"
                status_text += f"   - **Source**: {mol.source}\n"
                if mol.pathway:
                    status_text += f"   - **Pathway**: {' → '.join(mol.pathway[:3])}\n"
                status_text += "\n"

            return result_text, reasoning_text, status_text

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return self._mock_prediction(processed, property_query, cot_strategy)

    def _mock_prediction(
        self,
        smiles: str,
        property_query: str,
        cot_strategy: str
    ) -> Tuple[str, str, str]:
        """Error when databases not available - NO DEMO MODE"""
        result_text = f"""# ❌ Production System Not Ready

**Query Molecule**: `{smiles}`
**Property Question**: {property_query}

## 🚫 Real Data Required
This system operates with PRODUCTION DATA ONLY.
No demo mode or sample data available.

**Required Setup:**
1. ✅ Databases (Neo4j, Qdrant, Redis) running
2. ✅ API keys configured
3. ✅ **PrimeKG data loaded (130K+ nodes, 4M+ relationships)**

**Current Status**: ❌ System not initialized with real data

---

## 🚀 Production Setup Required

Run the setup script to download and load real PrimeKG data:

```bash
# 1. Add API key
echo "OPENAI_API_KEY=sk-your-key" >> .env

# 2. Setup production system (downloads PrimeKG ~500MB, 20 min)
./scripts/setup_real_data.sh

# 3. Restart this UI
python app.py
```

**What will be installed:**
- PrimeKG: 130,000+ biomedical entities
- 4,000,000+ curated relationships
- Drugs, proteins, diseases, pathways
- Real drug-target-disease data

**No demo/sample data mode available.**
"""

        reasoning_text = f"""# System Initialization Required

## Production Setup (20 minutes)

### Quick Method
```bash
./scripts/setup_real_data.sh
```

### Manual Method

1. **Start Databases**
```bash
docker-compose up -d
```

2. **Download PrimeKG** (~500MB, 15 min)
```bash
python scripts/download_real_data.py --dataset primekg
```

3. **Load into Databases** (5-10 min)
```bash
python scripts/load_knowledge_graphs.py
```

4. **Verify**
```bash
python -c "
from src.utils import Config, Neo4jConnector
config = Config()
neo4j = Neo4jConnector(config.neo4j_uri, config.neo4j_user, config.neo4j_password)
result = neo4j.execute_query('MATCH (n) RETURN count(n) as count')
print(f'Nodes: {result[0][\"count\"]:,}')
"
```

Expected: 130,000+ nodes

---

## Why No Demo Mode?

This is a production system for real drug discovery and biomedical research.
Operating with toy/sample data would provide misleading results.

**Real data ensures:**
- Accurate predictions
- Valid reasoning chains
- Reliable drug-target interactions
- Publication-quality results
"""

        status_text = """❌ **Production Data Required**

System requires real PrimeKG data (130K+ nodes, 4M+ relationships).

Setup:
```bash
./scripts/setup_real_data.sh
```

No sample/demo mode available.
"""

        return result_text, reasoning_text, status_text

    def create_ui(self) -> gr.Blocks:
        """Create Gradio UI"""

        with gr.Blocks(
            title="MolRAG - Molecular Property Prediction",
            theme=gr.themes.Soft()
        ) as demo:

            gr.Markdown("""
            # 🧬 MolRAG: Molecular Retrieval-Augmented Generation

            **Training-Free Molecular Property Prediction with Large Language Models and Knowledge Graphs**

            Combines:
            - 🔬 Molecular fingerprints (structural similarity)
            - 🧠 Knowledge graphs (biological mechanisms)
            - 🤖 Graph Neural Networks (knowledge-aware embeddings)
            - 💡 Large Language Models (reasoning and synthesis)
            """)

            with gr.Tabs():

                # Tab 1: Property Prediction
                with gr.Tab("🎯 Property Prediction"):
                    gr.Markdown("### Predict molecular properties using MolRAG")

                    with gr.Row():
                        with gr.Column():
                            pred_smiles = gr.Textbox(
                                label="SMILES String",
                                placeholder="CC(C)Cc1ccc(cc1)C(C)C(O)=O",
                                info="Enter molecular SMILES notation"
                            )

                            pred_query = gr.Textbox(
                                label="Property Query",
                                placeholder="Is this molecule toxic?",
                                info="Ask about molecular properties"
                            )

                            with gr.Row():
                                cot_strategy = gr.Dropdown(
                                    choices=["sim_cot", "struct_cot", "path_cot"],
                                    value="sim_cot",
                                    label="Reasoning Strategy",
                                    info="Sim-CoT is best on 6/7 datasets"
                                )

                                top_k = gr.Slider(
                                    minimum=5,
                                    maximum=20,
                                    value=10,
                                    step=1,
                                    label="Top-K Results",
                                    info="Number of similar molecules to retrieve"
                                )

                            predict_btn = gr.Button("🚀 Predict Property", variant="primary")

                        with gr.Column():
                            pred_result = gr.Markdown(label="Prediction Result")

                    with gr.Row():
                        pred_reasoning = gr.Markdown(label="Reasoning Chain")
                        pred_status = gr.Markdown(label="Retrieved Molecules")

                    predict_btn.click(
                        fn=self.predict_property,
                        inputs=[pred_smiles, pred_query, cot_strategy, top_k],
                        outputs=[pred_result, pred_reasoning, pred_status]
                    )

                    # Examples
                    gr.Examples(
                        examples=[
                            ["CC(C)Cc1ccc(cc1)C(C)C(O)=O", "Is this molecule toxic?", "sim_cot", 10],
                            ["CCO", "Is this molecule soluble in water?", "struct_cot", 10],
                            ["CC(=O)OC1=CC=CC=C1C(=O)O", "Does this molecule have anti-inflammatory properties?", "path_cot", 10],
                        ],
                        inputs=[pred_smiles, pred_query, cot_strategy, top_k],
                        label="Example Queries"
                    )

                # Tab 2: SMILES Analysis
                with gr.Tab("🔬 Molecular Analysis"):
                    gr.Markdown("### Analyze molecular structure and properties")

                    with gr.Row():
                        with gr.Column():
                            analysis_smiles = gr.Textbox(
                                label="SMILES String",
                                placeholder="CC(C)Cc1ccc(cc1)C(C)C(O)=O",
                                info="Enter molecular SMILES notation"
                            )

                            analyze_btn = gr.Button("🔍 Analyze Molecule", variant="primary")

                        with gr.Column():
                            analysis_props = gr.Markdown(label="Molecular Properties")

                    analysis_fp = gr.Markdown(label="Fingerprint Information")

                    analyze_btn.click(
                        fn=self.get_molecular_properties,
                        inputs=[analysis_smiles],
                        outputs=[analysis_props]
                    )

                    analyze_btn.click(
                        fn=self.generate_fingerprint_info,
                        inputs=[analysis_smiles],
                        outputs=[analysis_fp]
                    )

                    gr.Examples(
                        examples=[
                            ["CC(C)Cc1ccc(cc1)C(C)C(O)=O"],  # Ibuprofen
                            ["CCO"],  # Ethanol
                            ["CC(=O)OC1=CC=CC=C1C(=O)O"],  # Aspirin
                            ["CN1C=NC2=C1C(=O)N(C(=O)N2C)C"],  # Caffeine
                        ],
                        inputs=[analysis_smiles],
                        label="Example Molecules"
                    )

                # Tab 3: Similarity Comparison
                with gr.Tab("⚖️ Molecule Comparison"):
                    gr.Markdown("### Compare two molecules for structural similarity")

                    with gr.Row():
                        with gr.Column():
                            compare_smiles1 = gr.Textbox(
                                label="Molecule 1 SMILES",
                                placeholder="CC(C)Cc1ccc(cc1)C(C)C(O)=O"
                            )

                            compare_smiles2 = gr.Textbox(
                                label="Molecule 2 SMILES",
                                placeholder="CC(=O)OC1=CC=CC=C1C(=O)O"
                            )

                            compare_btn = gr.Button("🔄 Compare Molecules", variant="primary")

                        with gr.Column():
                            compare_result = gr.Markdown(label="Similarity Analysis")

                    compare_btn.click(
                        fn=self.compare_molecules,
                        inputs=[compare_smiles1, compare_smiles2],
                        outputs=[compare_result]
                    )

                    gr.Examples(
                        examples=[
                            ["CC(C)Cc1ccc(cc1)C(C)C(O)=O", "CC(=O)OC1=CC=CC=C1C(=O)O"],  # Ibuprofen vs Aspirin
                            ["CCO", "CCCO"],  # Ethanol vs Propanol
                            ["c1ccccc1", "c1ccncc1"],  # Benzene vs Pyridine
                        ],
                        inputs=[compare_smiles1, compare_smiles2],
                        label="Example Comparisons"
                    )

                # Tab 4: About
                with gr.Tab("ℹ️ About"):
                    gr.Markdown("""
                    ## About MolRAG

                    MolRAG is a training-free molecular property prediction system that combines:

                    ### 🔬 Triple Retrieval System
                    - **Vector Retrieval**: Morgan fingerprints with HNSW indexing (top-50)
                    - **Graph Retrieval**: Knowledge graph traversal with Neo4j (top-40)
                    - **GNN Retrieval**: KPGT knowledge-aware embeddings (top-30)
                    - **Hybrid Re-ranking**: 0.4×Tanimoto + 0.3×PathRelevance + 0.3×GNN → top-10

                    ### 🤖 Multi-Agent Architecture (CLADD Pattern)
                    1. **Planning Agent**: Query classification
                    2. **Vector Retrieval Agent**: Fingerprint search
                    3. **Graph Retrieval Agent**: KG traversal
                    4. **GNN Prediction Agent**: Embedding search
                    5. **Synthesis Agent**: GPT-4/Claude integration

                    ### 💡 Chain-of-Thought Strategies
                    - **Struct-CoT**: Structure-aware analysis
                    - **Sim-CoT**: Similarity-based reasoning (best on 6/7 datasets)
                    - **Path-CoT**: Biological pathway reasoning (NEW)

                    ### 📊 Performance
                    - **BACE**: +20.39% over LLM baseline (72.25% accuracy)
                    - **CYP450**: +21.22% improvement (72.29% accuracy)
                    - **Training**: Zero training required
                    - **Speed**: Sub-millisecond retrieval with parallel execution

                    ### 📚 Citation
                    ```bibtex
                    @article{krotkov2025nanostructured,
                      title={Nanostructured Material Design via RAG},
                      author={Krotkov, Nikita A and others},
                      journal={Journal of Chemical Information and Modeling},
                      volume={65},
                      pages={11064--11078},
                      year={2025}
                    }
                    ```

                    ### 🔗 Resources
                    - **GitHub**: [MolRAG Repository](https://github.com/yourusername/MOLRAG_KB)
                    - **Documentation**: See `README.md`
                    - **Paper**: `docs/papers/krotkov_et_al_2025_JCIM.pdf`

                    ### ⚙️ Setup
                    To enable full prediction capabilities:
                    1. Run `python scripts/setup_databases.py`
                    2. Load knowledge graphs (PrimeKG, DrugBank, ChEMBL)
                    3. Set API keys in `.env` file
                    4. Restart the UI
                    """)

            gr.Markdown("""
            ---
            **MolRAG v0.1.0** | Built with ❤️ for the molecular AI community
            """)

        return demo


def main():
    """Launch Gradio UI"""
    print("=" * 60)
    print("MolRAG - Molecular Property Prediction UI")
    print("=" * 60)

    # Initialize UI
    ui = MolRAGUI()
    demo = ui.create_ui()

    # Launch
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    main()
