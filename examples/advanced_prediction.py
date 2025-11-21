#!/usr/bin/env python3
"""
Advanced MolRAG Prediction Examples

Demonstrates:
1. Using different Chain-of-Thought strategies
2. Custom retrieval configurations
3. Ensemble predictions
4. Interpreting results in detail

Requirements: Databases must be set up (Neo4j, Qdrant, Redis)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.molrag import MolRAG
from src.retrieval import TripleRetriever


def example_1_cot_strategies():
    """Compare different Chain-of-Thought strategies"""
    print("=" * 70)
    print("Example 1: Comparing Chain-of-Thought Strategies")
    print("=" * 70)

    molrag = MolRAG(auto_init=True)

    # Test molecule: Aspirin
    smiles = "CC(=O)Oc1ccccc1C(=O)O"
    query = "Does this molecule have anti-inflammatory properties?"

    strategies = ["struct_cot", "sim_cot", "path_cot"]

    print(f"\nMolecule: {smiles}")
    print(f"Query: {query}\n")

    results = {}
    for strategy in strategies:
        print(f"\nTesting {strategy.upper()}...")
        result = molrag.predict(
            smiles=smiles,
            query=query,
            cot_strategy=strategy,
            top_k=10
        )

        results[strategy] = result

        print(f"  Prediction: {result.prediction}")
        print(f"  Confidence: {result.confidence:.2%}")
        print(f"  Time: {result.metadata['execution_time']:.2f}s")

    # Summary
    print("\n" + "=" * 70)
    print("Strategy Comparison:")
    print("=" * 70)
    for strategy, result in results.items():
        print(f"{strategy.upper():12s}: {result.prediction:15s} "
              f"({result.confidence:.0%} confidence)")

    print("\nRecommendation:")
    best_strategy = max(results.items(), key=lambda x: x[1].confidence)
    print(f"  Use {best_strategy[0].upper()} for this type of query "
          f"(highest confidence: {best_strategy[1].confidence:.0%})")


def example_2_custom_retrieval():
    """Use custom retrieval configuration"""
    print("\n" + "=" * 70)
    print("Example 2: Custom Retrieval Configuration")
    print("=" * 70)

    # Test with different retrieval weights
    configs = [
        {"name": "Fingerprint-Heavy", "vector": 0.7, "graph": 0.2, "gnn": 0.1},
        {"name": "Balanced", "vector": 0.4, "graph": 0.3, "gnn": 0.3},
        {"name": "Graph-Heavy", "vector": 0.2, "graph": 0.6, "gnn": 0.2},
    ]

    smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"  # Caffeine
    query = "Can this molecule cross the blood-brain barrier?"

    print(f"\nMolecule: Caffeine")
    print(f"Query: {query}\n")

    for config in configs:
        print(f"\n{config['name']} Configuration:")
        print(f"  Weights: Vector={config['vector']}, "
              f"Graph={config['graph']}, GNN={config['gnn']}")

        retriever = TripleRetriever(
            vector_weight=config['vector'],
            graph_weight=config['graph'],
            gnn_weight=config['gnn']
        )

        results = retriever.retrieve(
            query_smiles=smiles,
            top_k=10,
            enable_vector=True,
            enable_graph=True,
            enable_gnn=True,
            rerank=True
        )

        print(f"  Retrieved {len(results)} molecules")
        print(f"  Top result similarity: {results[0].score:.3f}")
        print(f"  Top molecule: {results[0].smiles[:30]}...")


def example_3_ensemble_prediction():
    """Use ensemble of predictions for higher confidence"""
    print("\n" + "=" * 70)
    print("Example 3: Ensemble Prediction")
    print("=" * 70)

    molrag = MolRAG(auto_init=True)

    smiles = "Cc1ccc(cc1)S(=O)(=O)N"  # Sulfonamide
    query = "Is this molecule hepatotoxic?"

    print(f"\nMolecule: {smiles}")
    print(f"Query: {query}\n")

    # Get predictions from all strategies
    strategies = ["struct_cot", "sim_cot", "path_cot"]
    predictions = []

    for strategy in strategies:
        result = molrag.predict(
            smiles=smiles,
            query=query,
            cot_strategy=strategy,
            top_k=10
        )
        predictions.append({
            'strategy': strategy,
            'prediction': result.prediction,
            'confidence': result.confidence
        })

        print(f"{strategy.upper()}:")
        print(f"  Prediction: {result.prediction}")
        print(f"  Confidence: {result.confidence:.2%}\n")

    # Ensemble: weighted average
    weights = {
        'struct_cot': 0.3,
        'sim_cot': 0.5,    # Best performer on most datasets
        'path_cot': 0.2
    }

    ensemble_confidence = sum(
        pred['confidence'] * weights[pred['strategy']]
        for pred in predictions
    )

    # Majority vote
    prediction_counts = {}
    for pred in predictions:
        prediction_counts[pred['prediction']] = \
            prediction_counts.get(pred['prediction'], 0) + 1

    majority_prediction = max(prediction_counts.items(), key=lambda x: x[1])[0]

    print("=" * 70)
    print("Ensemble Results:")
    print("=" * 70)
    print(f"Weighted Ensemble Confidence: {ensemble_confidence:.2%}")
    print(f"Majority Vote Prediction: {majority_prediction}")
    print(f"Agreement: {max(prediction_counts.values())}/{len(strategies)} strategies")


def example_4_detailed_analysis():
    """Detailed analysis of prediction results"""
    print("\n" + "=" * 70)
    print("Example 4: Detailed Result Analysis")
    print("=" * 70)

    molrag = MolRAG(auto_init=True)

    smiles = "CC(C)Cc1ccc(cc1)C(C)C(O)=O"  # Ibuprofen
    query = "What are the potential side effects of this molecule?"

    print(f"\nMolecule: Ibuprofen")
    print(f"Query: {query}\n")

    result = molrag.predict(
        smiles=smiles,
        query=query,
        cot_strategy="sim_cot",
        top_k=15
    )

    # Analyze retrieved molecules
    print("\nRetrieved Molecules:")
    print("=" * 70)
    for i, mol in enumerate(result.retrieved_molecules[:5], 1):
        print(f"{i}. {mol.smiles}")
        print(f"   Similarity: {mol.score:.3f}")
        print(f"   Known properties: {mol.properties}")
        print()

    # Analyze pathways
    print("Relevant Pathways:")
    print("=" * 70)
    for i, pathway in enumerate(result.pathways[:5], 1):
        print(f"{i}. {pathway.name}")
        print(f"   Description: {pathway.description}")
        print(f"   Relevance: {pathway.relevance_score:.3f}")
        print()

    # Full reasoning chain
    print("Chain-of-Thought Reasoning:")
    print("=" * 70)
    print(result.reasoning)

    # Citations
    print("\n" + "=" * 70)
    print("Citations:")
    print("=" * 70)
    for i, citation in enumerate(result.citations, 1):
        print(f"{i}. {citation}")

    # Metadata
    print("\n" + "=" * 70)
    print("Execution Metadata:")
    print("=" * 70)
    for key, value in result.metadata.items():
        print(f"  {key}: {value}")


def main():
    """Run all advanced examples"""
    print("\n" + "=" * 70)
    print("MolRAG Advanced Prediction Examples")
    print("=" * 70)
    print("These examples require databases to be set up.")
    print("See QUICKSTART.md for setup instructions.")
    print("=" * 70)

    try:
        # example_1_cot_strategies()
        # example_2_custom_retrieval()
        # example_3_ensemble_prediction()
        # example_4_detailed_analysis()

        print("\n" + "=" * 70)
        print("All examples completed successfully!")
        print("=" * 70)

    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure databases are set up:")
        print("  1. Start databases: docker-compose up -d")
        print("  2. Initialize: python scripts/setup_databases.py")
        print("  3. Load data: python scripts/load_knowledge_graphs.py")


if __name__ == "__main__":
    main()
