"""
Chain-of-Thought (CoT) reasoning strategies

Based on blueprint specifications:
1. Struct-CoT: Structural analysis (functional groups, chirality, rings)
2. Sim-CoT: Similarity-based reasoning (best performer on 6/7 datasets)
3. Path-CoT: Biological pathway reasoning (NEW)
"""

from typing import List, Dict, Any
from ..data.models import RetrievalResult
from ..utils.logger import get_logger

logger = get_logger(__name__)


class CoTStrategy:
    """Base class for Chain-of-Thought strategies"""

    def __init__(self, name: str):
        self.name = name

    def generate_reasoning(
        self,
        query_smiles: str,
        property_query: str,
        retrieval_results: List[RetrievalResult]
    ) -> str:
        """
        Generate chain-of-thought reasoning

        Args:
            query_smiles: Query molecule SMILES
            property_query: Property question
            retrieval_results: Retrieved similar molecules

        Returns:
            Reasoning chain as string
        """
        raise NotImplementedError


class StructCoT(CoTStrategy):
    """
    Structure-Aware Chain-of-Thought

    Focus:
    - Functional groups
    - Aromatic rings
    - Chirality
    - Chain length
    - Structural features → property mapping
    """

    def __init__(self):
        super().__init__("Struct-CoT")

    def generate_reasoning(
        self,
        query_smiles: str,
        property_query: str,
        retrieval_results: List[RetrievalResult]
    ) -> str:
        """Generate structure-aware reasoning"""

        reasoning = f"""Structure-Aware Analysis for {query_smiles}:

1. MOLECULAR STRUCTURE DECOMPOSITION:
"""

        # Analyze query molecule structure
        structural_features = self._analyze_structure(query_smiles)

        reasoning += "   Key Structural Features:\n"
        for feature, value in structural_features.items():
            reasoning += f"   - {feature}: {value}\n"

        reasoning += "\n2. SIMILAR MOLECULES STRUCTURAL COMPARISON:\n"

        # Compare with retrieved molecules
        for i, result in enumerate(retrieval_results[:3], 1):
            reasoning += f"\n   Molecule {i} (Similarity: {result.score:.3f}):\n"
            reasoning += f"   SMILES: {result.smiles}\n"

            similar_features = self._analyze_structure(result.smiles)
            reasoning += "   Shared Features:\n"

            # Identify shared features
            for feature, value in similar_features.items():
                if feature in structural_features:
                    if structural_features[feature] == value:
                        reasoning += f"     ✓ {feature}: {value}\n"

        reasoning += "\n3. STRUCTURE-PROPERTY RELATIONSHIP:\n"
        reasoning += self._map_structure_to_property(
            structural_features, property_query, retrieval_results
        )

        reasoning += "\n4. CONCLUSION:\n"
        reasoning += "   Based on structural analysis and comparison with similar molecules,\n"
        reasoning += "   the prediction considers key functional groups and structural motifs.\n"

        return reasoning

    def _analyze_structure(self, smiles: str) -> Dict[str, Any]:
        """Analyze molecular structure"""
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, Lipinski

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {}

            features = {
                'Aromatic Rings': Descriptors.NumAromaticRings(mol),
                'Aliphatic Rings': Descriptors.NumAliphaticRings(mol),
                'H-Bond Donors': Lipinski.NumHDonors(mol),
                'H-Bond Acceptors': Lipinski.NumHAcceptors(mol),
                'Rotatable Bonds': Lipinski.NumRotatableBonds(mol),
                'Molecular Weight': f"{Descriptors.MolWt(mol):.1f} Da",
                'LogP': f"{Descriptors.MolLogP(mol):.2f}"
            }

            return features

        except Exception as e:
            logger.error(f"Structure analysis failed: {e}")
            return {}

    def _map_structure_to_property(
        self,
        features: Dict[str, Any],
        property_query: str,
        results: List[RetrievalResult]
    ) -> str:
        """Map structural features to property"""
        mapping = "   Structural features relevant to property:\n"

        # Heuristic mappings (simplified)
        property_lower = property_query.lower()

        if 'toxic' in property_lower:
            mapping += "   - Aromatic rings: High number may increase toxicity\n"
            mapping += "   - LogP: Higher values associated with lipophilicity\n"
        elif 'solub' in property_lower:
            mapping += "   - H-Bond Donors/Acceptors: Important for water solubility\n"
            mapping += "   - Molecular Weight: Lower MW generally more soluble\n"
        elif 'bioavail' in property_lower or 'drug' in property_lower:
            mapping += "   - Lipinski's Rule of Five applies\n"
            mapping += "   - Balanced hydrophilicity/lipophilicity important\n"

        return mapping


class SimCoT(CoTStrategy):
    """
    Similarity-Aware Chain-of-Thought

    Principle: Similar molecules have similar properties
    Performance: Best on 6/7 datasets per blueprint
    """

    def __init__(self):
        super().__init__("Sim-CoT")

    def generate_reasoning(
        self,
        query_smiles: str,
        property_query: str,
        retrieval_results: List[RetrievalResult]
    ) -> str:
        """Generate similarity-based reasoning"""

        reasoning = f"""Similarity-Based Analysis for {query_smiles}:

PRINCIPLE: Structurally similar molecules tend to have similar properties (Property Continuity Principle)

1. RETRIEVED SIMILAR MOLECULES:
"""

        # Analyze similarity scores and properties
        for i, result in enumerate(retrieval_results[:5], 1):
            reasoning += f"\n   {i}. Similarity: {result.score:.3f}\n"
            reasoning += f"      SMILES: {result.smiles}\n"
            reasoning += f"      Properties: {result.properties}\n"
            reasoning += f"      Source: {result.source}\n"

        reasoning += "\n2. SIMILARITY DISTRIBUTION ANALYSIS:\n"

        similarities = [r.score for r in retrieval_results]
        if similarities:
            avg_sim = sum(similarities) / len(similarities)
            max_sim = max(similarities)
            reasoning += f"   - Average Similarity: {avg_sim:.3f}\n"
            reasoning += f"   - Maximum Similarity: {max_sim:.3f}\n"
            reasoning += f"   - Number of Similar Molecules: {len(retrieval_results)}\n"

        reasoning += "\n3. CROSS-MOLECULE CORRELATION:\n"

        # Analyze property patterns across similar molecules
        property_patterns = self._analyze_property_patterns(retrieval_results)
        reasoning += property_patterns

        reasoning += "\n4. PROPERTY CONTINUITY REASONING:\n"
        reasoning += "   Given the high similarity scores with retrieved molecules,\n"
        reasoning += "   we expect property continuity: the query molecule should exhibit\n"
        reasoning += "   properties similar to its structural neighbors.\n"

        reasoning += "\n5. CONFIDENCE ASSESSMENT:\n"
        if max(similarities) > 0.8:
            reasoning += "   HIGH confidence (similarity > 0.8)\n"
        elif max(similarities) > 0.6:
            reasoning += "   MEDIUM confidence (similarity 0.6-0.8)\n"
        else:
            reasoning += "   LOW confidence (similarity < 0.6)\n"

        return reasoning

    def _analyze_property_patterns(
        self,
        results: List[RetrievalResult]
    ) -> str:
        """Analyze property patterns across similar molecules"""
        patterns = "   Property distribution in similar molecules:\n"

        # Count property values (if available)
        property_counts = {}
        for result in results:
            for prop_name, prop_value in result.properties.items():
                if prop_name not in property_counts:
                    property_counts[prop_name] = {}
                val_str = str(prop_value)
                property_counts[prop_name][val_str] = \
                    property_counts[prop_name].get(val_str, 0) + 1

        for prop_name, value_counts in property_counts.items():
            patterns += f"   - {prop_name}:\n"
            for value, count in value_counts.items():
                patterns += f"     {value}: {count}/{len(results)} molecules\n"

        return patterns


class PathCoT(CoTStrategy):
    """
    Path-based Chain-of-Thought (NEW)

    Focus on biological pathways and mechanisms
    Example: molecule→binds→DPP-4→participates_in→Insulin_signaling→treats→Diabetes
    """

    def __init__(self):
        super().__init__("Path-CoT")

    def generate_reasoning(
        self,
        query_smiles: str,
        property_query: str,
        retrieval_results: List[RetrievalResult]
    ) -> str:
        """Generate pathway-based reasoning"""

        reasoning = f"""Pathway-Based Mechanistic Analysis for {query_smiles}:

1. BIOLOGICAL PATHWAY ANALYSIS:
"""

        # Extract pathway information from graph retrieval results
        pathway_results = [r for r in retrieval_results if r.pathway]

        if not pathway_results:
            reasoning += "   No pathway information available.\n"
            reasoning += "   Falling back to similarity-based reasoning.\n"
            # Fallback to similarity reasoning
            return SimCoT().generate_reasoning(query_smiles, property_query, retrieval_results)

        reasoning += f"   Found {len(pathway_results)} molecules with pathway connections.\n\n"

        # Analyze pathways
        pathway_counts = {}
        for result in pathway_results:
            pathway_str = " → ".join(result.pathway)
            pathway_counts[pathway_str] = pathway_counts.get(pathway_str, 0) + 1

        reasoning += "2. COMMON PATHWAYS:\n"
        sorted_pathways = sorted(pathway_counts.items(), key=lambda x: x[1], reverse=True)

        for pathway, count in sorted_pathways[:5]:
            reasoning += f"\n   {count} molecules via: {pathway}\n"

        reasoning += "\n3. MECHANISTIC CONNECTIONS:\n"
        reasoning += "   Molecules sharing pathways with query molecule:\n"

        for i, result in enumerate(pathway_results[:3], 1):
            reasoning += f"\n   {i}. SMILES: {result.smiles}\n"
            reasoning += f"      Pathway: {' → '.join(result.pathway)}\n"
            reasoning += f"      Hops: {result.hops}\n"
            reasoning += f"      Path Relevance: {result.path_relevance_score:.3f}\n"

        reasoning += "\n4. MECHANISM-OF-ACTION INFERENCE:\n"
        reasoning += self._infer_mechanism(pathway_results, property_query)

        reasoning += "\n5. PATHWAY-BASED PREDICTION:\n"
        reasoning += "   Based on shared biological pathways and mechanisms,\n"
        reasoning += "   we can infer property by mechanistic similarity.\n"

        return reasoning

    def _infer_mechanism(
        self,
        pathway_results: List[RetrievalResult],
        property_query: str
    ) -> str:
        """Infer mechanism of action from pathways"""
        mechanism = "   Inferred mechanisms:\n"

        # Extract common pathway elements
        all_relations = []
        for result in pathway_results:
            all_relations.extend(result.pathway)

        relation_counts = {}
        for rel in all_relations:
            relation_counts[rel] = relation_counts.get(rel, 0) + 1

        # Most common relationships
        top_relations = sorted(relation_counts.items(), key=lambda x: x[1], reverse=True)[:3]

        for rel, count in top_relations:
            mechanism += f"   - {rel}: appears in {count} pathways\n"

        return mechanism


def get_cot_strategy(strategy_name: str) -> CoTStrategy:
    """
    Factory function to get CoT strategy by name

    Args:
        strategy_name: Strategy name (struct_cot, sim_cot, path_cot)

    Returns:
        CoT strategy instance
    """
    strategy_map = {
        'struct_cot': StructCoT,
        'sim_cot': SimCoT,
        'path_cot': PathCoT
    }

    strategy_class = strategy_map.get(strategy_name.lower())
    if strategy_class is None:
        logger.warning(f"Unknown strategy: {strategy_name}, defaulting to Sim-CoT")
        strategy_class = SimCoT

    return strategy_class()
