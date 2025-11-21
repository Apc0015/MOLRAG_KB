"""
Generate sample knowledge graph data for testing MolRAG

Creates a small but realistic KG with:
- 100 molecules (drugs and compounds)
- 50 proteins (targets)
- 20 diseases
- 10 pathways
- ~500 relationships
"""

import csv
from pathlib import Path
from typing import List, Dict

# Sample molecules (real drug examples)
SAMPLE_MOLECULES = [
    # Anticancer drugs
    {"id": "CHEMBL25", "name": "Imatinib", "smiles": "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5", "type": "Drug"},
    {"id": "CHEMBL428647", "name": "Gefitinib", "smiles": "COC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4", "type": "Drug"},
    {"id": "CHEMBL1201585", "name": "Erlotinib", "smiles": "C#CC1=CC=C(C=C1)NC2=NC=NC3=C2C=C(C(=C3)OCCOC)OCCOC", "type": "Drug"},
    
    # Cardiovascular drugs
    {"id": "CHEMBL502", "name": "Aspirin", "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O", "type": "Drug"},
    {"id": "CHEMBL1308", "name": "Atorvastatin", "smiles": "CC(C)C1=C(C(=C(N1CC[C@H](C[C@H](CC(=O)O)O)O)C2=CC=C(C=C2)F)C3=CC=CC=C3)C(=O)NC4=CC=CC=C4", "type": "Drug"},
    
    # Antibiotics
    {"id": "CHEMBL1095", "name": "Amoxicillin", "smiles": "CC1([C@@H](N2[C@H](S1)[C@@H](C2=O)NC(=O)[C@@H](C3=CC=C(C=C3)O)N)C(=O)O)C", "type": "Drug"},
    {"id": "CHEMBL1771", "name": "Ciprofloxacin", "smiles": "C1CC1N2C=C(C(=O)C3=CC(=C(C=C32)N4CCNCC4)F)C(=O)O", "type": "Drug"},
    
    # Diabetes drugs
    {"id": "CHEMBL1431", "name": "Metformin", "smiles": "CN(C)C(=N)NC(=N)N", "type": "Drug"},
    {"id": "CHEMBL1276308", "name": "Sitagliptin", "smiles": "C[C@@H](CC(=O)N1CCN(CC1)CC2=CC(=C(C=C2F)F)F)[C@H](CC3=CC(=C(C=C3)F)F)N", "type": "Drug"},
    
    # Pain/inflammation
    {"id": "CHEMBL521", "name": "Ibuprofen", "smiles": "CC(C)CC1=CC=C(C=C1)[C@@H](C)C(=O)O", "type": "Drug"},
]

# Add more molecules to reach 100
for i in range(10, 100):
    SAMPLE_MOLECULES.append({
        "id": f"CHEMBL{1000+i}",
        "name": f"Compound_{i}",
        "smiles": "C" * (i % 10 + 5),  # Simple placeholder
        "type": "Molecule"
    })

# Sample proteins/targets
SAMPLE_PROTEINS = [
    {"id": "P00533", "name": "EGFR", "type": "Protein", "gene": "EGFR"},
    {"id": "P04626", "name": "ERBB2", "type": "Protein", "gene": "ERBB2"},
    {"id": "P10275", "name": "AR", "type": "Protein", "gene": "AR"},
    {"id": "P35354", "name": "PTGS2", "type": "Protein", "gene": "PTGS2"},
    {"id": "P23219", "name": "PTGS1", "type": "Protein", "gene": "PTGS1"},
    {"id": "P27361", "name": "MAPK3", "type": "Protein", "gene": "MAPK3"},
    {"id": "P28482", "name": "MAPK1", "type": "Protein", "gene": "MAPK1"},
    {"id": "P42574", "name": "CASP3", "type": "Protein", "gene": "CASP3"},
    {"id": "P10636", "name": "MAPT", "type": "Protein", "gene": "MAPT"},
    {"id": "P05067", "name": "APP", "type": "Protein", "gene": "APP"},
]

for i in range(10, 50):
    SAMPLE_PROTEINS.append({
        "id": f"P{10000+i}",
        "name": f"Protein_{i}",
        "type": "Protein",
        "gene": f"GENE{i}"
    })

# Sample diseases
SAMPLE_DISEASES = [
    {"id": "MONDO_0004992", "name": "Cancer", "type": "Disease"},
    {"id": "MONDO_0005148", "name": "Type 2 Diabetes", "type": "Disease"},
    {"id": "MONDO_0005180", "name": "Parkinson Disease", "type": "Disease"},
    {"id": "MONDO_0004975", "name": "Alzheimer Disease", "type": "Disease"},
    {"id": "MONDO_0005267", "name": "Heart Disease", "type": "Disease"},
    {"id": "MONDO_0005015", "name": "Diabetes Mellitus", "type": "Disease"},
    {"id": "MONDO_0005301", "name": "Multiple Sclerosis", "type": "Disease"},
    {"id": "MONDO_0008199", "name": "Schizophrenia", "type": "Disease"},
    {"id": "MONDO_0005090", "name": "Rheumatoid Arthritis", "type": "Disease"},
    {"id": "MONDO_0004979", "name": "Asthma", "type": "Disease"},
]

for i in range(10, 20):
    SAMPLE_DISEASES.append({
        "id": f"MONDO_{9000+i}",
        "name": f"Disease_{i}",
        "type": "Disease"
    })

# Sample pathways
SAMPLE_PATHWAYS = [
    {"id": "R-HSA-1640170", "name": "Cell Cycle", "type": "Pathway"},
    {"id": "R-HSA-162582", "name": "Signal Transduction", "type": "Pathway"},
    {"id": "R-HSA-109581", "name": "Apoptosis", "type": "Pathway"},
    {"id": "R-HSA-168256", "name": "Immune System", "type": "Pathway"},
    {"id": "R-HSA-392499", "name": "Metabolism", "type": "Pathway"},
    {"id": "R-HSA-1257604", "name": "MAPK Signaling", "type": "Pathway"},
    {"id": "R-HSA-1643685", "name": "Disease", "type": "Pathway"},
    {"id": "R-HSA-5663202", "name": "PI3K/AKT Signaling", "type": "Pathway"},
    {"id": "R-HSA-449147", "name": "Programmed Cell Death", "type": "Pathway"},
    {"id": "R-HSA-194315", "name": "Signaling by Rho GTPases", "type": "Pathway"},
]


def generate_nodes() -> List[Dict]:
    """Combine all nodes"""
    nodes = []
    
    # Add molecules
    for mol in SAMPLE_MOLECULES:
        nodes.append({
            "node_id": mol["id"],
            "node_type": mol["type"],
            "node_name": mol["name"],
            "node_source": "ChEMBL"
        })
    
    # Add proteins
    for prot in SAMPLE_PROTEINS:
        nodes.append({
            "node_id": prot["id"],
            "node_type": prot["type"],
            "node_name": prot["name"],
            "node_source": "UniProt"
        })
    
    # Add diseases
    for dis in SAMPLE_DISEASES:
        nodes.append({
            "node_id": dis["id"],
            "node_type": dis["type"],
            "node_name": dis["name"],
            "node_source": "MONDO"
        })
    
    # Add pathways
    for path in SAMPLE_PATHWAYS:
        nodes.append({
            "node_id": path["id"],
            "node_type": path["type"],
            "node_name": path["name"],
            "node_source": "Reactome"
        })
    
    return nodes


def generate_edges() -> List[Dict]:
    """Generate realistic relationships"""
    edges = []
    
    # Drug-Protein (targets) relationships
    drug_target_pairs = [
        ("CHEMBL25", "P00533", "inhibits"),  # Imatinib -> EGFR
        ("CHEMBL428647", "P00533", "inhibits"),  # Gefitinib -> EGFR
        ("CHEMBL1201585", "P00533", "inhibits"),  # Erlotinib -> EGFR
        ("CHEMBL502", "P35354", "inhibits"),  # Aspirin -> PTGS2
        ("CHEMBL502", "P23219", "inhibits"),  # Aspirin -> PTGS1
        ("CHEMBL521", "P35354", "inhibits"),  # Ibuprofen -> PTGS2
    ]
    
    for source, target, relation in drug_target_pairs:
        edges.append({
            "source": source,
            "target": target,
            "relation": relation,
            "source_type": "Drug",
            "target_type": "Protein"
        })
    
    # Add more drug-target relationships
    for i, mol in enumerate(SAMPLE_MOLECULES[:30]):
        target_prot = SAMPLE_PROTEINS[i % 10]
        edges.append({
            "source": mol["id"],
            "target": target_prot["id"],
            "relation": "binds" if i % 2 == 0 else "inhibits",
            "source_type": mol["type"],
            "target_type": "Protein"
        })
    
    # Drug-Disease (treats/contraindicates)
    drug_disease_pairs = [
        ("CHEMBL25", "MONDO_0004992", "treats"),  # Imatinib -> Cancer
        ("CHEMBL1431", "MONDO_0005148", "treats"),  # Metformin -> T2D
        ("CHEMBL502", "MONDO_0005267", "treats"),  # Aspirin -> Heart Disease
        ("CHEMBL1095", "MONDO_0004979", "contraindicates"),  # Amoxicillin -> Asthma
    ]
    
    for source, target, relation in drug_disease_pairs:
        edges.append({
            "source": source,
            "target": target,
            "relation": relation,
            "source_type": "Drug",
            "target_type": "Disease"
        })
    
    # Add more drug-disease relationships
    for i, mol in enumerate(SAMPLE_MOLECULES[:40]):
        disease = SAMPLE_DISEASES[i % len(SAMPLE_DISEASES)]
        edges.append({
            "source": mol["id"],
            "target": disease["id"],
            "relation": "treats" if i % 3 != 0 else "associated_with",
            "source_type": mol["type"],
            "target_type": "Disease"
        })
    
    # Protein-Disease (genetic_association)
    for i, prot in enumerate(SAMPLE_PROTEINS[:15]):
        disease = SAMPLE_DISEASES[i % len(SAMPLE_DISEASES)]
        edges.append({
            "source": prot["id"],
            "target": disease["id"],
            "relation": "genetic_association",
            "source_type": "Protein",
            "target_type": "Disease"
        })
    
    # Protein-Pathway (participates_in)
    for i, prot in enumerate(SAMPLE_PROTEINS[:20]):
        pathway = SAMPLE_PATHWAYS[i % len(SAMPLE_PATHWAYS)]
        edges.append({
            "source": prot["id"],
            "target": pathway["id"],
            "relation": "participates_in",
            "source_type": "Protein",
            "target_type": "Pathway"
        })
    
    # Molecule-Molecule (similar_to)
    for i in range(20):
        mol1 = SAMPLE_MOLECULES[i]
        mol2 = SAMPLE_MOLECULES[i + 10]
        edges.append({
            "source": mol1["id"],
            "target": mol2["id"],
            "relation": "similar_to",
            "source_type": mol1["type"],
            "target_type": mol2["type"]
        })
    
    return edges


def save_primekg_format(output_dir: Path):
    """Save data in PrimeKG format"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save nodes
    nodes_file = output_dir / "kg_nodes.csv"
    nodes = generate_nodes()
    
    print(f"Generating {len(nodes)} nodes...")
    with open(nodes_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['node_id', 'node_type', 'node_name', 'node_source'])
        writer.writeheader()
        writer.writerows(nodes)
    
    print(f"✓ Saved nodes to {nodes_file}")
    
    # Save edges
    edges_file = output_dir / "kg_edges.csv"
    edges = generate_edges()
    
    print(f"Generating {len(edges)} edges...")
    with open(edges_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['source', 'target', 'relation', 'source_type', 'target_type'])
        writer.writeheader()
        writer.writerows(edges)
    
    print(f"✓ Saved edges to {edges_file}")
    
    # Save molecule details (for fingerprint generation)
    molecules_file = output_dir / "molecules.csv"
    
    print(f"Generating molecules with SMILES...")
    with open(molecules_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'name', 'smiles', 'type'])
        writer.writeheader()
        writer.writerows(SAMPLE_MOLECULES)
    
    print(f"✓ Saved molecules to {molecules_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("📊 SAMPLE KNOWLEDGE GRAPH GENERATED")
    print("="*60)
    print(f"Nodes: {len(nodes)}")
    print(f"  - Molecules/Drugs: {len(SAMPLE_MOLECULES)}")
    print(f"  - Proteins: {len(SAMPLE_PROTEINS)}")
    print(f"  - Diseases: {len(SAMPLE_DISEASES)}")
    print(f"  - Pathways: {len(SAMPLE_PATHWAYS)}")
    print(f"\nEdges: {len(edges)}")
    print(f"\nFiles saved to: {output_dir}")
    print("="*60)
    print("\nNext steps:")
    print("1. Run: python scripts/load_knowledge_graphs.py")
    print("2. Launch: python app.py")
    print("="*60)


if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent / "data"
    save_primekg_format(data_dir)
