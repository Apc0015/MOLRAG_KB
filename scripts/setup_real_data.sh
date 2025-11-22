#!/bin/bash
# Setup Real Molecular Databases for MolRAG
# This script downloads and loads actual PrimeKG, ChEMBL data

set -e

echo "=========================================="
echo "MolRAG Real Data Setup"
echo "=========================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}This will download and load REAL molecular databases:${NC}"
echo ""
echo "  📦 PrimeKG:"
echo "     - Size: ~500MB download"
echo "     - Data: 4,000,000+ relationships"
echo "     - Nodes: 130,000+ (drugs, proteins, diseases, pathways)"
echo "     - Time: ~15-20 minutes"
echo ""
echo "  📦 ChEMBL (optional):"
echo "     - Sample: 1,000 molecules via API"
echo "     - Full download requires manual setup"
echo ""
echo "  ⚠️  Requirements:"
echo "     - 2GB free disk space"
echo "     - Stable internet connection"
echo "     - Databases running (docker-compose up -d)"
echo ""

read -p "Continue with real data download? (y/N) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled. Use './scripts/quick_start.sh' for sample data instead."
    exit 0
fi

# Check databases are running
echo -e "${GREEN}[1/4] Checking databases...${NC}"
if ! docker-compose ps | grep -q "molrag_neo4j.*Up"; then
    echo -e "${RED}Error: Neo4j not running!${NC}"
    echo "Start databases first: docker-compose up -d"
    exit 1
fi

if ! docker-compose ps | grep -q "molrag_qdrant.*Up"; then
    echo -e "${RED}Error: Qdrant not running!${NC}"
    echo "Start databases first: docker-compose up -d"
    exit 1
fi

echo -e "${GREEN}✓ Databases are running${NC}"

# Download real data
echo ""
echo -e "${GREEN}[2/4] Downloading PrimeKG (this will take 10-15 minutes)...${NC}"
python scripts/download_real_data.py --dataset primekg

if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to download PrimeKG${NC}"
    exit 1
fi

# Load into Neo4j and Qdrant
echo ""
echo -e "${GREEN}[3/4] Loading data into databases (this will take 5-10 minutes)...${NC}"
python scripts/load_knowledge_graphs.py

if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to load knowledge graphs${NC}"
    exit 1
fi

# Verify
echo ""
echo -e "${GREEN}[4/4] Verifying data...${NC}"
python -c "
from src.utils import Config, Neo4jConnector
config = Config()
neo4j = Neo4jConnector(config.neo4j_uri, config.neo4j_user, config.neo4j_password)
result = neo4j.execute_query('MATCH (n) RETURN count(n) as count')
node_count = result[0]['count'] if result else 0
result = neo4j.execute_query('MATCH ()-[r]->() RETURN count(r) as count')
rel_count = result[0]['count'] if result else 0
print(f'\n✓ Loaded {node_count:,} nodes')
print(f'✓ Loaded {rel_count:,} relationships')
neo4j.close()
"

# Complete
echo ""
echo "=========================================="
echo -e "${GREEN}✅ REAL DATA SETUP COMPLETE!${NC}"
echo "=========================================="
echo ""
echo "📊 You now have access to:"
echo "  - 130,000+ biomedical entities"
echo "  - 4,000,000+ relationships"
echo "  - Drugs, proteins, diseases, pathways"
echo ""
echo "🚀 Next steps:"
echo "  1. Start UI: python app.py"
echo "  2. Open: http://localhost:7860"
echo "  3. Try real molecular predictions!"
echo ""
echo "💡 Tips:"
echo "  - Use 'sim_cot' strategy for best results"
echo "  - Query about drug-target interactions"
echo "  - Explore disease-pathway relationships"
echo ""
echo "📝 View data:"
echo "  - Neo4j Browser: http://localhost:7474"
echo "  - Credentials: neo4j / molrag_password_2024"
echo ""
