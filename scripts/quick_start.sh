#!/bin/bash
# Quick Start Script for MolRAG
# This script will set up all databases and initialize the system

set -e

echo "=========================================="
echo "MolRAG Quick Start Setup"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo -e "${RED}Error: .env file not found!${NC}"
    echo "Please create .env file from .env.example and add your API keys"
    exit 1
fi

# Check if API keys are set
source .env
if [ -z "$OPENAI_API_KEY" ] && [ -z "$ANTHROPIC_API_KEY" ] && [ -z "$OPENROUTER_API_KEY" ]; then
    echo -e "${YELLOW}Warning: No API keys found in .env file${NC}"
    echo "Add at least one of: OPENAI_API_KEY, ANTHROPIC_API_KEY, or OPENROUTER_API_KEY"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Step 1: Start Docker containers
echo -e "${GREEN}[1/5] Starting Docker containers...${NC}"
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Error: docker-compose not found!${NC}"
    echo "Please install Docker and docker-compose first"
    exit 1
fi

docker-compose up -d

# Step 2: Wait for services to be ready
echo -e "${GREEN}[2/5] Waiting for services to be ready...${NC}"
echo "This may take 30-60 seconds..."

# Wait for Neo4j
echo -n "  - Neo4j: "
for i in {1..30}; do
    if docker-compose exec -T neo4j neo4j status &> /dev/null; then
        echo -e "${GREEN}Ready${NC}"
        break
    fi
    if [ $i -eq 30 ]; then
        echo -e "${RED}Timeout${NC}"
        exit 1
    fi
    sleep 2
    echo -n "."
done

# Wait for Qdrant
echo -n "  - Qdrant: "
for i in {1..30}; do
    if curl -s http://localhost:6333/healthz &> /dev/null; then
        echo -e "${GREEN}Ready${NC}"
        break
    fi
    if [ $i -eq 30 ]; then
        echo -e "${RED}Timeout${NC}"
        exit 1
    fi
    sleep 2
    echo -n "."
done

# Wait for Redis
echo -n "  - Redis: "
for i in {1..30}; do
    if docker-compose exec -T redis redis-cli -a molrag_redis_2024 ping &> /dev/null; then
        echo -e "${GREEN}Ready${NC}"
        break
    fi
    if [ $i -eq 30 ]; then
        echo -e "${RED}Timeout${NC}"
        exit 1
    fi
    sleep 2
    echo -n "."
done

# Step 3: Initialize databases
echo -e "${GREEN}[3/5] Initializing databases...${NC}"
python scripts/setup_databases.py

# Step 4: Load sample data (if available)
echo -e "${GREEN}[4/5] Loading sample data...${NC}"
if [ -f "scripts/load_sample_data.py" ]; then
    python scripts/load_sample_data.py
else
    echo -e "${YELLOW}Note: No sample data loader found. You can load your own data later.${NC}"
fi

# Step 5: Complete
echo -e "${GREEN}[5/5] Setup complete!${NC}"
echo ""
echo "=========================================="
echo -e "${GREEN}MolRAG is ready to use!${NC}"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Start the UI: python app.py"
echo "  2. Open browser: http://localhost:7860"
echo ""
echo "Database access:"
echo "  - Neo4j Browser: http://localhost:7474"
echo "  - Qdrant API: http://localhost:6333"
echo ""
echo "To stop databases: docker-compose down"
echo "To view logs: docker-compose logs -f"
echo ""
