# MolRAG Setup Guide

Complete guide to set up MolRAG with full database infrastructure.

## Prerequisites

- **Docker & Docker Compose** (for databases)
- **Python 3.9+**
- **API Key** (OpenAI, Anthropic, or OpenRouter)
- **8GB RAM minimum** (16GB recommended)

## Quick Start Options

### Option A: Quick Demo (5 Minutes)
For testing with sample data

### Option B: Real Data Setup (20 Minutes)
For production use with 4M+ real relationships

---

## Quick Start (Demo Mode)

### 1. Get API Keys

Choose one of these providers:
- **OpenAI**: https://platform.openai.com/api-keys
- **Anthropic**: https://console.anthropic.com/
- **OpenRouter**: https://openrouter.ai/keys

### 2. Configure Environment

Edit the `.env` file and add your API key:

```bash
# Add at least one API key
OPENAI_API_KEY=sk-...           # Recommended: GPT-4
ANTHROPIC_API_KEY=sk-ant-...    # Or Claude
OPENROUTER_API_KEY=sk-or-...    # Alternative
```

The `.env` file is already configured with database credentials that match the Docker setup.

### 3. Run Quick Start Script

```bash
# Make script executable (if not already)
chmod +x scripts/quick_start.sh

# Run the setup
./scripts/quick_start.sh
```

**You'll be prompted to choose:**
- **Option 1**: Quick demo with 10 sample molecules (<1 minute)
- **Option 2**: Real PrimeKG data with 4M+ relationships (~15 minutes)

This will:
- ✅ Start all required databases (Neo4j, Qdrant, Redis, PostgreSQL)
- ✅ Initialize database schemas and indexes
- ✅ Download and load molecular data (sample or real)
- ✅ Verify all connections

### 4. Launch the UI

```bash
python app.py
```

Open http://localhost:7860 in your browser.

## What Just Happened?

The setup created:

### 🗄️ Databases
- **Neo4j** (port 7474/7687): Knowledge graph with molecular relationships
- **Qdrant** (port 6333): Vector database with molecular fingerprints
- **Redis** (port 6379): Caching layer for fast queries
- **PostgreSQL** (port 5432): Metadata storage

### 🧬 Data Options
**Sample Data** (Quick demo):
- 10 molecules with known properties
- Ibuprofen, Aspirin, Caffeine, Ethanol, etc.
- Good for testing basic functionality

**Real Data** (Production):
- **PrimeKG**: 130,000+ nodes, 4,000,000+ relationships
- **Includes**: Drugs, proteins, diseases, pathways, biological processes
- **Sources**: Integrated from multiple databases (DrugBank, MONDO, GO, Reactome)
- **Use cases**: Real molecular predictions, drug discovery, biomedical research

## Verify Installation

### Test 1: Database Connectivity
```bash
# Test all databases
python -c "
from src.utils import Config
from src.utils.connectors import Neo4jConnector, QdrantConnector, RedisConnector

config = Config()

# Test Neo4j
neo4j = Neo4jConnector(config.neo4j_uri, config.neo4j_user, config.neo4j_password)
print('✅ Neo4j connected')

# Test Qdrant
qdrant = QdrantConnector(config.qdrant_url, None, 'molecular_fingerprints', 2048)
print('✅ Qdrant connected')

# Test Redis
redis = RedisConnector(config.redis_host, config.redis_port, config.redis_password)
print('✅ Redis connected')

print('\\n✅ All databases connected successfully!')
"
```

### Test 2: Sample Query
```bash
# Query sample molecule
python -c "
from src.utils import Config, Neo4jConnector

config = Config()
neo4j = Neo4jConnector(config.neo4j_uri, config.neo4j_user, config.neo4j_password)

result = neo4j.execute_query('MATCH (m:Molecule) RETURN count(m) as count')
print(f'Molecules in database: {result[0][\"count\"]}')
"
```

Expected output: `Molecules in database: 10`

### Test 3: UI Test
1. Open http://localhost:7860
2. Go to "🎯 Property Prediction" tab
3. Enter SMILES: `CCO`
4. Query: `Is this molecule toxic?`
5. Click "🚀 Predict Property"

**Expected**: Should return a real prediction (not "Demo Mode")

## Manual Setup (Alternative)

If you prefer manual setup:

### 1. Start Databases
```bash
docker-compose up -d
```

### 2. Wait for Services
```bash
# Check status
docker-compose ps

# All services should show "healthy"
```

### 3. Initialize Databases
```bash
python scripts/setup_databases.py
```

### 4. Load Sample Data
```bash
python scripts/load_sample_data.py
```

### 5. Start UI
```bash
python app.py
```

## Real Data Setup (Production)

For production use with real molecular databases instead of samples:

### Quick Method

```bash
chmod +x scripts/setup_real_data.sh
./scripts/setup_real_data.sh
```

This will:
- Download PrimeKG (~500MB, 15 minutes)
- Load 130,000+ nodes and 4M+ relationships
- Index all data in Neo4j and Qdrant
- Verify the installation

### Manual Method

#### 1. Download PrimeKG

```bash
python scripts/download_real_data.py --dataset primekg
```

This downloads the full PrimeKG knowledge graph:
- **Size**: ~500MB compressed
- **Nodes**: 130,000+ (drugs, proteins, diseases, pathways)
- **Relationships**: 4,000,000+
- **Source**: Harvard Medical School Dataverse
- **Time**: ~10-15 minutes depending on connection

#### 2. Load into Databases

```bash
python scripts/load_knowledge_graphs.py
```

This processes and loads:
- Neo4j: All nodes and relationships
- Qdrant: Molecular fingerprints for vector search
- Creates indexes for fast queries
- **Time**: ~5-10 minutes

#### 3. Verify

```bash
python -c "
from src.utils import Config, Neo4jConnector
config = Config()
neo4j = Neo4jConnector(config.neo4j_uri, config.neo4j_user, config.neo4j_password)
result = neo4j.execute_query('MATCH (n) RETURN count(n) as count')
print(f'Nodes: {result[0][\"count\"]:,}')
result = neo4j.execute_query('MATCH ()-[r]->() RETURN count(r) as count')
print(f'Relationships: {result[0][\"count\"]:,}')
"
```

Expected output:
```
Nodes: 130,000+
Relationships: 4,000,000+
```

### What's in PrimeKG?

**Node Types:**
- **Drugs** (~10,000): FDA-approved and experimental
- **Proteins** (~20,000): Human proteins and targets
- **Diseases** (~17,000): Disease ontology from MONDO
- **Pathways** (~3,000): Reactome biological pathways
- **Biological Processes** (~50,000): Gene Ontology terms
- **Molecular Functions** (~10,000)
- **Cellular Components** (~4,000)
- **Anatomical Entities** (~13,000)
- **Phenotypes** (~13,000)

**Relationship Types:**
- Drug-Protein interactions (targets, inhibits, activates)
- Protein-Protein interactions
- Disease-Protein associations
- Drug-Disease indications and contraindications
- Protein-Pathway memberships
- Disease-Phenotype associations
- And 30+ more relationship types

### Additional Datasets

#### ChEMBL (Optional)

For bioactivity data:

```bash
python scripts/download_real_data.py --dataset chembl
```

**Note**: This downloads a sample (1,000 molecules via API). For the full ChEMBL database:
1. Visit: https://www.ebi.ac.uk/chembl/
2. Download: Full database (SQLite or CSV)
3. Place in: `data/raw/chembl/`

#### DrugBank (Optional, Requires Account)

For detailed drug information:

1. Register at: https://go.drugbank.com/ (free for academics)
2. Download: "All drugs" CSV file
3. Place in: `data/raw/drugbank/drugbank.csv`
4. Load: `python scripts/load_drugbank.py`

## Database Management

### View Logs
```bash
docker-compose logs -f [service]

# Examples:
docker-compose logs -f neo4j
docker-compose logs -f qdrant
docker-compose logs -f redis
```

### Stop Databases
```bash
docker-compose down
```

### Stop and Remove Data
```bash
docker-compose down -v  # WARNING: Deletes all data
```

### Restart Databases
```bash
docker-compose restart
```

## Access Database UIs

### Neo4j Browser
- URL: http://localhost:7474
- Username: `neo4j`
- Password: `molrag_password_2024`

Browse the knowledge graph visually.

### Qdrant Dashboard
- URL: http://localhost:6333/dashboard

View vector collections and search.

### Redis CLI
```bash
docker-compose exec redis redis-cli -a molrag_redis_2024
```

## Loading Additional Data

### Load Full Knowledge Graphs
```bash
# Load PrimeKG (biomedical knowledge)
python scripts/load_knowledge_graphs.py --kg primekg

# Load DrugBank
python scripts/load_knowledge_graphs.py --kg drugbank

# Load ChEMBL
python scripts/load_knowledge_graphs.py --kg chembl
```

### Index Custom Molecules
```bash
python scripts/index_molecules.py --input your_molecules.csv
```

## Troubleshooting

### "Demo Mode" Still Showing

**Check 1**: API key configured?
```bash
grep "API_KEY=" .env
```

**Check 2**: Databases running?
```bash
docker-compose ps
```

**Check 3**: Data loaded?
```bash
python -c "
from src.utils import Config, Neo4jConnector
config = Config()
neo4j = Neo4jConnector(config.neo4j_uri, config.neo4j_user, config.neo4j_password)
result = neo4j.execute_query('MATCH (m:Molecule) RETURN count(m) as count')
print(f'Molecules: {result[0][\"count\"]}')
"
```

### Connection Errors

**Neo4j**:
```bash
# Check if running
docker-compose exec neo4j neo4j status

# View logs
docker-compose logs neo4j
```

**Qdrant**:
```bash
# Test connection
curl http://localhost:6333/healthz

# View logs
docker-compose logs qdrant
```

**Redis**:
```bash
# Test connection
docker-compose exec redis redis-cli -a molrag_redis_2024 ping

# Should return: PONG
```

### Port Conflicts

If ports are already in use, edit `docker-compose.yml`:

```yaml
services:
  neo4j:
    ports:
      - "17474:7474"  # Change host port
      - "17687:7687"
```

Then update `.env`:
```bash
NEO4J_URI=bolt://localhost:17687
```

### Memory Issues

If databases crash, increase Docker memory:
- Docker Desktop: Settings → Resources → Memory (8GB+)

Or reduce in `docker-compose.yml`:
```yaml
neo4j:
  environment:
    - NEO4J_dbms_memory_heap_max__size=1G  # Reduce from 2G
```

## Production Deployment

For production use:

### 1. Secure Passwords
```bash
# Generate strong passwords
openssl rand -base64 32
```

Update in both `docker-compose.yml` and `.env`

### 2. Enable SSL/TLS
Configure Neo4j with certificates:
```yaml
neo4j:
  environment:
    - NEO4J_dbms_ssl_policy_bolt_enabled=true
```

### 3. Use Docker Secrets
Instead of plain environment variables:
```yaml
secrets:
  neo4j_password:
    file: ./secrets/neo4j_password.txt
```

### 4. External Databases
For production, use managed services:
- Neo4j AuraDB
- Qdrant Cloud
- Redis Cloud

Update `.env` with external endpoints.

## Performance Tuning

### Neo4j Optimization
```yaml
environment:
  - NEO4J_dbms_memory_heap_max__size=4G
  - NEO4J_dbms_memory_pagecache_size=2G
```

### Qdrant Optimization
```yaml
environment:
  - QDRANT__SERVICE__MAX_REQUEST_SIZE_MB=10
```

### Redis Optimization
```bash
command: redis-server --maxmemory 2gb --maxmemory-policy allkeys-lru
```

## Next Steps

1. **Explore the UI**: http://localhost:7860
2. **Load More Data**: Use `scripts/load_knowledge_graphs.py`
3. **Customize**: Adjust CoT strategies in UI
4. **Evaluate**: Run benchmarks with `scripts/evaluate.py`
5. **API**: Build on top of MolRAG Python API

## Support

- **Documentation**: See `README.md`
- **Issues**: https://github.com/yourusername/MOLRAG_KB/issues
- **Paper**: `docs/papers/krotkov_et_al_2025_JCIM.pdf`

---

**Now go test some molecules! 🧬**
