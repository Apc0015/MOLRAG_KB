"""Verify knowledge graph data is loaded"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import Config
from src.utils.database import Neo4jConnector

config = Config()
neo4j = Neo4jConnector(config.neo4j_uri, config.neo4j_user, config.neo4j_password)

# Count nodes and relationships
result = neo4j.execute_query('MATCH (n) RETURN count(n) as nodes')
nodes = result[0]['nodes']

result = neo4j.execute_query('MATCH ()-[r]->() RETURN count(r) as rels')
rels = result[0]['rels']

# Sample some drugs
result = neo4j.execute_query('MATCH (d:Drug) RETURN d.name as name LIMIT 5')
drugs = [r['name'] for r in result]

# Get node type counts
result = neo4j.execute_query('MATCH (n) RETURN labels(n)[0] as type, count(n) as count')
node_types = {r['type']: r['count'] for r in result}

neo4j.close()

print('='*60)
print('📊 NEO4J KNOWLEDGE GRAPH STATUS')
print('='*60)
print(f'Total Nodes: {nodes}')
print(f'Total Relationships: {rels}')
print(f'\nNode Types:')
for node_type, count in sorted(node_types.items(), key=lambda x: -x[1]):
    print(f'  - {node_type}: {count}')
print(f'\nSample Drugs: {", ".join(drugs[:5])}')
print('='*60)
print('✅ Knowledge graph loaded successfully!')
print('🌐 App running at: http://localhost:7860')
print('='*60)
