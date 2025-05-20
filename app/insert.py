import os 
import asyncio
from dotenv import load_dotenv
from backend.core.pipeline.gemini.construct_level_1_graph import create_level1_graph
from backend.core.pipeline.gemini.construct_level_2_graph import create_level2_graph_from_db
from backend.db.neo4j_client import Neo4jClient
from backend.db.vector_db import VectorDBClient
from backend.llm.ollama_client import OllamaClient
from backend.core.pipeline.cross_level_rela import create_cross_level_relationships
load_dotenv()

print(os.getenv('NEO4J_URI'))

embedding = os.getenv('OLLAMA_EMBEDDING_MODEL')
neo4j_uri = os.getenv('NEO4J_URI')
neo4j_username = os.getenv('NEO4J_USERNAME')
neo4j_password = os.getenv('NEO4J_PASSWORD')
gemini_api_key = os.getenv('GEMINI_API_KEY_1')

qdrant_host = os.getenv('QDRANT_HOST')


neo4j_client = Neo4jClient(
    uri=neo4j_uri,
    username=neo4j_username,
    password=neo4j_password
)
vector_db_client = VectorDBClient(
    host=qdrant_host,
)

ollama_client = OllamaClient()

input_dir = '/home/hung/Documents/hung/code/KG_Hung/KGChat/data/level_1'
db_path = '/home/hung/Documents/hung/code/KG_Hung/KGChat/data/level_2/umls.db'

async def main(level=1, clear_existing=True):

    if level == 1:
        # Create the level 1 graph
        await create_level1_graph(
            input_directory=input_dir, 
            neo4j_uri=neo4j_uri, 
            neo4j_username=neo4j_username, 
            neo4j_password=neo4j_password, 
            clear_existing=clear_existing,
            embedding_model=embedding,
            save_batch_size=15,
            qdrant_host=qdrant_host,
            gemini_api_key=gemini_api_key,
        )

    elif level == 2:
        # Create the level 2 graph
        await create_level2_graph_from_db(
            db_path=db_path,
            neo4j_uri=neo4j_uri,
            neo4j_username=neo4j_username,
            neo4j_password=neo4j_password,
            clear_existing=clear_existing,
            qdrant_host=qdrant_host,
        )
    else:
        await create_cross_level_relationships(
            neo4j_uri=neo4j_uri,
            neo4j_username=neo4j_username,
            neo4j_password=neo4j_password,
            qdrant_host=qdrant_host,
            qdrant_port=6333,
            similarity_threshold=0.7,
            max_references_per_node=15,
        )

    

if __name__ == "__main__":
    asyncio.run(main(level=3, clear_existing=True))