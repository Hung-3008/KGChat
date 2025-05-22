import os 
import json
from dotenv import load_dotenv
from backend.llm.gemini_client import GeminiClient
from backend.db.neo4j_client import Neo4jClient
from backend.db.vector_db import VectorDBClient
from backend.llm.ollama_client import OllamaClient
from qdrant_client import QdrantClient
from backend.core.retrieval.query_analyzer import analyze_query
from backend.core.retrieval.keyword_extractor import extract_keywords
from backend.core.retrieval.dual_level_retriever import retrieve_from_knowledge_graph, format_retrieval_results, evaluate_and_expand_entities, format_triplets_for_evaluation


load_dotenv()

def initialize_clients():
    """Initialize and return all required clients."""
    load_dotenv()
    
    gemini_api_key = os.getenv("GEMINI_API_KEY_5")
    gemini = GeminiClient(api_key=gemini_api_key, model_name="gemini-2.0-flash")
    
    neo4j_uri = os.getenv('NEO4J_URI')
    neo4j_username = os.getenv('NEO4J_USERNAME')
    neo4j_password = os.getenv('NEO4J_PASSWORD')
    # neo4j_uri = "bolt://localhost:7687"
    # neo4j_username = "neo4j"
    # neo4j_password = "12345678"
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
    
    neo4j_client = Neo4jClient(
        uri=neo4j_uri,
        username=neo4j_username,
        password=neo4j_password
    )
    vector_db_client = VectorDBClient(host=qdrant_host)
    ollama_client = OllamaClient(embedding_model='mxbai-embed-large')
    qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
    
    return {
        "gemini": gemini,
        "neo4j_client": neo4j_client,
        "vector_db_client": vector_db_client,
        "ollama_client": ollama_client,
        "qdrant_client": qdrant_client
    }

async def run_query(query: str, grounding=False):
    