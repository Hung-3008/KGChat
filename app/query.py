import os 
from dotenv import load_dotenv
from backend.llm.gemini_client import GeminiClient
from backend.core.retrieval.query_analyzer import analyze_query
from backend.core.retrieval.keyword_extractor import extract_keywords

from backend.db.neo4j_client import Neo4jClient
from backend.db.vector_db import VectorDBClient
from backend.llm.ollama_client import OllamaClient
from qdrant_client import QdrantClient
from backend.core.retrieval.kg_query_processor import process_kg_query
from typing import List, Dict, Tuple, Optional, Any


async def initialize_clients():
    """Initialize and return all required clients."""
    load_dotenv()
    
    gemini_api_key = os.getenv("GEMINI_API_KEY_5")
    gemini = GeminiClient(api_key=gemini_api_key, model_name="gemini-2.0-flash")
    
    # neo4j_uri = os.getenv('NEO4J_URI')
    # neo4j_username = os.getenv('NEO4J_USERNAME')
    # neo4j_password = os.getenv('NEO4J_PASSWORD')
    neo4j_uri = "bolt://localhost:7687"
    neo4j_username = "neo4j"
    neo4j_password = "12345678"
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


async def analyze_user_query(query: str, 
                           history: List[Dict[str, str]], 
                           gemini_client: GeminiClient) -> Tuple[Any, List[str], List[str]]:

    intent_task = analyze_query(query=query, conversation_history=history, client=gemini_client)
    keywords_task = extract_keywords(query=query, conversation_history=None, llm_client=gemini_client)
    
    intent = await intent_task
    high_keywords, low_keywords = await keywords_task
    
    return intent, high_keywords, low_keywords


async def process_query(query: str, history: List[Dict[str, str]]) -> Any:

    clients = await initialize_clients()

    intent, high_keywords, low_keywords = await analyze_user_query(
        query=query,
        history=history,
        gemini_client=clients["gemini"]
    )

    print(f"High-level keywords: {high_keywords}")
    print(f"Low-level keywords: {low_keywords}")

    result = await process_kg_query(
        query=query,
        intent=intent,
        high_level_keywords=high_keywords,
        low_level_keywords=low_keywords,
        conversation_history=history,
        neo4j_client=clients["neo4j_client"],
        ollama_client=clients["ollama_client"],
        gemini_client=clients["gemini"],
        qdrant_client=clients["qdrant_client"]
    )
    
    return result


# Example usage
conversation_history = [
    {
        "role": "user",
        "content": "Hello, I'd like to ask some questions about diabetes."
    }
]

current_query = """In what case we can not use HbA1c to predict Diabetes?"""