from backend.core.llm.gemini_client import GeminiClient
from backend.utils.logging import get_logger
from backend.core.retrieval.dual_level_retriever import (
    retrieve_from_knowledge_graph,
    evaluate_and_expand_entities,
    format_retrieval_results
)
from backend.core.llm.ollama_client import OllamaClient
from backend.db.vector_db import QdrantClient
from backend.db.neo4j_client import Neo4jClient
import asyncio
import os
import sys
import logging
from dotenv import load_dotenv

# Add root path to sys.path to import modules
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..')))

# Load environment variables from .env file
load_dotenv()


# Configure logging
logger = get_logger(__name__)
logging.basicConfig(level=logging.INFO)


async def test_dual_level_retriever():
    """Test the functionality of dual_level_retriever.py"""

    # 1. Initialize required clients
    try:
        # Neo4j client
        neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        neo4j_user = os.getenv("NEO4J_USERNAME", "neo4j")
        neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
        neo4j_client = Neo4jClient(
            uri=neo4j_uri, username=neo4j_user, password=neo4j_password)

        # Qdrant client
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
        qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)

        # Ollama client
        ollama_base_url = os.getenv(
            "OLLAMA_BASE_URL", "http://localhost:11434")
        ollama_embedding_model = os.getenv(
            "OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
        ollama_client = OllamaClient(
            base_url=ollama_base_url, embedding_model=ollama_embedding_model)

        # Gemini client
        gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        gemini_client = GeminiClient(api_key=gemini_api_key)

        logger.info("All clients initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing clients: {str(e)}")
        return

    # 2. Prepare test data
    test_query = "What are the symptoms of diabetes and how is HbA1c used to diagnose it?"
    test_intent = "DIABETES_RELATED"
    high_level_keywords = ["Diabetes symptoms", "HbA1c", "Diabetes diagnosis"]
    low_level_keywords = ["Blood sugar", "Thirst",
                          "Urination", "Fatigue", "Glycated hemoglobin"]

    # 3. Test retrieve_from_knowledge_graph
    logger.info("Testing retrieve_from_knowledge_graph...")

    try:
        kg_context = await retrieve_from_knowledge_graph(
            query=test_query,
            intent=test_intent,
            high_level_keywords=high_level_keywords,
            low_level_keywords=low_level_keywords,
            neo4j_client=neo4j_client,
            ollama_client=ollama_client,
            qdrant_client=qdrant_client,
            gemini_client=gemini_client,
            top_k=5,
            max_distance=0.8,
            similarity_threshold=0.7,
            expansion_width=10,
            expansion_depth=10
        )

        level1_nodes = kg_context.get("level1_nodes", [])
        level2_nodes = kg_context.get("level2_nodes", [])
        relationships = kg_context.get("relationships", [])

        logger.info(f"Retrieved {len(level1_nodes)} Level 1 nodes")
        logger.info(f"Retrieved {len(level2_nodes)} Level 2 nodes")
        logger.info(f"Retrieved {len(relationships)} relationships")

        # 4. Test evaluate_and_expand_entities
        logger.info("Testing evaluate_and_expand_entities...")

        expanded_level1, expanded_level2 = await evaluate_and_expand_entities(
            query=test_query,
            level1_nodes=level1_nodes,
            level2_nodes=level2_nodes,
            relationships=relationships,
            neo4j_client=neo4j_client,
            ollama_client=ollama_client,
            qdrant_client=qdrant_client,
            gemini_client=gemini_client,
            min_level1_nodes=3,
            min_level2_nodes=5,
            max_iterations=2,
            similarity_threshold=0.7,
            expansion_width=10,
            expansion_depth=10
        )

        logger.info(f"After expansion: {len(expanded_level1)} Level 1 nodes")
        logger.info(f"After expansion: {len(expanded_level2)} Level 2 nodes")

        # 5. Test format_triplets_for_evaluation and format_retrieval_results
        logger.info("Testing formatters...")

        formatted_triplets = format_triplets_for_evaluation(
            expanded_level1,
            expanded_level2,
            relationships
        )

        logger.info(f"Formatted triplets length: {len(formatted_triplets)}")

        formatted_results = format_retrieval_results(
            expanded_level1,
            expanded_level2,
            relationships
        )

        logger.info(f"Formatted results length: {len(formatted_results)}")

        # 6. Display sample data
        if level1_nodes:
            logger.info(f"Sample Level 1 node: {level1_nodes[0]}")

        if level2_nodes:
            logger.info(f"Sample Level 2 node: {level2_nodes[0]}")

        if relationships:
            logger.info(f"Sample relationship: {relationships[0]}")

        logger.info("Test completed successfully")

    except Exception as e:
        logger.error(f"Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_dual_level_retriever())
