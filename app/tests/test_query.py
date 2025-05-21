from backend.llm.gemini_client import GeminiClient
from backend.utils.logging import get_logger
from backend.core.retrieval.query_analyzer import analyze_query, QueryIntent
from backend.core.retrieval.keyword_extractor import extract_keywords
from backend.core.retrieval.kg_query_processor import process_kg_query
from backend.llm.ollama_client import OllamaClient
from backend.db.vector_db import VectorDBClient
from backend.db.neo4j_client import Neo4jClient
from qdrant_client import QdrantClient
from app.query import process_query, analyze_user_query

import asyncio
import os
import sys
import time
import logging
from dotenv import load_dotenv
import datetime
import json
import unittest.mock as mock
from typing import Optional

# Add root path to sys.path to import modules
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../')))

# Load environment variables from .env file
load_dotenv()

# Configure logging
logger = get_logger(__name__)
logging.basicConfig(level=logging.INFO)

# Tạo thư mục để lưu kết quả nếu chưa tồn tại
TEST_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "test_results")
os.makedirs(TEST_RESULTS_DIR, exist_ok=True)


class GeminiKeyManager:
    def __init__(
        self,
        current_key_index: int = 1,
        max_keys: int = 6,
        key_pattern: str = "GEMINI_API_KEY_{}"
    ):
        self.current_key_index = current_key_index
        self.max_keys = max_keys
        self.key_pattern = key_pattern
        self.logger = logging.getLogger(__name__)

    def get_current_key(self) -> Optional[str]:
        key_name = self.key_pattern.format(self.current_key_index)
        api_key = os.getenv(key_name)

        if not api_key:
            self.logger.warning(
                f"API key {key_name} not found in environment variables")
            return None

        return api_key

    def rotate_key(self) -> Optional[str]:
        for _ in range(self.max_keys):
            self.current_key_index = (
                self.current_key_index % self.max_keys) + 1

            key_name = self.key_pattern.format(self.current_key_index)
            api_key = os.getenv(key_name)

            if api_key:
                self.logger.info(f"Rotated to API key {key_name}")
                return api_key

        self.logger.error("No valid API keys found after trying all options")
        return None


# Sửa hàm mock_initialize_clients để sử dụng key manager
key_manager = GeminiKeyManager(current_key_index=1, max_keys=6)


async def mock_initialize_clients():
    # Lấy API key Gemini từ key manager
    gemini_api_key = key_manager.get_current_key()
    if not gemini_api_key:
        # Thử rotation nếu key hiện tại không hợp lệ
        gemini_api_key = key_manager.rotate_key()
        if not gemini_api_key:
            logger.warning("No valid Gemini API keys found, using empty key")
            gemini_api_key = ""

    gemini_client = GeminiClient(api_key=gemini_api_key)

    # Các client khác giữ nguyên
    neo4j_uri = "bolt://localhost:7687"
    neo4j_user = "neo4j"
    neo4j_password = "12345678"

    neo4j_client = Neo4jClient(
        uri=neo4j_uri, username=neo4j_user, password=neo4j_password)

    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))

    vector_db_client = VectorDBClient(host=qdrant_host)
    ollama_client = OllamaClient(embedding_model='mxbai-embed-large')
    qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)

    return {
        "gemini": gemini_client,
        "neo4j_client": neo4j_client,
        "vector_db_client": vector_db_client,
        "ollama_client": ollama_client,
        "qdrant_client": qdrant_client
    }

# Thêm hàm retry với xoay vòng API key


async def retry_with_key_rotation(func, max_retries=3, *args, **kwargs):
    for attempt in range(max_retries):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            error_str = str(e).lower()
            logger.error(
                f"Error in attempt {attempt+1}/{max_retries}: {error_str}")

            # Kiểm tra lỗi rate limit hoặc quota
            if "resource_exhausted" in error_str or "rate limit" in error_str or "429" in error_str or "quota" in error_str:
                # Xoay vòng API key
                new_key = key_manager.rotate_key()
                if new_key:
                    logger.info(
                        f"Rotated to new Gemini API key (index: {key_manager.current_key_index})")
                    # Cập nhật API key cho clients
                    os.environ["GEMINI_API_KEY"] = new_key
                    # Đợi một khoảng thời gian ngắn trước khi thử lại
                    await asyncio.sleep(2)
                else:
                    logger.error("No more valid Gemini API keys available")
                    if attempt == max_retries - 1:
                        raise
            else:
                # Nếu không phải lỗi rate limit, không cần thử lại
                raise

    raise Exception(
        f"Failed after {max_retries} attempts with all available API keys")


def save_test_result(test_name, query, result, metadata=None, kg_context=None, final_prompt=None):
    """
    Lưu kết quả test vào file JSON.

    Args:
        test_name: Tên của test
        query: Câu truy vấn
        result: Kết quả phản hồi
        metadata: Thông tin bổ sung (từ khóa, intent, thời gian xử lý...)
        kg_context: Bối cảnh đồ thị tri thức đã truy xuất
        final_prompt: Final prompt được sử dụng để tạo câu trả lời
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{test_name}_{timestamp}.json"
    filepath = os.path.join(TEST_RESULTS_DIR, filename)

    if metadata is None:
        metadata = {}

    data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "test_name": test_name,
        "query": query,
        "result": result,
        "metadata": metadata
    }

    # Thêm thông tin về đồ thị tri thức và final prompt nếu có
    if kg_context:
        data["knowledge_graph_context"] = kg_context

    if final_prompt:
        data["final_prompt"] = final_prompt

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(f"Test result saved to {filepath}")
    return filepath


async def test_query_processing():
    """Test the functionality of query.py and related modules"""

    # Thiết lập biến môi trường cho process_query
    os.environ["NEO4J_URI"] = "neo4j+s://a1595a38.databases.neo4j.io"
    os.environ["NEO4J_USERNAME"] = "neo4j"
    os.environ["NEO4J_PASSWORD"] = "G2U-3Nl2BCUe7HLnlPYiO7OW77-9WlNAm9G61L7mviE"

    # 1. Initialize required clients
    try:
        # Neo4j client
        neo4j_uri = "neo4j+s://a1595a38.databases.neo4j.io"
        neo4j_user = "neo4j"
        neo4j_password = "G2U-3Nl2BCUe7HLnlPYiO7OW77-9WlNAm9G61L7mviE"
        neo4j_client = Neo4jClient(
            uri=neo4j_uri, username=neo4j_user, password=neo4j_password)

        # Qdrant client
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
        qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        vector_db_client = VectorDBClient(host=qdrant_host)

        # Ollama client
        ollama_base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        ollama_embedding_model = os.getenv(
            "OLLAMA_EMBEDDING_MODEL", "mxbai-embed-large")
        ollama_client = OllamaClient(
            host=ollama_base_url, embedding_model=ollama_embedding_model)

        # Gemini client
        gemini_api_key = os.getenv("GEMINI_API_KEY_5", "")
        gemini_client = GeminiClient(
            api_key=gemini_api_key)

        # Prepare client dictionary
        clients = {
            "gemini": gemini_client,
            "neo4j_client": neo4j_client,
            "vector_db_client": vector_db_client,
            "ollama_client": ollama_client,
            "qdrant_client": qdrant_client
        }

        logger.info("All clients initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing clients: {str(e)}")
        return

    # Kiểm tra cấu trúc đồ thị Neo4j
    logger.info("\n========== Checking Neo4j Graph Structure ==========")
    try:
        # Kiểm tra có bao nhiêu node Level 1
        level1_count = await neo4j_client.execute_query("MATCH (n:Level1) RETURN count(n) as count")
        logger.info(
            f"Level 1 nodes count: {level1_count[0]['count'] if level1_count else 0}")

        # Kiểm tra có bao nhiêu node Level 2
        level2_count = await neo4j_client.execute_query("MATCH (n:Level2) RETURN count(n) as count")
        logger.info(
            f"Level 2 nodes count: {level2_count[0]['count'] if level2_count else 0}")

        # Kiểm tra có bao nhiêu mối quan hệ REFERENCES
        rels_count = await neo4j_client.execute_query("MATCH (:Level1)-[r:REFERENCES]->(:Level2) RETURN count(r) as count")
        logger.info(
            f"REFERENCES relationships count: {rels_count[0]['count'] if rels_count else 0}")

        # Lấy 5 mẫu node Level 1 để kiểm tra
        sample_level1 = await neo4j_client.execute_query("MATCH (n:Level1) RETURN n.entity_id, n.name LIMIT 5")
        logger.info(f"Sample Level 1 nodes: {sample_level1}")
    except Exception as e:
        logger.error(f"Error checking Neo4j structure: {str(e)}")

    # 2. Chuẩn bị dữ liệu test
    test_query = "What are the symptoms of type 2 diabetes?"
    conversation_history = [
        {
            "role": "user",
            "content": "Hello, I'd like to ask some questions about diabetes."
        }
    ]

    # 3. Phân tích câu truy vấn
    logger.info("\n========== Analyzing Query ==========")
    try:
        start_time = time.time()

        intent, high_keywords, low_keywords = await analyze_user_query(
            query=test_query,
            history=conversation_history,
            gemini_client=clients["gemini"]
        )

        elapsed_time = time.time() - start_time

        logger.info(f"Query: '{test_query}'")
        logger.info(f"Detected intent: {intent.name}")
        logger.info(f"High-level keywords: {high_keywords}")
        logger.info(f"Low-level keywords: {low_keywords}")
        logger.info(f"Analysis time: {elapsed_time:.2f} seconds\n")
    except Exception as e:
        logger.error(f"Error analyzing query: {str(e)}")
        import traceback
        traceback.print_exc()

    # 4. Thử nghiệm với process_kg_query
    logger.info("\n========== Testing Process KG Query ==========")

    # Kiểm tra chi tiết quá trình truy xuất đồ thị kiến thức
    try:
        from backend.core.retrieval.dual_level_retriever import retrieve_from_knowledge_graph, format_retrieval_results

        logger.info(
            f"Retrieving knowledge graph data for query: '{test_query}'")
        kg_context = await retrieve_from_knowledge_graph(
            query=test_query,
            intent=str(intent),
            high_level_keywords=high_keywords,
            low_level_keywords=low_keywords,
            neo4j_client=clients["neo4j_client"],
            ollama_client=clients["ollama_client"],
            qdrant_client=clients["qdrant_client"],
            gemini_client=clients["gemini"],
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

        # Lưu kết quả chi tiết
        formatted_kg_context = format_retrieval_results(
            level1_nodes, level2_nodes, relationships)

        save_test_result(
            test_name="knowledge_graph_retrieval",
            query=test_query,
            result={
                "level1_count": len(level1_nodes),
                "level2_count": len(level2_nodes),
                "relationship_count": len(relationships)
            },
            metadata={
                "high_level_keywords": high_keywords,
                "low_level_keywords": low_keywords
            },
            kg_context=formatted_kg_context
        )

    except Exception as e:
        logger.error(f"Error retrieving from knowledge graph: {str(e)}")
        import traceback
        traceback.print_exc()

    # 5. Xử lý truy vấn cuối cùng
    logger.info("\n========== Testing Final Query Processing ==========")

    with mock.patch('app.query.initialize_clients', mock_initialize_clients):
        try:
            logger.info(f"Processing query: '{test_query}'")
            start_time = time.time()

            result = await process_query(test_query, conversation_history)

            elapsed_time = time.time() - start_time

            logger.info(f"Process time: {elapsed_time:.2f} seconds")
            logger.info(
                f"Response: {result.get('response', 'No response generated')[:200]}...")

            # Lưu kết quả
            save_test_result(
                test_name="final_process_query",
                query=test_query,
                result=result,
                metadata={
                    "processing_time": elapsed_time
                }
            )

        except Exception as e:
            logger.error(f"Error in final query processing: {str(e)}")
            import traceback
            traceback.print_exc()

    logger.info("\nTest completed")


# Hàm mock để bắt và lưu final prompt
async def mock_process_kg_query(*args, **kwargs):
    """
    Phiên bản mock của process_kg_query để thu thập final prompt
    và context của đồ thị tri thức.
    """
    # Thực hiện gọi hàm gốc
    from backend.core.retrieval.kg_query_processor import process_kg_query

    # Trích xuất các tham số quan trọng
    query = kwargs.get("query", args[0] if args else None)
    intent = kwargs.get("intent", args[1] if len(args) > 1 else None)
    high_level_keywords = kwargs.get(
        "high_level_keywords", args[2] if len(args) > 2 else [])
    low_level_keywords = kwargs.get(
        "low_level_keywords", args[3] if len(args) > 3 else [])

    # Thực hiện truy vấn đồ thị và lấy kết quả - dùng hàm retrieve_from_knowledge_graph
    from backend.core.retrieval.dual_level_retriever import retrieve_from_knowledge_graph, format_retrieval_results

    kg_context = await retrieve_from_knowledge_graph(
        query=query,
        intent=str(intent),
        high_level_keywords=high_level_keywords,
        low_level_keywords=low_level_keywords,
        neo4j_client=kwargs.get("neo4j_client"),
        ollama_client=kwargs.get("ollama_client"),
        qdrant_client=kwargs.get("qdrant_client"),
        gemini_client=kwargs.get("gemini_client"),
        similarity_threshold=kwargs.get("similarity_threshold", 0.7),
        expansion_width=kwargs.get("expansion_width", 20),
        expansion_depth=kwargs.get("expansion_depth", 20)
    )

    # Format kết quả truy vấn đồ thị để hiển thị
    level1_nodes = kg_context.get("level1_nodes", [])
    level2_nodes = kg_context.get("level2_nodes", [])
    relationships = kg_context.get("relationships", [])
    formatted_kg_context = format_retrieval_results(
        level1_nodes, level2_nodes, relationships)

    # Tạo final prompt (tương tự như trong KnowledgeGraphQueryProcessor._generate_diabetes_response)
    conversation_history = kwargs.get("conversation_history", [])
    response_type = kwargs.get("response_type", "concise")
    grounding_context = kg_context.get("grounding_context", "")

    final_prompt = f"""You are a specialized diabetes information assistant with access to both a knowledge graph of diabetes concepts and up-to-date web information.

Goal: Generate a comprehensive, evidence-based response to the user's diabetes-related query using both the provided knowledge graph information and recent web search results.

User Query: {query}

Knowledge Graph Context:
{formatted_kg_context}

Recent Web Information:
{grounding_context}

Instructions:
- Synthesize and critically evaluate information from both the knowledge graph and recent web sources
- When information from different sources conflicts:
  * Compare the reliability and recency of each source
  * Weigh medical consensus over isolated findings
  * Explain the differences if they are significant
- Prioritize information from peer-reviewed medical literature and authoritative health organizations
- Clearly distinguish between established diabetes knowledge and emerging research
- If contradictions exist between the knowledge graph and recent information, acknowledge this and explain the current understanding
- When discussing treatments or management approaches, note the level of evidence supporting them
- Use clinical reasoning to connect information to the user's specific query
- If the available information is insufficient to fully answer the query, acknowledge these limitations
- Format your response for readability with appropriate structure based on the requested length:
  * concise: Clear, focused response in 3-5 sentences that prioritizes the most clinically relevant information
  * detailed: Comprehensive explanation with relevant details, comparing different sources and explaining nuances
- Always note that this is informational only and not medical advice

Conversation History:
{conversation_history[-3:] if conversation_history else ''}

Response Type: {response_type}

Generate a well-reasoned, evidence-based response that integrates both knowledge sources."""

    # Tiếp tục gọi hàm process_kg_query thực tế
    result = await process_kg_query(*args, **kwargs)

    # Thêm thông tin về đồ thị và prompt vào kết quả
    result["kg_context"] = formatted_kg_context
    result["final_prompt"] = final_prompt

    return result


if __name__ == "__main__":
    asyncio.run(test_query_processing())
