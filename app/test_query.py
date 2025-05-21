from backend.llm.gemini_client import GeminiClient
from backend.utils.logging import get_logger
from backend.core.retrieval.query_analyzer import analyze_query, QueryIntent
from backend.core.retrieval.keyword_extractor import extract_keywords
from backend.core.retrieval.kg_query_processor import process_kg_query
from backend.llm.ollama_client import OllamaClient
from backend.db.vector_db import VectorDBClient
from backend.db.neo4j_client import Neo4jClient
from qdrant_client import QdrantClient
from query import process_query, analyze_user_query

import asyncio
import os
import time
import logging
from dotenv import load_dotenv
import datetime
import json
from typing import Optional

load_dotenv('/home/hung/Documents/hung/code/KG_Hung/KGChat/.env')

logger = get_logger(__name__)
logging.basicConfig(level=logging.INFO)

TEST_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "test_results")
os.makedirs(TEST_RESULTS_DIR, exist_ok=True)

