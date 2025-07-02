import os
from typing import Optional, Dict, Any
import logging
from dotenv import load_dotenv

from db.neo4j_client import Neo4jClient
from db.vector_db import VectorDBClient
from llm.ollama_client import OllamaClient
from llm.gemini_client import GeminiClient
from qdrant_client import QdrantClient

# Configure logger
logger = logging.getLogger(__name__)

class ClientManager:
    """
    Singleton class for managing connections to various services.
    Implements connection pooling to avoid creating new connections for each request.
    """
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """Get or create the singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        """Initialize client connections as None"""
        self.neo4j_client = None
        self.vector_db_client = None
        self.ollama_client = None
        self.gemini_client = None
        self.qdrant_client = None
        self._initialized = False
        self.current_key = 1
    
    async def initialize(self) -> bool:
        """Initialize all client connections"""
        if self._initialized:
            logger.info("ClientManager already initialized")
            return True
            
        load_dotenv()
        
        try:
            # Initialize Neo4j client
            neo4j_uri = os.getenv('NEO4J_URI')
            neo4j_username = os.getenv('NEO4J_USERNAME')
            neo4j_password = os.getenv('NEO4J_PASSWORD')
            
            self.neo4j_client = Neo4jClient(
                uri=neo4j_uri,
                username=neo4j_username,
                password=neo4j_password
            )
            
            # Check Neo4j connectivity
            if not await self.neo4j_client.verify_connectivity():
                logger.error("Failed to connect to Neo4j database")
                return False
                
            logger.info("Successfully connected to Neo4j database")
            
            # Initialize Vector DB client
            qdrant_host = os.getenv("QDRANT_HOST", "localhost")
            qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
            
            self.vector_db_client = VectorDBClient(
                host=qdrant_host,
                port=qdrant_port
            )
            
            # Create collections if they don't exist
            self.vector_db_client.create_collections()
            logger.info("Successfully initialized Vector DB client")
            
            # Initialize Qdrant client
            self.qdrant_client = QdrantClient(
                host=qdrant_host, 
                port=qdrant_port
            )
            logger.info("Successfully initialized Qdrant client")
            
            # Initialize Ollama client
            ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
            embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "mxbai-embed-large")
            
            self.ollama_client = OllamaClient(
                host=ollama_host,
                embedding_model=embedding_model
            )
            logger.info("Successfully initialized Ollama client")
            
            # Initialize Gemini client
            gemini_api_key = os.getenv(f"GEMINI_API_KEY_{self.current_key}")
            
            self.gemini_client = GeminiClient(
                api_key=gemini_api_key, 
                model_name="gemini-2.0-flash"
            )
            
            # Test API key
            if await self.gemini_client.test_api_key():
                logger.info("Successfully initialized Gemini client")
            else:
                logger.warning("Gemini API key may not be valid, but continuing with initialization")
            
            self._initialized = True
            logger.info("All clients successfully initialized")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing clients: {str(e)}")
            await self.shutdown()
            return False
    
    async def shutdown(self) -> None:
        """Close all client connections"""
        try:
            if self.neo4j_client:
                await self.neo4j_client.close()
                logger.info("Neo4j client closed")
                
            # Other clients may need specific cleanup logic
            
            self._initialized = False
            logger.info("All clients successfully shut down")
            
        except Exception as e:
            logger.error(f"Error shutting down clients: {str(e)}")
    
    def get_clients(self) -> Dict[str, Any]:
        """Get all clients as a dictionary"""
        if not self._initialized:
            logger.warning("Clients not initialized, returning None values")
            
        return {
            "neo4j_client": self.neo4j_client,
            "vector_db_client": self.vector_db_client,
            "ollama_client": self.ollama_client,
            "gemini_client": self.gemini_client,
            "qdrant_client": self.qdrant_client
        }

    def rotate_gemini_key(self):
        if self.current_key == 6:
            self.current_key = 1
        else:
            self.current_key += 1
        gemini_api_key = os.getenv(f"GEMINI_API_KEY_{self.current_key}")
        self.gemini_client = GeminiClient(
            api_key=gemini_api_key, 
            model_name="gemini-2.0-flash"
        )

        return self.gemini_client