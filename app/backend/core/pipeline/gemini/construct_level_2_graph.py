import sqlite3
import time
import os
import asyncio
import logging
import uuid
from typing import Dict, List, Any, Tuple, Set, Optional

from backend.llm.ollama_client import OllamaClient
from backend.db.neo4j_client import Neo4jClient
from backend.db.vector_db import VectorDBClient  # Import VectorDBClient
from backend.utils.logging import get_logger


# Initialize logger
logger = get_logger(__name__)

class Level2GraphBuilder:
    """
    Builds Level 2 of the knowledge graph (specific details) from a UMLS database 
    and saves directly to Neo4j.
    
    This class orchestrates the extraction of detailed medical concepts and relationships
    from a UMLS SQLite database and their direct storage in Neo4j.
    """
    
    def __init__(
        self,
        neo4j_client: Neo4jClient,
        ollama_client: OllamaClient,
        vector_db_client: Optional[VectorDBClient] = None,  # Add vector_db_client
        embedding_dim: int = 1024,
        batch_size: int = 100
    ):
        """
        Initialize the Level 2 graph builder.
        
        Args:
            neo4j_client: Client for Neo4j database operations
            ollama_client: Client for LLM and embedding operations
            vector_db_client: Client for vector database operations
            embedding_dim: Dimension of vector embeddings
            batch_size: Batch size for processing nodes
        """
        self.neo4j_client = neo4j_client
        self.ollama_client = ollama_client
        self.vector_db_client = vector_db_client  # Store vector_db_client
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        
        # Create a fallback embedding for error cases
        self.fallback_embedding = [0.0] * embedding_dim
        
        # Initialize vector database collections if client is provided
        if self.vector_db_client:
            self.vector_db_client.create_collections()
            logger.info("Vector database collections initialized")
    
    async def process_db_nodes(self, conn: sqlite3.Connection) -> Dict[Tuple[str, str], str]:
        """
        Process all nodes from the UMLS database into Neo4j Level2 nodes.
        
        Args:
            conn: SQLite database connection
            
        Returns:
            Mapping from (CUI, AUI) to node_id for relationship creation
        """
        # Get total count for progress reporting
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) 
            FROM mrconso m 
            LEFT JOIN mrdef d ON m.CUI = d.CUI AND m.AUI = d.AUI 
            WHERE m.LAT = 'ENG' AND d.DEF IS NOT NULL
        """)
        total_nodes = cursor.fetchone()[0]
        logger.info(f"Total nodes to process: {total_nodes}")
        
        cui_aui_to_id = {}  # Mapping from (CUI, AUI) to node_id
        processed = 0
        offset = 0
        
        while processed < total_nodes:
            # Fetch a batch of nodes
            cursor.execute("""
                SELECT m.CUI, m.AUI, m.STR, d.DEF 
                FROM mrconso m
                LEFT JOIN mrdef d ON m.CUI = d.CUI AND m.AUI = d.AUI
                WHERE m.LAT = 'ENG' AND d.DEF IS NOT NULL
                LIMIT ? OFFSET ?
            """, (self.batch_size, offset))
            
            rows = cursor.fetchall()
            if not rows:
                break
            
            # Prepare texts for batch embedding - using only STR values
            str_values = []
            batch_data = []
            
            for row in rows:
                cui, aui, str_value, definition = row
                
                if not all([cui, aui, str_value, definition]):
                    continue
                
                # Use only the STR value for embedding
                str_values.append(str_value)
                batch_data.append((cui, aui, str_value, definition))
            
            # Generate embeddings for each STR value
            try:
                batch_embeddings = await self.ollama_client.embed(str_values)
                logger.info(f"Generated {len(batch_embeddings)} embeddings for STR values")
            except Exception as e:
                logger.error(f"Error generating embeddings: {e}")
                # Use fallback embeddings if error occurs
                batch_embeddings = [self.fallback_embedding] * len(str_values)
            
            # Create nodes with their embeddings
            nodes_batch = []
            vector_db_nodes_batch = []  # Batch for vector database
            
            for i, (cui, aui, str_value, definition) in enumerate(batch_data):
                node_id = f"{cui}_{aui}"
                cui_aui_to_id[(cui, aui)] = node_id
                
                # Get embedding or use fallback
                try:
                    if i < len(batch_embeddings):
                        embedding = batch_embeddings[i]
                    else:
                        logger.warning(f"Missing embedding for node {i}, using fallback")
                        embedding = self.fallback_embedding
                except Exception as e:
                    logger.error(f"Error accessing embedding for node {i}: {e}")
                    embedding = self.fallback_embedding
                
                # Create node dictionary with its unique embedding based on STR
                node = {
                    "entity_id": node_id,
                    "name": str_value,  
                    "description": definition,
                    "vector_embedding": embedding,
                    "knowledge_level": 2,
                    "cui": cui,
                    "aui": aui
                }
                
                nodes_batch.append(node)
                
                # Create a copy for the vector database
                if self.vector_db_client:
                    vector_db_nodes_batch.append(node.copy())
                
                processed += 1
            
            # Import nodes batch directly to Neo4j
            if nodes_batch:
                success = await self.neo4j_client.import_nodes(nodes_batch, "Level2")
                if not success:
                    logger.error(f"Failed to import batch of {len(nodes_batch)} nodes to Neo4j")
                else:
                    logger.info(f"Imported batch of {len(nodes_batch)} Level2 nodes to Neo4j")
            
            # Store nodes in vector database
            if self.vector_db_client and vector_db_nodes_batch:
                try:
                    stored_count = self.vector_db_client.store_nodes_batch(
                        vector_db_nodes_batch, 
                        "level2_nodes"
                    )
                    logger.info(f"Stored {stored_count} Level2 nodes in vector database")
                except Exception as e:
                    logger.error(f"Error storing nodes in vector database: {str(e)}")
            
            offset += self.batch_size
            logger.info(f"Processed {processed}/{total_nodes} nodes")
        
        logger.info(f"Completed processing {processed} Level2 nodes")
        return cui_aui_to_id
    
    def _derive_entity_type(self, str_value: str, definition: str) -> str:
        """
        Derive an appropriate entity type from the string value and definition.
        
        Args:
            str_value: The string value of the concept
            definition: The definition of the concept
            
        Returns:
            Derived entity type
        """
        # A simple heuristic for determining the entity type based on common patterns
        str_lower = str_value.lower()
        def_lower = definition.lower()
        
        if "disease" in str_lower or "disease" in def_lower or "disorder" in str_lower or "disorder" in def_lower:
            return "DISEASE"
        elif "symptom" in str_lower or "symptom" in def_lower or "sign" in str_lower:
            return "SYMPTOM"
        elif "medicine" in str_lower or "drug" in str_lower or "medication" in def_lower:
            return "MEDICATION"
        elif "treatment" in str_lower or "therapy" in def_lower or "procedure" in str_lower:
            return "TREATMENT"
        elif "anatomy" in str_lower or "body" in str_lower or "organ" in str_lower:
            return "ANATOMY"
        elif "test" in str_lower or "diagnosis" in str_lower or "examination" in def_lower:
            return "DIAGNOSTIC_TEST"
        elif "biomarker" in str_lower or "biomarker" in def_lower:
            return "BIOMARKER"
        else:
            return "MEDICAL_CONCEPT"  # Default type
    
    async def process_db_relationships(
        self,
        conn: sqlite3.Connection,
        cui_aui_to_id: Dict[Tuple[str, str], str],
        batch_size: int = 5000
    ) -> int:
        """
        Process relationships from the UMLS database into Neo4j.
        
        Args:
            conn: SQLite database connection
            cui_aui_to_id: Mapping from (CUI, AUI) tuples to node IDs
            batch_size: Number of relationships to process in each batch
            
        Returns:
            Number of relationships processed
        """
        # Get total count for progress reporting
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COUNT(*)
            FROM mrrel r
            JOIN mrconso m1 ON r.CUI1 = m1.CUI AND r.AUI1 = m1.AUI
            JOIN mrconso m2 ON r.CUI2 = m2.CUI AND r.AUI2 = m2.AUI
            WHERE m1.LAT = 'ENG' AND m2.LAT = 'ENG'
        """)
        total_rels = cursor.fetchone()[0]
        logger.info(f"Total relationships to process: {total_rels}")
        
        processed = 0
        offset = 0
        
        while processed < total_rels:
            cursor.execute("""
                SELECT r.CUI1, r.AUI1, r.RELA, r.CUI2, r.AUI2
                FROM mrrel r
                JOIN mrconso m1 ON r.CUI1 = m1.CUI AND r.AUI1 = m1.AUI
                JOIN mrconso m2 ON r.CUI2 = m2.CUI AND r.AUI2 = m2.AUI
                WHERE m1.LAT = 'ENG' AND m2.LAT = 'ENG'
                LIMIT ? OFFSET ?
            """, (batch_size, offset))
            
            rows = cursor.fetchall()
            if not rows:
                break
            
            # Process relationships in this batch
            relationships_batch = []
            
            for row in rows:
                cui1, aui1, rela, cui2, aui2 = row
                
                # Skip if either node wasn't in our processed nodes
                if (cui1, aui1) not in cui_aui_to_id or (cui2, aui2) not in cui_aui_to_id:
                    continue
                
                # Use a default relationship type if none provided
                rela = rela if rela else "RELATED_TO"
                
                source_id = cui_aui_to_id[(cui1, aui1)]
                target_id = cui_aui_to_id[(cui2, aui2)]
                
                relationship = {
                    "relationship_id": f"rel_{cui1}_{aui1}_{cui2}_{aui2}",
                    "source_id": source_id,
                    "target_id": target_id,
                    "source_entity_id": source_id,
                    "target_entity_id": target_id,
                    "type": rela.upper(),
                    "description": f"{rela} relationship from {cui1} to {cui2}",
                    "strength": 0.8,  # Default strength
                    "keywords": [rela.lower()],
                    "vector_embedding": [],  # Empty embedding for now
                    "knowledge_level": 2  # Level 2 for specific details
                }
                
                relationships_batch.append(relationship)
                processed += 1
            
            # Import relationships batch directly to Neo4j
            if relationships_batch:
                success = await self.neo4j_client.import_relationships(
                    relationships_batch, "Level2", "Level2"
                )
                if not success:
                    logger.error(f"Failed to import batch of {len(relationships_batch)} relationships to Neo4j")
                else:
                    logger.info(f"Imported batch of {len(relationships_batch)} Level2 relationships to Neo4j")
            
            offset += batch_size
            logger.info(f"Processed {processed}/{total_rels} relationships")
        
        logger.info(f"Completed processing {processed} Level2 relationships")
        return processed
    
    async def build_graph_from_db(self, db_path: str) -> bool:
        """
        Build the Level 2 knowledge graph from a UMLS SQLite database.
        
        Args:
            db_path: Path to the UMLS SQLite database
            
        Returns:
            True if successful, False otherwise
        """
        try:
            start_time = time.time()
            
            # Verify Neo4j connectivity
            connected = await self.neo4j_client.verify_connectivity()
            if not connected:
                logger.error("Failed to connect to Neo4j, exiting")
                return False
            
            # Setup Neo4j schema
            schema_success = await self.neo4j_client.setup_schema()
            if not schema_success:
                logger.warning("Neo4j schema setup had issues, but continuing")
            
            # Connect to SQLite database
            logger.info(f"Connecting to SQLite database: {db_path}")
            conn = sqlite3.connect(db_path)
            conn.execute("PRAGMA temp_store = MEMORY")
            conn.execute("PRAGMA cache_size = 10000")
            
            try:
                # Process in two phases: nodes first, then relationships
                logger.info("Phase 1: Processing nodes...")
                cui_aui_to_id = await self.process_db_nodes(conn)
                
                logger.info("Phase 2: Processing relationships...")
                rel_count = await self.process_db_relationships(conn, cui_aui_to_id)
                
                # Log statistics
                elapsed_time = time.time() - start_time
                logger.info(f"Level 2 graph construction completed in {elapsed_time:.2f} seconds")
                logger.info(f"Created {len(cui_aui_to_id)} nodes and {rel_count} relationships")
                
                # Get statistics from Neo4j
                stats = await self.neo4j_client.get_graph_statistics()
                logger.info(f"Neo4j graph statistics: {stats}")
                
                return True
            finally:
                conn.close()
        
        except Exception as e:
            logger.error(f"Error building Level 2 graph: {str(e)}")
            return False


async def create_level2_graph_from_db(
    db_path: str,
    neo4j_uri: Optional[str] = None,
    neo4j_username: Optional[str] = None,
    neo4j_password: Optional[str] = None,
    ollama_host: str = "http://localhost:11434",
    embedding_model: str = "mxbai-embed-large",
    embedding_dim: int = 1024,
    clear_existing: bool = False,
    qdrant_host: Optional[str] = None,
    qdrant_port: int = 6333,
    qdrant_api_key: Optional[str] = None,
    qdrant_url: Optional[str] = None
) -> bool:
    """
    Create Level 2 of the knowledge graph from a UMLS SQLite database.
    
    This function is the main entry point for constructing Level 2 of the
    knowledge graph, handling client initialization and cleanup.
    
    Args:
        db_path: Path to the UMLS SQLite database
        neo4j_uri: Neo4j server URI
        neo4j_username: Neo4j username
        neo4j_password: Neo4j password
        ollama_host: Ollama server host
        embedding_model: Model to use for embeddings
        embedding_dim: Dimension of vector embeddings
        clear_existing: Whether to clear existing Level 2 nodes
        qdrant_host: Qdrant server host
        qdrant_port: Qdrant server port
        qdrant_api_key: API key for Qdrant Cloud
        qdrant_url: URL for Qdrant Cloud
        
    Returns:
        True if creation was successful, False otherwise
    """
    neo4j_client = Neo4jClient(
        uri=neo4j_uri,
        username=neo4j_username,
        password=neo4j_password
    )
    
    ollama_client = OllamaClient(
        host=ollama_host,
        embedding_model=embedding_model
    )
    
    vector_db_client = None
    if qdrant_host or qdrant_url:
        vector_db_client = VectorDBClient(
            host=qdrant_host,
            port=qdrant_port,
            api_key=qdrant_api_key,
            url=qdrant_url,
            vector_size=embedding_dim
        )
        logger.info("Initialized Vector Database client")
    
    try:
        # Verify Neo4j connectivity
        connected = await neo4j_client.verify_connectivity()
        if not connected:
            logger.error("Failed to connect to Neo4j, exiting")
            return False
        
        # Clear existing Level 2 nodes if requested
        if clear_existing:
            logger.info("Clearing existing Level 2 nodes and relationships")
            await neo4j_client.execute_query("MATCH (n:Level2) DETACH DELETE n")
            
            # Clear vector database collection if it exists
            if vector_db_client:
                try:
                    # Check if collection exists before attempting to delete
                    collections = vector_db_client.client.get_collections()
                    collection_names = [c.name for c in collections.collections]
                    
                    if "level2_nodes" in collection_names:
                        logger.info("Clearing existing Level 2 nodes from vector database")
                        vector_db_client.client.delete_collection("level2_nodes")
                        vector_db_client.create_collections()  # Recreate collection
                except Exception as e:
                    logger.error(f"Error clearing vector database collection: {str(e)}")
        
        # Initialize the graph builder
        builder = Level2GraphBuilder(
            neo4j_client=neo4j_client,
            ollama_client=ollama_client,
            vector_db_client=vector_db_client,
            embedding_dim=embedding_dim
        )
        
        # Build the graph
        success = await builder.build_graph_from_db(db_path)
        
        return success
    finally:
        # Close client connections
        await neo4j_client.close()


if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv
    
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Build Level 2 of the knowledge graph from UMLS database")
    parser.add_argument("--db_path", type=str, required=True, help="Path to UMLS SQLite database")
    parser.add_argument("--neo4j_uri", default=os.getenv("NEO4J_URI"), help="Neo4j URI")
    parser.add_argument("--neo4j_user", default=os.getenv("NEO4J_USERNAME"), help="Neo4j username")
    parser.add_argument("--neo4j_pass", default=os.getenv("NEO4J_PASSWORD"), help="Neo4j password")
    parser.add_argument("--ollama_host", default="http://localhost:11434", help="Ollama host")
    parser.add_argument("--embedding_model", default="mxbai-embed-large", help="Embedding model to use")
    parser.add_argument("--embedding_dim", type=int, default=1024, help="Dimension of vector embeddings")
    parser.add_argument("--clear", action="store_true", help="Clear existing Level 2 nodes")
    parser.add_argument("--qdrant_host", default=os.getenv("QDRANT_HOST"), help="Qdrant host")
    parser.add_argument("--qdrant_port", type=int, default=int(os.getenv("QDRANT_PORT", "6333")), help="Qdrant port")
    parser.add_argument("--qdrant_url", default=os.getenv("QDRANT_URL"), help="Qdrant Cloud URL")
    parser.add_argument("--qdrant_api_key", default=os.getenv("QDRANT_API_KEY"), help="Qdrant API key")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the graph creation
    asyncio.run(create_level2_graph_from_db(
        db_path=args.db_path,
        neo4j_uri=args.neo4j_uri,
        neo4j_username=args.neo4j_user,
        neo4j_password=args.neo4j_pass,
        ollama_host=args.ollama_host,
        embedding_model=args.embedding_model,
        embedding_dim=args.embedding_dim,
        clear_existing=args.clear,
        qdrant_host=args.qdrant_host,
        qdrant_port=args.qdrant_port,
        qdrant_api_key=args.qdrant_api_key,
        qdrant_url=args.qdrant_url
    ))