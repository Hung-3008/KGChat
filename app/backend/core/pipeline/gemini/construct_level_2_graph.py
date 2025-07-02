import sqlite3
import time
import os
import asyncio
import logging
import uuid
from typing import Dict, List, Any, Tuple, Optional

from llm.ollama_client import OllamaClient
from db.neo4j_client import Neo4jClient
from db.vector_db import VectorDBClient
from utils.logging import get_logger

logger = get_logger(__name__)

class Level2GraphBuilder:
    def __init__(
        self,
        neo4j_client: Neo4jClient,
        ollama_client: OllamaClient,
        vector_db_client: Optional[VectorDBClient] = None,
        embedding_dim: int = 1024,
        batch_size: int = 100
    ):
        self.neo4j_client = neo4j_client
        self.ollama_client = ollama_client
        self.vector_db_client = vector_db_client
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.fallback_embedding = [0.0] * embedding_dim
        
        if self.vector_db_client:
            self.vector_db_client.create_collections()
            logger.info("Vector database collections initialized")
    
    async def process_db_nodes(self, conn: sqlite3.Connection) -> Dict[Tuple[str, str], str]:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) 
            FROM mrconso m 
            LEFT JOIN mrdef d ON m.CUI = d.CUI AND m.AUI = d.AUI 
            WHERE m.LAT = 'ENG' AND d.DEF IS NOT NULL
        """)
        total_nodes = cursor.fetchone()[0]
        logger.info(f"Total nodes to process: {total_nodes}")
        
        cui_aui_to_id = {}
        processed = 0
        offset = 0
        
        while processed < total_nodes:
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
            
            str_values = []
            batch_data = []
            
            for row in rows:
                cui, aui, str_value, definition = row
                
                if not all([cui, aui, str_value, definition]):
                    continue
                
                str_values.append(str_value)
                batch_data.append((cui, aui, str_value, definition))
            
            try:
                batch_embeddings = await self.ollama_client.embed(str_values)
                logger.info(f"Generated {len(batch_embeddings)} embeddings for STR values")
            except Exception as e:
                logger.error(f"Error generating embeddings: {e}")
                batch_embeddings = [self.fallback_embedding] * len(str_values)
            
            nodes_batch = []
            vector_db_nodes_batch = []
            
            for i, (cui, aui, str_value, definition) in enumerate(batch_data):
                node_id = f"{cui}_{aui}"
                cui_aui_to_id[(cui, aui)] = node_id
                
                try:
                    embedding = batch_embeddings[i] if i < len(batch_embeddings) else self.fallback_embedding
                except Exception:
                    embedding = self.fallback_embedding
                
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
                
                if self.vector_db_client:
                    vector_db_nodes_batch.append(node.copy())
                
                processed += 1
            
            if nodes_batch:
                success = await self.neo4j_client.import_nodes(nodes_batch, "Level2")
                if success:
                    logger.info(f"Imported batch of {len(nodes_batch)} Level2 nodes to Neo4j")
                else:
                    logger.error(f"Failed to import batch of {len(nodes_batch)} nodes to Neo4j")
            
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
    
    async def process_db_relationships(
        self,
        conn: sqlite3.Connection,
        cui_aui_to_id: Dict[Tuple[str, str], str],
        batch_size: int = 5000
    ) -> int:
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
            
            relationships_batch = []
            
            for row in rows:
                cui1, aui1, rela, cui2, aui2 = row
                
                if (cui1, aui1) not in cui_aui_to_id or (cui2, aui2) not in cui_aui_to_id:
                    continue
                
                rela = rela if rela else "RELATED_TO"
                
                source_id = cui_aui_to_id[(cui1, aui1)]
                target_id = cui_aui_to_id[(cui2, aui2)]
                
                relationship = {
                    "relationship_id": f"rel_{cui1}_{aui1}_{cui2}_{aui2}",
                    "source_id": source_id,
                    "target_id": target_id,
                    "type": rela.upper(),
                    "description": f"{rela} relationship from {cui1} to {cui2}",
                    "strength": 0.8,
                    "keywords": [rela.lower()],
                    "knowledge_level": 2
                }
                
                relationships_batch.append(relationship)
                processed += 1
            
            if relationships_batch:
                success = await self.neo4j_client.import_relationships(
                    relationships_batch, "Level2", "Level2"
                )
                if success:
                    logger.info(f"Imported batch of {len(relationships_batch)} Level2 relationships to Neo4j")
                else:
                    logger.error(f"Failed to import batch of {len(relationships_batch)} relationships to Neo4j")
            
            offset += batch_size
            logger.info(f"Processed {processed}/{total_rels} relationships")
        
        logger.info(f"Completed processing {processed} Level2 relationships")
        return processed
    
    async def build_graph_from_db(self, db_path: str) -> bool:
        try:
            start_time = time.time()
            
            if not await self.neo4j_client.verify_connectivity():
                logger.error("Failed to connect to Neo4j, exiting")
                return False
            
            await self.neo4j_client.setup_schema()
            
            logger.info(f"Connecting to SQLite database: {db_path}")
            conn = sqlite3.connect(db_path)
            conn.execute("PRAGMA temp_store = MEMORY")
            conn.execute("PRAGMA cache_size = 10000")
            
            try:
                logger.info("Phase 1: Processing nodes...")
                cui_aui_to_id = await self.process_db_nodes(conn)
                
                logger.info("Phase 2: Processing relationships...")
                rel_count = await self.process_db_relationships(conn, cui_aui_to_id)
                
                elapsed_time = time.time() - start_time
                logger.info(f"Level 2 graph construction completed in {elapsed_time:.2f} seconds")
                logger.info(f"Created {len(cui_aui_to_id)} nodes and {rel_count} relationships")
                
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
        if not await neo4j_client.verify_connectivity():
            logger.error("Failed to connect to Neo4j, exiting")
            return False
        
        if clear_existing:
            logger.info("Clearing existing Level 2 nodes and relationships")
            await neo4j_client.execute_query("MATCH (n:Level2) DETACH DELETE n")
            
            if vector_db_client:
                try:
                    collections = vector_db_client.client.get_collections()
                    collection_names = [c.name for c in collections.collections]
                    
                    if "level2_nodes" in collection_names:
                        logger.info("Clearing existing Level 2 nodes from vector database")
                        vector_db_client.client.delete_collection("level2_nodes")
                        vector_db_client.create_collections()
                except Exception as e:
                    logger.error(f"Error clearing vector database collection: {str(e)}")
        
        builder = Level2GraphBuilder(
            neo4j_client=neo4j_client,
            ollama_client=ollama_client,
            vector_db_client=vector_db_client,
            embedding_dim=embedding_dim
        )
        
        return await builder.build_graph_from_db(db_path)
    finally:
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
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
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