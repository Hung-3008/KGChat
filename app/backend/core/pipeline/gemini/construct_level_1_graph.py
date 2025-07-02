import os
import logging
import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

from db.neo4j_client import Neo4jClient
from db.vector_db import VectorDBClient
from llm.ollama_client import OllamaClient
from llm.gemini_client import GeminiClient
from core.pipeline.gemini.chunking import DocumentChunker
from core.pipeline.gemini.graph_extraction import extract_graph_elements_from_chunks
from utils.logging import get_logger

logger = get_logger(__name__)

class GeminiKeyManager:
    def __init__(self, current_key_index: int = 1, max_keys: int = 6, key_pattern: str = "GEMINI_API_KEY_{}"):
        self.current_key_index = current_key_index
        self.max_keys = max_keys
        self.key_pattern = key_pattern
        self.logger = logging.getLogger(__name__)
    
    def get_current_key(self) -> Optional[str]:
        key_name = self.key_pattern.format(self.current_key_index)
        api_key = os.getenv(key_name)
        
        if not api_key:
            self.logger.warning(f"API key {key_name} not found in environment variables")
            return None
            
        return api_key
    
    def rotate_key(self) -> Optional[str]:
        for _ in range(self.max_keys):
            self.current_key_index = (self.current_key_index % self.max_keys) + 1
            key_name = self.key_pattern.format(self.current_key_index)
            api_key = os.getenv(key_name)
            
            if api_key:
                self.logger.info(f"Rotated to API key {key_name}")
                return api_key
        
        self.logger.error("No valid API keys found after trying all options")
        return None

class Level1GraphConstructor:
    def __init__(
        self,
        neo4j_client: Neo4jClient,
        ollama_client: OllamaClient,
        gemini_api_key: str,
        vector_db_client: Optional[VectorDBClient] = None,
        max_chunk_tokens: int = 12000,
        chunk_overlap: int = 1000,
        max_retry_attempts: int = 3
    ):
        self.neo4j_client = neo4j_client
        self.ollama_client = ollama_client
        self.vector_db_client = vector_db_client
        self.gemini_api_key = gemini_api_key
        self.max_chunk_tokens = max_chunk_tokens
        self.chunk_overlap = chunk_overlap
        self.max_retry_attempts = max_retry_attempts
        self.gemini_client = None
        
        self.key_manager = GeminiKeyManager(current_key_index=1, max_keys=6)
        
        if self.vector_db_client:
            self.vector_db_client.create_collections()
            logger.info("Vector database collections initialized")
    
    def _create_gemini_client(self) -> Optional[GeminiClient]:
        return GeminiClient(api_key=self.gemini_api_key)
    
    async def process_document(self, document_path: str, document_id: Optional[str] = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        if document_id is None:
            document_id = os.path.basename(document_path)
            
        logger.info(f"Processing document: {document_id}")
        
        content = await self._read_document(document_path)
        if not content:
            return [], []
            
        chunker = DocumentChunker(
            max_chunk_tokens=self.max_chunk_tokens,
            overlap_tokens=self.chunk_overlap
        )
        
        document_metadata = {"document_id": document_id, "knowledge_level": 1}
        chunks = await chunker.create_chunks(content, document_metadata)
        
        if not chunks:
            logger.warning(f"No chunks generated for document {document_id}")
            return [], []
        time.sleep(1)
        return await self._process_chunks(chunks)
    
    async def _read_document(self, file_path: str) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content
        except Exception as e:
            logger.error(f"Error reading document {file_path}: {str(e)}")
            return ""
    
    async def _process_chunks(self, chunks: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        time.sleep(2)
        all_nodes = []
        all_relationships = []
        
        processed_node_ids = set()
        processed_relationship_ids = set()
        
        total_chunks = len(chunks)
        chunks_processed = 0
        extraction_attempts = 0
        extraction_failures = 0
        
        chunk_groups = self._group_chunks(chunks, 5)
        
        logger.info(f"Starting processing of {total_chunks} chunks in {len(chunk_groups)} groups")
        
        max_keys = 6
        key_index = 0
        
        for i, chunk_group in enumerate(chunk_groups):
            logger.info(f"Processing chunk group {i+1}/{len(chunk_groups)} ({len(chunk_group)} chunks)")
            
            chunk_processed = False
            
            # Try each API key until successful or all keys are exhausted
            while key_index < max_keys and not chunk_processed:
                current_key = self.key_manager.get_current_key()
                if not current_key:
                    logger.warning(f"Invalid or missing API key at index {self.key_manager.current_key_index}")
                    current_key = self.key_manager.rotate_key()
                    if not current_key:
                        logger.error("No valid API keys available")
                        break
                    key_index += 1
                    continue
                
                self.gemini_api_key = current_key
                os.environ["GEMINI_API_KEY"] = current_key
                self.gemini_client = self._create_gemini_client()
                logger.info(f"Using API key index {self.key_manager.current_key_index} for chunk group {i+1}")
                
                max_retries = 3
                retries_attempted = 0
                
                # Try extraction with current key up to max_retries times
                while retries_attempted < max_retries and not chunk_processed:
                    try:
                        extraction_attempts += 1
                        retries_attempted += 1
                        
                        if not self.gemini_client:
                            self.gemini_client = self._create_gemini_client()
                            if not self.gemini_client:
                                logger.error("Failed to create Gemini client, unable to continue")
                                break
                        
                        logger.info(f"Attempt {retries_attempted}/{max_retries} for chunk group {i+1} with key index {self.key_manager.current_key_index}")
                        
                        res = await extract_graph_elements_from_chunks(
                            chunks=chunk_group,
                            gemini_api_key=self.gemini_api_key,
                            embedding_client=self.ollama_client,
                            vector_db_client=self.vector_db_client
                        )

                        if res is None:
                            logger.warning(f"Extraction returned None on attempt {retries_attempted}/{max_retries}")
                            extraction_failures += 1
                            
                            if retries_attempted >= max_retries:
                                logger.warning(f"All {max_retries} attempts failed with key index {self.key_manager.current_key_index}, will try next key")
                            
                            await asyncio.sleep(2)
                            continue
                        
                        nodes, relationships, _ = res

                        valid_nodes = [node for node in nodes if node.get('entity_id') is not None]
                        invalid_nodes_count = len(nodes) - len(valid_nodes)
                        
                        if invalid_nodes_count > 0:
                            logger.warning(f"Extracted {len(nodes)} nodes but {invalid_nodes_count} missing entity_id")
                        
                        valid_relationships = [rel for rel in relationships if rel.get('source_id') and rel.get('target_id')]
                        invalid_rels_count = len(relationships) - len(valid_relationships)
                        
                        if invalid_rels_count > 0:
                            logger.warning(f"Extracted {len(relationships)} relationships but {invalid_rels_count} missing source/target")
                        
                        if not valid_nodes and not valid_relationships:
                            logger.warning(f"No valid nodes or relationships in attempt {retries_attempted}/{max_retries}")
                            extraction_failures += 1
                            
                            if retries_attempted >= max_retries:
                                logger.warning(f"All {max_retries} attempts failed with key index {self.key_manager.current_key_index}, will try next key")
                            
                            await asyncio.sleep(2)
                            continue
                        
                        # Add new nodes to result set
                        new_nodes = 0
                        for node in valid_nodes:
                            node_id = node.get('entity_id')
                            if node_id not in processed_node_ids:
                                all_nodes.append(node)
                                processed_node_ids.add(node_id)
                                new_nodes += 1
                        
                        # Add new relationships to result set
                        new_relationships = 0
                        for rel in valid_relationships:
                            rel_id = rel.get('relationship_id', f"{rel.get('source_id')}_{rel.get('target_id')}")
                            if rel_id not in processed_relationship_ids:
                                all_relationships.append(rel)
                                processed_relationship_ids.add(rel_id)
                                new_relationships += 1
                        
                        chunks_processed += len(chunk_group)
                        
                        logger.info(f"Chunk group {i+1}: Extracted {len(nodes)} nodes ({new_nodes} new, {len(nodes) - invalid_nodes_count} valid) "
                                f"and {len(relationships)} relationships ({new_relationships} new, {len(relationships) - invalid_rels_count} valid). "
                                f"Total: {len(all_nodes)} nodes, {len(all_relationships)} relationships")
                        
                        chunk_processed = True
                        break
                    
                    except Exception as e:
                        extraction_failures += 1
                        logger.error(f"Error in attempt {retries_attempted}/{max_retries} with key index {self.key_manager.current_key_index}: {str(e)}")
                        
                        if retries_attempted >= max_retries:
                            logger.warning(f"All {max_retries} attempts failed with key index {self.key_manager.current_key_index}, will try next key")
                        
                        await asyncio.sleep(2)
                
                if chunk_processed:
                    break
                
                # Try next API key
                new_key = self.key_manager.rotate_key()
                if new_key:
                    logger.info(f"Rotated to new API key at index {self.key_manager.current_key_index}")
                    key_index += 1
                else:
                    logger.error("No more API keys available to try")
                    break
            
            if not chunk_processed:
                logger.warning(f"Failed to process chunk group {i+1} after trying all available keys")

        logger.info(f"Processing complete: {chunks_processed}/{total_chunks} chunks processed")
        logger.info(f"Extraction statistics: {extraction_attempts} attempts, {extraction_failures} failures")
        logger.info(f"Retrieved {len(all_nodes)} unique valid nodes and {len(all_relationships)} unique valid relationships")
        
        return all_nodes, all_relationships
    
    def _group_chunks(self, chunks: List[Dict[str, Any]], group_size: int) -> List[List[Dict[str, Any]]]:
        return [chunks[i:i + group_size] for i in range(0, len(chunks), group_size)]
    
    async def save_to_neo4j(self, nodes: List[Dict[str, Any]], relationships: List[Dict[str, Any]]) -> bool:
        try:
            await self.neo4j_client.setup_schema()
            
            logger.info(f"Preparing to save {len(nodes)} nodes and {len(relationships)} relationships to Neo4j")
            
            valid_nodes = [node for node in nodes if node.get('entity_id') is not None]
            invalid_nodes_count = len(nodes) - len(valid_nodes)
            
            if invalid_nodes_count > 0:
                logger.warning(f"Filtered out {invalid_nodes_count} nodes with missing entity_id ({len(valid_nodes)} valid nodes remaining)")
            
            logger.info(f"Importing {len(valid_nodes)} nodes as Level1 nodes")
            node_success = await self.neo4j_client.import_nodes(valid_nodes, "Level1")
            
            if not node_success:
                logger.error("Failed to import nodes to Neo4j")
                return False
            
            valid_relationships = [rel for rel in relationships if rel.get('source_id') and rel.get('target_id')]
            invalid_rels_count = len(relationships) - len(valid_relationships)
            
            if invalid_rels_count > 0:
                logger.warning(f"Filtered out {invalid_rels_count} relationships with missing source/target ({len(valid_relationships)} valid relationships remaining)")
                
            logger.info(f"Importing {len(valid_relationships)} relationships between Level1 nodes")
            rel_success = await self.neo4j_client.import_relationships(
                valid_relationships, "Level1", "Level1"
            )
            
            if not rel_success:
                logger.warning("Some relationships failed to import")
            
            stats = await self.neo4j_client.get_graph_statistics()
            logger.info(f"Level 1 graph statistics after import: {stats}")
            
            return True
        except Exception as e:
            logger.error(f"Error saving to Neo4j: {str(e)}")
            return False
    
    async def build_from_directory(self, directory_path: str, batch_size: int = 10) -> bool:
        try:
            if not os.path.exists(directory_path):
                logger.error(f"Directory not found: {directory_path}")
                return False
                
            files = [f for f in os.listdir(directory_path) 
                    if os.path.isfile(os.path.join(directory_path, f))]
                    
            if not files:
                logger.warning(f"No files found in directory: {directory_path}")
                return False
                
            logger.info(f"Found {len(files)} files in {directory_path}")
            
            current_nodes = []
            current_relationships = []
            total_nodes = 0
            total_relationships = 0
            processed_files = 0
            successful_imports = 0
            
            for i, file in enumerate(files):
                file_path = os.path.join(directory_path, file)
                logger.info(f"Processing file {i+1}/{len(files)}: {file}")
                
                nodes, relationships = await self.process_document(
                    document_path=file_path,
                    document_id=file
                )
                
                current_nodes.extend(nodes)
                current_relationships.extend(relationships)
                processed_files += 1
                
                logger.info(f"Completed processing {file}, extracted {len(nodes)} nodes and {len(relationships)} relationships")
                
                is_last_file = (i == len(files) - 1)
                should_save_batch = (processed_files % batch_size == 0) or is_last_file
                
                if should_save_batch and (current_nodes or current_relationships):
                    logger.info(f"Saving batch of {len(current_nodes)} nodes and {len(current_relationships)} relationships to Neo4j")
                    
                    batch_success = await self.save_to_neo4j(current_nodes, current_relationships)
                    
                    if batch_success:
                        logger.info(f"Successfully saved batch to Neo4j")
                        successful_imports += 1
                        
                        total_nodes += len(current_nodes)
                        total_relationships += len(current_relationships)
                        
                        current_nodes = []
                        current_relationships = []
                    else:
                        logger.error(f"Failed to save batch to Neo4j")
                        if is_last_file:
                            return False
            
            logger.info(f"Completed processing {len(files)} documents")
            logger.info(f"Imported a total of {total_nodes} nodes and {total_relationships} relationships")
            logger.info(f"Successfully saved {successful_imports} batches to Neo4j")
            
            stats = await self.neo4j_client.get_graph_statistics()
            logger.info(f"Final Neo4j graph statistics: {stats}")
            
            return successful_imports > 0
            
        except Exception as e:
            logger.error(f"Error building Level 1 graph: {str(e)}", exc_info=True)
            return False

async def create_level1_graph(
    input_directory: str,
    neo4j_uri: Optional[str] = None,
    neo4j_username: Optional[str] = None,
    neo4j_password: Optional[str] = None,
    ollama_host: str = "http://localhost:11434",
    ollama_model: str = "llama3:8b",
    embedding_model: str = "mxbai-embed-large",
    gemini_api_key: Optional[str] = None,
    clear_existing: bool = False,
    save_batch_size: int = 10,
    qdrant_host: Optional[str] = None,
    qdrant_port: int = 6333,
    qdrant_api_key: Optional[str] = None,
    qdrant_url: Optional[str] = None
) -> bool:
    if not gemini_api_key:
        gemini_api_key = os.getenv("GEMINI_API_KEY_1")
        if not gemini_api_key:
            logger.error("No Gemini API key available")
            return False

    neo4j_client = Neo4jClient(
        uri=neo4j_uri, 
        username=neo4j_username, 
        password=neo4j_password
    )
    
    ollama_client = OllamaClient(
        host=ollama_host,  
        model_name=ollama_model,
        embedding_model=embedding_model
    )
    
    vector_db_client = None
    if qdrant_host or qdrant_url:
        vector_db_client = VectorDBClient(
            host=qdrant_host,
            port=qdrant_port,
            api_key=qdrant_api_key,
            url=qdrant_url,
            vector_size=1024
        )
        logger.info("Initialized Vector Database client")
    
    try:
        connected = await neo4j_client.verify_connectivity()
        if not connected:
            logger.error("Failed to connect to Neo4j, exiting")
            return False
        
        if clear_existing:
            logger.info("Clearing existing Level 1 nodes and relationships from Neo4j")
            try:
                await neo4j_client.execute_query("""
                        CALL apoc.periodic.iterate(
                        'MATCH (n:Level1) RETURN n',
                        'DETACH DELETE n',
                        {batchSize:1000}
                        ) 
                        """)
                logger.info("Cleared existing Level 1 nodes and relationships from Neo4j")
            except:
                await neo4j_client.execute_query("""
                        MATCH (n:Level1)
                        DETACH DELETE n
                        """)
                logger.info("Cleared all nodes and relationships from Neo4j")
            
            if vector_db_client:
                try:
                    collection_names = vector_db_client.get_collection_names()
                    if "level1_nodes" in collection_names:
                        vector_db_client.delete_points("level1_nodes")
                        logger.info("Cleared existing Level 1 nodes from vector database")
                    else:
                        vector_db_client.create_collections()
                        logger.info("Created collections in vector database")

                except Exception as e:
                    logger.error(f"Error clearing Level 1 nodes from vector database: {str(e)}")

        constructor = Level1GraphConstructor(
            neo4j_client=neo4j_client,
            ollama_client=ollama_client,
            gemini_api_key=gemini_api_key,
            vector_db_client=vector_db_client
        )
        
        success = await constructor.build_from_directory(
            input_directory, 
            batch_size=save_batch_size
        )
        
        return success
    finally:
        await neo4j_client.close()