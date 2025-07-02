import logging
import uuid
import os
import aiohttp
from typing import List, Dict, Any, Optional, Tuple

from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from llm.gemini_client import GeminiClient
from core.pipeline.gemini.prompts import GEMINI_ENTITY_EXTRACTION_PROMPT
from db.vector_db import VectorDBClient
#from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
logger = logging.getLogger(__name__)

def remove_duplicate_relationships(relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    unique_relationships = {}
    for relationship in relationships:
        # Create a unique key using source_id, target_id, and type
        key = (
            relationship['source_id'], 
            relationship['target_id'], 
            relationship['type']
        )
        
        # If no relationship with this key exists, add it
        if key not in unique_relationships:
            unique_relationships[key] = relationship
        else:
            # If a relationship already exists, keep the one with higher knowledge level
            existing_relationship = unique_relationships[key]
            if (relationship.get('knowledge_level', 1) > 
                existing_relationship.get('knowledge_level', 1)):
                unique_relationships[key] = relationship
            elif (relationship.get('knowledge_level', 1) == 
                  existing_relationship.get('knowledge_level', 1)):
                # If knowledge levels are the same, prefer the more descriptive relationship
                if len(relationship.get('description', '')) > len(existing_relationship.get('description', '')):
                    unique_relationships[key] = relationship
    
    return list(unique_relationships.values())

class GraphElementExtractor:
    def __init__(self, gemini_client: Optional[GeminiClient] = None, api_key: Optional[str] = None, model_name: str = "gemini-2.0-flash", max_retries: int = 3, embedding_client = None, embedding_model: str = "mxbai-embed-large", vector_db_client: Optional[VectorDBClient] = None):
        self.api_key = api_key
        self.model_name = model_name
        self.max_retries = max_retries
        self.embedding_client = embedding_client
        self.embedding_model = embedding_model
        self.total_requests = 0
        self.vector_db_client = vector_db_client
        self.gemini_client = gemini_client or self._create_gemini_client()
        self._setup_llm()
        # self.prompt_template = PromptTemplate(
        #     input_variables=["text"],
        #     template=GEMINI_ENTITY_EXTRACTION_PROMPT
        # )

        self.prompt_template = ChatPromptTemplate.from_messages(
            messages=[
                ("system", GEMINI_ENTITY_EXTRACTION_PROMPT),
                ("human", "Please extract the specified nodes and relationships from this text: {input}")
            ]
        )
        self.llm_transformer = None
        if self.vector_db_client:
            self.vector_db_client.create_collections()
    
    def _create_gemini_client(self) -> Optional[GeminiClient]:
        if not self.api_key:
            logger.error("No Gemini API key available, cannot create client")
            return None
        return GeminiClient(api_key=self.api_key, model_name=self.model_name)
    
    def _setup_llm(self):
        if self.api_key:
            os.environ["GOOGLE_API_KEY"] = self.api_key
        self.llm = ChatGoogleGenerativeAI(model=self.model_name, temperature=0)
    
    def _initialize_transformer(self):
        if self.llm_transformer is None:
            self.llm_transformer = LLMGraphTransformer(llm=self.llm, 
                                                       prompt=self.prompt_template, 
                                                       node_properties=["entity_description"], 
                                                       relationship_properties=["relationship_type"])
    
    def _deduplicate_relationships(self, relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return remove_duplicate_relationships(relationships)

    async def extract_graph_elements(self, chunks: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        if not chunks:
            logger.warning("No chunks provided for graph element extraction")
            return [], []
        logger.info(f"Starting graph element extraction for {len(chunks)} chunks")
        self._initialize_transformer()
        all_nodes = []
        all_relationships = []
        for i, chunk in enumerate(chunks):
            try:
                chunk_id = chunk.get('chunk_id', f'chunk-{i}')
                document_id = chunk.get('document_id', 'unknown')
                knowledge_level = chunk.get('knowledge_level', 1)
                content = chunk.get('content', '')
                if not content.strip():
                    logger.warning(f"Empty content in chunk {chunk_id}")
                    continue
                
                # Process the chunk directly without key rotation
                nodes, relationships, requests_made = await self._process_chunk(
                    content=content, 
                    chunk_id=chunk_id, 
                    document_id=document_id, 
                    knowledge_level=knowledge_level
                )
                
                self.total_requests += requests_made
                if self.embedding_client and nodes:
                    nodes = await self._add_embeddings_to_nodes(nodes)
                all_nodes.extend(nodes)
                all_relationships.extend(relationships)
                logger.info(f"Processed chunk {i+1}/{len(chunks)}: extracted {len(nodes)} nodes and {len(relationships)} relationships")
            except Exception as e:
                logger.error(f"Error processing chunk {i}: {str(e)}")
                continue
        if self.vector_db_client and all_nodes:
            await self._store_nodes_in_vector_db(all_nodes)
        logger.info(f"Completed extraction with {len(all_nodes)} nodes and {len(all_relationships)} relationships")
        return all_nodes, all_relationships
    
    async def _process_chunk(self, content: str, chunk_id: str, document_id: str, knowledge_level: int = 1) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
        """
        Process a chunk without retry or key rotation - single attempt.
        """
        num_requests = 1  # We'll make exactly one request
        
        try:
            logger.debug(f"Processing chunk {chunk_id}")
            
            # Create a document for langchain
            documents = [Document(page_content=content)]
            
            # Process with LLM transformer
            graph_elements = self.llm_transformer.convert_to_graph_documents(documents)
            
            if not graph_elements or not hasattr(graph_elements[0], 'nodes'):
                logger.warning(f"No graph elements extracted from chunk {chunk_id}")
                return [], [], num_requests
            
            # Convert to our format
            formatted_nodes = self._convert_nodes_to_format(
                nodes=graph_elements[0].nodes, 
                chunk_id=chunk_id, 
                document_id=document_id, 
                knowledge_level=knowledge_level
            )
            
            formatted_relationships = self._convert_relationships_to_format(
                relationships=graph_elements[0].relationships, 
                chunk_id=chunk_id, 
                document_id=document_id, 
                knowledge_level=knowledge_level
            )
            
            return formatted_nodes, formatted_relationships, num_requests
                
        except Exception as e:
            logger.error(f"Error processing chunk {chunk_id}: {str(e)}")
            return [], [], num_requests
    
    def _convert_nodes_to_format(self, nodes: List[Any], chunk_id: str, document_id: str, knowledge_level: int) -> List[Dict[str, Any]]:
        formatted_nodes = []
        for node in nodes:
            entity_name = node.id
            entity_type = node.type
            entity_description = ""
            if 'entity_description' in node.properties:
                entity_description = node.properties['entity_description']
            elif 'description' in node.properties:
                entity_description = node.properties['description']
            else:
                entity_description = f"A {entity_type.lower()} entity"
            if not entity_name:
                continue
            formatted_node = {
                "entity_id": entity_name,
                "entity_type": entity_type,
                "description": entity_description,
                "knowledge_level": knowledge_level,
            }
            formatted_nodes.append(formatted_node)
        return formatted_nodes
    
    def _convert_relationships_to_format(self, relationships: List[Any], chunk_id: str, document_id: str, knowledge_level: int) -> List[Dict[str, Any]]:
        formatted_relationships = []
        for rel in relationships:
            source_entity = rel.source.id if hasattr(rel.source, 'id') else str(rel.source)
            target_entity = rel.target.id if hasattr(rel.target, 'id') else str(rel.target)
            if not source_entity or not target_entity:
                continue
            relationship_type = rel.type or "RELATED_TO"
            relationship_description = ""
            if 'relationship_description' in rel.properties:
                relationship_description = rel.properties['relationship_description']
            elif 'description' in rel.properties:
                relationship_description = rel.properties['description']
            else:
                relationship_description = f"Relationship between {source_entity} and {target_entity}"
            
            # Generate relationship_id using the format "source id_target id"
            relationship_id = f"{source_entity}_{target_entity}"
            
            formatted_relationship = {
                "relationship_id": relationship_id,
                "type": relationship_type,
                "source_id": source_entity,
                "target_id": target_entity,
                "description": relationship_description,
                "knowledge_level": knowledge_level
            }
            formatted_relationships.append(formatted_relationship)
        
        # Remove duplicate relationships based on the new relationship_id format
        unique_relationships = {}
        for rel in formatted_relationships:
            rel_id = rel["relationship_id"]
            if rel_id not in unique_relationships:
                unique_relationships[rel_id] = rel
            else:
                # If duplicate found, keep the one with higher knowledge level
                existing_rel = unique_relationships[rel_id]
                if rel.get("knowledge_level", 1) > existing_rel.get("knowledge_level", 1):
                    unique_relationships[rel_id] = rel
                elif rel.get("knowledge_level", 1) == existing_rel.get("knowledge_level", 1):
                    # If knowledge levels are the same, prefer the more descriptive relationship
                    if len(rel.get("description", "")) > len(existing_rel.get("description", "")):
                        unique_relationships[rel_id] = rel
        
        return list(unique_relationships.values())

    async def _add_embeddings_to_nodes(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Add vector embeddings to nodes.
        
        Args:
            nodes: Node dictionaries to add embeddings to
            
        Returns:
            Nodes with embeddings added
        """
        if not self.embedding_client:
            return nodes
        try:
            texts = [node.get('entity_id', '') for node in nodes]
            embeddings = await self.embedding_client.embed(texts=texts, model=self.embedding_model)
            for i, node in enumerate(nodes):
                if i < len(embeddings):
                    node["vector_embedding"] = embeddings[i]
            return nodes
        except Exception as e:
            logger.error(f"Error generating embeddings for nodes: {str(e)}")
            return nodes
    
    async def _store_nodes_in_vector_db(self, nodes: List[Dict[str, Any]]) -> bool:
        """
        Store nodes in the vector database.
        
        Args:
            nodes: Node dictionaries to store
            
        Returns:
            True if successful, False otherwise
        """
        if not self.vector_db_client:
            logger.info("No vector database client provided, skipping storage")
            return False
        try:
            level1_nodes = [node for node in nodes if node.get('knowledge_level') == 1]
            if level1_nodes:
                level1_count = self.vector_db_client.store_nodes_batch(nodes=level1_nodes, collection_name="level1_nodes")
                logger.info(f"Stored {level1_count} Level 1 nodes in vector database")
            return True
        except Exception as e:
            logger.error(f"Error storing nodes in vector database: {str(e)}")
            return False


async def extract_graph_elements_from_chunks(
    chunks: List[Dict[str, Any]], 
    gemini_api_key: Optional[str] = None, 
    gemini_model: str = "gemini-2.0-flash", 
    embedding_client = None, 
    embedding_model: str = "mxbai-embed-large", 
    vector_db_client: Optional[VectorDBClient] = None
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
    """
    Extract graph elements from document chunks using Gemini.
    
    Args:
        chunks: List of document chunks
        gemini_api_key: Gemini API key for extraction
        gemini_model: Gemini model to use
        embedding_client: Client for embedding generation
        embedding_model: Model to use for embeddings
        vector_db_client: Optional Vector database client
        
    Returns:
        Tuple of (nodes, relationships, total_requests)
    """
    # Initialize extractor with the provided API key
    extractor = GraphElementExtractor(
        api_key=gemini_api_key, 
        model_name=gemini_model, 
        embedding_client=embedding_client, 
        embedding_model=embedding_model, 
        vector_db_client=vector_db_client
    )
    
    try:
        # Process the chunks
        nodes, relationships = await extractor.extract_graph_elements(chunks)
        total_requests = extractor.total_requests
        
        return nodes, relationships, total_requests
    except Exception as e:
        logger.error(f"Unexpected error in extract_graph_elements_from_chunks: {str(e)}")
        return None