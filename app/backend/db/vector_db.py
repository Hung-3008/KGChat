import os
import logging
from typing import List, Dict, Any, Optional, Union
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import ScoredPoint, PointStruct
import uuid
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class VectorDBClient:
    """
    Client for interacting with Qdrant vector database for knowledge graph nodes.
    
    This class handles database connections and operations for storing and retrieving
    vector embeddings of knowledge graph nodes.
    """
    
    def __init__(
        self,
        host: str = None,
        port: int = 6333,
        api_key: str = None,
        url: str = None,
        vector_size: int = 1024
    ):
        """
        Initialize the Qdrant client.
        
        Args:
            host: Qdrant server host (default: localhost)
            port: Qdrant server port (default: 6333)
            api_key: API key for Qdrant Cloud
            url: URL for Qdrant Cloud
            vector_size: Dimension of vector embeddings
        """
        # Load environment variables if not provided
        load_dotenv()
        
        self.host = host or os.getenv("QDRANT_HOST", "localhost")
        self.port = port or int(os.getenv("QDRANT_PORT", "6333"))
        self.api_key = api_key or os.getenv("QDRANT_API_KEY")
        self.url = url or os.getenv("QDRANT_URL")
        self.vector_size = vector_size
        
        # Initialize the client
        if self.url:
            # Cloud connection
            self.client = QdrantClient(url=self.url, api_key=self.api_key)
            logger.info(f"Initialized Qdrant client with cloud URL: {self.url}")
        else:
            # Local connection
            self.client = QdrantClient(host=self.host, port=self.port)
            logger.info(f"Initialized Qdrant client with local host: {self.host}:{self.port}")
    
    def create_collections(self):
        """
        Set up collections for level 1 and level 2 nodes if they don't exist.
        
        Creates two collections:
        - level1_nodes: For general concepts (Level 1)
        - level2_nodes: For specific details (Level 2)
        """
        for collection_name in ["level1_nodes", "level2_nodes"]:
            try:
                # Check if collection exists
                self.client.get_collection(collection_name)
                logger.info(f"Collection {collection_name} already exists")
            except Exception as e:
                logger.info(f"Creating collection {collection_name}: {str(e)}")
                # Create new collection if it doesn't exist
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=self.vector_size,
                        distance=models.Distance.COSINE
                    )
                )
                logger.info(f"Created collection {collection_name}")
    
    def get_collection_names(self) -> List[str]:
        """
        Get a list of all collection names in the vector database.
        
        Returns:
            List of collection names as strings
        """
        try:
            # Get all collections from the database
            collections = self.client.get_collections()
            
            # Extract just the names from the collection objects
            collection_names = []
            if hasattr(collections, 'collections'):
                collection_names = [collection.name for collection in collections.collections]
            
            logger.info(f"Retrieved {len(collection_names)} collection names from vector database")
            return collection_names
            
        except Exception as e:
            logger.error(f"Error retrieving collection names from vector database: {str(e)}")
            return []


    def store_node(self, node: Dict[str, Any], collection_name: str) -> bool:
        try:
            # Tạo UUID mới làm ID
            point_id = str(uuid.uuid4())
            
            # Lấy vector embedding
            vector = node.get("vector_embedding", [])
            
            # Skip if missing required fields
            if not vector:
                logger.warning(f"Skipping node with missing vector_embedding: {node}")
                return False
            
            # Tạo payload, lưu entity_id như metadata
            payload = {
                **{k: v for k, v in node.items() if k != "vector_embedding"},
                "original_entity_id": node.get("entity_id", point_id)
            }
            
            # Upsert into Qdrant
            self.client.upsert(
                collection_name=collection_name,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload
                    )
                ]
            )
            logger.info(f"Stored node {point_id} in {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing node in vector database: {str(e)}")
            return False

    def store_nodes_batch(self, nodes: List[Dict[str, Any]], collection_name: str) -> int:

        try:
            logger.info(f"Received {len(nodes)} nodes for storing in collection {collection_name}")
            
            valid_nodes = []
            for node in nodes:
                point_id = str(uuid.uuid4())
                
                vector = node.get("vector_embedding", [])
                
                if not vector:
                    logger.warning(f"Node {node.get('entity_id', 'unknown')} has no vector embedding, skipping")
                    continue
                
                payload = {
                "original_entity_id": node.get("entity_id", point_id),
                "knowledge_level": node.get("knowledge_level", 1),
                **{k: v for k, v in node.items() 
                   if k not in ["entity_id", "vector_embedding", "name", "type", "description", "knowledge_level", "entity_name", "entity_type"]}
                }
                
                valid_nodes.append(
                    models.PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload
                    )
                )
            
            logger.info(f"Found {len(valid_nodes)} valid nodes to store")
            
            if valid_nodes:
                self.client.upsert(
                    collection_name=collection_name,
                    points=valid_nodes
                )
                
                logger.info(f"Stored {len(valid_nodes)} nodes in {collection_name}")
                return len(valid_nodes)
            
            logger.warning("No valid nodes to store")
            return 0
            
        except Exception as e:
            logger.error(f"Error storing nodes batch in vector database: {str(e)}")
            return 0
    
    def retrieve_from_id(
        self,
        collection_name: str,
        ids: List[str],
        with_vectors: bool = False,
        with_payload: bool = True,
        limit: int = 10,
        score_threshold: float = None
    ):
        try:
            points = self.client.retrieve(
                collection_name=collection_name,
                ids=ids,
                with_vectors=with_vectors,
                with_payload=with_payload
            )
            
            if score_threshold is None:
                return [
                    ScoredPoint(
                        id=point.id,
                        payload=point.payload,
                        vector=point.vector if with_vectors else None,
                        score=1.0  
                    )
                    for point in points[:limit]
                ]
            
            if score_threshold is not None and not with_vectors:
                
                points_with_vectors = self.client.retrieve(
                    collection_name=collection_name,
                    ids=ids,
                    with_vectors=True,
                    with_payload=with_payload
                )
                result_points = []
                
                for i, point in enumerate(points_with_vectors[:limit]):
                    similarity_score = 0.9  
                    

                    if similarity_score >= score_threshold:
                        result_points.append(
                            ScoredPoint(
                                id=point.id,
                                payload=point.payload,
                                vector=point.vector if with_vectors else None,
                                score=similarity_score
                            )
                        )
                
                return result_points
            

            return [
                ScoredPoint(
                    id=point.id,
                    payload=point.payload,
                    vector=point.vector if with_vectors else None,
                    score=1.0
                )
                for point in points[:limit]
            ]
            
        except Exception as e:
            self.logger.error(f"Error retrieving points by ID from collection {collection_name}: {str(e)}")
            return []
    
    def delete_points(
        self,
        collection_name: str,
        point_ids: Optional[List[str]] = None,
        filter_condition: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Delete points from a specified collection.
        
        Args:
            collection_name: Name of the collection to delete points from (e.g., "level1_nodes" or "level2_nodes")
            point_ids: Optional list of specific point IDs to delete. If None, deletes based on filter_condition.
            filter_condition: Optional filter condition to select points for deletion. If None and point_ids is None, deletes all points.
            
        Returns:
            True if the deletion was successful, False otherwise
        """
        try:
            if not collection_name:
                logger.error("Collection name must be provided")
                return False
                
            # Check if collection exists
            try:
                self.client.get_collection(collection_name)
            except Exception as e:
                logger.error(f"Collection {collection_name} does not exist: {str(e)}")
                return False
                
            # Different deletion strategies based on provided parameters
            if point_ids:
                # Delete specific points by IDs
                logger.info(f"Deleting {len(point_ids)} points by ID from collection {collection_name}")
                self.client.delete(
                    collection_name=collection_name,
                    points=point_ids
                )
                logger.info(f"Successfully deleted {len(point_ids)} points from {collection_name}")
                
            elif filter_condition:
                # Delete points matching a filter condition
                logger.info(f"Deleting points matching filter from collection {collection_name}")
                self.client.delete(
                    collection_name=collection_name,
                    filters=filter_condition
                )
                logger.info(f"Successfully deleted points matching filter from {collection_name}")
                
            else:
                # Delete all points in the collection (scroll API with batch deletion)
                logger.info(f"Deleting all points from collection {collection_name}")
                
                # Get the total count for logging purposes
                collection_info = self.client.get_collection(collection_name)
                total_points = collection_info.vectors_count
                
                # Use scroll parameter for efficient deletion of all points
                batch_size = 1000
                self.client.scroll(
                    collection_name=collection_name,
                    limit=batch_size,
                    with_payload=False,
                    with_vectors=False
                )
                
                # Use the `recreate_collection` approach which is more efficient for deleting all points
                vector_size = self.vector_size
                distance = models.Distance.COSINE
                
                # Store current collection config
                try:
                    collection_config = self.client.get_collection(collection_name)
                    if hasattr(collection_config, 'config') and hasattr(collection_config.config, 'params'):
                        if hasattr(collection_config.config.params, 'vectors'):
                            vector_size = collection_config.config.params.vectors.size
                        if hasattr(collection_config.config.params, 'vectors'):
                            distance = collection_config.config.params.vectors.distance
                except Exception as e:
                    logger.warning(f"Could not retrieve existing collection config, using defaults: {str(e)}")
                
                # Delete and recreate the collection
                self.client.delete_collection(collection_name)
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=distance
                    )
                )
                
                logger.info(f"Successfully deleted all {total_points} points from {collection_name} by recreating the collection")
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting points from collection {collection_name}: {str(e)}")
            return False