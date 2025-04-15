"""
Cross-Level Relationship Builder

This module builds cross-level relationships between Level 1 and Level 2 nodes
by using vector similarity to find connections between general concepts and specific details.
"""
import os
import asyncio
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import uuid

from qdrant_client import QdrantClient
from backend.db.neo4j_client import Neo4jClient
from backend.utils.logging import get_logger

logger = get_logger(__name__)

class CrossLevelRelationshipBuilder:
    def __init__(
        self,
        neo4j_client: Neo4jClient,
        qdrant_client: QdrantClient,
        similarity_threshold: float = 0.7,
        max_references_per_node: int = 5,
        batch_size: int = 100
    ):
        self.neo4j_client = neo4j_client
        self.qdrant_client = qdrant_client
        self.similarity_threshold = similarity_threshold
        self.max_references_per_node = max_references_per_node
        self.batch_size = batch_size
    
    async def get_level1_nodes(self) -> List[Dict[str, Any]]:
        query = """
        MATCH (n:Level1)
        WHERE n.vector_embedding IS NOT NULL
        RETURN n
        """
        
        try:
            results = await self.neo4j_client.execute_query(query)
            nodes = [dict(record['n']) for record in results if 'n' in record]
            logger.info(f"Retrieved {len(nodes)} Level 1 nodes with vector embeddings")
            return nodes
        except Exception as e:
            logger.error(f"Error retrieving Level 1 nodes: {str(e)}")
            return []
    
    async def find_similar_level2_nodes(
        self, 
        vector_embedding: List[float], 
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        try:
            if not vector_embedding:
                return []
                
            results = self.qdrant_client.query_points(
                collection_name="level2_nodes",
                query=vector_embedding,
                limit=limit,
            )
            
            similar_nodes = []
            for scored_point in results.points:
                score = scored_point.score
                if score > self.similarity_threshold:
                    payload = scored_point.payload
                    level2_node = {
                        "entity_id": payload.get("original_entity_id"),
                        "similarity_score": scored_point.score,
                        "knowledge_level": 2,
                        "cui": payload.get("cui", ""),
                        "aui": payload.get("aui", "")
                    }
                    similar_nodes.append(level2_node)
            
            if similar_nodes:
                logger.info(f"Found {len(similar_nodes)} similar Level 2 nodes")
            
            return similar_nodes
        except Exception as e:
            logger.error(f"Error finding similar Level 2 nodes: {str(e)}")
            return []
    
    async def create_cross_level_relationships(
        self, 
        level1_node: Dict[str, Any], 
        level2_nodes: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        relationships = []
        level1_id = level1_node.get("entity_id")
        
        for level2_node in level2_nodes:
            level2_id = level2_node.get("entity_id")
            similarity_score = level2_node.get("similarity_score", 0)
            relationship = {
                "source_id": level1_id,
                "target_id": level2_id,
                "type": "REFERENCES",
                "similarity_score": similarity_score
            }
            relationships.append(relationship)
        
        return relationships
    
    async def save_relationships_to_neo4j(self, relationships: List[Dict[str, Any]]) -> bool:
        try:
            logger.info(f"Saving {len(relationships)} cross-level relationships to Neo4j")
            
            for i in range(0, len(relationships), self.batch_size):
                batch = relationships[i:i + self.batch_size]
                success = await self.neo4j_client.import_relationships(batch, "Level1", "Level2")
                if not success:
                    logger.warning(f"Failed to import batch {i//self.batch_size + 1}")
                else:
                    logger.info(f"Successfully imported batch {i//self.batch_size + 1}")
            
            return True
        except Exception as e:
            logger.error(f"Error saving relationships to Neo4j: {str(e)}")
            return False
    
    async def build_cross_level_relationships(self) -> bool:
        try:
            if not await self.neo4j_client.verify_connectivity():
                logger.error("Failed to connect to Neo4j, exiting")
                return False
            
            try:
                collections = self.qdrant_client.get_collections()
                if "level2_nodes" not in [c.name for c in collections.collections]:
                    logger.error("level2_nodes collection not found in Qdrant")
                    return False
            except Exception as e:
                logger.error(f"Failed to connect to Qdrant: {str(e)}")
                return False
            
            level1_nodes = await self.get_level1_nodes()
            if not level1_nodes:
                logger.warning("No Level 1 nodes found with vector embeddings")
                return False
            
            total_relationships = []
            
            for i, level1_node in enumerate(level1_nodes):
                vector_embedding = level1_node.get("vector_embedding", [])
                
                level2_nodes = await self.find_similar_level2_nodes(
                    vector_embedding, 
                    limit=self.max_references_per_node
                )
                
                relationships = await self.create_cross_level_relationships(
                    level1_node, 
                    level2_nodes
                )
                
                total_relationships.extend(relationships)
                
                if (i + 1) % 10 == 0 or i + 1 == len(level1_nodes):
                    logger.info(f"Processed {i+1}/{len(level1_nodes)} Level 1 nodes")
            
            if total_relationships:
                logger.info(f"Found {len(total_relationships)} cross-level relationships")
                if await self.save_relationships_to_neo4j(total_relationships):
                    logger.info("Successfully saved cross-level relationships")
                else:
                    logger.error("Failed to save cross-level relationships")
                    return False
            
            stats = await self.neo4j_client.get_graph_statistics()
            logger.info(f"Cross-level relationships: {stats.get('cross_level_relationships', 0)}")
            
            return True
        except Exception as e:
            logger.error(f"Error building cross-level relationships: {str(e)}")
            return False


async def create_cross_level_relationships(
    neo4j_uri: Optional[str] = None,
    neo4j_username: Optional[str] = None,
    neo4j_password: Optional[str] = None,
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    qdrant_api_key: Optional[str] = None,
    qdrant_url: Optional[str] = None,
    similarity_threshold: float = 0.7,
    max_references_per_node: int = 5
) -> bool:
    load_dotenv()
    
    neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI")
    neo4j_username = neo4j_username or os.getenv("NEO4J_USERNAME")
    neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD")
    qdrant_host = qdrant_host or os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = qdrant_port or int(os.getenv("QDRANT_PORT", "6333"))
    qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")
    qdrant_url = qdrant_url or os.getenv("QDRANT_URL")
    
    neo4j_client = Neo4jClient(
        uri=neo4j_uri,
        username=neo4j_username,
        password=neo4j_password
    )
    
    qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key) if qdrant_url else QdrantClient(host=qdrant_host, port=qdrant_port)
    
    try:
        builder = CrossLevelRelationshipBuilder(
            neo4j_client=neo4j_client,
            qdrant_client=qdrant_client,
            similarity_threshold=similarity_threshold,
            max_references_per_node=max_references_per_node
        )
        
        return await builder.build_cross_level_relationships()
    finally:
        await neo4j_client.close()
