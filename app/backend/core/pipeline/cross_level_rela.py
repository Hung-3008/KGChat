import os
import asyncio
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import uuid
import asyncio
from tqdm import tqdm

from qdrant_client import QdrantClient
from db.neo4j_client import Neo4jClient
from utils.logging import get_logger

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
        """
        Save relationships to Neo4j in smaller sub-batches to optimize memory usage.
        
        Args:
            relationships: List of relationship dictionaries
            
        Returns:
            True if saving was successful, False otherwise
        """
        try:
            if not relationships:
                logger.info("No relationships to save")
                return True
                
            logger.info(f"Saving {len(relationships)} cross-level relationships to Neo4j")
            
            # Use smaller sub-batches for better memory management
            sub_batch_size = min(1000, self.batch_size)
            
            for i in range(0, len(relationships), sub_batch_size):
                sub_batch = relationships[i:i + sub_batch_size]
                
                # Optimize the import query for better performance
                query = """
                UNWIND $batch AS rel
                MATCH (source:Level1 {entity_id: rel.source_id})
                MATCH (target:Level2 {entity_id: rel.target_id})
                MERGE (source)-[r:REFERENCES]->(target)
                ON CREATE SET r.similarity_score = rel.similarity_score,
                            r.created_at = timestamp()
                ON MATCH SET r.similarity_score = CASE 
                    WHEN rel.similarity_score > r.similarity_score THEN rel.similarity_score
                    ELSE r.similarity_score
                END,
                r.updated_at = timestamp()
                """
                
                try:
                    await self.neo4j_client.execute_query(query, {"batch": sub_batch})
                    logger.info(f"Successfully imported sub-batch {i//sub_batch_size + 1}/{(len(relationships)-1)//sub_batch_size + 1}")
                except Exception as e:
                    logger.error(f"Error importing sub-batch {i//sub_batch_size + 1}: {str(e)}")
                    
                # Add a small delay between sub-batches to reduce database load
                if i + sub_batch_size < len(relationships):
                    await asyncio.sleep(0.2)
            
            return True
        except Exception as e:
            logger.error(f"Error saving relationships to Neo4j: {str(e)}")
            return False
    

    async def build_cross_level_relationships(self) -> bool:
        """
        Build cross-level relationships between Level 1 and Level 2 nodes with batch processing
        and visual progress tracking using tqdm.
        
        Returns:
            True if successful, False otherwise
        """
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
            
            # First, count total number of Level 1 nodes with embeddings
            count_query = """
            MATCH (n:Level1)
            WHERE n.vector_embedding IS NOT NULL
            RETURN COUNT(n) AS total
            """
            result = await self.neo4j_client.execute_query(count_query)
            total_nodes = result[0]["total"] if result else 0
            
            if total_nodes == 0:
                logger.warning("No Level 1 nodes found with vector embeddings")
                return False
                
            # Process Level 1 nodes in batches
            batch_size = 500  # Adjust based on your memory constraints
            total_batches = (total_nodes + batch_size - 1) // batch_size  # Ceiling division
            offset = 0
            total_processed = 0
            total_relationships = []
            relationships_batch_size = 5000  # Batch size for saving relationships
            
            logger.info(f"Starting process for {total_nodes} Level 1 nodes in {total_batches} batches")
            
            # Create progress bar
            with tqdm(total=total_nodes, desc="Processing Level 1 nodes") as pbar:
                for batch_num in range(total_batches):
                    # Get batch of Level 1 nodes
                    query = f"""
                    MATCH (n:Level1)
                    WHERE n.vector_embedding IS NOT NULL
                    RETURN n
                    SKIP {offset} LIMIT {batch_size}
                    """
                    
                    try:
                        results = await self.neo4j_client.execute_query(query)
                        level1_batch = [dict(record['n']) for record in results if 'n' in record]
                        
                        if not level1_batch:
                            logger.info(f"No more Level 1 nodes to process, processed {total_processed} nodes in total")
                            break
                        
                        batch_size_actual = len(level1_batch)
                        logger.info(f"Processing batch {batch_num+1}/{total_batches}: {batch_size_actual} nodes (offset: {offset})")
                        batch_relationships = []
                        
                        # Process each node in this batch
                        for i, level1_node in enumerate(level1_batch):
                            vector_embedding = level1_node.get("vector_embedding", [])
                            
                            level2_nodes = await self.find_similar_level2_nodes(
                                vector_embedding, 
                                limit=self.max_references_per_node
                            )
                            
                            relationships = await self.create_cross_level_relationships(
                                level1_node, 
                                level2_nodes
                            )
                            
                            batch_relationships.extend(relationships)
                            
                            # Update progress bar every node
                            pbar.update(1)
                            
                            # Log detailed progress periodically
                            if (i + 1) % 100 == 0:
                                pbar.set_postfix({"batch": f"{batch_num+1}/{total_batches}", 
                                                "rels_found": len(batch_relationships)})
                        
                        total_processed += batch_size_actual
                        total_relationships.extend(batch_relationships)
                        
                        # Set postfix info for the progress bar to show relationship count
                        pbar.set_postfix({"total_rels": len(total_relationships)})
                        logger.info(f"Batch {batch_num+1} complete: Found {len(batch_relationships)} relationships, total {len(total_relationships)}")
                        
                        # Save relationships when we reach the threshold
                        if len(total_relationships) >= relationships_batch_size:
                            logger.info(f"Saving batch of {len(total_relationships)} relationships to Neo4j")
                            save_success = await self.save_relationships_to_neo4j(total_relationships)
                            if save_success:
                                logger.info(f"Successfully saved {len(total_relationships)} relationships")
                                total_relationships = []  # Clear after saving
                            else:
                                logger.error(f"Failed to save batch of relationships")
                                return False
                        
                        # Move to next batch
                        offset += batch_size
                        
                        # Optional: Add a small delay to reduce database load
                        await asyncio.sleep(0.1)
                        
                    except Exception as e:
                        logger.error(f"Error processing batch {batch_num+1}: {str(e)}")
                        offset += batch_size  # Skip problematic batch and continue
                        # Update progress bar for skipped nodes
                        pbar.update(batch_size)
                
                # Save any remaining relationships
                if total_relationships:
                    logger.info(f"Saving final batch of {len(total_relationships)} relationships to Neo4j")
                    if await self.save_relationships_to_neo4j(total_relationships):
                        logger.info(f"Successfully saved final batch of relationships")
                    else:
                        logger.error(f"Failed to save final batch of relationships")
                        return False
            
            # Get final stats
            stats = await self.neo4j_client.get_graph_statistics()
            logger.info(f"Process complete. Cross-level relationships: {stats.get('cross_level_relationships', 0)}")
            
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
