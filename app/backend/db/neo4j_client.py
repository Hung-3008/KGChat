import logging
import asyncio
import json
from typing import List, Dict, Any, Optional, Union
from neo4j import GraphDatabase, AsyncGraphDatabase
from neo4j.exceptions import Neo4jError
import os
from dotenv import load_dotenv
import uuid

logger = logging.getLogger(__name__)

class Neo4jClient:
    def __init__(self, uri: str = None, username: str = None, password: str = None, database: str = "neo4j", use_async: bool = True):
        if uri is None or username is None or password is None:
            load_dotenv()
            uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
            username = username or os.getenv("NEO4J_USERNAME", "neo4j")
            password = password or os.getenv("NEO4J_PASSWORD", "12345678")
        
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self.use_async = use_async
        self._driver = None
        self._initialize_driver()
        
    def _initialize_driver(self):
        try:
            if self.use_async:
                self._driver = AsyncGraphDatabase.driver(self.uri, auth=(self.username, self.password))
            else:
                self._driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            logger.info(f"Successfully initialized Neo4j driver for {self.uri}")
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j driver: {str(e)}")
            raise
    
    async def close(self):
        if self._driver:
            if self.use_async:
                await self._driver.close()
            else:
                self._driver.close()
            logger.info("Neo4j driver closed")
    
    def close_sync(self):
        if self._driver and not self.use_async:
            self._driver.close()
            logger.info("Neo4j driver closed")
    
        # Initialize the driver
        self._initialize_driver()
        
    def _initialize_driver(self):
        """Initialize the Neo4j driver with provided credentials."""
        try:
            if self.use_async:
                self._driver = AsyncGraphDatabase.driver(
                    self.uri, 
                    auth=(self.username, self.password)
                )
            else:
                self._driver = GraphDatabase.driver(
                    self.uri, 
                    auth=(self.username, self.password)
                )
            logger.info(f"Successfully initialized Neo4j driver for {self.uri}")
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j driver: {str(e)}")
            raise
    
    async def close(self):
        """Close the Neo4j driver connection."""
        if self._driver:
            if self.use_async:
                await self._driver.close()
            else:
                self._driver.close()
            logger.info("Neo4j driver closed")
    
    def close_sync(self):
        """Synchronous version of close method."""
        if self._driver and not self.use_async:
            self._driver.close()
            logger.info("Neo4j driver closed")
    
    async def verify_connectivity(self) -> bool:
        """
        Verify that the Neo4j connection is working.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            if self.use_async:
                await self._driver.verify_connectivity()
            else:
                self._driver.verify_connectivity()
            logger.info("Neo4j connectivity verified")
            return True
        except Exception as e:
            logger.error(f"Neo4j connectivity check failed: {str(e)}")
            return False
    
    async def setup_schema(self) -> bool:
        """
        Set up Neo4j schema with constraints and indexes for the knowledge graph.
        
        Returns:
            True if schema setup is successful, False otherwise
        """
        schema_queries = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Level1) REQUIRE n.entity_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Level2) REQUIRE n.entity_id IS UNIQUE",
            "CREATE INDEX level1_name_idx IF NOT EXISTS FOR (n:Level1) ON (n.name)",
            "CREATE INDEX level2_name_idx IF NOT EXISTS FOR (n:Level2) ON (n.name)",
        ]
        
        try:
            if self.use_async:
                async with self._driver.session(database=self.database) as session:
                    for query in schema_queries:
                        try:
                            await session.run(query)
                            logger.info(f"Successfully executed schema query: {query}")
                        except Neo4jError as ne:
                            # Handle case where vector indexes aren't supported
                            if "vector index" in str(ne).lower():
                                logger.warning(f"Vector index not supported: {str(ne)}")
                            else:
                                logger.error(f"Neo4j error executing schema query: {str(ne)}")
                                raise
            else:
                with self._driver.session(database=self.database) as session:
                    for query in schema_queries:
                        try:
                            session.run(query)
                            logger.info(f"Successfully executed schema query: {query}")
                        except Neo4jError as ne:
                            # Handle case where vector indexes aren't supported
                            if "vector index" in str(ne).lower():
                                logger.warning(f"Vector index not supported: {str(ne)}")
                            else:
                                logger.error(f"Neo4j error executing schema query: {str(ne)}")
                                raise
            
            logger.info("Neo4j schema setup completed successfully")
            return True
        except Exception as e:
            logger.error(f"Error setting up Neo4j schema: {str(e)}")
            return False
    
    

    async def import_nodes(self, nodes: List[Dict[str, Any]], label: str, batch_size: int = 1000) -> bool:
        """
        Import nodes into Neo4j in batches.
        
        Args:
            nodes: List of node dictionaries
            label: Node label (Level1 or Level2)
            batch_size: Number of nodes to import in each batch
            
        Returns:
            True if import is successful, False otherwise
        """
        if not nodes:
            logger.warning(f"No {label} nodes to import")
            return True
            
        try:
            # Check for duplicate entity_ids
            entity_ids = [node.get('entity_id') for node in nodes]
            duplicate_ids = [id for id in set(entity_ids) if entity_ids.count(id) > 1]
            if duplicate_ids:
                logger.warning(f"Found {len(duplicate_ids)} duplicate entity_ids in the input data")
                
            # Log sample node
            if nodes:
                sample_node = {k: v for k, v in nodes[0].items() if k != 'vector_embedding'}
                logger.info(f"Sample node structure: {sample_node}")
                
            # Process nodes in batches
            for i in range(0, len(nodes), batch_size):
                batch = nodes[i:i + batch_size]
                
                # Create Cypher query for importing nodes
                query = f"""
                UNWIND $nodes AS node
                MERGE (n:{label} {{entity_id: node.entity_id}})
                SET n = node
                """
                
                # Clean nodes for Neo4j compatibility
                cleaned_batch = self._clean_node_properties(batch)
                
                if self.use_async:
                    async with self._driver.session(database=self.database) as session:
                        try:
                            await session.run(query, nodes=cleaned_batch)
                            logger.info(f"Successfully executed node import query for batch of {len(cleaned_batch)} nodes")
                        except Exception as e:
                            logger.error(f"Error during node import transaction: {str(e)}")
                            if cleaned_batch:
                                logger.error(f"Sample node that caused error: {cleaned_batch[0].get('entity_id', 'unknown')}")
                            raise
                else:
                    with self._driver.session(database=self.database) as session:
                        session.run(query, nodes=cleaned_batch)
                        logger.info(f"Successfully executed node import query for batch of {len(cleaned_batch)} nodes")
                
                logger.info(f"Imported batch of {len(batch)} {label} nodes ({i+1}-{i+len(batch)} of {len(nodes)})")
            
            # Verify the number of nodes actually imported
            async with self._driver.session(database=self.database) as session:
                result = await session.run(f"MATCH (n:{label}) RETURN count(n) as count")
                record = await result.single()
                actual_count = record["count"] if record else 0
                logger.info(f"Actual {label} node count after import: {actual_count}")
            
            logger.info(f"Successfully imported {len(nodes)} {label} nodes")
            return True
        except Exception as e:
            logger.error(f"Error importing {label} nodes: {str(e)}")
            return False


    
    def _clean_relationship_properties(self, relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Clean relationship properties to make them compatible with Neo4j.
        
        Args:
            relationships: List of relationship dictionaries
            
        Returns:
            Cleaned relationship dictionaries
        """
        cleaned_relationships = []
        
        for rel in relationships:
            cleaned_rel = rel.copy()
            
            # Remove any None values
            cleaned_rel = {k: v for k, v in cleaned_rel.items() if v is not None}
            
            # Handle potential missing fields
            if 'relationship_id' not in cleaned_rel:
                # Generate a random ID if missing
                cleaned_rel['relationship_id'] = f"rel_{uuid.uuid4().hex[:8]}"
                #logger.warning(f"Added missing relationship_id: {cleaned_rel['relationship_id']}")
            
            # Ensure required fields exist
            required_fields = ['source_id', 'target_id', 'type']
            missing_fields = [field for field in required_fields if field not in cleaned_rel]
            if missing_fields:
                logger.warning(f"Relationship missing required fields: {missing_fields}. Relationship: {cleaned_rel}")
                
                # Try to fill missing fields from alternative properties
                if 'source_id' not in cleaned_rel and 'source' in cleaned_rel:
                    cleaned_rel['source_id'] = cleaned_rel['source']
                    
                if 'target_id' not in cleaned_rel and 'target' in cleaned_rel:
                    cleaned_rel['target_id'] = cleaned_rel['target']
                
                if 'type' not in cleaned_rel:
                    cleaned_rel['type'] = 'RELATES_TO'
            
            # Ensure source_id and target_id exist at this point, or skip the relationship
            if not cleaned_rel.get('source_id') or not cleaned_rel.get('target_id'):
                logger.warning(f"Skipping relationship with missing source_id or target_id after cleanup: {cleaned_rel}")
                continue
                    
            # Handle vector embedding if present
            if 'vector_embedding' in cleaned_rel and not cleaned_rel['vector_embedding']:
                cleaned_rel.pop('vector_embedding')
            
            # Ensure keywords is a list
            if 'keywords' in cleaned_rel and not isinstance(cleaned_rel['keywords'], list):
                cleaned_rel['keywords'] = list(cleaned_rel['keywords']) if cleaned_rel['keywords'] else []
            
            # Ensure all keys are strings (Neo4j requirement)
            cleaned_rel = {str(k): v for k, v in cleaned_rel.items()}
                
            cleaned_relationships.append(cleaned_rel)
        
        return cleaned_relationships
    async def import_relationships(self, relationships: List[Dict[str, Any]], source_label: str, target_label: str, batch_size: int = 1000) -> bool:
        """
        Import relationships into Neo4j in batches with optimized performance for large graphs.
        
        Args:
            relationships: List of relationship dictionaries
            source_label: Label for source nodes
            target_label: Label for target nodes
            batch_size: Number of relationships to import in each batch
                
        Returns:
            True if import is successful, False otherwise
        """
        if not relationships:
            logger.warning(f"No relationships to import from {source_label} to {target_label}")
            return True
                
        try:
            # Track statistics
            total_processed = 0
            total_imported = 0
            total_skipped = 0
            
            # Log sample relationship for debugging
            if relationships:
                logger.info(f"Sample relationship structure: {relationships[0]}")
            
            # Pre-process relationships once to filter obvious duplicates and invalid entries
            filtered_relationships = []
            seen_pairs = set()
            
            for rel in relationships:
                # Clean the relationship (just extract what we need to check uniqueness)
                source_id = rel.get('source_id')
                target_id = rel.get('target_id')
                
                # Skip if missing required fields
                if not source_id or not target_id:
                    total_skipped += 1
                    continue
                
                # Check for duplicates within this batch
                rel_key = (source_id, target_id, rel.get('type', 'RELATES_TO'))
                if rel_key in seen_pairs:
                    total_skipped += 1
                    continue
                    
                seen_pairs.add(rel_key)
                filtered_relationships.append(rel)
            
            logger.info(f"Pre-processing: {len(filtered_relationships)} unique relationships after filtering duplicates")
            
            # Process relationships in batches
            for i in range(0, len(filtered_relationships), batch_size):
                batch = filtered_relationships[i:i + batch_size]
                
                # Clean relationships for Neo4j compatibility
                cleaned_batch = self._clean_relationship_properties(batch)
                # Group by relationship type for better performance
                relationships_by_type = {}
                for rel in cleaned_batch:
                    rel_type = rel.get('type', 'RELATES_TO')
                    if rel_type not in relationships_by_type:
                        relationships_by_type[rel_type] = []
                    relationships_by_type[rel_type].append(rel)

                # Process each relationship type in a separate query
                for rel_type, type_batch in relationships_by_type.items():
                    # Use APOC for bulk imports if available, otherwise use standard Cypher
                    import_query = f"""
                    UNWIND $relationships AS rel
                    MATCH (source:{source_label} {{entity_id: rel.source_id}})
                    MATCH (target:{target_label} {{entity_id: rel.target_id}})
                    MERGE (source)-[r:{rel_type}]->(target)
                    ON CREATE SET r = rel, r._imported_at = timestamp()
                    ON MATCH SET r = rel, r._updated_at = timestamp()
                    RETURN count(r) as count
                    """
                
#                 # Group by relationship type for better performance
#                 relationships_by_type = {}
#                 for rel in cleaned_batch:
#                     rel_type = rel.get('type', 'RELATES_TO')
#                     if rel_type not in relationships_by_type:
#                         relationships_by_type[rel_type] = []
#                     relationships_by_type[rel_type].append(rel)
                
#                 # Process each relationship type in a separate query
#                 for rel_type, type_batch in relationships_by_type.items():
#                     # Use APOC for bulk imports if available, otherwise use standard Cypher
#                     # Here we use MERGE directly and let Neo4j handle existence checking
#                     import_query = """
# UNWIND $relationships AS rel
# MATCH (source:{source_label} {{entity_id: rel.source_id}}), 
#     (target:{target_label} {{entity_id: rel.target_id}})
# MERGE (source)-[r:{relationship_type}]->(target)
# """.format(
#     source_label=source_label, 
#     target_label=target_label, 
#     relationship_type=rel_type
# )              
                    
                    try:
                        if self.use_async:
                            async with self._driver.session(database=self.database) as session:
                                # Use an explicit transaction for better control
                                result = await session.run(import_query, relationships=type_batch)
                                summary = await result.consume()
                                
                                # Get count of relationships created
                                if hasattr(summary, 'counters'):
                                    created = summary.counters.relationships_created
                                    total_imported += created
                                    logger.info(f"Successfully imported {created} new {rel_type} relationships")
                                else:
                                    # Fallback if summary counters not available
                                    logger.info(f"Imported batch of {rel_type} relationships (count unknown)")
                                    total_imported += len(type_batch)
                        else:
                            # Add synchronous version if needed
                            pass
                    except Exception as e:
                        logger.error(f"Error importing {rel_type} relationships: {str(e)}")
                        # Continue with next batch despite errors
                
                total_processed += len(batch)
                logger.info(f"Progress: {total_processed}/{len(filtered_relationships)} relationships processed")
                
                # Optional: Add a small delay between batches to reduce database load
                if i + batch_size < len(filtered_relationships):
                    await asyncio.sleep(0.1)
            
            # Get final relationship count (optional - can be skipped for very large graphs)
            try:
                async with self._driver.session(database=self.database) as session:
                    result = await session.run(f"MATCH (:{source_label})-[r]->(:{target_label}) RETURN count(r) as count")
                    record = await result.single()
                    actual_count = record["count"] if record else 0
                    logger.info(f"Final relationship count from {source_label} to {target_label}: {actual_count}")
            except Exception as e:
                logger.warning(f"Error getting final relationship count: {str(e)}")
            
            logger.info(f"Import summary: {total_processed} processed, ~{total_imported} imported, {total_skipped} skipped")
            return True
        except Exception as e:
            logger.error(f"Error importing relationships from {source_label} to {target_label}: {str(e)}")
            return False
    
    def _clean_node_properties(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Clean node properties to make them compatible with Neo4j.
        
        Args:
            nodes: List of node dictionaries
            
        Returns:
            Cleaned node dictionaries
        """
        cleaned_nodes = []
        
        for node in nodes:
            cleaned_node = node.copy()
            
            # Remove any None values
            cleaned_node = {k: v for k, v in cleaned_node.items() if v is not None}
            
            # Verify required fields
            if 'entity_id' not in cleaned_node:
                logger.warning(f"Node missing entity_id, skipping: {cleaned_node.get('entity_name', 'unknown')}")
                continue
                
            # Ensure entity_type exists
            if 'entity_type' not in cleaned_node:
                cleaned_node['entity_type'] = 'CONCEPT'
                
            # Handle vector_embedding if empty
            if 'vector_embedding' in cleaned_node and not cleaned_node['vector_embedding']:
                cleaned_node.pop('vector_embedding')
            
            # Ensure all keys are strings (Neo4j requirement)
            cleaned_node = {str(k): v for k, v in cleaned_node.items()}
            
            cleaned_nodes.append(cleaned_node)
        
        return cleaned_nodes
    
    async def import_knowledge_graph(self, graph_data: Dict[str, Any]) -> bool:
        """
        Import the entire knowledge graph into Neo4j.
        
        Args:
            graph_data: Dictionary containing nodes and edges
            
        Returns:
            True if import is successful, False otherwise
        """
        try:
            # Setup schema
            schema_success = await self.setup_schema()
            if not schema_success:
                logger.warning("Schema setup had issues, but continuing with import")
            
            # Split nodes by level
            level1_nodes = [node for node in graph_data.get('nodes', []) if node.get('knowledge_level') == 1]
            level2_nodes = [node for node in graph_data.get('nodes', []) if node.get('knowledge_level') == 2]
            
            # Import Level 1 nodes
            level1_success = await self.import_nodes(level1_nodes, 'Level1')
            
            # Import Level 2 nodes
            level2_success = await self.import_nodes(level2_nodes, 'Level2')
            
            if not level1_success or not level2_success:
                logger.error("Failed to import all nodes, aborting relationship import")
                return False
            
            # Split relationships by level
            level1_edges = [edge for edge in graph_data.get('edges', []) 
                           if edge.get('knowledge_level') == 1 and edge.get('type') != 'REFERENCES']
            
            level2_edges = [edge for edge in graph_data.get('edges', []) 
                           if edge.get('knowledge_level') == 2]
            
            cross_level_edges = [edge for edge in graph_data.get('edges', []) 
                               if edge.get('type') == 'REFERENCES']
            
            # Import Level 1 relationships
            level1_rel_success = await self.import_relationships(level1_edges, 'Level1', 'Level1')
            
            # Import Level 2 relationships
            level2_rel_success = await self.import_relationships(level2_edges, 'Level2', 'Level2')
            
            # Import cross-level relationships
            cross_level_success = await self.import_relationships(cross_level_edges, 'Level1', 'Level2')
            
            if not level1_rel_success or not level2_rel_success or not cross_level_success:
                logger.warning("Some relationships failed to import")
            
            logger.info("Knowledge graph import completed")
            
            # Log some statistics
            stats = {
                "nodes": {
                    "level1": len(level1_nodes),
                    "level2": len(level2_nodes),
                    "total": len(level1_nodes) + len(level2_nodes)
                },
                "relationships": {
                    "level1": len(level1_edges),
                    "level2": len(level2_edges),
                    "cross_level": len(cross_level_edges),
                    "total": len(level1_edges) + len(level2_edges) + len(cross_level_edges)
                }
            }
            
            logger.info(f"Import statistics: {json.dumps(stats)}")
            
            return True
        except Exception as e:
            logger.error(f"Error importing knowledge graph: {str(e)}")
            return False
    
    async def clear_database(self) -> bool:
        """
        Clear all data from the Neo4j database.
        
        WARNING: This deletes all nodes and relationships!
        
        Returns:
            True if clearing is successful, False otherwise
        """
        try:
            query = "MATCH (n) DETACH DELETE n"
            
            if self.use_async:
                async with self._driver.session(database=self.database) as session:
                    await session.run(query)
            else:
                with self._driver.session(database=self.database) as session:
                    session.run(query)
            
            logger.info("Successfully cleared all data from Neo4j database")
            return True
        except Exception as e:
            logger.error(f"Error clearing Neo4j database: {str(e)}")
            return False
    
    async def get_graph_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge graph in Neo4j.
        
        Returns:
            Dictionary with graph statistics
        """
        try:
            queries = {
                "total_nodes": "MATCH (n) RETURN count(n) AS count",
                "level1_nodes": "MATCH (n:Level1) RETURN count(n) AS count",
                "level2_nodes": "MATCH (n:Level2) RETURN count(n) AS count",
                "total_relationships": "MATCH ()-[r]->() RETURN count(r) AS count",
                "level1_relationships": "MATCH (:Level1)-[r]->(:Level1) RETURN count(r) AS count",
                "level2_relationships": "MATCH (:Level2)-[r]->(:Level2) RETURN count(r) AS count",
                "cross_level_relationships": "MATCH (:Level1)-[r]->(:Level2) RETURN count(r) AS count"
            }
            
            stats = {}
            
            if self.use_async:
                async with self._driver.session(database=self.database) as session:
                    for key, query in queries.items():
                        result = await session.run(query)
                        record = await result.single()
                        stats[key] = record["count"] if record else 0
            else:
                with self._driver.session(database=self.database) as session:
                    for key, query in queries.items():
                        result = session.run(query)
                        record = result.single()
                        stats[key] = record["count"] if record else 0
            
            logger.info(f"Graph statistics retrieved: {json.dumps(stats)}")
            return stats
        except Exception as e:
            logger.error(f"Error getting graph statistics: {str(e)}")
            return {}

    async def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Execute a custom Cypher query against the Neo4j database.
        
        Args:
            query: Cypher query to execute
            params: Parameters for the query
            
        Returns:
            List of query result records as dictionaries
        """
        params = params or {}
        results = []
        
        try:
            if self.use_async:
                async with self._driver.session(database=self.database) as session:
                    result = await session.run(query, params)
                    # records = await result.fetch()
                    # for record in records:
                    #     results.append(dict(record))
                    records = await result.data()
            else:
                with self._driver.session(database=self.database) as session:
                    result = session.run(query, params)
                    # for record in result:
                    #     results.append(dict(record))
                    records = result.data()
            
            return records
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            return []



async def import_graph_from_file(file_path: str, neo4j_client: Neo4jClient) -> bool:
    """
    Import a knowledge graph from a JSON file into Neo4j.
    
    Args:
        file_path: Path to the JSON file containing the graph data
        neo4j_client: Initialized Neo4jClient instance
        
    Returns:
        True if import is successful, False otherwise
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)
        
        if 'nodes' not in graph_data or 'edges' not in graph_data:
            logger.error(f"Invalid graph data: missing 'nodes' or 'edges' key")
            return False
        
        success = await neo4j_client.import_knowledge_graph(graph_data)
        
        if success:
            stats = await neo4j_client.get_graph_statistics()
            logger.info(f"Import completed. Graph statistics: {json.dumps(stats)}")
        
        return success
    except Exception as e:
        logger.error(f"Error importing graph from file: {str(e)}")
        return False


async def main():
    """Example main function demonstrating the Neo4jClient usage."""
    client = Neo4jClient()
    
    try:
        connected = await client.verify_connectivity()
        if not connected:
            logger.error("Failed to connect to Neo4j, exiting")
            return

        schema_success = await client.setup_schema()
        if not schema_success:
            logger.warning("Schema setup had issues")
        

        graph_path = "output/combined_graph.json"
        import_success = await import_graph_from_file(graph_path, client)
        
        if import_success:
            logger.info("Graph successfully imported")
            

            query_result = await client.execute_query(
                "MATCH (n:Level1)-[r:REFERENCES]->(m:Level2) RETURN n.name, m.name LIMIT 5"
            )
            logger.info(f"Sample cross-level references: {query_result}")
        
    finally:
        await client.close()


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    asyncio.run(main())