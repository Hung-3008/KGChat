"""
Dual-Level Knowledge Graph Retriever

This module retrieves relevant information from the two-level knowledge graph
based on high-level and low-level keywords extracted from user queries.
It uses vector similarity search to find relevant concepts in Level 1,
then traverses connections to more specific Level 2 nodes.
"""
import logging
import asyncio
from typing import List, Dict, Any, Optional, Set, Union, Tuple

from backend.db.neo4j_client import Neo4jClient
from backend.db.vector_db import VectorDBClient
from backend.utils.logging import get_logger

# Configure logger
logger = get_logger(__name__)

async def retrieve_from_knowledge_graph(
    high_level_keywords: List[str],
    low_level_keywords: List[str],
    neo4j_client: Neo4jClient,
    ollama_client: Any,
    qdrant_client: Optional[VectorDBClient] = None,
    top_k: int = 5,
    max_distance: float = 0.8,
    similarity_threshold: float = 0.7
) -> Dict[str, Any]:
    """
    Retrieve relevant information from the two-level knowledge graph.
    
    Args:
        high_level_keywords: High-level keywords from the query
        low_level_keywords: Low-level keywords from the query
        neo4j_client: Neo4j database client
        ollama_client: Ollama client for embedding generation
        qdrant_client: Optional Vector database client for similarity search
        top_k: Number of top results to retrieve for each keyword
        max_distance: Maximum vector distance for retrievals
        similarity_threshold: Minimum similarity score threshold
        
    Returns:
        Dictionary with retrieved context from both knowledge graph levels
    """
    retrieval_context = {
        "level1_nodes": [],
        "level2_nodes": [],
        "relationships": [],
        "sources": [],
        "combined_text": ""
    }
    
    if not high_level_keywords and not low_level_keywords:
        logger.warning("No keywords provided for knowledge graph retrieval")
        return retrieval_context
    
    logger.info(f"Retrieving knowledge with high-level keywords: {high_level_keywords}")
    logger.info(f"Retrieving knowledge with low-level keywords: {low_level_keywords}")
    
    all_keywords = high_level_keywords + low_level_keywords
    
    try:
        # STEP 1: Generate embeddings for all keywords
        embeddings = await ollama_client.embed(all_keywords)
        
        #STEP 2: Retrieve relevant Level 1 nodes using vector similarity
        level1_entities = await retrieve_level1_nodes(
            embeddings,
            qdrant_client,
            neo4j_client,
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )
        
        # Store Level 1 nodes in context
        retrieval_context["level1_nodes"] = level1_entities
        
        # STEP 3: For each retrieved Level 1 node, find referenced Level 2 nodes
        level2_entities, relationships = await retrieve_level2_references(
            level1_entities,
            neo4j_client,
            max_references=top_k
        )
        
        # Store Level 2 nodes and relationships in context
        retrieval_context["level2_nodes"] = level2_entities
        retrieval_context["relationships"] = relationships
        
        # STEP 4: Format the retrieved information into a combined text
        combined_text = format_retrieval_results(
            level1_entities, 
            level2_entities,
            relationships
        )
        
        retrieval_context["combined_text"] = combined_text
        
        return retrieval_context
        
    except Exception as e:
        logger.error(f"Error during knowledge graph retrieval: {str(e)}")
        return retrieval_context


async def retrieve_level1_nodes(
    embeddings: List[List[float]],
    qdrant_client: Any,
    neo4j_client: Neo4jClient,
    top_k: int = 5,
    similarity_threshold: float = 0.7
) -> List[Dict[str, Any]]:
    """
    Retrieve Level 1 nodes using vector similarity search.
    
    Args:
        embeddings: List of embedding vectors for keywords
        qdrant_client: Vector database client for similarity search
        neo4j_client: Neo4j database client for retrieving node data
        top_k: Number of top results to retrieve for each keyword
        similarity_threshold: Minimum similarity score threshold
    
    Returns:
        List of retrieved Level 1 node dictionaries
    """
    retrieved_entities = []
    unique_entity_ids = set()
    
    try:
        for embedding in embeddings:
            try:
                # Query Qdrant for similar vectors
                similar_nodes = qdrant_client.query_points(
                    collection_name="level1_nodes",
                    query=embedding,
                    limit=top_k
                )
                
                logger.info(f"Found {len(similar_nodes.points) if hasattr(similar_nodes, 'points') else 0} similar Level 1 nodes")
                
                for node in similar_nodes.points:
                    # Get original entity ID from payload
                    entity_id = node.payload.get("original_entity_id")
                    
                    if entity_id and entity_id not in unique_entity_ids:
                        # Retrieve specific node data from Neo4j
                        query = """
                        MATCH (n:Level1 {entity_id: $entity_id})
                        RETURN n.entity_id as entity_id, n.entity_type as entity_type, 
                               n.description as description, n.name as name
                        """
                        
                        results = await neo4j_client.execute_query(query, {"entity_id": entity_id})
                        
                        if results and len(results) > 0:
                            # Create node data with only the required fields
                            node_data = {
                                "entity_id": results[0].get("entity_id", entity_id),
                                "entity_type": results[0].get("entity_type", "CONCEPT"),
                                "description": results[0].get("description", ""),
                                "name": results[0].get("entity_id", entity_id),
                                "similarity_score": node.score
                            }
                            
                            retrieved_entities.append(node_data)
                            unique_entity_ids.add(entity_id)
                            
            except Exception as e:
                logger.error(f"Error retrieving similar nodes: {str(e)}")
        
        # Sort by similarity score
        retrieved_entities.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
        
        # Limit to top_k * 2 most relevant nodes overall
        max_nodes = top_k * 2
        if len(retrieved_entities) > max_nodes:
            retrieved_entities = retrieved_entities[:max_nodes]
        
        logger.info(f"Retrieved {len(retrieved_entities)} unique Level 1 nodes")
        return retrieved_entities
    
    except Exception as e:
        logger.error(f"Error in Level 1 node retrieval: {str(e)}")
        return []


async def retrieve_level2_references(
    level1_nodes: List[Dict[str, Any]],
    neo4j_client: Neo4jClient,
    max_references: int = 5
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Retrieve Level 2 nodes referenced by Level 1 nodes.
    
    Args:
        level1_nodes: List of Level 1 node dictionaries
        neo4j_client: Neo4j database client
        max_references: Maximum number of references to retrieve per Level 1 node
    
    Returns:
        Tuple of (level2_nodes, relationships) lists
    """
    level2_nodes = []
    relationships = []
    unique_level2_ids = set()
    unique_relationship_ids = set()
    
    try:
        for level1_node in level1_nodes:
            entity_id = level1_node.get("entity_id")
            if not entity_id:
                continue
                
            # Retrieve Level 2 nodes referenced by this Level 1 node focusing on name and description
            query = """
            MATCH (l1:Level1 {entity_id: $entity_id})-[r:REFERENCES]->(l2:Level2)
            RETURN l2.name AS name, l2.entity_type AS entity_type, l2.description AS description
            LIMIT $limit
            """
            
            results = await neo4j_client.execute_query(
                query, 
                {"entity_id": entity_id, "limit": max_references}
            )
            
            for record in results:
                # Create Level 2 node data with only the required fields
                level2_data = {
                    "name": record.get("name", "Unknown"),
                    "entity_type": record.get("entity_type", "CONCEPT"),
                    "description": record.get("description", "")
                }
                
                # Generate a unique ID for the node based on name
                level2_id = record.get("name", "")
                
                # Add Level 2 node if not already added
                if level2_id and level2_id not in unique_level2_ids:
                    level2_nodes.append(level2_data)
                    unique_level2_ids.add(level2_id)
                
                # Create basic relationship data
                rel_id = f"{entity_id}_to_{level2_id}"
                if rel_id and rel_id not in unique_relationship_ids:
                    rel_data = {
                        "source_id": entity_id,
                        "target_name": record.get("name", "Unknown"),
                        "source_name": level1_node.get("name", "Unknown"),
                        "type": "REFERENCES",
                        "description": f"{level1_node.get('name', 'Unknown')} is related to {record.get('name', 'Unknown')}"
                    }
                    
                    relationships.append(rel_data)
                    unique_relationship_ids.add(rel_id)
        
        logger.info(f"Retrieved {len(level2_nodes)} Level 2 nodes and {len(relationships)} relationships")
        return level2_nodes, relationships
    
    except Exception as e:
        logger.error(f"Error retrieving Level 2 references: {str(e)}")
        return [], []


def format_retrieval_results(
    level1_nodes: List[Dict[str, Any]],
    level2_nodes: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]]
) -> str:
    """
    Format the retrieval results into a more informative structured text representation.
    
    Args:
        level1_nodes: List of Level 1 node dictionaries
        level2_nodes: List of Level 2 node dictionaries
        relationships: List of relationship dictionaries
    
    Returns:
        Formatted text representation of the retrieved information
    """
    # Group Level 2 nodes by the Level 1 nodes that reference them
    level1_to_level2 = {}
    
    # Create a dictionary to quickly look up Level 2 nodes by name
    level2_by_name = {node.get('name', 'Unknown'): node for node in level2_nodes}
    
    # Group relationships by source entity ID
    for rel in relationships:
        source_id = rel.get('source_id')
        target_name = rel.get('target_name')
        
        if source_id and target_name:
            if source_id not in level1_to_level2:
                level1_to_level2[source_id] = []
            
            # Add the target Level 2 node to the list if it exists
            if target_name in level2_by_name:
                level1_to_level2[source_id].append({
                    'name': target_name,
                    'node': level2_by_name[target_name],
                    'relationship': rel
                })
    
    # Format the text with each Level 1 node and its related Level 2 nodes
    sections = []
    
    # Add main content section with detailed information
    main_content = []
    
    for level1_node in level1_nodes:
        entity_id = level1_node.get('entity_id')
        entity_name = level1_node.get('name', 'Unknown').upper()
        entity_type = level1_node.get('entity_type', 'Unknown')
        entity_desc = level1_node.get('description', 'No description available')
        
        # Add Level 1 node info
        node_section = [
            f"## {entity_name} ({entity_type})",
            f"{entity_desc}",
            ""
        ]
        
        # Add related Level 2 nodes if any
        related_nodes = level1_to_level2.get(entity_id, [])
        if related_nodes:
            node_section.append(f"### Related Concepts:")
            for item in related_nodes:
                level2_node = item['node']
                level2_name = level2_node.get('name', 'Unknown')
                level2_type = level2_node.get('entity_type', 'CONCEPT')
                level2_desc = level2_node.get('description', '')
                
                # Truncate very long descriptions
                if len(level2_desc) > 300:
                    level2_desc = level2_desc[:297] + "..."
                
                node_section.append(f"* **{level2_name}** ({level2_type}): {level2_desc}")
            
            node_section.append("")
        
        main_content.extend(node_section)
    
    # Create a key concepts summary section
    concept_summary = ["# KEY CONCEPTS", ""]
    
    # Group Level 1 nodes by entity type
    entity_types = {}
    for node in level1_nodes:
        entity_type = node.get('entity_type', 'Unknown')
        if entity_type not in entity_types:
            entity_types[entity_type] = []
        entity_types[entity_type].append(node)
    
    # Add a summary for each entity type
    for entity_type, nodes in entity_types.items():
        concept_summary.append(f"## {entity_type.upper()}S")
        for node in nodes:
            name = node.get('name', 'Unknown')
            desc = node.get('description', '')
            # Create a short description (first sentence or truncated)
            short_desc = desc.split('.')[0] if '.' in desc else desc[:50]
            concept_summary.append(f"* **{name}**: {short_desc}")
        concept_summary.append("")
    
    # Add a relationships summary
    relationship_summary = ["# RELATIONSHIPS", ""]
    
    # Group relationships by type
    rel_types = {}
    for rel in relationships:
        rel_type = rel.get('type', 'RELATED_TO')
        if rel_type not in rel_types:
            rel_types[rel_type] = []
        rel_types[rel_type].append(rel)
    
    # Add a summary for each relationship type
    for rel_type, rels in rel_types.items():
        relationship_summary.append(f"## {rel_type}")
        # List only unique source-target pairs to avoid repetition
        unique_pairs = set()
        for rel in rels:
            source = rel.get('source_name', 'Unknown')
            target = rel.get('target_name', 'Unknown')
            pair = f"{source} → {target}"
            if pair not in unique_pairs:
                unique_pairs.add(pair)
                relationship_summary.append(f"* {pair}")
        relationship_summary.append("")
    
    # Combine all sections
    sections.append("\n".join(concept_summary))
    sections.append("\n".join(relationship_summary))
    sections.append("# DETAILED INFORMATION\n")
    sections.append("\n".join(main_content))
    
    return "\n".join(sections)