"""
Dual-Level Knowledge Graph Retriever

This module implements a two-level knowledge graph retrieval system that:
1. Retrieves Level 1 nodes (papers/documents) using vector similarity search
2. Expands to Level 2 nodes (UMLS concepts) through graph traversal
3. Supports both width expansion (Level 1) and depth expansion (Level 2)
4. Evaluates and expands entities based on query relevance
"""
import logging
import asyncio
import numpy as np
from scipy import spatial
from typing import List, Dict, Any, Optional, Set, Union, Tuple
from pydantic import BaseModel

from db.neo4j_client import Neo4jClient
from db.vector_db import VectorDBClient
from utils.logging import get_logger
from core.pipeline.gemini.prompts import PROMPTS
from sklearn.metrics.pairwise import cosine_similarity

logger = get_logger(__name__)


async def retrieve_from_knowledge_graph(
    query: str,
    intent: str,
    high_level_keywords: List[str],
    low_level_keywords: List[str],
    neo4j_client: Neo4jClient,
    ollama_client: Any,
    qdrant_client: Optional[VectorDBClient] = None,
    gemini_client: Optional[Any] = None,
    top_k: int = 5,
    max_distance: float = 0.8,
    similarity_threshold: float = 0.7,
    expansion_width: int = 20,
    expansion_depth: int = 20
) -> Dict[str, Any]:
    """
    Main function to retrieve information from the two-level knowledge graph.

    Args:
        query: User's query
        intent: Query intent
        high_level_keywords: List of high-level keywords
        low_level_keywords: List of low-level keywords
        neo4j_client: Neo4j database client
        ollama_client: Ollama client for embeddings
        qdrant_client: Vector database client
        gemini_client: Gemini client for evaluations
        top_k: Minimum number of nodes to retrieve
        max_distance: Maximum distance for vector similarity
        similarity_threshold: Threshold for similarity search
        expansion_width: Maximum width for Level 1 node expansion
        expansion_depth: Maximum depth for Level 2 node expansion

    Returns:
        Dictionary containing retrieval context
    """
    # Validate input parameters
    if not isinstance(high_level_keywords, list) or not isinstance(low_level_keywords, list):
        raise ValueError("Keywords must be provided as lists")

    if not neo4j_client or not ollama_client:
        raise ValueError("Required clients not provided")

    retrieval_context = {
        "high_level1_nodes": [],
        "low_level1_nodes": [],
        "high_level2_nodes": [],
        "low_level2_nodes": [],
        "relationships": [],
        "sources": [],
        "high_combined_text": "",
        "low_combined_text": "",

    }

    if not high_level_keywords and not low_level_keywords:
        logger.warning("No keywords provided for knowledge graph retrieval")
        return retrieval_context

    try:
        # STEP 1: Generate embeddings
        logger.info(f"Generating embeddings for {len(high_level_keywords) + len(low_level_keywords)} keywords")
        #embeddings = await ollama_client.embed(high_level_keywords + low_level_keywords)
        high_embeddings = await ollama_client.embed(high_level_keywords)
        low_embeddings = await ollama_client.embed(low_level_keywords)


        # STEP 2: Retrieve High and Low Level 1 nodes
        logger.info("Retrieving high-Level 1 nodes...")
        high_level1_entities = await retrieve_level1_nodes(
            embeddings=high_embeddings,
            qdrant_client=qdrant_client,
            neo4j_client=neo4j_client,
            min_width=top_k,
            max_width=expansion_width,
            similarity_threshold=similarity_threshold
        )

        logger.info("Retrieving low-Level 1 nodes...")
        low_level1_entities = await retrieve_level1_nodes(
            embeddings=low_embeddings,
            qdrant_client=qdrant_client,
            neo4j_client=neo4j_client,
            min_width=top_k,
            max_width=expansion_width,
            similarity_threshold=similarity_threshold
        )
        logger.info(f"Retrieved {len(high_level1_entities)} high-Level 1 nodes and {len(low_level1_entities)} low-Level 1 nodes")

        # STEP 3: Retrieve High and Low Level 2 nodes 
        logger.info("Retriving High-Level 2 nodes...")
        high_level2_entities, high_relationships = await retrieve_level2_references(
            level1_nodes=high_level1_entities,
            neo4j_client=neo4j_client,
            min_depth=top_k,
            max_depth=expansion_depth
        )

        logger.info("Retrieving Low-Level 2 nodes...")
        low_level2_entities, low_relationships = await retrieve_level2_references(
            level1_nodes=low_level1_entities,
            neo4j_client=neo4j_client,
            min_depth=top_k,
            max_depth=expansion_depth
        )
        logger.info(f"Retrieved {len(high_level2_entities)} high-Level 2 nodes and {len(low_level2_entities)} low-Level 2 nodes")

        # Step 4: Format results
        logger.info("Formatting retrieval results")
        high_combined_text = format_retrieval_results(
            level1_nodes=high_level1_entities,
            level2_nodes=high_level2_entities,
            relationships=high_relationships
        )

        low_combined_text = format_retrieval_results(
            level1_nodes=low_level1_entities,
            level2_nodes=low_level2_entities,
            relationships=low_relationships
        )

        retrieval_context.update({
            "high_level1_nodes": high_level1_entities,
            "low_level1_nodes": low_level1_entities,
            "high_level2_nodes": high_level2_entities,
            "low_level2_nodes": low_level2_entities,
            "relationships": high_relationships + low_relationships,
            "high_combined_text": high_combined_text,
            "low_combined_text": low_combined_text,
        })

        return retrieval_context

    except Exception as e:
        logger.error(f"Error during knowledge graph retrieval: {str(e)}")
        return retrieval_context


async def retrieve_level1_nodes(
    embeddings: List[List[float]],
    qdrant_client: Any,  # Vẫn giữ tham số này để tương thích
    neo4j_client: Neo4jClient,
    min_width: int = 5,
    max_width: int = 20,
    similarity_threshold: float = 0.7
) -> List[Dict[str, Any]]:
    """
    Retrieve Level 1 nodes using vector similarity search in Neo4j.
    
    Args:
        embeddings: List of embedding vectors for keywords
        qdrant_client: Vector database client (kept for compatibility)
        neo4j_client: Neo4j database client
        min_width: Minimum number of nodes to retrieve per embedding
        max_width: Maximum number of nodes to retrieve in total
        similarity_threshold: Minimum similarity score threshold
    
    Returns:
        List of retrieved Level 1 node dictionaries
    """
    retrieved_entities = []
    unique_entity_ids = set()
    
    try:
        try:
            index_check = await neo4j_client.execute_query(
                "SHOW INDEXES WHERE name = 'level1_vector_embedding'"
            )
            index_state = "ONLINE" if any(index_check) else ""
            
            if index_state != 'ONLINE':
                logger.warning(f"Vector index 'level1_vector_embedding' may not be online. Vector search may not work properly.")
        except Exception as e:
            logger.warning(f"Could not verify vector index: {str(e)}")
        
        for i, embedding in enumerate(embeddings):
            try:
                query = """
                CALL db.index.vector.queryNodes('level1_vector_embedding', $limit, $vector_param) 
                YIELD node, score 
                WHERE score >= $threshold
                RETURN 
                    node.entity_id as entity_id, 
                    node.entity_type as entity_type,
                    node.description as description,
                    COALESCE(node.name, node.entity_id) as name,
                    score as similarity_score
                """
                
                params = {
                    "vector_param": embedding,
                    "limit": min_width,
                    "threshold": similarity_threshold
                }
                
                results = await neo4j_client.execute_query(query, params)
                
                logger.info(f"Found {len(results)} similar Level 1 nodes for embedding {i+1}/{len(embeddings)}")
                
                for record in results:
                    entity_id = record.get("entity_id")
                    
                    if entity_id and entity_id not in unique_entity_ids:
                        node_data = {
                            "entity_id": entity_id,
                            "entity_type": record.get("entity_type", "CONCEPT"),
                            "description": record.get("description", ""),
                            "name": record.get("name", entity_id),
                            "similarity_score": record.get("similarity_score", 0)
                        }
                        
                        retrieved_entities.append(node_data)
                        unique_entity_ids.add(entity_id)
                        
            except Exception as e:
                logger.error(f"Error retrieving similar nodes for embedding {i+1}: {str(e)}")
                logger.error(f"Skipping this embedding and continuing with others")
        
        retrieved_entities.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
        
        if len(retrieved_entities) > max_width:
            retrieved_entities = retrieved_entities[:max_width]
        elif len(retrieved_entities) < min_width and len(embeddings) > 0:
            logger.warning(f"Retrieved only {len(retrieved_entities)} nodes, which is less than the minimum width {min_width}")
        
        logger.info(f"Retrieved {len(retrieved_entities)} unique Level 1 nodes in total")
        return retrieved_entities
    
    except Exception as e:
        logger.error(f"Error in Level 1 node retrieval: {str(e)}")
        return []



async def retrieve_level2_references(
    level1_nodes: List[Dict[str, Any]],
    neo4j_client: Neo4jClient,
    min_depth: int = 5,
    max_depth: int = 20
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Retrieve Level 2 nodes referenced by Level 1 nodes with depth expansion.
    """
    if not level1_nodes or not neo4j_client:
        logger.warning(
            "Missing required parameters for Level 2 node retrieval")
        return [], []

    level2_nodes = []
    relationships = []
    unique_level2_ids = set()
    unique_relationship_ids = set()

    try:
        for level1_node in level1_nodes:
            entity_id = level1_node.get("entity_id")
            if not entity_id:
                continue

            # Query Neo4j for connected Level 2 nodes
            results = await _get_connected_level2_nodes(neo4j_client, entity_id, max_depth)

            for record in results:
                level2_data = _create_level2_node_data(record)
                level2_id = level2_data.get("entity_id")

                if level2_id and level2_id not in unique_level2_ids:
                    level2_nodes.append(level2_data)
                    unique_level2_ids.add(level2_id)

                    # Create relationship
                    rel_data = _create_relationship_data(
                        level1_node, level2_data)
                    rel_id = rel_data.get("rel_id")

                    if rel_id and rel_id not in unique_relationship_ids:
                        relationships.append(rel_data)
                        unique_relationship_ids.add(rel_id)

        # Validate and limit results
        if len(level2_nodes) < min_depth:
            logger.warning(
                f"Insufficient Level 2 nodes found: {len(level2_nodes)} < {min_depth}")
        else:
            level2_nodes = level2_nodes[:max_depth]

        logger.info(
            f"Retrieved {len(level2_nodes)} Level 2 nodes and {len(relationships)} relationships")
        return level2_nodes, relationships

    except Exception as e:
        logger.error(f"Error retrieving Level 2 references: {str(e)}")
        return [], []


async def _get_node_details(neo4j_client: Neo4jClient, entity_id: str) -> Optional[Dict[str, Any]]:
    """Helper function to get node details from Neo4j."""
    query = """
    MATCH (n:Level1 {entity_id: $entity_id})
    RETURN n.entity_id as entity_id, 
           n.entity_type as entity_type, 
           n.description as description, 
           n.name as name
    """
    results = await neo4j_client.execute_query(query, {"entity_id": entity_id})
    return results[0] if results else None


async def _get_connected_level2_nodes(neo4j_client: Neo4jClient, entity_id: str, limit: int) -> List[Dict[str, Any]]:
    """Helper function to get connected Level 2 nodes."""
    query = """
    MATCH (l1:Level1 {entity_id: $entity_id})-[r:REFERENCES]->(l2:Level2)
    RETURN l2.name AS name, 
           l2.entity_type AS entity_type, 
           l2.description AS description,
           l2.entity_id AS entity_id
    LIMIT $limit
    """
    return await neo4j_client.execute_query(query, {"entity_id": entity_id, "limit": limit})


def _create_level2_node_data(record: Dict[str, Any]) -> Dict[str, Any]:
    """Helper function to create Level 2 node data structure."""
    return {
        "name": record.get("name", "Unknown"),
        "entity_type": record.get("entity_type", "CONCEPT"),
        "description": record.get("description", ""),
        "entity_id": record.get("entity_id", "")
    }


def _create_relationship_data(level1_node: Dict[str, Any], level2_data: Dict[str, Any]) -> Dict[str, Any]:
    """Helper function to create relationship data structure."""
    source_id = level1_node.get("entity_id", "")
    target_id = level2_data.get("entity_id", "")
    source_name = level1_node.get("name", "Unknown")
    target_name = level2_data.get("name", "Unknown")

    # Create a unique relationship ID
    rel_id = f"{source_id}_{target_id}" if source_id and target_id else ""

    return {
        "source_id": source_id,
        "target_id": target_id,
        "source_name": source_name,
        "target_name": target_name,
        "type": "REFERENCES",
        "description": f"{source_name} references {target_name}",
        "rel_id": rel_id  # Add the relationship ID
    }


async def retrieve_node_relationships(
    source_nodes: List[Dict[str, Any]],
    neo4j_client: Neo4jClient,
    max_references: int = 100
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Retrieve relationships and target nodes for a given list of source nodes.
    
    Args:
        source_nodes: List of source node dictionaries
        neo4j_client: Neo4j database client
        max_references: Maximum number of references to retrieve per source node
    
    Returns:
        Tuple of (target_nodes, relationships) lists
    """
    target_nodes = []
    relationships = []
    unique_target_ids = set()
    unique_relationship_ids = set()
    
    try:
        for source_node in source_nodes:
            entity_id = source_node.get('entity_id')
            if not entity_id:
                continue
            
            # Retrieve outgoing relationships and target nodes
            query = """
            MATCH (source {entity_id: $entity_id})-[r]->(target)
            RETURN 
                type(r) as relationship_type, 
                properties(r) as relationship_properties,
                target.entity_id as target_id,
                target.name as target_name,
                target.entity_type as target_type,
                target.description as target_description
            LIMIT $limit
            """
            
            results = await neo4j_client.execute_query(
                query, 
                {"entity_id": entity_id, "limit": max_references}
            )
            
            for record in results:
                # Process target node
                target_id = record.get('target_id')
                
                if target_id and target_id not in unique_target_ids:
                    target_node = {
                        "entity_id": target_id,
                        "name": record.get('target_name', 'Unknown'),
                        "entity_type": record.get('target_type', 'CONCEPT'),
                        "description": record.get('target_description', ''),
                        "vector_embedding": record.get('vector_embedding', []),
                    }
                    target_nodes.append(target_node)
                    unique_target_ids.add(target_id)
                
                # Process relationship
                relationship_properties = record.get('relationship_properties', {})
                relationship_type = record.get('relationship_type', 'RELATES_TO')
                
                relationship = {
                    "source_id": entity_id,
                    "target_id": target_id,
                    "type": relationship_type,
                    **relationship_properties
                }
                
                # Generate a unique relationship ID
                rel_id = f"{entity_id}_{target_id}_{relationship_type}"
                if rel_id not in unique_relationship_ids:
                    relationships.append(relationship)
                    unique_relationship_ids.add(rel_id)
        
        logger.info(f"Retrieved {len(target_nodes)} unique target nodes and {len(relationships)} relationships")
        return target_nodes, relationships
    
    except Exception as e:
        logger.error(f"Error retrieving node relationships: {str(e)}")
        return [], []

class EvaluationResult(BaseModel):
    is_sufficient: bool
    message: str

async def evaluate_and_expand_entities(
    query: str,
    level1_nodes: List[Dict[str, Any]],
    level2_nodes: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
    neo4j_client: Neo4jClient,
    ollama_client: Any,
    qdrant_client: Any,
    gemini_client: Any,
    min_level1_nodes: int = 5,
    min_level2_nodes: int = 5,
    max_iterations: int = 2,
    similarity_threshold: float = 0.7,
    expansion_width: int = 20,
    expansion_depth: int = 20,
    high_level_keywords: Optional[List[str]] = None,
    low_level_keywords: Optional[List[str]] = None,
    grounding: Optional[bool] = False

) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Evaluate and expand entities based on the query and the retrieved knowledge graph.

    Args:
        query: User query to evaluate against
        level1_nodes: List of Level 1 node dictionaries
        level2_nodes: List of Level 2 node dictionaries
        relationships: List of relationship dictionaries
        neo4j_client: Neo4j database client
        ollama_client: Ollama client for embeddings generation
        qdrant_client: Vector database client
        gemini_client: Gemini client for evaluation
        min_level1_nodes: Minimum number of Level 1 nodes required
        min_level2_nodes: Minimum number of Level 2 nodes required
        max_iterations: Maximum number of expansion iterations
        similarity_threshold: Threshold for similarity search
        expansion_width: Maximum width for Level 1 node expansion
        expansion_depth: Maximum depth for Level 2 node expansion

    Returns:
        Tuple of expanded Level 1 and Level 2 nodes
    """
    current_level1_count = len(level1_nodes)
    current_level2_count = len(level2_nodes)
    iteration = 0

    while iteration < max_iterations:
        # Format triplets for evaluation
        triplets = format_triplets_for_evaluation(
            level1_nodes,
            level2_nodes,
            relationships
        )

        evaluation_prompt = PROMPTS["evaluate_information"].format(
            query=query,
            triplets=triplets
        )

        try:
            evaluation_response = await gemini_client.generate(prompt=evaluation_prompt)
            if isinstance(evaluation_response, dict) and "message" in evaluation_response:
                response_text = evaluation_response["message"]["content"]
                is_sufficient = "yes" in response_text.lower()
            else:
                is_sufficient = False

            if is_sufficient:
                logger.info(
                    f"Enough information found after {iteration} iterations")
                break

            logger.info(
                f"Insufficient information found. Expanding knowledge graph")

            # Generate embeddings from query for expansion
            if current_level1_count < min_level1_nodes:
                # Generate embeddings from query
                logger.info(
                    f"Generating embeddings from query for Level 1 node expansion")
                
                keywords_embeddings = await ollama_client.embed(
                    high_level_keywords + low_level_keywords
                )
                additional_nodes, relationships = await retrieve_node_relationships(
                    source_nodes=level1_nodes,
                    neo4j_client=neo4j_client
                )

                for key in keywords_embeddings:
                    similarities = cosine_similarity(
                        np.array(key).reshape(1, -1),
                        np.array([node['vector_embedding'] for node in additional_nodes]).reshape(1, -1)
                    )
                    top_indices = np.argsort(similarities[0])[-expansion_width:]
                    
                    for index in top_indices:
                        additional_node = additional_nodes[index]
                        if additional_node not in level1_nodes:
                            level1_nodes.append(additional_node)
                            relationships.extend(await retrieve_node_relationships(
                                source_nodes=[additional_node],
                                neo4j_client=neo4j_client
                            ))
            iteration += 1

        except Exception as e:
            logger.error(f"Error in evaluation and expand entities: {str(e)}")
            break

    additional_info = ""
    
    if iteration > 0:
        # Final evaluation for logging purposes
        triplets = format_triplets_for_evaluation(
            level1_nodes,
            level2_nodes,
            relationships
        )
        evaluation_prompt = PROMPTS["evaluate_information"].format(
            query=query,
            triplets=triplets
        )
        try:
            evaluation_response = await gemini_client.generate(prompt=evaluation_prompt, format=EvaluationResult)
            evaluation_response = evaluation_response[0]
            if evaluation_response.is_sufficient:
                logger.info(f"Final evaluation: {evaluation_response.message[:150]}")
            else:
                if grounding:
                    internet_searched = await gemini_client.grounding(query)
                    internet_searched = internet_searched['message']['content']
                    additional_info = f"## Additional information:\n{internet_searched}"
                    triplets = triplets + "\n\n ## Additional information:" + internet_searched
                    evaluation_response = await gemini_client.generate(prompt=triplets)
                    logger.info(f"Final evaluation: {evaluation_response['message']['content']}")
                else:
                    logger.info(f"Final evaluation: {evaluation_response.message[:150]}")
                

        except Exception as e:
            logger.error(f"Error in final evaluation: {str(e)}")

    return level1_nodes, level2_nodes, additional_info


def format_triplets_for_evaluation(
    level1_nodes: List[Dict[str, Any]],
    level2_nodes: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]]
) -> str:
    """
    Format the triplets for evaluation in a comprehensive way.

    Args:
        level1_nodes: List of Level 1 node dictionaries
        level2_nodes: List of Level 2 node dictionaries
        relationships: List of relationship dictionaries

    Returns:
        String representation of triplets for evaluation
    """
    triplets = []

    # Format Level 1 nodes first
    triplets.append("# LEVEL 1 NODES (Papers/Documents)")
    for node in level1_nodes:
        name = node.get('name', 'Unknown')
        entity_type = node.get('entity_type', 'Unknown')
        description = node.get('description', '')
        triplets.append(f"{name}, {entity_type}, {description}")

    # Format Level 2 nodes next
    triplets.append("\n# LEVEL 2 NODES (UMLS Concepts)")
    for node in level2_nodes:
        name = node.get('name', 'Unknown')
        entity_type = node.get('entity_type', 'Unknown')
        description = node.get('description', '')
        triplets.append(f"{name}, {entity_type}, {description}")

    # Format relationships last
    triplets.append("\n# RELATIONSHIPS")
    for rel in relationships:
        source_name = rel.get('source_name', 'Unknown')
        target_name = rel.get('target_name', 'Unknown')
        rel_type = rel.get('type', 'RELATED_TO')
        rel_description = rel.get('description', '')

        # Include relationship description if available
        if rel_description:
            triplets.append(
                f"{source_name}, {rel_type}, {target_name}, {rel_description}")
        else:
            triplets.append(f"{source_name}, {rel_type}, {target_name}")

    return "\n".join(triplets)


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

    # Create a dictionary for quick lookup of Level 2 nodes by name
    level2_by_name = {node.get('name', 'Unknown')
                               : node for node in level2_nodes}

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
        entity_name = level1_node.get('name', 'Unknown')
        entity_type = level1_node.get('entity_type', 'Unknown')
        entity_desc = level1_node.get(
            'description', 'No description available')

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

                node_section.append(
                    f"* **{level2_name}** ({level2_type}): {level2_desc}")

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
        concept_summary.append(f"## {entity_type}S")
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
