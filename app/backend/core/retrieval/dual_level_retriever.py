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
from typing import List, Dict, Any, Optional, Set, Union, Tuple

from backend.db.neo4j_client import Neo4jClient
from backend.db.vector_db import VectorDBClient
from backend.utils.logging import get_logger
from backend.core.pipeline.gemini.prompts import PROMPTS

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
        "level1_nodes": [],
        "level2_nodes": [],
        "relationships": [],
        "sources": [],
        "combined_text": ""
    }

    if not high_level_keywords and not low_level_keywords:
        logger.warning("No keywords provided for knowledge graph retrieval")
        return retrieval_context

    try:
        # STEP 1: Generate embeddings
        logger.info(
            f"Generating embeddings for {len(high_level_keywords) + len(low_level_keywords)} keywords")
        embeddings = await ollama_client.embed(high_level_keywords + low_level_keywords)

        # STEP 2: Retrieve Level 1 nodes
        logger.info("Retrieving Level 1 nodes")
        level1_entities = await retrieve_level1_nodes(
            embeddings=embeddings,
            qdrant_client=qdrant_client,
            neo4j_client=neo4j_client,
            min_width=top_k,
            max_width=expansion_width,
            similarity_threshold=similarity_threshold
        )

        # STEP 3: Retrieve Level 2 nodes
        logger.info("Retrieving Level 2 nodes")
        level2_entities, relationships = await retrieve_level2_references(
            level1_nodes=level1_entities,
            neo4j_client=neo4j_client,
            min_depth=top_k,
            max_depth=expansion_depth
        )

        # STEP 4: Format results
        logger.info("Formatting retrieval results")
        combined_text = format_retrieval_results(
            level1_entities,
            level2_entities,
            relationships
        )

        retrieval_context.update({
            "level1_nodes": level1_entities,
            "level2_nodes": level2_entities,
            "relationships": relationships,
            "combined_text": combined_text
        })

        # Add validation and review if gemini_client is provided
        if gemini_client:
            # Add validation and review results to context
            retrieval_context.update({
                "level1_nodes": level1_entities,
                "level2_nodes": level2_entities,
                "relationships": relationships,
                "combined_text": combined_text,
            })

        return retrieval_context

    except Exception as e:
        logger.error(f"Error during knowledge graph retrieval: {str(e)}")
        return retrieval_context


async def retrieve_level1_nodes(
    embeddings: List[List[float]],
    qdrant_client: Any,
    neo4j_client: Neo4jClient,
    min_width: int = 5,
    max_width: int = 20,
    similarity_threshold: float = 0.7
) -> List[Dict[str, Any]]:
    """
    Retrieve Level 1 nodes using vector similarity search with width expansion.
    """
    if not embeddings or not qdrant_client:
        logger.warning(
            "Missing required parameters for Level 1 node retrieval")
        return []

    retrieved_entities = []
    unique_entity_ids = set()

    try:
        for embedding in embeddings:
            # Query Qdrant for similar vectors
            similar_nodes = qdrant_client.query_points(
                collection_name="level1_nodes",
                query=embedding,
                limit=max_width
            )

            # Process similar nodes
            for node in similar_nodes.points:
                if node.score < similarity_threshold:
                    continue

                entity_id = node.payload.get("original_entity_id")
                if not entity_id or entity_id in unique_entity_ids:
                    continue

                # Retrieve detailed node information from Neo4j
                node_data = await _get_node_details(neo4j_client, entity_id)
                if node_data:
                    node_data["similarity_score"] = node.score
                    retrieved_entities.append(node_data)
                    unique_entity_ids.add(entity_id)

        # Sort and limit results
        retrieved_entities.sort(key=lambda x: x.get(
            "similarity_score", 0), reverse=True)

        if len(retrieved_entities) < min_width:
            logger.warning(
                f"Insufficient Level 1 nodes found: {len(retrieved_entities)} < {min_width}")
        else:
            retrieved_entities = retrieved_entities[:max_width]

        logger.info(f"Retrieved {len(retrieved_entities)} Level 1 nodes")
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
    max_iterations: int = 3,
    similarity_threshold: float = 0.7,
    expansion_width: int = 20,
    expansion_depth: int = 20
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
                query_embeddings = await ollama_client.embed([query])

                additional_level1 = await retrieve_level1_nodes(
                    embeddings=query_embeddings,
                    qdrant_client=qdrant_client,
                    neo4j_client=neo4j_client,
                    min_width=min_level1_nodes - current_level1_count,
                    max_width=expansion_width,
                    similarity_threshold=similarity_threshold
                )

                # Add unique nodes only
                unique_ids = {node.get("entity_id")
                              for node in level1_nodes}
                new_nodes = [node for node in additional_level1
                             if node.get("entity_id") and node.get("entity_id") not in unique_ids]

                level1_nodes.extend(new_nodes)
                current_level1_count = len(level1_nodes)
                logger.info(f"Added {len(new_nodes)} new Level 1 nodes")

            if current_level2_count < min_level2_nodes:
                additional_level2, additional_relationships = await retrieve_level2_references(
                    level1_nodes=level1_nodes,
                    neo4j_client=neo4j_client,
                    min_depth=min_level2_nodes - current_level2_count,
                    max_depth=expansion_depth
                )

                # Add unique Level 2 nodes only
                unique_level2_ids = {node.get("entity_id")
                                     for node in level2_nodes}
                new_level2_nodes = [node for node in additional_level2
                                    if node.get("entity_id") and node.get("entity_id") not in unique_level2_ids]

                # Add unique relationships only
                unique_rel_ids = {rel.get("rel_id")
                                  for rel in relationships if rel.get("rel_id")}
                new_relationships = [rel for rel in additional_relationships
                                     if rel.get("rel_id") and rel.get("rel_id") not in unique_rel_ids]

                level2_nodes.extend(new_level2_nodes)
                relationships.extend(new_relationships)
                current_level2_count = len(level2_nodes)
                logger.info(
                    f"Added {len(new_level2_nodes)} new Level 2 nodes and {len(new_relationships)} relationships")

            iteration += 1

        except Exception as e:
            logger.error(f"Error in evaluation and expand entities: {str(e)}")
            break

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
            evaluation_response = await gemini_client.generate(prompt=evaluation_prompt)
            if isinstance(evaluation_response, dict) and "message" in evaluation_response:
                response_text = evaluation_response["message"]["content"]
                logger.info(f"Final evaluation response: {response_text}")
            else:
                logger.info("Could not get evaluation response text")
        except Exception as e:
            logger.error(f"Error in final evaluation: {str(e)}")

    return level1_nodes, level2_nodes


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
        entity_name = level1_node.get('name', 'Unknown').upper()
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
