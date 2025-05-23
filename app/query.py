import os 
import json
from dotenv import load_dotenv
from backend.llm.gemini_client import GeminiClient
from backend.db.neo4j_client import Neo4jClient
from backend.db.vector_db import VectorDBClient
from backend.llm.ollama_client import OllamaClient
from qdrant_client import QdrantClient
from backend.core.retrieval.query_analyzer import analyze_query
from backend.core.retrieval.keyword_extractor import extract_keywords
from backend.core.retrieval.dual_level_retriever import retrieve_from_knowledge_graph, format_retrieval_results, evaluate_and_expand_entities, format_triplets_for_evaluation
from typing import Dict, Any, List, Tuple
import time
load_dotenv()


async def run_query(query: str, conversation_history: List[Dict[str, str]], clients: Dict[str, Any], grounding=False, language="English"):

    time_start = time.time()
    # step 1: get intent, high_keywords, low_keywords
    query = "In what case we can not use HbA1c to predict Diabetes?"
    history = [
        {
            "role": "user",
            "content": "Hello, I'd like to ask some questions about diabetes."
        }
    ]
    intent = await analyze_query(
        query=query,
        conversation_history=history,
        client=clients["gemini_client"]
    )

    high_keywords, low_keywords = await extract_keywords(
        query=query,
        conversation_history=history,
        llm_client=clients["gemini_client"]
    )


    # Step 2: Get relevant nodes and relationships from Neo4j
    kg_context = await retrieve_from_knowledge_graph(
                query=query,
                intent=str(intent.name),
                high_level_keywords=high_keywords,
                low_level_keywords=low_keywords,
                neo4j_client=clients["neo4j_client"],
                ollama_client=clients["ollama_client"],
                qdrant_client=clients["qdrant_client"],
                gemini_client=clients["gemini"],
                similarity_threshold=0.7,
                expansion_width=10,
                expansion_depth=10
            )

    level1_nodes = kg_context["high_level1_nodes"] + kg_context["low_level1_nodes"]
    level2_nodes = kg_context["high_level2_nodes"] + kg_context["low_level2_nodes"]
    relationships = kg_context["relationships"]

    expanded_level1, expanded_level2, additional_info = await evaluate_and_expand_entities(
            query=query,
            level1_nodes=level1_nodes,
            level2_nodes=level2_nodes,
            relationships=relationships,
            neo4j_client=clients["neo4j_client"],
            ollama_client=clients["ollama_client"],
            qdrant_client=clients["qdrant_client"],
            gemini_client=clients["gemini"],
            min_level1_nodes=3,
            min_level2_nodes=5,
            max_iterations=2,
            similarity_threshold=0.7,
            expansion_width=10,
            expansion_depth=10, 
            grounding=grounding
        )
    formatted_triplets = format_retrieval_results(expanded_level1, expanded_level2, relationships)
    formatted_triplets += additional_info

    final_prompt = f"""You are DTG chatbot a specialized diabetes information assistant with access to both a knowledge graph of diabetes concepts and up-to-date web information.

    Goal: Generate anevidence-based response to the user's diabetes-related query using both the provided knowledge graph information.

    User Query: {query}

    Knowledge Graph Context:
    {formatted_triplets}

    Instructions:
    - Synthesize and critically evaluate information from both the knowledge graph
    - When information from different sources conflicts:
    * Compare the reliability and recency of each source
    * Weigh medical consensus over isolated findings
    * Explain the differences if they are significant
    - Prioritize information from peer-reviewed medical literature and authoritative health organizations
    - Clearly distinguish between established diabetes knowledge and emerging research
    - If contradictions exist between the knowledge graph and recent information, acknowledge this and explain the current understanding
    - When discussing treatments or management approaches, note the level of evidence supporting them
    - Use clinical reasoning to connect information to the user's specific query
    - If the available information is insufficient to fully answer the query, acknowledge these limitations
    - Format your response for readability with appropriate structure based on the requested length:
    * concise: Clear, focused response in 3-5 sentences that prioritizes the most clinically relevant information
    * detailed: Comprehensive explanation with relevant details, comparing different sources and explaining nuances


    Conversation History:
    {conversation_history[-3:] if conversation_history else ''}

    Important: 
    - Give your response in {language}.
    - Give polite, brief and understandable answers.
    """
    response = await clients["gemini"].generate(
        prompt=final_prompt,
        conversation_history=conversation_history,
        language=language
    )

    time_end = time.time()
    processing_time = time_end - time_start

    return {
        "query": query,
        "intent": str(intent.name),
        "response": response,
        "keywords": {
            "high_level": high_keywords,
            "low_level": low_keywords
        },
        "processing_time_seconds": processing_time,
        "sources": formatted_triplets
    }
