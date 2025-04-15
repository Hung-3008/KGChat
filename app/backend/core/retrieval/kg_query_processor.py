import json
import logging
import time
import asyncio
import os
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
from pydantic import BaseModel, Field

from backend.core.retrieval.query_analyzer import QueryIntent
from backend.core.retrieval.dual_level_retriever import retrieve_from_knowledge_graph
from backend.db.neo4j_client import Neo4jClient
from backend.utils.logging import get_logger

# Configure logger
logger = get_logger(__name__)

class ResponseFormat(BaseModel):
    """Pydantic model for structured response output."""
    response: str = Field(..., description="The response text answering the user's query")
    sources: List[str] = Field(default_factory=list, description="Source references used in the response")
    confidence: float = Field(default=0.0, description="Confidence score for the response")

class KnowledgeGraphQueryProcessor:
    """
    Processes queries against the two-level knowledge graph.
    
    Implements a comprehensive query processing pipeline that:
    1. Analyzes query intent
    2. Retrieves relevant information from knowledge graph
    3. Generates appropriate responses based on intent and context
    4. Handles different query types with specialized prompts
    """
    
    def __init__(
        self, 
        neo4j_client: Neo4jClient, 
        ollama_client: Any,
        gemini_client: Any,
        qdrant_client: Optional[Any] = None,
        max_tokens: int = 10000,
        top_k: int = 5,
        max_distance: float = 0.8
    ):
        """
        Initialize the query processor.
        
        Args:
            neo4j_client: Neo4j database client
            ollama_client: Ollama LLM client for embeddings
            gemini_client: Gemini LLM client for response generation
            qdrant_client: Optional Qdrant vector database client
            max_tokens: Maximum tokens for context
            top_k: Number of top results to retrieve
            max_distance: Maximum vector distance for retrieval
        """
        self.neo4j_client = neo4j_client
        self.ollama_client = ollama_client
        self.gemini_client = gemini_client
        self.qdrant_client = qdrant_client
        self.max_tokens = max_tokens
        self.top_k = top_k
        self.max_distance = max_distance
        
        # Define base data directory for personal information
        self.user_data_dir = os.path.join(os.getcwd(), "user_data")
        os.makedirs(self.user_data_dir, exist_ok=True)
    
    async def process_query(
        self, 
        query: str,
        intent: QueryIntent,
        high_level_keywords: List[str],
        low_level_keywords: List[str], 
        conversation_history: Optional[List[Dict[str, str]]] = None,
        user_id: Optional[str] = None,
        response_type: str = "concise"
    ) -> Dict[str, Any]:
        """
        Main method to process a user query against the knowledge graph.
        
        Args:
            query: User's natural language query
            intent: Analyzed query intent
            high_level_keywords: High-level keywords extracted from query
            low_level_keywords: Low-level keywords extracted from query
            conversation_history: Optional list of previous conversation messages
            user_id: Optional user identifier for personal info management
            response_type: Type of response ('concise' or 'detailed')
            
        Returns:
            Dictionary containing query processing results
        """
        try:
            start_time = datetime.now()
            
            # Log the incoming query and intent
            logger.info(f"Processing query: '{query}' with intent: {intent.name}")
            if high_level_keywords or low_level_keywords:
                logger.info(f"High-level keywords: {high_level_keywords}")
                logger.info(f"Low-level keywords: {low_level_keywords}")
            
            # Create initial result structure
            result = {
                "query": query,
                "intent": intent.value,
                "keywords": {
                    "high_level": high_level_keywords,
                    "low_level": low_level_keywords
                },
                "response": "",
                "processing_time_seconds": 0,
                "sources": []
            }
            
            # Process based on intent type
            if intent == QueryIntent.GREETING:
                # For greetings, generate a friendly response without KG retrieval
                result["response"] = await self._generate_greeting_response(
                    query, 
                    conversation_history
                )
                
            elif intent == QueryIntent.PERSONAL_INFO:
                # For personal info, save to user profile and acknowledge
                if user_id:
                    save_result = await self._save_personal_info(
                        query, 
                        user_id
                    )
                    result["response"] = save_result.get("message", "I've saved your information to your profile.")
                else:
                    result["response"] = "I'd be happy to save your personal information, but I need a user ID to do so. Please log in first."
                    
            elif intent == QueryIntent.DIABETES_RELATED:
                # For diabetes queries, parallelize retrieval and ground truth gathering
                kg_retrieval_task = None
                grounding_task = None
                
                # Only attempt retrieval if we have keywords
                if high_level_keywords or low_level_keywords:
                    kg_retrieval_task = asyncio.create_task(
                        retrieve_from_knowledge_graph(
                            high_level_keywords=high_level_keywords,
                            low_level_keywords=low_level_keywords,
                            neo4j_client=self.neo4j_client,
                            ollama_client=self.ollama_client,
                            qdrant_client=self.qdrant_client,
                            top_k=self.top_k,
                            max_distance=self.max_distance
                        )
                    )
                
                # Start grounding in parallel
                if self.gemini_client:
                    try:
                        grounding_task = asyncio.create_task(
                            self.gemini_client.grounding(
                                f"Latest medical research about: {query} related to diabetes",
                                model="gemini-2.0-flash"
                            )
                        )
                    except Exception as e:
                        logger.warning(f"Failed to start grounding task: {str(e)}")
                        grounding_task = None
                
                # Wait for both tasks to complete
                kg_context = {"combined_text": ""}
                grounding_context = ""
                
                if kg_retrieval_task:
                    try:
                        kg_context = await kg_retrieval_task
                    except Exception as e:
                        logger.error(f"Error in KG retrieval: {str(e)}")
                        kg_context = {"combined_text": ""}
                
                if grounding_task:
                    try:
                        grounding_result = await grounding_task
                        grounding_context = grounding_result.get("message", {}).get("content", "")
                    except Exception as e:
                        logger.warning(f"Error retrieving grounding context: {str(e)}")
                        grounding_context = ""
                
                # Generate response with both contexts
                result["response"] = await self._generate_diabetes_response(
                    query,
                    kg_context.get("combined_text", ""),
                    conversation_history,
                    response_type,
                    grounding_context=grounding_context
                )
                
                # Include sources if available
                result["sources"] = kg_context.get("sources", [])
                
            else:  # QueryIntent.GENERAL
                # For general queries, retrieve from KG and generate general response
                kg_context = await retrieve_from_knowledge_graph(
                    high_level_keywords=high_level_keywords,
                    low_level_keywords=low_level_keywords,
                    neo4j_client=self.neo4j_client,
                    ollama_client=self.ollama_client,
                    qdrant_client=self.qdrant_client,
                    top_k=self.top_k,
                    max_distance=self.max_distance
                )
                
                result["response"] = await self._generate_general_response(
                    query,
                    kg_context["combined_text"],
                    conversation_history,
                    response_type
                )
                
                # Include sources if available
                result["sources"] = kg_context.get("sources", [])
                result["context"] = kg_context.get("combined_text", "")
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            result["processing_time_seconds"] = processing_time
            
            logger.info(f"Query processed in {processing_time:.2f} seconds")
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "query": query,
                "error": str(e),
                "response": "I'm sorry, but I couldn't process your query successfully."
            }
    
    async def _generate_greeting_response(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Generate a friendly greeting response.
        
        Args:
            query: User's greeting query
            conversation_history: Optional conversation history
            
        Returns:
            Greeting response text
        """
        prompt = f"""You are a friendly and helpful diabetes assistant engaging in conversation with a user.

Goal: Generate a warm, natural greeting response to the user that establishes rapport and sets a positive tone for the conversation. 

Instructions:
- Respond in a friendly and conversational manner
- Keep your response concise (1-3 sentences)
- Do not make claims about specific diabetes knowledge or medical advice in your greeting
- Acknowledge the user's greeting in a natural way
- If the user is returning, acknowledge the continued conversation
- If the user expresses feelings or a mood in their greeting, acknowledge appropriately

Conversation History: 
{conversation_history[-3:] if conversation_history else ''}

User Query: {query}

Generate a friendly greeting response."""
        
        try:
            # Generate greeting using Gemini
            response = await self.gemini_client.generate(
                prompt=prompt
            )
            
            # Extract content from response
            if isinstance(response, dict):
                return response.get("message", {}).get("content", 
                    "Hello! How can I help you with your diabetes questions today?")
            return str(response)
            
        except Exception as e:
            logger.error(f"Error generating greeting response: {str(e)}")
            return "Hello! How can I help you with your diabetes questions today?"
    
    # The rest of the methods remain the same...
    
    async def _save_personal_info(
        self,
        query: str,
        user_id: str
    ) -> Dict[str, Any]:
        """
        Extract and save personal information from user query.
        
        Args:
            query: User's query containing personal information
            user_id: User identifier
            
        Returns:
            Dict with operation status and message
        """
        # Implementation remains the same as before...
        # Create filename for user data
        file_path = os.path.join(self.user_data_dir, f"{user_id}_personal_info.json")
        
        # Load existing data if available
        existing_data = {}
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    existing_data = json.load(f)
            except Exception as e:
                logger.error(f"Error loading existing user data: {str(e)}")
        
        # Create prompt to extract structured personal info
        prompt = f"""You are a personal information extraction assistant for a diabetes management application.

Goal: Extract structured personal information from the user's message for their diabetes profile, without requesting any additional information.

Instructions:
- Extract ONLY information that is explicitly stated in the user's message
- DO NOT invent or assume any details that aren't explicitly mentioned
- Output a JSON object with the following possible fields (only include fields that have information provided):
    - name: Full name if provided
    - age: Numeric age if provided
    - diabetes_type: Type of diabetes if mentioned (Type 1, Type 2, Gestational, etc.)
    - diagnosis_date: When they were diagnosed, if mentioned
    - medications: List of medications mentioned
    - medical_history: Any medical history or conditions mentioned
    - symptoms: Any symptoms mentioned
    - glucose_levels: Any blood glucose readings mentioned
    - a1c: A1C test results if mentioned
    - diet: Any diet information mentioned
    - exercise: Any exercise information mentioned
    - contact_info: Any contact information provided (email, phone)

User Message: {query}

Output Format:
{{
    "extracted_fields": {{
        // only include fields with information present in the message
    }},
    "acknowledgment": "A brief, friendly acknowledgment message confirming what information was saved"
}}"""
        
        try:
            # Use Gemini to extract structured information
            response = await self.gemini_client.generate(
                prompt=prompt
            )
            
            # Parse the response
            content = response.get("message", {}).get("content", "{}")
            if isinstance(content, str):
                try:
                    extracted_data = json.loads(content)
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse response as JSON: {content}")
                    extracted_data = {
                        "extracted_fields": {},
                        "acknowledgment": "I've noted your information, but there was an issue processing it. Could you try again?"
                    }
            else:
                extracted_data = content
            
            # Get extracted fields and acknowledgment
            extracted_fields = extracted_data.get("extracted_fields", {})
            acknowledgment = extracted_data.get("acknowledgment", "I've saved your information to your profile.")
            
            # Update existing data with new data
            for key, value in extracted_fields.items():
                existing_data[key] = value
            
            # Add timestamp
            existing_data["last_updated"] = datetime.now().isoformat()
            
            # Save to file
            with open(file_path, 'w') as f:
                json.dump(existing_data, f, indent=2)
            
            logger.info(f"Saved personal information for user {user_id}")
            
            return {
                "status": "success",
                "message": acknowledgment,
                "file_path": file_path
            }
            
        except Exception as e:
            logger.error(f"Error saving personal information: {str(e)}")
            return {
                "status": "error",
                "message": "I had trouble saving your information. Could you try again?"
            }
    
    async def _generate_diabetes_response(
        self,
        query: str,
        context: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        response_type: str = "concise",
        grounding_context: str = ""
    ) -> str:
        """
        Generate a response for diabetes-related queries.
        
        Args:
            query: User's query
            context: Knowledge graph context
            conversation_history: Optional conversation history
            response_type: Type of response ('concise' or 'detailed')
            grounding_context: Additional context from web search
            
        Returns:
            Generated response
        """
        prompt = f"""You are a specialized diabetes information assistant with access to both a knowledge graph of diabetes concepts and up-to-date web information.

Goal: Generate a comprehensive, evidence-based response to the user's diabetes-related query using both the provided knowledge graph information and recent web search results.

User Query: {query}

Knowledge Graph Context:
{context}

Recent Web Information:
{grounding_context}

Instructions:
- Synthesize and critically evaluate information from both the knowledge graph and recent web sources
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
- Always note that this is informational only and not medical advice

Conversation History:
{conversation_history[-3:] if conversation_history else ''}

Response Type: {response_type}

Generate a well-reasoned, evidence-based response that integrates both knowledge sources."""

        try:
            # Generate response using Gemini
            response = await self.gemini_client.generate(
                prompt=prompt
            )
            
            # Extract content from response
            if isinstance(response, dict):
                return response.get("message", {}).get("content", 
                    "I don't have enough information about that specific diabetes topic in my knowledge sources. Would you like to ask about something else related to diabetes?")
            return str(response)
            
        except Exception as e:
            logger.error(f"Error generating diabetes response: {str(e)}")
            return "I apologize, but I encountered an error generating a response. Could you try rephrasing your question?"
    
    async def _generate_general_response(
        self,
        query: str,
        context: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        response_type: str = "concise"
    ) -> str:
        """
        Generate a response for general queries.
        
        Args:
            query: User's query
            context: Knowledge graph context
            conversation_history: Optional conversation history
            response_type: Type of response ('concise' or 'detailed')
            
        Returns:
            Generated response
        """
        prompt = f"""You are a helpful assistant responding to a general query from a user.

Goal: Generate a thoughtful response to the user's query based on the provided context and conversation history.

User Query: {query}

Available Context:
{context}

Instructions:
- Provide a helpful and informative response based on the available context
- Be conversational and friendly in your tone
- If the context doesn't contain sufficient information to answer the query, acknowledge this
- Format your response for readability with appropriate structure based on the requested length:
  * concise: Clear, focused response in 2-4 sentences
  * detailed: More comprehensive explanation with relevant details
- Do not claim to know information that isn't present in the context

Conversation History:
{conversation_history[-3:] if conversation_history else ''}

Response Type: {response_type}

Generate a thoughtful response."""

        try:
            # Generate response using Gemini
            response = await self.gemini_client.generate(
                prompt=prompt
            )
            
            # Extract content from response
            if isinstance(response, dict):
                return response.get("message", {}).get("content", 
                    "I don't have enough information to answer that question properly. Is there something else I can help you with?")
            return str(response)
            
        except Exception as e:
            logger.error(f"Error generating general response: {str(e)}")
            return "I apologize, but I encountered an error generating a response. Could you try rephrasing your question?"


async def process_kg_query(
    query: str,
    intent: QueryIntent,
    high_level_keywords: List[str],
    low_level_keywords: List[str],
    conversation_history: Optional[List[Dict[str, str]]] = None,
    neo4j_client: Neo4jClient = None,
    ollama_client: Any = None,
    gemini_client: Any = None,
    qdrant_client: Any = None,
    user_id: Optional[str] = None,
    response_type: str = "concise"
) -> Dict[str, Any]:
    """
    Process a complete query against the dual-level knowledge graph.
    
    Args:
        query: User's natural language query
        intent: Analyzed query intent (GREETING, DIABETES_RELATED, etc.)
        high_level_keywords: High-level conceptual keywords extracted from query
        low_level_keywords: Low-level specific keywords extracted from query
        conversation_history: Optional conversation history for context
        neo4j_client: Neo4j database client
        ollama_client: Ollama client for embeddings
        gemini_client: Gemini client for text generation
        qdrant_client: Qdrant vector database client
        user_id: Optional user identifier for personal info storage
        response_type: Type of response ('concise' or 'detailed')
        
    Returns:
        Dictionary with processed query results
    """
    start_time = time.time()
    logger.info(f"Processing query: '{query}' with intent: {intent}")
    logger.info(f"High-level keywords: {high_level_keywords}")
    logger.info(f"Low-level keywords: {low_level_keywords}")
    
    # Create a processor instance
    processor = KnowledgeGraphQueryProcessor(
        neo4j_client=neo4j_client,
        ollama_client=ollama_client,
        gemini_client=gemini_client,
        qdrant_client=qdrant_client
    )
    
    # Process the query using the processor
    result = await processor.process_query(
        query=query,
        intent=intent,
        high_level_keywords=high_level_keywords,
        low_level_keywords=low_level_keywords,
        conversation_history=conversation_history,
        user_id=user_id,
        response_type=response_type
    )
    
    # Add processing time
    processing_time = time.time() - start_time
    logger.info(f"Query processed in {processing_time:.2f} seconds")
    
    # If result doesn't already include processing time, add it
    if "processing_time_seconds" not in result:
        result["processing_time_seconds"] = processing_time
    
    return result