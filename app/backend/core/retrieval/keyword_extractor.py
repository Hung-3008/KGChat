"""
Keyword Extraction Module

This module extracts high-level and low-level keywords from user queries and
conversation history using the LLM. These keywords are used in the knowledge
graph retrieval pipeline to improve query accuracy.
"""
import json
import logging
from typing import List, Dict, Any, Tuple, Optional, Union
import asyncio
from pydantic import BaseModel, Field

from core.pipeline.gemini.prompts import PROMPTS
from utils.logging import get_logger

# Configure logging
logger = get_logger(__name__)

class KeywordExtraction(BaseModel):
    """Pydantic model for structured keyword extraction output."""
    high_level_keywords: List[str] = Field(
        description="Overarching concepts or themes from the query"
    )
    low_level_keywords: List[str] = Field(
        description="Specific entities, details, or concrete terms from the query"
    )

async def extract_keywords(
    query: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    llm_client = None
) -> Tuple[List[str], List[str]]:

    if not query or not query.strip():
        logger.warning("Empty query provided for keyword extraction")
        return [], []
    
    history_text = ""
    if conversation_history:
        history_text = "\n".join([
            f"{msg.get('role', 'unknown').capitalize()}: {msg.get('content', '')}" 
            for msg in conversation_history[-3:]  # Only use last 3 messages for context
        ])
    
    
    extraction_prompt = PROMPTS.get("keywords_extraction", "").format(
        query=query,
        history=history_text,
    )
    
    try:
        
        response = await llm_client.generate(
            prompt=extraction_prompt,
            format=KeywordExtraction
        )
        
        if isinstance(response[0], KeywordExtraction):
            keywords_data = response[0]
            high_level_keywords = keywords_data.high_level_keywords
            low_level_keywords = keywords_data.low_level_keywords
        else:
            logger.error("Unexpected response format from LLM")
            return [], []
        logger.info(f"Extracted keywords: {high_level_keywords}, {low_level_keywords}") 
        return high_level_keywords, low_level_keywords
                    
    except Exception as e:
        logger.error(f"Error during keyword extraction: {str(e)}")
        return [], []





