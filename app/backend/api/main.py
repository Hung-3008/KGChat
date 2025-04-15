import os
import asyncio
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import logging
import time
import uvicorn

from backend.core.retrieval.query_analyzer import analyze_query, QueryIntent
from backend.core.retrieval.keyword_extractor import extract_keywords
from backend.core.retrieval.kg_query_processor import process_kg_query

# Import the ClientManager
from backend.api.client_manager import ClientManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(
    title="Knowledge Graph Query API",
    description="API for processing diabetes-related queries using a knowledge graph",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

class QueryRequest(BaseModel):
    query: str = Field(..., description="The user's query text")
    conversation_history: Optional[List[Dict[str, str]]] = Field(
        default=None, 
        description="Optional conversation history as a list of message dictionaries"
    )
    user_id: Optional[str] = Field(
        default=None, 
        description="Optional user identifier for personal information tracking"
    )
    response_type: Optional[str] = Field(
        default="concise", 
        description="Response type: 'concise' or 'detailed'"
    )

class IntentResponse(BaseModel):
    intent: str = Field(..., description="The classified intent of the query")
    intent_description: str = Field(..., description="Description of the intent")

class KeywordsResponse(BaseModel):
    high_level: List[str] = Field(..., description="High-level conceptual keywords")
    low_level: List[str] = Field(..., description="Low-level specific keywords")

class QueryResponse(BaseModel):
    query: str = Field(..., description="The original query")
    response: str = Field(..., description="The generated response")
    intent: Optional[str] = Field(None, description="The classified intent")
    keywords: Optional[Dict[str, List[str]]] = Field(None, description="Extracted keywords")
    processing_time_seconds: Optional[float] = Field(None, description="Processing time in seconds")
    sources: Optional[List[str]] = Field(None, description="Sources used for the response")

# Initialize the client manager at startup
@app.on_event("startup")
async def startup_event():
    """Initialize all clients when the application starts"""
    client_manager = ClientManager.get_instance()
    success = await client_manager.initialize()
    
    if not success:
        logger.error("Failed to initialize clients. Application may not function correctly.")
    else:
        logger.info("All clients initialized successfully")

# Cleanup at shutdown
@app.on_event("shutdown")
async def shutdown_event():
    """Close all client connections when the application shuts down"""
    client_manager = ClientManager.get_instance()
    await client_manager.shutdown()
    logger.info("All clients shut down successfully")

# Dependency to get clients
async def get_clients():
    """Get all clients from the ClientManager as a dependency"""
    client_manager = ClientManager.get_instance()
    
    # If not initialized yet, try to initialize
    if not client_manager._initialized:
        await client_manager.initialize()
        
    clients = client_manager.get_clients()
    
    # Check if any clients are None and log a warning
    for name, client in clients.items():
        if client is None:
            logger.warning(f"{name} is not initialized")
    
    return clients

# API endpoints
@app.get("/")
async def root():
    """Root endpoint providing API information."""
    return {
        "message": "Knowledge Graph Query API",
        "version": "1.0.0",
        "endpoints": {
            "/api/query": "Process a query against the knowledge graph",
            "/api/analyze_intent": "Analyze the intent of a query",
            "/api/extract_keywords": "Extract keywords from a query"
        }
    }

@app.post("/api/analyze_intent", response_model=IntentResponse)
async def analyze_intent(
    request: QueryRequest,
    clients: Dict[str, Any] = Depends(get_clients)
):
    """Analyze the intent of a query."""
    try:
        intent = await analyze_query(
            query=request.query,
            conversation_history=request.conversation_history,
            client=clients["gemini_client"],
            user_id=request.user_id
        )
        
        intent_descriptions = {
            QueryIntent.GREETING: "General greeting or conversation starter",
            QueryIntent.DIABETES_RELATED: "Query related to diabetes information",
            QueryIntent.PERSONAL_INFO: "Sharing or requesting personal information",
            QueryIntent.GENERAL: "General query not fitting other categories"
        }
        
        return {
            "intent": intent.value,
            "intent_description": intent_descriptions.get(intent, "Unknown intent")
        }
    except Exception as e:
        logger.error(f"Error analyzing intent: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Intent analysis failed: {str(e)}")

@app.post("/api/extract_keywords", response_model=KeywordsResponse)
async def extract_query_keywords(
    request: QueryRequest,
    clients: Dict[str, Any] = Depends(get_clients)
):
    """Extract high-level and low-level keywords from a query."""
    try:
        high_level, low_level = await extract_keywords(
            query=request.query,
            conversation_history=request.conversation_history,
            llm_client=clients["gemini_client"]
        )
        
        return {
            "high_level": high_level,
            "low_level": low_level
        }
    except Exception as e:
        logger.error(f"Error extracting keywords: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Keyword extraction failed: {str(e)}")

@app.post("/api/query", response_model=QueryResponse)
async def process_user_query(
    request: QueryRequest,
    clients: Dict[str, Any] = Depends(get_clients)
):
    """Process a complete query against the knowledge graph."""
    try:
        start_time = time.time()
        
        # Step 1: Analyze intent
        intent_task = asyncio.create_task(
            analyze_query(
                query=request.query,
                conversation_history=request.conversation_history,
                client=clients["gemini_client"],
                user_id=request.user_id
            )
        )
        
        # Wait for intent analysis to complete
        intent = await intent_task
        
        # Step 2: Extract keywords only for diabetes-related queries
        high_level = []
        low_level = []
        
        if intent == QueryIntent.DIABETES_RELATED:
            # Only extract keywords if the query is diabetes-related
            keywords_task = asyncio.create_task(
                extract_keywords(
                    query=request.query,
                    conversation_history=request.conversation_history,
                    llm_client=clients["gemini_client"]
                )
            )
            high_level, low_level = await keywords_task
        
        logger.info(f"Query: '{request.query}'")
        logger.info(f"Intent: {intent.name}")
        
        if intent == QueryIntent.DIABETES_RELATED:
            logger.info(f"High-level keywords: {high_level}")
            logger.info(f"Low-level keywords: {low_level}")
        
        # Step 3: Process the query with the knowledge graph
        # Note: We're using our pooled clients here by passing them in
        result = await process_kg_query(
            query=request.query,
            intent=intent,
            high_level_keywords=high_level,
            low_level_keywords=low_level,
            conversation_history=request.conversation_history,
            neo4j_client=clients["neo4j_client"],
            ollama_client=clients["ollama_client"],
            gemini_client=clients["gemini_client"],
            qdrant_client=clients["qdrant_client"],
            user_id=request.user_id,
            response_type=request.response_type
        )
        
        processing_time = time.time() - start_time
        
        # Return the complete response
        return {
            "query": request.query,
            "response": result.get("response", "Sorry, I couldn't generate a response."),
            "intent": intent.value,
            "keywords": {
                "high_level": high_level,
                "low_level": low_level
            },
            "processing_time_seconds": processing_time,
            "sources": result.get("sources", [])
        }
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint that also verifies all clients are connected."""
    client_manager = ClientManager.get_instance()
    clients = client_manager.get_clients()
    
    status = {
        "api": "ok",
        "clients": {}
    }
    
    # Check Neo4j connectivity
    try:
        if clients["neo4j_client"] and await clients["neo4j_client"].verify_connectivity():
            status["clients"]["neo4j"] = "connected"
        else:
            status["clients"]["neo4j"] = "disconnected"
    except Exception:
        status["clients"]["neo4j"] = "error"
    
    # Add other client checks as needed
    for client_name in ["gemini_client", "ollama_client", "qdrant_client", "vector_db_client"]:
        status["clients"][client_name.replace("_client", "")] = "available" if clients[client_name] else "unavailable"
    
    return status

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)