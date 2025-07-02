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

from core.retrieval.query_analyzer import analyze_query, QueryIntent
from core.retrieval.keyword_extractor import extract_keywords
from core.retrieval.kg_query_processor import run_query

# Import the ClientManager
from api.client_manager import ClientManager

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

# client_manager = ClientManager.get_instance()
# if not client_manager._initialized:
#     asyncio.run(client_manager.initialize())
# clients = client_manager.get_clients()
    


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
    
    return clients, client_manager



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
    #clients: Dict[str, Any] = Depends(get_clients)
):
    clients, client_manager = await get_clients()

    max_retries = 3
    while max_retries > 0:
        try:
            result = await run_query(
                query=request.query,
                conversation_history=request.conversation_history,
                clients = clients, 
                grounding=False
            )
            max_retries = 0
            return result

        except Exception as e:
            logger.warning(f"Retrying due to error: {str(e)}")
            max_retries -= 1
            time.sleep(10)
            clients['gemini_client'] = client_manager.rotate_gemini_key()

            try:
                result = await run_query(
                    query=request.query,
                    conversation_history=request.conversation_history,
                    clients = clients, 
                )
                max_retries = 0
                return result
            except Exception as e:
                continue
            

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