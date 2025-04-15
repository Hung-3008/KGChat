import os
import logging
from typing import List, Dict, Any, Optional, Union, AsyncIterator

import ollama
from ollama import AsyncClient, ChatResponse
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

logger = logging.getLogger(__name__)

class OllamaClient:
    """
    A client for interacting with Ollama API for LightRAG.
    
    This class provides methods for text generation and text embedding
    through the Ollama Python library, with support for both streaming 
    and non-streaming responses.
    """
    
    def __init__(
        self,
        host: str = "http://localhost:11434",
        model_name: str = "llama3.2:3b",
        embedding_model: Optional[str] = None,
        max_retries: int = 3,
        **kwargs
    ):
        """
        Initialize the Ollama client.
        
        Args:
            host: URL of the Ollama API endpoint
            model_name: Default model to use for completions
            embedding_model: Model to use for embeddings (defaults to model_name if None)
            max_retries: Number of retry attempts for API calls
            **kwargs: Additional parameters to pass to AsyncClient
        """
        self.host = host
        self.model_name = model_name
        self.embedding_model = embedding_model or model_name
        self.max_retries = max_retries
        self.client_kwargs = kwargs
        
        # Create the async client
        self.client = AsyncClient(host=host, **kwargs)
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=(retry_if_exception_type(ollama.ResponseError))
    )

    # system prompt: you are a helpful assistant
    # prompt: user query

    # Lam phep tinh 1+1 
    # 1 + 1 = 2 

    # {"answer": "1 + 1 = 2"}
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: Optional[List[Dict[str, str]]] = None,
        stream: bool = False,
        model: Optional[str] = None,
        format: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Union[ChatResponse, AsyncIterator[ChatResponse]]:
        """
        Generate text using Ollama API.
        
        Args:
            prompt: The user prompt text
            system_prompt: Optional system prompt to guide the model
            history_messages: List of past conversation messages
            stream: Whether to stream the response
            model: Model to use (defaults to self.model_name)
            format: JSON schema for structured output format
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            Generated text response or an async iterator of text chunks if streaming
        """
        if history_messages is None:
            history_messages = []
            
        # Prepare messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add history messages
        messages.extend(history_messages)
        
        # Add the current prompt
        messages.append({"role": "user", "content": prompt})
        
        model = model or self.model_name
        
        try:
            logger.debug(f"Generating completion with model {model}")
            
            # Add format parameter for structured output if provided
            if format:
                kwargs["format"] = format
                
            response = await self.client.chat(
                model=model,
                messages=messages,
                stream=stream,
                **kwargs
            )
            return response
        except ollama.ResponseError as e:
            logger.error(f"Ollama API error: {e.status_code} - {e.error}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error calling Ollama API: {str(e)}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=(retry_if_exception_type(ollama.ResponseError))
    )
    async def embed(
        self, 
        texts: Union[str, List[str]], 
        model: Optional[str] = None,
        **kwargs
    ) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for a text or list of texts using Ollama API.
        
        Args:
            texts: Single text or list of texts to embed
            model: Model to use (defaults to self.embedding_model)
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            List of embedding vectors or list of lists of embedding vectors
        """
        model = model or self.embedding_model
        
        try:
            logger.debug(f"Generating embeddings with model {model}")
            response = await self.client.embed(
                model=model,
                input=texts,
                **kwargs
            )
            return response['embeddings']
        except ollama.ResponseError as e:
            logger.error(f"Ollama API error: {e.status_code} - {e.error}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error calling Ollama API: {str(e)}")
            raise
    
    async def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get a list of available models from the Ollama API.
        
        Returns:
            List of model information dictionaries
        """
        try:
            logger.debug("Fetching available models")
            response = await self.client.list()
            return response["models"]
        except ollama.ResponseError as e:
            logger.error(f"Ollama API error when listing models: {e.status_code} - {e.error}")
            return []
        except Exception as e:
            logger.error(f"Error fetching available models: {str(e)}")
            return []
            
    async def pull_model(self, model_name: str) -> bool:
        """
        Pull a model from Ollama registry.
        
        Args:
            model_name: Name of the model to pull
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Pulling model: {model_name}")
            await self.client.pull(model=model_name)
            return True
        except ollama.ResponseError as e:
            logger.error(f"Ollama API error when pulling model: {e.status_code} - {e.error}")
            return False
        except Exception as e:
            logger.error(f"Error pulling model {model_name}: {str(e)}")
            return False
            
    async def create_model(
        self, 
        model_name: str, 
        base_model: str, 
        system_prompt: str,
        **kwargs
    ) -> bool:
        """
        Create a new model based on an existing one.
        
        Args:
            model_name: Name for the new model
            base_model: Base model to derive from
            system_prompt: System prompt to customize the model behavior
            **kwargs: Additional parameters for model creation
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Creating model {model_name} from {base_model}")
            await self.client.create(
                model=model_name,
                from_=base_model,
                system=system_prompt,
                **kwargs
            )
            return True
        except ollama.ResponseError as e:
            logger.error(f"Ollama API error when creating model: {e.status_code} - {e.error}")
            return False
        except Exception as e:
            logger.error(f"Error creating model {model_name}: {str(e)}")
            return False

    @classmethod
    def from_env(cls) -> 'OllamaClient':
        """
        Create an OllamaClient instance from environment variables.
        
        Environment variables:
        - OLLAMA_HOST: Base URL for Ollama API
        - OLLAMA_MODEL: Default model name
        - OLLAMA_EMBEDDING_MODEL: Model to use for embeddings
        
        Returns:
            Configured OllamaClient instance
        """
        return cls(
            host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            model_name=os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
            embedding_model=os.getenv("OLLAMA_EMBEDDING_MODEL"),
        )