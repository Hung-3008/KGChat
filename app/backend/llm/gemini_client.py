import os
import logging
import json
import enum
from typing import List, Dict, Any, Optional, Union, AsyncIterator
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

class GeminiClient:
    """
    A client for interacting with Google's Gemini API.
    
    This class provides methods for text generation and text embedding
    similar to the OllamaClient implementation.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.0-flash", #gemini-1.5-flash 
        embedding_model: Optional[str] = "embedding-001",
        max_retries: int = 3,
        **kwargs
    ):
        """
        Initialize the Gemini client.
        
        Args:
            api_key: Gemini API key (if None, tries to load from environment variable)
            model_name: Default model to use for completions
            embedding_model: Model to use for embeddings
            max_retries: Number of retry attempts for API calls
            **kwargs: Additional parameters to pass to the API
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY_1")
        if not self.api_key:
            raise ValueError("No API key provided. Please provide an API key or set the GEMINI_API_KEY environment variable.")
            
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.max_retries = max_retries
        self.client_kwargs = kwargs
        if api_key:
            os.environ["GOOGLE_API_KEY"] = self.api_key 
        
        # Initialize the client
        self.client = genai.Client(api_key=self.api_key, **kwargs)

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: Optional[List[Dict[str, str]]] = None,
        stream: bool = False,
        model: Optional[str] = None,
        format: Optional[Any] = None,
        **kwargs
    ) -> Dict[str, Any]:

        model = model or self.model_name
        
        try:
            logger.debug(f"Generating completion with model {model}")
            
            generation_config = {}
            if "temperature" in kwargs:
                generation_config["temperature"] = kwargs.pop("temperature")
            
            contents = []
            if system_prompt:
                contents.append({"role": "system", "parts": [{"text": system_prompt}]})
            
            if history_messages:
                for msg in history_messages:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if role and content:
                        contents.append({"role": role, "parts": [{"text": content}]})
            
            contents.append({"role": "user", "parts": [{"text": prompt}]})
            
            if format:
                generation_config["response_mime_type"] = "application/json"
                generation_config["response_schema"] = list[format]
            
            from google.genai import types
            
            content_config = types.GenerateContentConfig(
                temperature=generation_config.get("temperature", 0.7) if generation_config else 0.7,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                response_mime_type=generation_config.get("response_mime_type") if generation_config else None,
                response_schema=generation_config.get("response_schema") if generation_config else None
            )
            
            config_args = {
                "model": model,
                "contents": contents if len(contents) > 1 else contents[0]["parts"][0]["text"],
                "config": content_config
            }
            
            if stream:
                config_args["stream"] = True
                
            response = self.client.models.generate_content(**config_args)
            

            if format:
                try:
                    structured_data = response.parsed
                    return structured_data
                    
                except Exception as e:
                    logger.warning(f"Failed to process structured output: {str(e)}")
                    return {
                        "message": {
                            "content": response.text
                        }
                    }
            else:
                return {
                    "message": {
                        "content": response.text
                    }
                }
                
        except Exception as e:
            logger.error(f"Error generating text with Gemini API: {str(e)}")
            raise
    

    async def embed(
        self, 
        texts: Union[str, List[str]], 
        model: Optional[str] = None,
        **kwargs
    ) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for a text or list of texts using Gemini API.
        
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
            
            if isinstance(texts, str):
                text_list = [texts]
            else:
                text_list = texts
            
            embedding_model = self.client.models.embeddings
            embeddings = []
            
            for text in text_list:
                result = embedding_model.batch_embed_content(
                    model=model,
                    contents=[{"parts": [{"text": text}]}],
                    **kwargs
                )
                if result.embeddings:
                    embeddings.append(result.embeddings[0].values)
            
            if isinstance(texts, str):
                return embeddings[0] if embeddings else []
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings with Gemini API: {str(e)}")
            raise

    async def grounding(
        self, 
        query: str, 
        model: Optional[str] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[Dict[str, Any], AsyncIterator[Any]]:
        """
        Search the internet and retrieve results using Gemini's Google Search tool.
        
        Args:
            query: The search query text
            model: Model to use (defaults to self.model_name)
            stream: Whether to stream the response
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            Search results as a dictionary or async iterator if streaming
        """
        model = model or self.model_name
        
        try:
            logger.debug(f"Performing grounding search with model {model}")
            
            # Prepare content with user query
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=query)],
                ),
            ]
            
            # Define Google Search tool
            tools = [types.Tool(google_search=types.GoogleSearch())]
            
            # Configure generation
            generate_content_config = types.GenerateContentConfig(
                tools=tools,
                **kwargs
            )
            
            if stream:
                # For streaming responses
                response_stream = self.client.models.generate_content_stream(
                    model=model,
                    contents=contents,
                    config=generate_content_config,
                )
                
                return response_stream
            else:
                # For non-streaming responses
                response = self.client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=generate_content_config,
                )
                
                return {
                    "message": {
                        "content": response.text
                    },
                    "used_tools": getattr(response, "used_tools", None),
                    "citation_metadata": getattr(response, "citation_metadata", None)
                }
                
        except Exception as e:
            logger.error(f"Error with grounding search: {str(e)}")
            raise
    
    async def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get a list of available models from the Gemini API.
        
        Returns:
            List of model information dictionaries
        """
        try:
            logger.debug("Fetching available models")
            models = [model._asdict() for model in self.client.list_models()]
            return models
        except Exception as e:
            logger.error(f"Error fetching available models: {str(e)}")
            return []
        
    async def test_api_key(self) -> bool:
        try:
            test_prompt = "Hello, this is a test."
            response = await self.generate(prompt=test_prompt)
            
            if response['message']['content']:
                logger.info("API key is valid")
                return True
            else:
                logger.error("API key is invalid: No response content")
                return False
                
        except Exception as e:
            error_str = str(e).lower()
            logger.error(f"API key verification failed with error: {error_str}")
            
            # Kiểm tra nếu lỗi là do API key không hợp lệ
            if "api key not valid" in error_str or "api_key_invalid" in error_str:
                logger.error("Invalid API key detected during verification")
                return False
            
            # Các lỗi khác có thể là do mạng, giới hạn tốc độ, v.v.
            return False
    async def extract_pdf_raw_text(
        self,
        pdf_path: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Extract raw text from a PDF file using Gemini 2.0 Flash.

        Args:
            pdf_path: Path to the PDF file.
            system_prompt: Optional system prompt to customize text extraction behavior.
            model: Model to use (defaults to self.model_name).
            **kwargs: Additional parameters to pass to the generate method.

        Returns:
            String containing the raw text extracted from the PDF.

        Raises:
            FileNotFoundError: If the PDF file does not exist.
            Exception: If there is an error during processing.
        """
        try:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"Tệp PDF không tồn tại: {pdf_path}")


            with open(pdf_path, "rb") as file:
                pdf_content = file.read()


            import base64
            pdf_base64 = base64.b64encode(pdf_content).decode("utf-8")
            default_system_prompt = """
            You are an expert in extracting text from PDF documents. Extract all text content from the provided PDF and return it as plain text.
            
            Instructions:
            - Include all visible text in the PDF, preserving the content as accurately as possible.
            - Do not format or structure the output; return raw text only.
            - If the PDF contains images or scanned content, extract text using OCR capabilities.
            """
            system_prompt = system_prompt or default_system_prompt
            prompt = f"Extract raw text from the following PDF content:\n\n[Base64 encoded PDF: {pdf_base64}]"
            response = await self.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                model=model or self.model_name,
                **kwargs
            )
            return response["message"]["content"]

        except FileNotFoundError as e:
            logger.error(f"PDF file not found: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise
    @classmethod
    def from_env(cls) -> 'GeminiClient':
        """
        Create a GeminiClient instance from environment variables.
        
        Environment variables:
        - GEMINI_API_KEY: API key for authentication
        - GEMINI_MODEL: Default model name
        - GEMINI_EMBEDDING_MODEL: Model to use for embeddings
        
        Returns:
            Configured GeminiClient instance
        """
        return cls(
            api_key=os.getenv("GEMINI_API_KEY"),
            model_name=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
            embedding_model=os.getenv("GEMINI_EMBEDDING_MODEL", "embedding-001"),
        )