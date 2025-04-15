import re
import logging
from typing import List, Optional
import tiktoken

logger = logging.getLogger(__name__)


DEFAULT_TOKENIZER_MODEL = "cl100k_base"

def count_tokens(text: str, model: Optional[str] = None) -> int:
    """
    Count the number of tokens in a text string.
    
    Args:
        text: The text to count tokens for
        model: The model name to use for tokenization (defaults to cl100k_base)
        
    Returns:
        Number of tokens in the text
    """
    if not text:
        return 0
        
    try:
        # Get the appropriate encoding for the model
        encoding_name = model or DEFAULT_TOKENIZER_MODEL
        
        try:
            encoding = tiktoken.get_encoding(encoding_name)
        except KeyError:
            logger.warning(f"Encoding {encoding_name} not found, falling back to cl100k_base")
            encoding = tiktoken.get_encoding("cl100k_base")
            
        # Count tokens
        tokens = encoding.encode(text)
        return len(tokens)
        
    except Exception as e:
        # Fallback to a simple approximation if tiktoken fails
        logger.warning(f"Error counting tokens with tiktoken: {str(e)}. Using approximate count.")
        return _approximate_token_count(text)

def _approximate_token_count(text: str) -> int:
    """
    Approximate token count when tiktoken is unavailable.
    
    This is a fallback method that provides a rough estimate.
    English texts average ~4 characters per token.
    
    Args:
        text: The text to count tokens for
        
    Returns:
        Approximate number of tokens
    """
    # Split on whitespace and punctuation
    words = re.findall(r'\w+|[^\w\s]', text)
    
    # Count characters (excluding whitespace)
    char_count = sum(len(word) for word in words)
    
    # Approximate: 4 chars per token on average for English text
    return max(1, round(char_count / 4))

def split_text_by_separator(text: str, separator: str = "\n\n") -> List[str]:
    """
    Split text into chunks using a separator.
    
    Args:
        text: Text to split
        separator: String separator to split on
        
    Returns:
        List of text chunks
    """
    if not text:
        return []
        
    chunks = text.split(separator)
    return [chunk.strip() for chunk in chunks if chunk.strip()]

def clean_text(text: str) -> str:
    """
    Clean text by removing excessive whitespace and normalizing line breaks.
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
        
    # Replace multiple spaces with a single space
    text = re.sub(r' +', ' ', text)
    
    # Normalize line breaks
    text = re.sub(r'\r\n', '\n', text)
    
    # Remove more than 3 consecutive line breaks
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

def truncate_text(text: str, max_tokens: int, model: Optional[str] = None) -> str:
    """
    Truncate text to a maximum number of tokens.
    
    Args:
        text: Text to truncate
        max_tokens: Maximum number of tokens
        model: Model name for tokenization
        
    Returns:
        Truncated text
    """
    if not text:
        return ""
        
    current_tokens = count_tokens(text, model)
    
    if current_tokens <= max_tokens:
        return text
        
    try:
        encoding_name = model or DEFAULT_TOKENIZER_MODEL
        encoding = tiktoken.get_encoding(encoding_name)
        
        # Encode the text and truncate
        tokens = encoding.encode(text)
        truncated_tokens = tokens[:max_tokens]
        
        # Decode back to text
        return encoding.decode(truncated_tokens)
        
    except Exception as e:
        logger.warning(f"Error truncating text with tiktoken: {str(e)}. Using approximate truncation.")
        
        # Fallback: Approximate truncation based on character count
        ratio = max_tokens / current_tokens
        char_limit = int(len(text) * ratio)
        
        # Try to truncate at a sentence boundary
        last_period = text[:char_limit].rfind('.')
        if last_period > char_limit * 0.7:  # Only if the period is reasonably close to the end
            return text[:last_period + 1]
        
        return text[:char_limit]