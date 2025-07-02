import logging
import re
import uuid
from typing import List, Dict, Any, Optional, Tuple
import asyncio

from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils.text_processing import count_tokens

logger = logging.getLogger(__name__)

class DocumentChunker:
    """
    Implements document chunking for the Gemini knowledge graph pipeline.
    
    This chunker divides documents into fixed-length chunks with optional 
    overlap, using LangChain's RecursiveCharacterTextSplitter.
    """
    
    def __init__(
        self, 
        max_chunk_tokens: int = 10000,
        overlap_tokens: int = 800,
        min_chunk_tokens: int = 100
    ):
        """
        Initialize the Gemini chunker.
        
        Args:
            max_chunk_tokens: Maximum number of tokens per chunk
            overlap_tokens: Number of tokens to overlap between chunks
            min_chunk_tokens: Minimum number of tokens for a valid chunk
        """
        self.max_chunk_tokens = max_chunk_tokens
        self.overlap_tokens = overlap_tokens
        self.min_chunk_tokens = min_chunk_tokens
        
        self.max_chunk_size = max_chunk_tokens
        self.overlap_size = overlap_tokens
    
    def split_into_paragraphs(self, text: str) -> List[str]:
        """
        Split text into paragraphs based on line breaks.
        
        Args:
            text: Raw document text
            
        Returns:
            List of paragraphs
        """
        # Split on double line breaks first (preferred paragraph separator)
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Process each potential paragraph to handle single line breaks
        processed_paragraphs = []
        for para in paragraphs:
            # If paragraph is long enough, keep as is
            if len(para.strip()) > 100:
                processed_paragraphs.append(para.strip())
            else:
                # For shorter text blocks, check if they contain single line breaks
                # that might indicate separate paragraphs
                sub_paras = re.split(r'\n', para)
                sub_paras = [sp.strip() for sp in sub_paras if sp.strip()]
                processed_paragraphs.extend(sub_paras)
        
        # Remove empty paragraphs and ensure each has meaningful content
        return [p.strip() for p in processed_paragraphs if p.strip() and len(p.strip()) > 10]
    
    async def create_chunks(
        self,
        text: str,
        document_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Create fixed-length chunks from a document with metadata using LangChain.
        
        Args:
            text: Raw document text
            document_metadata: Additional metadata to include with chunks
                (document_id, knowledge_level, etc.)
            
        Returns:
            List of chunks with metadata
        """
        # Extract key metadata
        document_id = document_metadata.get("document_id", str(uuid.uuid4())) if document_metadata else str(uuid.uuid4())
        knowledge_level = document_metadata.get("knowledge_level", 1) if document_metadata else 1
        
        # Initialize the LangChain text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_chunk_size,
            chunk_overlap=self.overlap_size,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Split the text into chunks
        raw_chunks = text_splitter.split_text(text)
        logger.info(f"Document split into {len(raw_chunks)} chunks using LangChain")
        
        if not raw_chunks:
            return []
        
        # Process chunks and add metadata
        chunks = []
        for i, chunk_content in enumerate(raw_chunks):
            # Count tokens for each chunk
            chunk_tokens = count_tokens(chunk_content)
            
            # Skip chunks that are too small
            if chunk_tokens < self.min_chunk_tokens:
                continue
                
            # Create chunk dictionary with metadata
            chunk_id = str(uuid.uuid4())
            chunk = {
                "content": chunk_content,
                "tokens": chunk_tokens,
                "paragraph_count": len(self.split_into_paragraphs(chunk_content)),
                "chunk_index": i,
                "chunk_id": chunk_id,
                "total_chunks": len(raw_chunks),
                "document_id": document_id,
                "knowledge_level": knowledge_level
            }
            
            # Add any additional document metadata that should be propagated
            if document_metadata:
                for key, value in document_metadata.items():
                    if key not in chunk and key not in ["document_id", "knowledge_level"]:
                        chunk[key] = value
            
            chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} fixed-length chunks for document {document_id} (Level {knowledge_level})")
        return chunks


async def process_document(
    document_text: str,
    document_id: str,
    document_metadata: Optional[Dict[str, Any]] = None,
    max_chunk_tokens: int = 1000,
    overlap_tokens: int = 200
) -> List[Dict[str, Any]]:
    """
    Process a document into fixed-length chunks.
    
    Args:
        document_text: Raw document text
        document_id: Unique document identifier
        document_metadata: Additional metadata for the document
        max_chunk_tokens: Maximum tokens per chunk
        overlap_tokens: Number of tokens to overlap between chunks
        
    Returns:
        List of chunk objects with content and metadata
    """
    # Create metadata dictionary if not provided
    if document_metadata is None:
        document_metadata = {}
    
    # Ensure document_id is in the metadata
    document_metadata["document_id"] = document_id
    
    # Default knowledge_level to 1 if not specified
    if "knowledge_level" not in document_metadata:
        document_metadata["knowledge_level"] = 1
    
    chunker = DocumentChunker(
        max_chunk_tokens=max_chunk_tokens,
        overlap_tokens=overlap_tokens
    )
    
    chunks = await chunker.create_chunks(document_text, document_metadata)
    
    return chunks