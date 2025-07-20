"""
Embedding utilities for vector database operations using transformers.
"""
import os
import logging
import numpy as np
from typing import List
from dotenv import load_dotenv

from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()


class TransformerEmbeddings(Embeddings):
    """Custom embeddings class using SentenceTransformer."""
    
    def __init__(
        self, 
        model_name: str,
        device: str = None
    ):
        """Initialize with a SentenceTransformer model."""
        self.model_name = model_name
        if device is None:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.model = SentenceTransformer(model_name, device=self.device)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        embedding = self.model.encode(text)
        return embedding.tolist()


class EmbeddingManager:
    """Handles document embedding and vector database operations."""

    def __init__(
        self, 
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        persist_directory: str = "./data/chroma_db",
        local_files_only: bool = False
    ):
        """
        Initialize the embedding manager.
        
        Args:
            model_name: The embedding model to use.
            persist_directory: Directory to store the vector database.
            local_files_only: Whether to use only local model files (offline mode).
        """
        # Fallback to a simpler embedding model that's likely to be cached
        fallback_model = "distilbert-base-uncased"
        
        try:
            logger.info(f"Loading embedding model: {model_name}")
            # Initialize embeddings using SentenceTransformer
            self.embeddings = TransformerEmbeddings(model_name=model_name)
            logger.info(f"Successfully loaded embedding model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading embedding model {model_name}: {str(e)}")
            
            # Try the fallback model if the first one fails
            try:
                logger.info(f"Attempting to load fallback embedding model: {fallback_model}")
                self.embeddings = TransformerEmbeddings(model_name=fallback_model)
                logger.info(f"Successfully loaded fallback embedding model: {fallback_model}")
            except Exception as fallback_error:
                logger.error(f"Failed to load fallback embedding model: {str(fallback_error)}")
                raise fallback_error
                
        self.persist_directory = persist_directory
    
    def store_documents(
        self, 
        documents: List[Document], 
        collection_name: str = "pdf_documents"
    ) -> Chroma:
        """
        Store documents in the vector database.
        
        Args:
            documents: The documents to store.
            collection_name: The name of the collection.
            
        Returns:
            The ChromaDB instance.
        """
        db = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name=collection_name
        )
        return db
    
    def load_vector_store(
        self, 
        collection_name: str = "pdf_documents"
    ) -> Chroma:
        """
        Load an existing vector store.
        
        Args:
            collection_name: The name of the collection.
            
        Returns:
            The ChromaDB instance.
        """
        return Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name=collection_name
        )
    
    def search_documents(
        self, 
        query: str, 
        db: Chroma, 
        k: int = 4
    ) -> List[Document]:
        """
        Search for relevant documents.
        
        Args:
            query: The search query.
            db: The vector database.
            k: Number of documents to retrieve.
            
        Returns:
            List of relevant documents.
        """
        return db.similarity_search(query, k=k)
