"""
Embedding utilities for vector database operations using transformers.
"""
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import stat
import logging
import numpy as np
from typing import List, Optional
from dotenv import load_dotenv
import time
import shutil

from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()


def ensure_directory_permissions(directory_path):
    """Ensure directory exists and has proper write permissions."""
    try:
        if not os.path.exists(directory_path):
            os.makedirs(directory_path, mode=0o755)
            logger.info(f"Created directory: {directory_path}")
        else:
            # Set directory permissions to allow writing
            os.chmod(directory_path, 0o755)
            
            # Set permissions for existing database files
            for root, dirs, files in os.walk(directory_path):
                for dir_name in dirs:
                    dir_path = os.path.join(root, dir_name)
                    os.chmod(dir_path, 0o755)
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        os.chmod(file_path, 0o644)
                    except Exception as e:
                        logger.warning(f"Could not set permissions for {file_path}: {e}")
            
            logger.info(f"Updated permissions for directory: {directory_path}")
            
    except Exception as e:
        logger.error(f"Error setting directory permissions: {e}")
        raise


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
        self._db_cache = {}  # Cache for database connections
        
        # Ensure directory exists and has proper permissions
        ensure_directory_permissions(self.persist_directory)
    
    def _cleanup_database_if_corrupted(self, collection_name: str):
        """Clean up potentially corrupted database files."""
        try:
            collection_path = os.path.join(self.persist_directory, f"{collection_name}.sqlite3")
            if os.path.exists(collection_path):
                # Check if file is corrupted or has permission issues
                try:
                    # Try to get file stats
                    stat_info = os.stat(collection_path)
                    if stat_info.st_size == 0:
                        logger.warning(f"Database file is empty, removing: {collection_path}")
                        os.remove(collection_path)
                except Exception as e:
                    logger.warning(f"Database file may be corrupted, removing: {collection_path}")
                    try:
                        os.remove(collection_path)
                    except:
                        pass
        except Exception as e:
            logger.warning(f"Error during database cleanup: {e}")
    
    def store_documents(
        self, 
        documents: List[Document], 
        collection_name: str = "pdf_documents",
        max_retries: int = 3
    ) -> Chroma:
        """
        Store documents in the vector database.
        
        Args:
            documents: The documents to store.
            collection_name: The name of the collection.
            max_retries: Maximum number of retry attempts.
            
        Returns:
            The ChromaDB instance.
        """
        for attempt in range(max_retries):
            try:
                # Clear any cached connections
                self._clear_cache()
                
                # Ensure permissions are correct
                ensure_directory_permissions(self.persist_directory)
                
                # Try to load existing database first
                try:
                    existing_db = self.load_vector_store(collection_name)
                    # Test if we can write to it
                    existing_db.add_documents([Document(page_content="test", metadata={})])
                    # If successful, remove the test document and add real ones
                    existing_db.delete(where={"page_content": "test"})
                    existing_db.add_documents(documents)
                    logger.info(f"Added {len(documents)} documents to existing collection")
                    return existing_db
                except Exception as load_error:
                    logger.warning(f"Could not load existing database: {load_error}")
                    # Clean up potentially corrupted files
                    self._cleanup_database_if_corrupted(collection_name)
                
                # Create new database
                logger.info(f"Creating new ChromaDB collection: {collection_name}")
                db = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    persist_directory=self.persist_directory,
                    collection_name=collection_name
                )
                
                # Verify the database was created successfully
                test_search = db.similarity_search("test", k=1)
                logger.info(f"Successfully created database with {len(documents)} documents")
                return db
                
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
                
                if "readonly database" in str(e).lower():
                    # Try to fix permissions and clean up
                    self._cleanup_database_if_corrupted(collection_name)
                    ensure_directory_permissions(self.persist_directory)
                    time.sleep(1)  # Brief pause between attempts
                
                if attempt == max_retries - 1:
                    # Last attempt failed, try with a fresh directory
                    backup_dir = f"{self.persist_directory}_backup_{int(time.time())}"
                    if os.path.exists(self.persist_directory):
                        shutil.move(self.persist_directory, backup_dir)
                        logger.warning(f"Moved corrupted database to {backup_dir}")
                    
                    ensure_directory_permissions(self.persist_directory)
                    
                    try:
                        db = Chroma.from_documents(
                            documents=documents,
                            embedding=self.embeddings,
                            persist_directory=self.persist_directory,
                            collection_name=collection_name
                        )
                        logger.info(f"Successfully created database with fresh directory")
                        return db
                    except Exception as final_error:
                        logger.error(f"Final attempt failed: {final_error}")
                        raise final_error
        
        raise Exception(f"Failed to store documents after {max_retries} attempts")
    
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
        # Check cache first
        cache_key = f"{self.persist_directory}_{collection_name}"
        if cache_key in self._db_cache:
            return self._db_cache[cache_key]
        
        # Ensure permissions before loading
        ensure_directory_permissions(self.persist_directory)
        
        db = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name=collection_name
        )
        
        # Cache the database connection
        self._db_cache[cache_key] = db
        return db
    
    def search_documents(
        self, 
        query: str, 
        db: Optional[Chroma] = None, 
        k: int = 4,
        collection_name: str = "pdf_documents"
    ) -> List[Document]:
        """
        Search for relevant documents.
        
        Args:
            query: The search query.
            db: The vector database (optional, will load if not provided).
            k: Number of documents to retrieve.
            collection_name: The collection name (used if db is None).
            
        Returns:
            List of relevant documents.
        """
        if db is None:
            db = self.load_vector_store(collection_name)
        
        return db.similarity_search(query, k=k)
    
    def _clear_cache(self):
        """Clear the database cache."""
        self._db_cache.clear()
    
    def close_connections(self):
        """Close all database connections and clear cache."""
        try:
            # Clear the cache
            for db in self._db_cache.values():
                try:
                    # Try to close the connection if the method exists
                    if hasattr(db, '_client') and hasattr(db._client, 'reset'):
                        db._client.reset()
                except Exception as e:
                    logger.debug(f"Error closing individual connection: {e}")
            
            self._clear_cache()
            
            # Force garbage collection
            import gc
            gc.collect()
            
        except Exception as e:
            logger.warning(f"Error closing database connections: {str(e)}")
    
    def __del__(self):
        """Destructor to ensure connections are closed."""
        try:
            self.close_connections()
        except:
            pass
