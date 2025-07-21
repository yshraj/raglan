"""
Embedding utilities for vector database operations using transformers.
Fixed version with better error handling and database management.
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
import tempfile
import sqlite3

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
                    try:
                        os.chmod(dir_path, 0o755)
                    except Exception as e:
                        logger.warning(f"Could not set permissions for directory {dir_path}: {e}")
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        os.chmod(file_path, 0o666)  # More permissive for files
                    except Exception as e:
                        logger.warning(f"Could not set permissions for {file_path}: {e}")
            
            logger.info(f"Updated permissions for directory: {directory_path}")
            
    except Exception as e:
        logger.error(f"Error setting directory permissions: {e}")
        raise


def test_sqlite_write_permissions(db_path):
    """Test if we can write to SQLite database in the given path."""
    test_db_path = os.path.join(db_path, "test_write.db")
    try:
        conn = sqlite3.connect(test_db_path)
        conn.execute("CREATE TABLE IF NOT EXISTS test (id INTEGER)")
        conn.execute("INSERT INTO test (id) VALUES (1)")
        conn.commit()
        conn.close()
        os.remove(test_db_path)
        return True
    except Exception as e:
        logger.warning(f"SQLite write test failed: {e}")
        if os.path.exists(test_db_path):
            try:
                os.remove(test_db_path)
            except:
                pass
        return False


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
        local_files_only: bool = False,
        use_temp_fallback: bool = True
    ):
        """
        Initialize the embedding manager.
        
        Args:
            model_name: The embedding model to use.
            persist_directory: Directory to store the vector database.
            local_files_only: Whether to use only local model files (offline mode).
            use_temp_fallback: Whether to use temporary directory as fallback.
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
                
        self.original_persist_directory = persist_directory
        self.use_temp_fallback = use_temp_fallback
        self._db_cache = {}  # Cache for database connections
        
        # Try to setup the persist directory
        self.persist_directory = self._setup_persist_directory()
    
    def _setup_persist_directory(self):
        """Setup the persist directory with proper permissions or fallback to temp."""
        try:
            # Try the original directory first
            ensure_directory_permissions(self.original_persist_directory)
            
            # Test if we can actually write SQLite databases here
            if test_sqlite_write_permissions(self.original_persist_directory):
                logger.info(f"Using original persist directory: {self.original_persist_directory}")
                return self.original_persist_directory
            else:
                raise Exception("Cannot write SQLite databases to original directory")
                
        except Exception as e:
            logger.warning(f"Cannot use original persist directory: {e}")
            
            if self.use_temp_fallback:
                # Fallback to a temporary directory
                temp_dir = tempfile.mkdtemp(prefix="chroma_db_")
                logger.warning(f"Falling back to temporary directory: {temp_dir}")
                return temp_dir
            else:
                raise e
    
    def _cleanup_database_if_corrupted(self, collection_name: str):
        """Clean up potentially corrupted database files."""
        try:
            # Look for various ChromaDB files that might exist
            patterns = [
                f"chroma.sqlite3",
                f"{collection_name}.sqlite3",
                f"*.sqlite3"
            ]
            
            for pattern in patterns:
                if '*' in pattern:
                    import glob
                    files = glob.glob(os.path.join(self.persist_directory, pattern))
                else:
                    files = [os.path.join(self.persist_directory, pattern)]
                
                for file_path in files:
                    if os.path.exists(file_path):
                        try:
                            # Check if file is corrupted or has permission issues
                            stat_info = os.stat(file_path)
                            if stat_info.st_size == 0:
                                logger.warning(f"Database file is empty, removing: {file_path}")
                                os.remove(file_path)
                            else:
                                # Try to open and check the database
                                conn = sqlite3.connect(file_path)
                                conn.execute("SELECT 1")
                                conn.close()
                        except Exception as e:
                            logger.warning(f"Database file may be corrupted, removing: {file_path}")
                            try:
                                os.remove(file_path)
                            except:
                                pass
                                
        except Exception as e:
            logger.warning(f"Error during database cleanup: {e}")
    
    def _create_fresh_directory(self):
        """Create a fresh directory for the database."""
        backup_dir = f"{self.original_persist_directory}_backup_{int(time.time())}"
        
        # Backup existing directory if it exists
        if os.path.exists(self.persist_directory):
            try:
                shutil.move(self.persist_directory, backup_dir)
                logger.warning(f"Moved corrupted database to {backup_dir}")
            except Exception as e:
                logger.warning(f"Could not backup directory: {e}")
                # Try to remove it instead
                try:
                    shutil.rmtree(self.persist_directory)
                except Exception as rm_error:
                    logger.error(f"Could not remove corrupted directory: {rm_error}")
        
        # Create fresh directory
        if self.use_temp_fallback and not test_sqlite_write_permissions(os.path.dirname(self.original_persist_directory)):
            # Use temp directory
            self.persist_directory = tempfile.mkdtemp(prefix="chroma_db_fresh_")
            logger.info(f"Created fresh temp directory: {self.persist_directory}")
        else:
            # Use original path
            self.persist_directory = self.original_persist_directory
            ensure_directory_permissions(self.persist_directory)
            logger.info(f"Created fresh directory: {self.persist_directory}")
    
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
        last_error = None
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempting to store documents (attempt {attempt + 1}/{max_retries})")
                
                # Clear any cached connections
                self._clear_cache()
                
                # Ensure permissions are correct
                ensure_directory_permissions(self.persist_directory)
                
                # For the first attempt, try to load existing database
                if attempt == 0:
                    try:
                        existing_db = self.load_vector_store(collection_name)
                        # Test if we can write to it with a simple operation
                        test_doc = Document(page_content="connection_test", metadata={"test": True})
                        existing_db.add_documents([test_doc])
                        
                        # If successful, remove the test document and add real ones
                        try:
                            existing_db.delete(where={"test": True})
                        except:
                            # Some versions might not support delete, that's ok
                            pass
                        
                        existing_db.add_documents(documents)
                        logger.info(f"Successfully added {len(documents)} documents to existing collection")
                        return existing_db
                        
                    except Exception as load_error:
                        logger.warning(f"Could not load/write to existing database: {load_error}")
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
                last_error = e
                logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
                
                if "readonly database" in str(e).lower() or "database is locked" in str(e).lower():
                    # Try to fix permissions and clean up
                    self._cleanup_database_if_corrupted(collection_name)
                    
                    # If this isn't the last attempt, try creating a fresh directory
                    if attempt < max_retries - 1:
                        self._create_fresh_directory()
                    
                    time.sleep(1)  # Brief pause between attempts
                
                elif attempt == max_retries - 1:
                    # Last attempt - try one more time with a completely fresh setup
                    logger.warning("Final attempt with fresh directory")
                    self._create_fresh_directory()
                    
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
                        last_error = final_error
        
        raise Exception(f"Failed to store documents after {max_retries} attempts. Last error: {last_error}")
    
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
        for db in self._db_cache.values():
            try:
                # Try to close the connection if the method exists
                if hasattr(db, '_client') and hasattr(db._client, 'reset'):
                    db._client.reset()
            except Exception as e:
                logger.debug(f"Error closing cached connection: {e}")
        
        self._db_cache.clear()
    
    def close_connections(self):
        """Close all database connections and clear cache."""
        try:
            self._clear_cache()
            
            # Force garbage collection
            import gc
            gc.collect()
            
        except Exception as e:
            logger.warning(f"Error closing database connections: {str(e)}")
    
    def get_persist_directory(self):
        """Get the current persist directory being used."""
        return self.persist_directory
    
    def __del__(self):
        """Destructor to ensure connections are closed."""
        try:
            self.close_connections()
        except:
            pass
