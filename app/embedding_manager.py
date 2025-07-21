"""
Embedding utilities optimized for Streamlit deployment.
Uses in-memory storage and session state to handle ChromaDB limitations.
"""
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import stat
import logging
import numpy as np
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
import time
import tempfile
import streamlit as st
import pickle
import uuid

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


class StreamlitEmbeddingManager:
    """
    Streamlit-compatible embedding manager that uses session state 
    and in-memory storage to avoid file system issues.
    """

    def __init__(
        self, 
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        use_session_state: bool = True
    ):
        """
        Initialize the embedding manager for Streamlit.
        
        Args:
            model_name: The embedding model to use.
            use_session_state: Whether to use Streamlit session state for persistence.
        """
        self.model_name = model_name
        self.use_session_state = use_session_state
        self.session_key = f"embedding_manager_{model_name.replace('/', '_')}"
        
        # Initialize embeddings
        try:
            logger.info(f"Loading embedding model: {model_name}")
            self.embeddings = TransformerEmbeddings(model_name=model_name)
            logger.info(f"Successfully loaded embedding model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading embedding model {model_name}: {str(e)}")
            # Fallback to a simpler model
            fallback_model = "sentence-transformers/all-MiniLM-L6-v2"
            if fallback_model != model_name:
                try:
                    logger.info(f"Attempting fallback model: {fallback_model}")
                    self.embeddings = TransformerEmbeddings(model_name=fallback_model)
                    logger.info(f"Successfully loaded fallback model: {fallback_model}")
                except Exception as fallback_error:
                    logger.error(f"Failed to load fallback model: {str(fallback_error)}")
                    raise fallback_error
            else:
                raise e
        
        # Initialize session state if using Streamlit
        if self.use_session_state and 'st' in sys.modules:
            self._init_session_state()
    
    def _init_session_state(self):
        """Initialize Streamlit session state for document storage."""
        if self.session_key not in st.session_state:
            st.session_state[self.session_key] = {
                'documents': {},
                'embeddings_cache': {},
                'vector_stores': {}
            }
    
    def _get_session_data(self) -> Dict[str, Any]:
        """Get session data, creating it if it doesn't exist."""
        if self.use_session_state and 'st' in sys.modules:
            if self.session_key not in st.session_state:
                self._init_session_state()
            return st.session_state[self.session_key]
        else:
            # Fallback to instance variable if not using Streamlit
            if not hasattr(self, '_local_storage'):
                self._local_storage = {
                    'documents': {},
                    'embeddings_cache': {},
                    'vector_stores': {}
                }
            return self._local_storage
    
    def store_documents(
        self, 
        documents: List[Document], 
        collection_name: str = "pdf_documents"
    ) -> str:
        """
        Store documents using in-memory approach compatible with Streamlit.
        
        Args:
            documents: The documents to store.
            collection_name: The name of the collection.
            
        Returns:
            Collection ID for retrieval.
        """
        try:
            logger.info(f"Storing {len(documents)} documents in memory for collection: {collection_name}")
            
            # Create a unique temp directory for this session
            temp_dir = tempfile.mkdtemp(prefix=f"streamlit_chroma_{collection_name}_")
            logger.info(f"Using temporary directory: {temp_dir}")
            
            # Create ChromaDB in the temp directory
            try:
                db = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    persist_directory=temp_dir,
                    collection_name=collection_name
                )
                
                # Test the database
                test_results = db.similarity_search("test query", k=1)
                logger.info(f"Successfully created ChromaDB with {len(documents)} documents")
                
                # Store in session state
                session_data = self._get_session_data()
                collection_id = f"{collection_name}_{uuid.uuid4().hex[:8]}"
                session_data['vector_stores'][collection_id] = {
                    'db': db,
                    'temp_dir': temp_dir,
                    'documents': documents,
                    'collection_name': collection_name
                }
                
                logger.info(f"Stored collection with ID: {collection_id}")
                return collection_id
                
            except Exception as chroma_error:
                logger.error(f"ChromaDB creation failed: {chroma_error}")
                # Fallback: store documents directly in session state
                return self._store_documents_fallback(documents, collection_name)
                
        except Exception as e:
            logger.error(f"Error storing documents: {str(e)}")
            # Ultimate fallback
            return self._store_documents_fallback(documents, collection_name)
    
    def _store_documents_fallback(self, documents: List[Document], collection_name: str) -> str:
        """Fallback method that stores documents and embeddings directly in session state."""
        logger.warning("Using fallback storage method - storing documents and embeddings in memory")
        
        try:
            # Generate embeddings for all documents
            texts = [doc.page_content for doc in documents]
            document_embeddings = self.embeddings.embed_documents(texts)
            
            # Store in session state
            session_data = self._get_session_data()
            collection_id = f"{collection_name}_fallback_{uuid.uuid4().hex[:8]}"
            
            session_data['documents'][collection_id] = documents
            session_data['embeddings_cache'][collection_id] = document_embeddings
            
            logger.info(f"Stored {len(documents)} documents using fallback method with ID: {collection_id}")
            return collection_id
            
        except Exception as fallback_error:
            logger.error(f"Fallback storage also failed: {fallback_error}")
            raise Exception(f"All storage methods failed. Last error: {fallback_error}")
    
    def load_vector_store(self, collection_id: str) -> Any:
        """
        Load a vector store by collection ID.
        
        Args:
            collection_id: The collection ID returned by store_documents.
            
        Returns:
            The vector store or fallback search object.
        """
        session_data = self._get_session_data()
        
        if collection_id in session_data['vector_stores']:
            store_info = session_data['vector_stores'][collection_id]
            return store_info['db']
        elif collection_id in session_data['documents']:
            # Return a fallback search object
            return FallbackVectorStore(
                documents=session_data['documents'][collection_id],
                embeddings_list=session_data['embeddings_cache'][collection_id],
                embeddings_function=self.embeddings
            )
        else:
            raise ValueError(f"Collection ID {collection_id} not found")
    
    def search_documents(
        self, 
        query: str, 
        collection_id: str,
        k: int = 4
    ) -> List[Document]:
        """
        Search for relevant documents.
        
        Args:
            query: The search query.
            collection_id: The collection ID.
            k: Number of documents to retrieve.
            
        Returns:
            List of relevant documents.
        """
        try:
            vector_store = self.load_vector_store(collection_id)
            return vector_store.similarity_search(query, k=k)
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            # Return empty list if search fails
            return []
    
    def list_collections(self) -> List[str]:
        """List all stored collection IDs."""
        session_data = self._get_session_data()
        vector_store_ids = list(session_data['vector_stores'].keys())
        fallback_ids = list(session_data['documents'].keys())
        return vector_store_ids + fallback_ids
    
    def clear_collection(self, collection_id: str):
        """Clear a specific collection from storage."""
        session_data = self._get_session_data()
        
        if collection_id in session_data['vector_stores']:
            # Clean up temp directory if it exists
            store_info = session_data['vector_stores'][collection_id]
            if 'temp_dir' in store_info:
                try:
                    import shutil
                    shutil.rmtree(store_info['temp_dir'], ignore_errors=True)
                except:
                    pass
            del session_data['vector_stores'][collection_id]
        
        if collection_id in session_data['documents']:
            del session_data['documents'][collection_id]
        
        if collection_id in session_data['embeddings_cache']:
            del session_data['embeddings_cache'][collection_id]
    
    def clear_all(self):
        """Clear all stored data."""
        session_data = self._get_session_data()
        
        # Clean up temp directories
        for store_info in session_data['vector_stores'].values():
            if 'temp_dir' in store_info:
                try:
                    import shutil
                    shutil.rmtree(store_info['temp_dir'], ignore_errors=True)
                except:
                    pass
        
        # Clear all data
        session_data['documents'].clear()
        session_data['embeddings_cache'].clear()
        session_data['vector_stores'].clear()


class FallbackVectorStore:
    """Simple fallback vector store using cosine similarity."""
    
    def __init__(self, documents: List[Document], embeddings_list: List[List[float]], embeddings_function):
        self.documents = documents
        self.embeddings_list = np.array(embeddings_list)
        self.embeddings_function = embeddings_function
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Perform similarity search using cosine similarity."""
        try:
            # Get query embedding
            query_embedding = np.array(self.embeddings_function.embed_query(query))
            
            # Calculate cosine similarities
            similarities = np.dot(self.embeddings_list, query_embedding) / (
                np.linalg.norm(self.embeddings_list, axis=1) * np.linalg.norm(query_embedding)
            )
            
            # Get top k most similar documents
            top_k_indices = np.argsort(similarities)[::-1][:k]
            
            return [self.documents[i] for i in top_k_indices if i < len(self.documents)]
        except Exception as e:
            logger.error(f"Error in fallback similarity search: {e}")
            return self.documents[:k]  # Return first k documents as fallback


# Convenience function for Streamlit apps
@st.cache_resource
def get_embedding_manager(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Get cached embedding manager for Streamlit app."""
    return StreamlitEmbeddingManager(model_name=model_name)
