"""
PDF document loader and processor.
"""
from typing import List, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class PDFProcessor:
    """Handles PDF document loading, parsing and chunking."""

    def __init__(
        self, 
        chunk_size: int = 1000, 
        chunk_overlap: int = 200
    ):
        """
        Initialize the PDF processor.
        
        Args:
            chunk_size: The size of each chunk.
            chunk_overlap: The overlap between chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def load_pdf(self, file_path: str) -> List[Document]:
        """
        Load a PDF file and split it into chunks.
        
        Args:
            file_path: Path to the PDF file.
            
        Returns:
            List of document chunks.
        """
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks.
        
        Args:
            documents: The documents to split.
            
        Returns:
            List of document chunks.
        """
        return self.text_splitter.split_documents(documents)
    
    def process_pdf(self, file_path: str) -> List[Document]:
        """
        Load and process a PDF file.
        
        Args:
            file_path: Path to the PDF file.
            
        Returns:
            List of document chunks.
        """
        documents = self.load_pdf(file_path)
        chunks = self.split_documents(documents)
        return chunks
