"""
LangGraph orchestrator for RAG workflow.
"""
from typing import Dict, List, Annotated, TypedDict, Sequence
import os

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

from app.pdf_processor import PDFProcessor
from app.embedding_manager import EmbeddingManager
from app.llm_manager import LLMManager


class RAGState(TypedDict):
    """The state of the RAG workflow."""
    question: str
    documents: List[Document]
    context: str
    answer: str


def create_rag_graph(
    embedding_manager: EmbeddingManager,
    llm_manager: LLMManager,
    collection_name: str = "pdf_documents",
    top_k: int = 4
):
    """
    Create a RAG workflow graph.
    
    Args:
        embedding_manager: The embedding manager.
        llm_manager: The LLM manager.
        collection_name: The vector DB collection name.
        top_k: Number of documents to retrieve.
        
    Returns:
        A LangGraph StateGraph.
    """
    # Define the workflow states
    workflow = StateGraph(RAGState)
    
    # Step 1: Retrieve relevant documents
    def retrieve_documents(state: RAGState) -> RAGState:
        """Retrieve relevant documents from the vector store."""
        question = state["question"]
        
        # Load the vector store
        db = embedding_manager.load_vector_store(collection_name=collection_name)
        
        # Search for relevant documents
        docs = embedding_manager.search_documents(question, db, k=top_k)
        
        return {"documents": docs}
    
    # Step 2: Create context from documents
    def create_context(state: RAGState) -> RAGState:
        """Create a context from the retrieved documents."""
        docs = state["documents"]
        
        # Extract and format document content
        context_parts = []
        for i, doc in enumerate(docs):
            context_parts.append(f"Document {i+1}:\n{doc.page_content}")
        
        context = "\n\n".join(context_parts)
        
        return {"context": context}
    
    # Step 3: Generate answer using LLM
    def generate_answer(state: RAGState) -> RAGState:
        """Generate an answer using the LLM."""
        question = state["question"]
        context = state["context"]
        
        prompt = f"""You are a helpful assistant that answers questions based on the provided context.
        
Context:
{context}

Question: {question}

Answer the question based only on the provided context. If the context doesn't contain enough information to answer the question, say "I don't have enough information to answer this question."
"""
        answer = llm_manager.generate(prompt)
        
        return {"answer": answer}
    
    # Add nodes to the graph
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("create_context", create_context)
    workflow.add_node("generate", generate_answer)
    
    # Add edges
    workflow.add_edge("retrieve", "create_context")
    workflow.add_edge("create_context", "generate")
    workflow.add_edge("generate", END)
    
    # Set the entry point
    workflow.set_entry_point("retrieve")
    
    # Compile the graph
    return workflow.compile()


class RAGOrchestrator:
    """Orchestrates the RAG workflow."""
    
    def __init__(
        self,
        persist_directory: str = "./data/chroma_db",
        collection_name: str = "pdf_documents",
        top_k: int = 4,
        local_files_only: bool = False
    ):
        """
        Initialize the RAG orchestrator.
        
        Args:
            persist_directory: Directory to store the vector database.
            collection_name: The vector DB collection name.
            top_k: Number of documents to retrieve.
            local_files_only: Whether to use only local model files (offline mode).
        """
        # Using fixed default models
        embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        llm_model = "mistralai/mistral-7b-instruct:free"
        self.collection_name = collection_name
        self.top_k = top_k
        
        # Initialize components
        self.pdf_processor = PDFProcessor()
        self.embedding_manager = EmbeddingManager(
            model_name=embedding_model,
            persist_directory=persist_directory
        )
        self.llm_manager = LLMManager(
            model_name=llm_model,
            temperature=0.7,
            max_tokens=512
        )
        
        # Create the graph
        self.graph = create_rag_graph(
            self.embedding_manager,
            self.llm_manager,
            collection_name=collection_name,
            top_k=top_k
        )
    
    def process_pdf(self, pdf_path: str) -> None:
        """
        Process a PDF and store it in the vector database.
        
        Args:
            pdf_path: Path to the PDF file.
        """
        # Load and chunk the PDF
        chunks = self.pdf_processor.process_pdf(pdf_path)
        
        # Store the chunks in the vector database
        self.embedding_manager.store_documents(
            chunks,
            collection_name=self.collection_name
        )
    
    def answer_question(self, question: str) -> str:
        """
        Answer a question using the RAG workflow.
        
        Args:
            question: The user's question.
            
        Returns:
            The generated answer.
        """
        # Run the workflow
        result = self.graph.invoke({"question": question})
        
        return result["answer"]
