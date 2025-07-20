"""
Main application entry point.
Handles both CLI and Streamlit interfaces.
"""
import os
import sys
import argparse
import subprocess
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.orchestrator import RAGOrchestrator


def run_cli():
    """
    Command line interface for the PDF Q&A system.
    """
    parser = argparse.ArgumentParser(description="PDF Q&A with RAG")
    parser.add_argument("--pdf", type=str, help="Path to the PDF file")
    parser.add_argument("--question", type=str, help="Question to ask about the PDF")
    parser.add_argument("--top-k", type=int, default=4,
                       help="Number of documents to retrieve")
    
    args = parser.parse_args()
    
    # Define paths
    data_dir = Path("./data")
    chroma_dir = data_dir / "chroma_db"
    
    # Ensure directories exist
    data_dir.mkdir(exist_ok=True)
    chroma_dir.mkdir(exist_ok=True)
    
    # Initialize the orchestrator
    orchestrator = RAGOrchestrator(
        persist_directory=str(chroma_dir),
        top_k=args.top_k
    )
    
    # Process the PDF if provided
    if args.pdf:
        print(f"Processing PDF: {args.pdf}")
        orchestrator.process_pdf(args.pdf)
        print("PDF processed successfully!")
    
    # Answer the question if provided
    if args.question:
        if not args.pdf and not orchestrator.embedding_manager.load_vector_store().get("_collection"):
            print("Error: No PDF has been processed. Please provide a PDF file first.")
            return
            
        print(f"\nQuestion: {args.question}")
        print("\nGenerating answer...")
        
        answer = orchestrator.answer_question(args.question)
        
        print("\nAnswer:")
        print(answer)


def run_streamlit():
    """Run the Streamlit application."""
    # Get the absolute path to the app directory
    app_dir = Path(__file__).parent
    app_file = app_dir / "app" / "streamlit_app.py"
    
    # Run Streamlit
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_file), "--server.headless", "true"]
    print(f"Starting Streamlit app: {app_file}")
    subprocess.run(cmd)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="RAGLan PDF Q&A System")
    parser.add_argument("--ui", action="store_true", help="Launch the Streamlit UI")
    
    args, unknown = parser.parse_known_args()
    
    if args.ui:
        run_streamlit()
    else:
        run_cli()
