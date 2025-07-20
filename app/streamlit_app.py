"""
Streamlit application for PDF Q&A.
"""
import os
import sys
import tempfile
import streamlit as st
import shutil
from pathlib import Path
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load environment variables
load_dotenv()

from app.orchestrator import RAGOrchestrator
from huggingface_hub import try_to_load_from_cache
from huggingface_hub.utils import EntryNotFoundError

# Configure page
st.set_page_config(
    page_title="PDF Q&A with RAG",
    page_icon="📚",
    layout="centered"
)

# Define paths
DATA_DIR = Path("./data")
CHROMA_DIR = DATA_DIR / "chroma_db"
UPLOADS_DIR = DATA_DIR / "uploads"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)

def check_model_cached(model_id, filename="config.json"):
    """Check if a model is already downloaded in the cache."""
    try:
        path = try_to_load_from_cache(repo_id=model_id, filename=filename)
        return path is not None
    except EntryNotFoundError:
        return False


def clear_all_data():
    """Clear all embeddings and uploaded PDFs."""
    try:
        # First, clear the orchestrator to close any database connections
        if st.session_state.orchestrator is not None:
            try:
                # Try to delete the orchestrator and its connections
                del st.session_state.orchestrator
                st.session_state.orchestrator = None
            except:
                pass
        
        # Force garbage collection to help release file handles
        import gc
        gc.collect()
        
        # Small delay to allow file handles to be released
        import time
        time.sleep(0.5)
        
        # Clear ChromaDB directory
        if CHROMA_DIR.exists():
            try:
                shutil.rmtree(CHROMA_DIR)
            except PermissionError:
                # If we can't delete the folder, try to delete individual files
                for root, dirs, files in os.walk(CHROMA_DIR, topdown=False):
                    for file in files:
                        try:
                            os.remove(os.path.join(root, file))
                        except:
                            pass
                    for dir in dirs:
                        try:
                            os.rmdir(os.path.join(root, dir))
                        except:
                            pass
                # Try to remove the main directory
                try:
                    os.rmdir(CHROMA_DIR)
                except:
                    pass
            
            # Recreate the directory
            CHROMA_DIR.mkdir(exist_ok=True)
        
        # Clear uploads directory
        if UPLOADS_DIR.exists():
            shutil.rmtree(UPLOADS_DIR)
            UPLOADS_DIR.mkdir(exist_ok=True)
        
        # Clear session state
        st.session_state.orchestrator = None
        st.session_state.uploaded_pdfs = []
        
        return True
    except Exception as e:
        st.error(f"Error clearing data: {str(e)}")
        # Even if there's an error, still clear session state
        st.session_state.orchestrator = None
        st.session_state.uploaded_pdfs = []
        return False


def initialize_session_state():
    """Initialize session state variables."""
    if "orchestrator" not in st.session_state:
        st.session_state.orchestrator = None
    if "uploaded_pdfs" not in st.session_state:
        st.session_state.uploaded_pdfs = []
    
    # Clear data on app startup
    if "app_initialized" not in st.session_state:
        clear_all_data()
        st.session_state.app_initialized = True


def main():
    """Main application."""
    initialize_session_state()
    
    # App header
    st.title("📚 PDF Q&A with RAG")
    st.markdown("""
    Upload PDFs and ask questions about their contents.
    The system uses LangGraph, LangChain, and HuggingFace to provide answers.
    """)
    
    # Clear Data Button
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🗑️ Clear All Data", type="secondary"):
            with st.spinner("Clearing all data..."):
                if clear_all_data():
                    st.success("All data cleared successfully!")
                    # Force a rerun to refresh the UI
                    st.rerun()
                else:
                    st.warning("Some files might still be in use. Please restart the app if needed.")
    
    # Sidebar for model configuration
    with st.sidebar:
        st.header("⚙️ Model Configuration")
        
        st.markdown("---")
        
        # Default models information
        with st.container():
            st.markdown("#### Models in Use")
            st.markdown("""
            <div style="background-color:#f0f2f6;padding:10px;border-radius:5px">
            <p><strong>Embeddings:</strong> sentence-transformers/all-MiniLM-L6-v2</p>
            <p><strong>Text Generation:</strong> mistralai/mistral-7b-instruct</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Add a button to pre-download the default model
        if st.button("Download Default Models for Offline Use"):
            with st.status("Downloading models for offline use..."):
                try:
                    from sentence_transformers import SentenceTransformer
                    
                    # Download the embedding model
                    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
                    st.write(f"Downloading embedding model: {embedding_model}")
                    SentenceTransformer(embedding_model)
                    
                    st.success("✅ Default embedding model downloaded successfully for offline use!")
                except Exception as e:
                    st.error(f"Error downloading models: {str(e)}")
                    st.info("Try again with a valid Hugging Face token.")
    
    # Display uploaded PDFs
    if st.session_state.uploaded_pdfs:
        st.subheader("📄 Uploaded PDFs")
        for pdf_name in st.session_state.uploaded_pdfs:
            st.write(f"• {pdf_name}")
        st.markdown("---")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
    
    # Process the uploaded PDF
    if uploaded_file:
        # Check if this PDF is already uploaded
        if uploaded_file.name not in st.session_state.uploaded_pdfs:
            with st.spinner(f"Processing PDF: {uploaded_file.name}..."):
                # Save the uploaded file
                pdf_path = UPLOADS_DIR / uploaded_file.name
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                # Initialize orchestrator if not exists
                if st.session_state.orchestrator is None:
                    st.session_state.orchestrator = RAGOrchestrator(
                        persist_directory=str(CHROMA_DIR),
                        local_files_only=False,
                        top_k=4
                    )
                
                # Process the PDF
                st.session_state.orchestrator.process_pdf(str(pdf_path))
                
                # Add to uploaded PDFs list
                st.session_state.uploaded_pdfs.append(uploaded_file.name)
                
            st.success(f"PDF '{uploaded_file.name}' processed and added successfully!")
            st.rerun()
    
    # Q&A interface
    if st.session_state.uploaded_pdfs:
        st.subheader("❓ Ask Questions")
        
        # Question input
        question = st.text_input("Your question:")
        
        if st.button("Get Answer", type="primary"):
            if question:
                with st.spinner("Generating answer..."):
                    answer = st.session_state.orchestrator.answer_question(question)
                
                # Display the answer
                st.markdown("### Answer")
                st.markdown(answer)
            else:
                st.warning("Please enter a question.")
    else:
        st.info("Upload a PDF to start asking questions!")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Built with LangGraph, LangChain, and HuggingFace."
    )


if __name__ == "__main__":
    main()