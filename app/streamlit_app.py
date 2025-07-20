"""
Streamlit application for PDF Q&A.
"""
import os
import sys
import tempfile
import streamlit as st
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


def initialize_session_state():
    """Initialize session state variables."""
    if "orchestrator" not in st.session_state:
        st.session_state.orchestrator = None
    if "pdf_uploaded" not in st.session_state:
        st.session_state.pdf_uploaded = False
    if "pdf_name" not in st.session_state:
        st.session_state.pdf_name = None


def main():
    """Main application."""
    initialize_session_state()
    
    # App header
    st.title("📚 PDF Q&A with RAG")
    st.markdown("""
    Upload a PDF and ask questions about its contents.
    The system uses LangGraph, LangChain, and HuggingFace to provide answers.
    """)
    
    # Offline mode information
    if os.environ.get("TRANSFORMERS_OFFLINE") == "1" or not os.environ.get("HF_HUB_ENABLE_HF_TRANSFER"):
        st.warning("⚠️ Running in system-wide offline mode. Models must be pre-downloaded.")
        st.info("""
        To run in online mode, set these environment variables:
        ```
        set TRANSFORMERS_OFFLINE=0
        set HF_HUB_ENABLE_HF_TRANSFER=1
        ```
        """)
    
    # Sidebar for model configuration
    with st.sidebar:
        st.header("⚙️ Model Configuration")
        
        st.markdown("---")
        
        # Default models information with improved design
        with st.container():
            st.markdown("#### Models in Use")
            st.markdown("""
            <div style="background-color:#f0f2f6;padding:10px;border-radius:5px">
            <p><strong>Embeddings:</strong> sentence-transformers/all-MiniLM-L6-v2</p>
            <p><strong>Text Generation:</strong> mistralai/mistral-7b-instruct</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Hidden but used variable for number of documents to retrieve
        top_k = 4
        
        # Offline mode option
        local_files_only = st.checkbox(
            "Offline Mode (Use local models only)", 
            value=False,
            help="Enable to use only locally cached models without connecting to Hugging Face servers"
        )
        
        # Add a button to pre-download the default model
        if st.button("Download Default Models for Offline Use"):
            with st.status("Downloading models for offline use..."):
                try:
                    from huggingface_hub import snapshot_download
                    from sentence_transformers import SentenceTransformer
                    
                    # Download the embedding model
                    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
                    st.write(f"Downloading embedding model: {embedding_model}")
                    SentenceTransformer(embedding_model)
                    
                    st.success("✅ Default embedding model downloaded successfully for offline use!")
                except Exception as e:
                    st.error(f"Error downloading models: {str(e)}")
                    st.info("Try again with a valid Hugging Face token.")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
    
    # Process the uploaded PDF
    if uploaded_file and not st.session_state.pdf_uploaded:
        with st.spinner("Processing PDF..."):
            # Save the uploaded file
            temp_dir = tempfile.TemporaryDirectory()
            temp_path = os.path.join(temp_dir.name, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            # Initialize the orchestrator
            orchestrator = RAGOrchestrator(
                persist_directory=str(CHROMA_DIR),
                local_files_only=local_files_only,
                top_k=4  # Using fixed value instead of UI slider
            )
            
            # Process the PDF
            orchestrator.process_pdf(temp_path)
            
            # Update session state
            st.session_state.orchestrator = orchestrator
            st.session_state.pdf_uploaded = True
            st.session_state.pdf_name = uploaded_file.name
            
        st.success(f"PDF '{uploaded_file.name}' processed successfully!")
    
    # Q&A interface
    if st.session_state.pdf_uploaded:
        st.subheader(f"Ask questions about '{st.session_state.pdf_name}'")
        
        # Question input
        question = st.text_input("Your question:")
        
        if st.button("Get Answer"):
            if question:
                with st.spinner("Generating answer..."):
                    answer = st.session_state.orchestrator.answer_question(question)
                
                # Display the answer
                st.markdown("### Answer")
                st.markdown(answer)
            else:
                st.warning("Please enter a question.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Built with LangGraph, LangChain, and HuggingFace."
    )


if __name__ == "__main__":
    main()
