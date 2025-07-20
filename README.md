# 📚 RAGLan: LangGraph-Powered PDF Q&A

A simple RAG-based system that allows users to upload PDF documents and ask questions about their content. This project uses LangGraph, OpenRouter and SentenceTransformers to create a powerful question-answering system based on the Retrieval-Augmented Generation (RAG) pattern.

## 🛠️ Tech Stack

- **LangGraph**: Orchestrates the QnA flow
- **LangChain**: For document parsing and chaining
- **ChromaDB**: Local vector DB to store document embeddings
- **OpenRouter**: Cloud API for accessing powerful LLMs
- **Sentence Transformers**: For document embeddings
- **Streamlit**: Simple web interface for file upload + Q&A

## ✨ Features

- Upload and process PDF documents
- Parse and chunk PDFs using LangChain
- Embed chunks using SentenceTransformers
- Store vectors in a local ChromaDB instance
- Query the system with natural language questions
- Get context-aware answers from powerful cloud LLMs via OpenRouter
- Support for both online and offline embedding models

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Pip
- Optional: CUDA-compatible GPU for faster inference

### Installation

1. Clone the repository
```bash
git clone https://github.com/raglan-team/raglan.git
cd raglan
```

2. Create a virtual environment
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix/macOS
source venv/bin/activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Set up API keys
```bash
# Copy the sample env file
cp .env.sample .env

# Edit the .env file with your API keys
# - Get an OpenRouter API key from https://openrouter.ai/keys
# - Get a Hugging Face token from https://huggingface.co/settings/tokens
```

5. Run the health check to validate your environment
```bash
python healthcheck.py
```

This will check if your environment is properly configured and if all required dependencies are installed.

### Running the Application

Use the unified entry point:

```bash
# Run the Streamlit UI
python app.py --ui

# Run the CLI version
python app.py --pdf path/to/document.pdf --question "Your question about the document"
```

### Using Offline Mode

If you're experiencing connectivity issues with Hugging Face servers, you can use the application in offline mode:

#### Option 1: Use the Streamlit UI

1. Run the app with internet connectivity first
2. Click the "Download Default Models for Offline Use" button to cache the default embedding model
3. When you encounter connectivity issues later, enable the "Offline Mode" checkbox in the sidebar

#### Option 2: Use the download script

1. Run the download script to pre-cache models:
```bash
# Download the default set of models (recommended for offline use)
python download_models.py

# Or specify specific models to download
python download_models.py --models "gpt2" "distilbert-base-uncased"
```

2. When running the app offline, enable the "Offline Mode" checkbox in the sidebar

#### Option 3: Set environment variables for system-wide offline mode

```bash
# On Windows:
set TRANSFORMERS_OFFLINE=1
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
set HF_HUB_OFFLINE=1

# On Linux/macOS:
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export HF_HUB_OFFLINE=1
```

For more information on Hugging Face's offline mode, see: https://huggingface.co/docs/transformers/installation#offline-mode

### Troubleshooting Offline Mode

If you're having issues with offline mode:

1. Try smaller models like 'distilgpt2' for LLM and 'distilbert-base-uncased' for embeddings
2. Check if models are cached by looking for the green checkmark in the UI
3. Run the download script before going offline

## 🧪 Running Tests

To run the tests, you'll need a test PDF. You can generate one automatically:

```bash
# Generate a test PDF
python create_test_pdf.py

# Run the tests
pytest
```

This will create a sample PDF in the `tests/resources` directory and run the test suite.

## 📖 Usage Guide

### Command Line Interface

You can use RAGLan from the command line:

```bash
# Process a PDF and ask a question
python main.py --pdf path/to/your/document.pdf --question "What is the document about?"

# Customize number of retrieved documents
python main.py --pdf path/to/your/document.pdf --question "What is the document about?" --top-k 6
```

### Web Interface

For a more interactive experience, use the Streamlit web interface:

```bash
python run_app.py
```

Then open your browser to http://localhost:8501

### Installation as a Package

You can also install RAGLan as a package:

```bash
pip install -e .
```

Then use the provided commands:

```bash
# Command line
raglan --pdf path/to/your/document.pdf --question "What is the document about?"

# Web interface
raglan-ui
```
4. If a specific model doesn't work offline, the app will try to use fallback models

4. Configure OpenRouter API token
```bash
# Option 1: Create/edit .env file
echo "OPENROUTER_API_KEY=your_token_here" > .env

# Option 2: Set environment variable directly
# On Windows
set OPENROUTER_API_KEY=your_token_here
# On Unix/macOS
export OPENROUTER_API_KEY=your_token_here
```

> **Note**: You need an OpenRouter API key to access LLM models. Get it from [openrouter.ai/keys](https://openrouter.ai/keys)

### Free OpenRouter Models

The application uses free models by default:

- **mistralai/mistral-7b-instruct:free** - Primary model
- **google/gemma-7b-it:free** - Fallback model

Other free models you can use:
- **meta-llama/llama-3-8b-instruct:free**
- **openchat/openchat-7b:free**
- **microsoft/phi-3-mini-4k-instruct:free**
- **nousresearch/nous-hermes-2-mixtral-8x7b-dpo:free**

For more models, check the [OpenRouter documentation](https://openrouter.ai/docs#models)

### Running the Application

#### CLI Mode

Process a PDF and ask a question:
```bash
python main.py --pdf path/to/your/document.pdf --question "What is the main topic of the document?"
```

#### Web Interface

Launch the Streamlit app:
```bash
# Option 1: Using the launcher script
python run_app.py

# Option 2: Direct streamlit command
streamlit run app/streamlit_app.py
```

Then open your browser to http://localhost:8501

## 📂 Project Structure

```
raglan/
├── app/
│   ├── __init__.py
│   ├── pdf_processor.py    # PDF loading and chunking
│   ├── embedding_manager.py  # Vector DB operations
│   ├── llm_manager.py      # HuggingFace model wrapper
│   ├── orchestrator.py     # LangGraph workflow
│   └── streamlit_app.py    # Web interface
├── data/
│   ├── chroma_db/          # Vector DB storage
│   └── uploads/            # PDF uploads (web interface)
├── tests/
│   ├── resources/          # Test resources
│   └── test_pdf_processor.py  # Unit tests
├── main.py                 # CLI entry point
├── requirements.txt        # Dependencies
└── README.md               # Documentation
```

## 🎛️ Configuration Options

- **Embedding Models**: Choose from various SentenceTransformers models
- **Language Models**: Support for different HuggingFace models
  - mistralai/Mistral-7B-Instruct-v0.2 (default)
  - google/flan-t5-base (lighter alternative)
- **Chunking Parameters**: Customize document chunking size and overlap
- **Retrieval Settings**: Adjust the number of chunks retrieved for context

## 📊 Performance Considerations

- **GPU Acceleration**: Using CUDA can significantly speed up inference
- **Quantization**: Models are loaded with 4-bit quantization when a GPU is available
- **Model Size**: Smaller models trade accuracy for speed and memory usage
- **Chunking Strategy**: Smaller chunks improve retrieval precision but may lose context

## 🧪 Running Tests

```bash
pytest tests/
```

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- LangChain team for the document processing tools
- HuggingFace for providing open-access models
- LangGraph team for the workflow orchestration framework

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py