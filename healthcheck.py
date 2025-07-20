"""
Healthcheck script to validate the environment and dependencies.
"""
import os
import sys
import importlib
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_python_version():
    """Check if Python version is compatible."""
    print(f"Python Version: {sys.version}")
    major, minor, *_ = sys.version_info
    if major < 3 or (major == 3 and minor < 9):
        print("❌ Python version must be 3.9 or higher")
        return False
    print("✅ Python version is compatible")
    return True

def check_dependencies():
    """Check if all required dependencies are installed."""
    print("\nChecking dependencies...")
    
    required_packages = [
        "langchain",
        "langchain_community",
        "langchain_openai",
        "langchain_core",
        "langgraph",
        "sentence_transformers",
        "chromadb",
        "transformers",
        "torch",
        "pypdf",
        "streamlit",
        "openai",
        "requests",
        "python-dotenv",
        "tqdm"
    ]
    
    all_installed = True
    for package in required_packages:
        try:
            importlib.import_module(package.replace("-", "_"))
            print(f"✅ {package} is installed")
        except ImportError:
            print(f"❌ {package} is not installed")
            all_installed = False
    
    return all_installed

def check_environment_variables():
    """Check if necessary environment variables are set."""
    print("\nChecking environment variables...")
    
    env_vars = {
        "OPENROUTER_API_KEY": "OpenRouter API key",
        "HUGGINGFACE_TOKEN": "Hugging Face API token (optional but recommended)"
    }
    
    all_set = True
    for var, description in env_vars.items():
        if var == "HUGGINGFACE_TOKEN":
            if os.getenv(var):
                print(f"✅ {description} is set")
            else:
                print(f"⚠️ {description} is not set (optional)")
        else:
            if os.getenv(var):
                print(f"✅ {description} is set")
            else:
                print(f"❌ {description} is not set")
                all_set = False
    
    return all_set

def check_directories():
    """Check if necessary directories exist."""
    print("\nChecking directories...")
    
    directories = [
        Path("./data"),
        Path("./data/chroma_db"),
        Path("./data/uploads")
    ]
    
    all_exist = True
    for directory in directories:
        if directory.exists():
            print(f"✅ Directory {directory} exists")
        else:
            print(f"❌ Directory {directory} does not exist")
            all_exist = False
            # Create the directory
            try:
                directory.mkdir(parents=True, exist_ok=True)
                print(f"✅ Created directory {directory}")
            except Exception as e:
                print(f"❌ Failed to create directory {directory}: {str(e)}")
    
    return all_exist

def check_gpu():
    """Check if CUDA is available."""
    print("\nChecking GPU support...")
    
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            print(f"✅ CUDA is available ({device_count} devices, using {device_name})")
            return True
        else:
            print("⚠️ CUDA is not available - the app will run on CPU, which might be slower")
            return False
    except ImportError:
        print("❌ PyTorch is not installed properly")
        return False

def check_app_file():
    """Check if app.py exists and is executable."""
    print("\nChecking application file...")
    
    app_file = Path("./app.py")
    if app_file.exists():
        print(f"✅ Application entry point {app_file} exists")
        return True
    else:
        print(f"❌ Application entry point {app_file} does not exist")
        return False

def main():
    """Run all health checks."""
    print("🏥 RAGLan Health Check\n")
    
    results = {
        "Python Version": check_python_version(),
        "Dependencies": check_dependencies(),
        "Environment Variables": check_environment_variables(),
        "Directories": check_directories(),
        "Application File": check_app_file(),
        "GPU Support": check_gpu()
    }
    
    print("\n🏥 Health Check Summary:")
    all_passed = True
    for check, passed in results.items():
        if check == "GPU Support":
            # GPU support is optional
            status = "✅" if passed else "⚠️"
            print(f"{status} {check}: {'OK' if passed else 'Not Available (CPU mode)'}")
        else:
            status = "✅" if passed else "❌"
            print(f"{status} {check}: {'OK' if passed else 'Failed'}")
            if not passed and check != "GPU Support":
                all_passed = False
    
    if all_passed:
        print("\n✅ All critical checks passed! The application should work correctly.")
        print("\nYou can run the application with:")
        print("  - python app.py --ui           (for the web interface)")
        print("  - python app.py --pdf <file> --question \"<your question>\"  (for CLI)")
    else:
        print("\n❌ Some checks failed. Please fix the issues and run the health check again.")
        print("   Check the README.md for more information on setting up the environment.")

if __name__ == "__main__":
    main()
