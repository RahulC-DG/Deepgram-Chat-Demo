import sys
import importlib
import os
from dotenv import load_dotenv

def test_imports():
    """Test if all required packages can be imported."""
    packages = {
        'flask': 'Flask web framework',
        'flask_socketio': 'Flask-SocketIO for WebSocket support',
        'deepgram': 'Deepgram SDK',
        'langchain': 'LangChain core',
        'langchain_community': 'LangChain community packages',
        'langchain_openai': 'LangChain OpenAI integration',
        'openai': 'OpenAI API client',
        'faiss': 'FAISS vector store',
        'numpy': 'NumPy for numerical operations',
        'python_engineio': 'Python Engine.IO client',
        'python_socketio': 'Python Socket.IO client'
    }
    
    print("Testing package imports...")
    all_imports_ok = True
    for package, description in packages.items():
        try:
            importlib.import_module(package)
            print(f"✓ Successfully imported {package} ({description})")
        except ImportError as e:
            print(f"✗ Failed to import {package}: {str(e)}")
            all_imports_ok = False
    return all_imports_ok

def test_env_vars():
    """Test if required environment variables are set."""
    load_dotenv()
    required_vars = {
        'OPENAI_API_KEY': 'OpenAI API key for LLM and embeddings',
        'DEEPGRAM_API_KEY': 'Deepgram API key for voice processing',
        'EMBEDDING_MODEL': 'Model for text embeddings',
        'LLM_MODEL': 'Model for language generation'
    }
    
    print("\nTesting environment variables...")
    all_vars_ok = True
    for var, description in required_vars.items():
        if os.getenv(var):
            print(f"✓ {var} is set ({description})")
        else:
            print(f"✗ {var} is not set ({description})")
            all_vars_ok = False
    return all_vars_ok

def test_vector_stores():
    """Test if vector store directories exist."""
    required_dirs = {
        'data/vector_db/docs_store': 'Documentation vector store',
        'data/vector_db/sdk_store': 'SDK code vector store',
        'data/cache': 'Cache directory'
    }
    
    print("\nTesting vector stores and cache directories...")
    all_dirs_ok = True
    for dir_path, description in required_dirs.items():
        if os.path.exists(dir_path):
            print(f"✓ {dir_path} exists ({description})")
        else:
            print(f"✗ {dir_path} does not exist ({description})")
            all_dirs_ok = False
    return all_dirs_ok

def main():
    print("Starting dependency test...\n")
    
    # Test Python version
    print(f"Python version: {sys.version}")
    
    # Run all tests
    imports_ok = test_imports()
    env_ok = test_env_vars()
    stores_ok = test_vector_stores()
    
    # Print summary
    print("\nTest Summary:")
    print(f"Package imports: {'✓ OK' if imports_ok else '✗ Failed'}")
    print(f"Environment variables: {'✓ OK' if env_ok else '✗ Failed'}")
    print(f"Vector stores: {'✓ OK' if stores_ok else '✗ Failed'}")
    
    if all([imports_ok, env_ok, stores_ok]):
        print("\n✓ All tests passed! The installation is complete.")
        print("\nYou can now run either:")
        print("  - python chat.py (for text interface)")
        print("  - python Agent.py (for voice interface)")
    else:
        print("\n✗ Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main() 