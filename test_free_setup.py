#!/usr/bin/env python3
"""
Simple test to verify the FREE local setup is working.

Tests:
1. Ollama connection
2. Sentence Transformers embeddings
3. Local storage
4. Config loading
"""

import sys
from pathlib import Path

# Add Admin to path
sys.path.insert(0, str(Path(__file__).parent / "Admin"))

def test_ollama():
    """Test Ollama is installed and working."""
    print("🔍 Testing Ollama...")
    try:
        import ollama
        # Try to list models
        models = ollama.list()
        print(f"✅ Ollama working! Found {len(models.get('models', []))} models")
        
        # Check for llama3.2
        model_names = [m['name'] for m in models.get('models', [])]
        if any('llama3.2' in name for name in model_names):
            print("✅ llama3.2 model found!")
        else:
            print("⚠️  llama3.2 not found. Run: ollama pull llama3.2")
        return True
    except Exception as e:
        print(f"❌ Ollama test failed: {e}")
        print("💡 Install Ollama: https://ollama.com/download")
        return False

def test_embeddings():
    """Test Sentence Transformers embeddings."""
    print("\n🔍 Testing Sentence Transformers...")
    try:
        from config import config
        
        # Try to load embedding model
        embeddings = config.bedrock_embeddings
        
        # Test embedding generation
        test_text = "This is a test."
        vector = embeddings.embed_query(test_text)
        
        print(f"✅ Embeddings working! Vector dimension: {len(vector)}")
        return True
    except Exception as e:
        print(f"❌ Embeddings test failed: {e}")
        print("💡 Run: pip install sentence-transformers")
        return False

def test_local_storage():
    """Test local storage setup."""
    print("\n🔍 Testing local storage...")
    try:
        from config import config
        
        # Check directories exist
        if config.data_dir.exists():
            print(f"✅ Data directory exists: {config.data_dir}")
        else:
            print(f"⚠️  Data directory will be created: {config.data_dir}")
        
        if config.vector_store_dir.exists():
            print(f"✅ Vector store directory exists: {config.vector_store_dir}")
            
            # Count existing vector stores
            stores = [d for d in config.vector_store_dir.iterdir() if d.is_dir()]
            print(f"📁 Found {len(stores)} existing vector stores")
        else:
            print(f"⚠️  Vector store directory will be created: {config.vector_store_dir}")
        
        return True
    except Exception as e:
        print(f"❌ Storage test failed: {e}")
        return False

def test_config():
    """Test configuration loading."""
    print("\n🔍 Testing configuration...")
    try:
        from config import config
        
        print(f"✅ Config loaded")
        print(f"   Embedding model: {config.embedding_model_name}")
        print(f"   LLM model: {config.llm_model_name}")
        print(f"   Chunk size: {config.chunk_size}")
        print(f"   Chunk overlap: {config.chunk_overlap}")
        return True
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 FREE Local AI Setup Test")
    print("=" * 50)
    
    results = []
    
    results.append(("Ollama", test_ollama()))
    results.append(("Embeddings", test_embeddings()))
    results.append(("Local Storage", test_local_storage()))
    results.append(("Configuration", test_config()))
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print("=" * 50)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    
    print(f"\n🎯 Summary: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! You're ready to use the FREE version!")
        print("\n📝 Next steps:")
        print("1. Run Admin interface: streamlit run Admin/admin.py --server.port 8501")
        print("2. Process some PDFs")
        print("3. Run User interface: streamlit run User/app.py --server.port 8502")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        print("\n💡 Quick fixes:")
        print("- Install Ollama: https://ollama.com/download")
        print("- Download model: ollama pull llama3.2")
        print("- Install packages: pip install -r Admin/requirements.txt")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
