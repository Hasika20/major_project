"""
Configuration module for the admin interface.
FREE VERSION - Uses local AI models (no cloud costs!)
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from typing import List
import ollama

# Load environment variables
load_dotenv()

class SentenceTransformerEmbeddings(Embeddings):
    """Wrapper for sentence-transformers to work with LangChain."""
    
    def __init__(self, model):
        self.model = model
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        return self.model.encode(texts, show_progress_bar=True).tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        return self.model.encode([text])[0].tolist()

class Config:
    """Configuration class for FREE local AI settings (no AWS needed!)."""
    
    def __init__(self):
        # Local storage paths (replaces S3)
        self.project_root = Path(__file__).parent.parent
        self.data_dir = self.project_root / "data"
        self.vector_store_dir = self.data_dir / "vector_stores"
        
        # Create directories if they don't exist
        self.data_dir.mkdir(exist_ok=True)
        self.vector_store_dir.mkdir(exist_ok=True)
        
        # Processing parameters
        self.chunk_size = 1000
        self.chunk_overlap = 200
        
        # FREE AI Models
        self.embedding_model_name = "all-MiniLM-L6-v2"  # 80MB, free, local
        self.llm_model_name = "llama3.2"  # Free via Ollama (install: ollama pull llama3.2)
        
        # Initialize models (lazy loading)
        self._embedding_model = None
        self._embeddings = None
    
    @property
    def embedding_model(self):
        """Lazy initialization of sentence-transformer model."""
        if self._embedding_model is None:
            print(f"📥 Loading free embedding model: {self.embedding_model_name}...")
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
            print("✅ Embedding model loaded!")
        return self._embedding_model
    
    @property
    def bedrock_embeddings(self):
        """Lazy initialization of embeddings (compatible with old code)."""
        if self._embeddings is None:
            self._embeddings = SentenceTransformerEmbeddings(self.embedding_model)
        return self._embeddings
    
    def generate_response(self, prompt, max_tokens=512, temperature=0.2):
        """Generate response using FREE local Ollama LLM."""
        try:
            response = ollama.chat(
                model=self.llm_model_name,
                messages=[{'role': 'user', 'content': prompt}],
                options={
                    'num_predict': max_tokens,
                    'temperature': temperature
                }
            )
            return response['message']['content']
        except Exception as e:
            return f"Error: {str(e)}\n\nMake sure Ollama is installed and running:\n1. Install: https://ollama.com/download\n2. Pull model: ollama pull {self.llm_model_name}"

# Global configuration instance
config = Config()
