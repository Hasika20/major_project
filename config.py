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
        return self.model.encode(texts, show_progress_bar=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text])[0].tolist()


class Config:
    """Central configuration for the RAG pipeline (local, no cloud needed)."""

    def __init__(self):
        # Project root is the directory this file lives in
        self.project_root = Path(__file__).parent
        self.data_dir = self.project_root / "data"
        self.vector_store_dir = self.data_dir / "vector_stores"

        # Create directories if needed
        self.data_dir.mkdir(exist_ok=True)
        self.vector_store_dir.mkdir(exist_ok=True)

        # Chunking parameters
        self.chunk_size = 1000
        self.chunk_overlap = 200

        # Free local AI models
        self.embedding_model_name = "all-MiniLM-L6-v2"
        self.llm_model_name = "llama3.2"

        # Lazy-loaded model handles
        self._embedding_model = None
        self._embeddings = None

    @property
    def embedding_model(self):
        if self._embedding_model is None:
            print(f"Loading embedding model: {self.embedding_model_name}...")
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
        return self._embedding_model

    @property
    def bedrock_embeddings(self):
        if self._embeddings is None:
            self._embeddings = SentenceTransformerEmbeddings(self.embedding_model)
        return self._embeddings

    def generate_response(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
        """Generate a response using the local Ollama LLM."""
        try:
            response = ollama.chat(
                model=self.llm_model_name,
                messages=[{"role": "user", "content": prompt}],
                options={"num_predict": max_tokens, "temperature": temperature},
            )
            return response["message"]["content"]
        except Exception as e:
            return (
                f"Error: {str(e)}\n\n"
                f"Make sure Ollama is installed and running:\n"
                f"  1. Install: https://ollama.com/download\n"
                f"  2. Pull model: ollama pull {self.llm_model_name}"
            )


# Singleton
config = Config()
