from abc import ABC, abstractmethod
from typing import List
import numpy as np
import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

class BaseEmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        pass

    @abstractmethod
    def chroma_embedding_function(self):
        pass

class GeminiEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, model_name: str = "text-embedding-004"):
        """
        Initializes the Gemini Embedding Provider.
        The prefix 'models/' is removed if present for SDK compatibility.
        """
        self.model_name = model_name.replace("models/", "")
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
            
        # FIX: Correct initialization for the google-genai library
        # Keyword argument 'api_key' is correct for the Client constructor.
        self.client = genai.Client(api_key=api_key)

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generates embeddings using Gemini API."""
        if not texts:
            return []
        try:
            # Call the embedding model
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=texts
            )
            # Map the response to a list of float vectors
            return [embedding.values for embedding in response.embeddings]
        except Exception as e:
            print(f"DEBUG: Embedding Error with model '{self.model_name}': {e}")
            raise RuntimeError(f"Error during Gemini embedding: {str(e)}")

    def chroma_embedding_function(self):
        """Standard wrapper for ChromaDB integration."""
        class ChromaGeminiEF:
            def __init__(self, provider: GeminiEmbeddingProvider):
                self.provider = provider
            def __call__(self, input: List[str]):
                return self.provider.embed([str(text) for text in input])
        return ChromaGeminiEF(self)

class SentenceTransformerEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Local embedding using SentenceTransformers."""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
        except ImportError:
            raise ImportError("Please install sentence-transformers: pip install sentence-transformers")

    def embed(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts)
        return embeddings.tolist()

    def chroma_embedding_function(self):
        class ChromaSTEF:
            def __init__(self, provider: SentenceTransformerEmbeddingProvider):
                self.provider = provider
            def __call__(self, input: List[str]):
                return self.provider.embed(input)
        return ChromaSTEF(self)