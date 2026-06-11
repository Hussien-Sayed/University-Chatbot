import os
from typing import List, Optional

from huggingface_hub import InferenceClient


class EmbeddingAPI:
    """Class for generating embeddings using HuggingFace Inference Client"""

    def __init__(self, model_name: Optional[str] = None, api_key: Optional[str] = None):
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.api_key = api_key or os.getenv("HUGGINGFACE_API_KEY")
        self.embedding_call_count = 0

        if not self.api_key:
            raise ValueError("api_key must be provided or set in .env as HUGGINGFACE_API_KEY")

        self.client = InferenceClient(
            model=self.model_name,
            token=self.api_key
        )

    def reset_counters(self) -> None:
        """Reset the embedding API call counter."""
        self.embedding_call_count = 0

    def _increment_counter(self) -> None:
        """Increment the embedding API call counter."""
        self.embedding_call_count += 1

    def generate_embedding(self, text: str) -> List[float]:
        if not text or not text.strip():
            raise ValueError("text cannot be empty")

        self._increment_counter()
        embedding = self.client.feature_extraction(text, normalize=True, truncate=True)

        if hasattr(embedding, 'tolist'):
            embedding = embedding.tolist()

        if isinstance(embedding, list) and len(embedding) > 0:
            if isinstance(embedding[0], list):
                return embedding[0]
            return embedding

        raise RuntimeError(f"Unexpected embedding format: {type(embedding)}")

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            raise ValueError("texts list cannot be empty")

        embeddings = []
        for text in texts:
            embeddings.append(self.generate_embedding(text))

        return embeddings
