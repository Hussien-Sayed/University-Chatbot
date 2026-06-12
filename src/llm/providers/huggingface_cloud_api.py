"""HuggingFace cloud embedding provider using InferenceClient."""
import os
from typing import List, Optional


class HuggingFaceCloudProvider:
    """Embedding provider using HuggingFace Inference API (cloud)."""

    def __init__(self, model_name: Optional[str] = None, api_key: Optional[str] = None):
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.api_key = api_key or os.getenv("HUGGINGFACE_API_KEY")

        if not self.api_key:
            raise ValueError("api_key must be provided or set in .env as HUGGINGFACE_API_KEY")

        try:
            from huggingface_hub import InferenceClient
            self.client = InferenceClient(model=self.model_name, token=self.api_key)
        except ImportError:
            raise ImportError("huggingface_hub package is required. Install it with: pip install huggingface_hub")

    def _generate_embedding(self, text: str) -> List[float]:
        """Generate a single embedding via HuggingFace Inference API.

        Args:
            text: Input text to embed

        Returns:
            Embedding as a list of floats
        """
        embedding = self.client.feature_extraction(text, normalize=True, truncate=True)

        if hasattr(embedding, 'tolist'):
            embedding = embedding.tolist()

        if isinstance(embedding, list) and len(embedding) > 0:
            if isinstance(embedding[0], list):
                return embedding[0]
            return embedding

        raise RuntimeError(f"Unexpected embedding format: {type(embedding)}")
