import os
from typing import List, Optional

from .providers.huggingface_cloud_api import HuggingFaceCloudProvider
from .providers.huggingface_local_api import HuggingFaceLocalProvider


class EmbeddingAPI:
    """Class for generating embeddings supporting cloud and local providers."""

    def __init__(
        self,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        """Initialize EmbeddingAPI with specified provider.

        Args:
            provider: Embedding provider to use ("cloud" or "local"). Defaults to env var EMBEDDING_PROVIDER or "cloud".
            model_name: Model name to use. Defaults to EMBEDDING_MODEL env var.
            api_key: API key (only used for cloud provider).
        """
        self.provider_name = (provider or os.getenv("EMBEDDING_PROVIDER", "cloud")).lower()
        self.embedding_call_count = 0

        if self.provider_name == "cloud":
            self.provider = HuggingFaceCloudProvider(model_name=model_name, api_key=api_key)
            self.model_name = self.provider.model_name
            self.api_key = self.provider.api_key
        elif self.provider_name == "local":
            self.provider = HuggingFaceLocalProvider(model_name=model_name)
            self.model_name = self.provider.model_name
            self.api_key = None
        else:
            raise ValueError(f"Unknown embedding provider: {self.provider_name}. Supported: cloud, local")

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
        return self.provider._generate_embedding(text)

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            raise ValueError("texts list cannot be empty")

        embeddings = []
        for text in texts:
            embeddings.append(self.generate_embedding(text))

        return embeddings
