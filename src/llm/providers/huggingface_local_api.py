"""HuggingFace local embedding provider using transformers pipeline."""
import os
from typing import List, Optional


class HuggingFaceLocalProvider:
    """Embedding provider using HuggingFace Transformers locally (no API key required)."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

        try:
            from transformers import pipeline
            self._pipeline = pipeline(
                "feature-extraction",
                model=self.model_name,
                tokenizer=self.model_name
            )
        except ImportError:
            raise ImportError("transformers package is required. Install it with: pip install transformers torch")

    def _generate_embedding(self, text: str) -> List[float]:
        """Generate a single embedding using local transformers pipeline.

        Args:
            text: Input text to embed

        Returns:
            Embedding as a list of floats (mean-pooled over tokens)
        """
        output = self._pipeline(text, return_tensors=False)

        if isinstance(output, list) and len(output) > 0:
            token_embeddings = output[0]
            if isinstance(token_embeddings[0], list):
                num_tokens = len(token_embeddings)
                dim = len(token_embeddings[0])
                mean_embedding = [
                    sum(token_embeddings[t][d] for t in range(num_tokens)) / num_tokens
                    for d in range(dim)
                ]
                return mean_embedding

        raise RuntimeError(f"Unexpected embedding format from local pipeline: {type(output)}")
