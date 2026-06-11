"""Ollama API provider for local LLM calls."""
import os
from typing import List, Dict, Any, Optional


class OllamaProvider:
    """Provider for local Ollama API calls using the official ollama Python package."""

    def __init__(self, model_name: Optional[str] = None, host: Optional[str] = None):
        env_model = os.getenv("LLM_MODEL")
        if env_model:
            env_model = env_model.strip().strip('"\'')  # Strip whitespace and quotes
        self.model_name = model_name or env_model
        self.host = host or os.getenv("OLLAMA_HOST", "localhost:11434")

        try:
            from ollama import chat
            self._chat = chat
        except ImportError:
            raise ImportError(
                "ollama package is required. Install it with: pip install ollama"
            )

    def _call_api(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        top_p: float = 1.0,
        stream: bool = False
    ) -> str:
        """Make a call to the Ollama API.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            temperature: Sampling temperature
            max_tokens: Maximum completion tokens
            top_p: Nucleus sampling parameter (ignored for Ollama)
            stream: Whether to stream the response

        Returns:
            The generated content string
        """
        response = self._chat(
            model=self.model_name,
            messages=messages,
            stream=stream,
            options={
                "temperature": temperature,
                "num_predict": max_tokens
            }
        )

        content = response.message.content
        if not content:
            raise RuntimeError("LLM returned empty response")

        return content.strip()
