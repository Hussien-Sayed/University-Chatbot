"""Groq API provider for LLM calls."""
import os
from typing import List, Dict, Any, Optional


class GroqProvider:
    """Provider for Groq API calls."""

    def __init__(self, model_name: Optional[str] = None, api_key: Optional[str] = None):
        env_model = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
        if env_model:
            env_model = env_model.strip().strip('"\'')  # Strip whitespace and quotes
        self.model_name = model_name or env_model
        self.api_key = api_key or os.getenv("GROQ_API_KEY")

        if not self.api_key:
            raise ValueError("api_key must be provided or set in .env as GROQ_API_KEY")

        try:
            from groq import Groq
            self.client = Groq(api_key=self.api_key)
        except ImportError:
            raise ImportError("groq package is required. Install it with: pip install groq")

    def _call_api(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        top_p: float = 1.0,
        stream: bool = False
    ) -> str:
        """Make a call to the Groq API.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            temperature: Sampling temperature
            max_tokens: Maximum completion tokens
            top_p: Nucleus sampling parameter
            stream: Whether to stream the response

        Returns:
            The generated content string
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_tokens,
            top_p=top_p,
            stream=stream
        )

        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("LLM returned empty response")

        return content.strip()
