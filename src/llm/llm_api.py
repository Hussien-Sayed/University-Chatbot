import os
from typing import Optional


class LLMAPI:
    """Class for LLM API using Groq"""

    def __init__(self, model_name: Optional[str] = None, api_key: Optional[str] = None):
        self.model_name = model_name or os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
        self.api_key = api_key or os.getenv("GROQ_API_KEY")

        if not self.api_key:
            raise ValueError("api_key must be provided or set in .env as GROQ_API_KEY")

        try:
            from groq import Groq
            self.client = Groq(api_key=self.api_key)
        except ImportError:
            raise ImportError("groq package is required. Install it with: pip install groq")

    def generate_response(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        if not prompt or not prompt.strip():
            raise ValueError("prompt cannot be empty")

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_completion_tokens=max_tokens,
            top_p=1,
            stream=False
        )

        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("LLM returned empty response")

        return content.strip()

    def generate_response_with_context(self, prompt: str, context: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        if not prompt or not prompt.strip():
            raise ValueError("prompt cannot be empty")

        system_message = f"Use the following context to answer the question:\n\n{context}"

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_completion_tokens=max_tokens,
            top_p=1,
            stream=False
        )

        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("LLM returned empty response")

        return content.strip()
