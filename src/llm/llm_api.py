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

    def evaluate_chunk_relevance(self, query: str, chunk_content: str, max_tokens: int = 10) -> float:
        """Evaluate how relevant a chunk is to the query. Returns score 0-1."""
        prompt = f"""Rate how relevant this text is to answering the question.

Question: {query}

Text: {chunk_content}

Rate relevance from 0 to 1, where:
- 0 = completely irrelevant
- 0.5 = somewhat relevant
- 1 = highly relevant

Respond with only a number between 0 and 1."""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_completion_tokens=max_tokens,
                top_p=1,
                stream=False
            )
            content = response.choices[0].message.content.strip()
            # Extract number from response
            import re
            numbers = re.findall(r'0?\.\d+|[01]', content)
            if numbers:
                score = float(numbers[0])
                return max(0.0, min(1.0, score))  # Clamp to [0, 1]
            return 0.5  # Default if parsing fails
        except Exception:
            return 0.5  # Default on error

    def evaluate_response_confidence(self, query: str, response: str, context: str, max_tokens: int = 10) -> float:
        """Evaluate confidence in the generated response. Returns score 0-1."""
        prompt = f"""Rate your confidence that this answer is correct and well-supported by the context.

Question: {query}

Context: {context}

Answer: {response}

Rate confidence from 0 to 1, where:
- 0 = no confidence, answer is unsupported or wrong
- 0.5 = moderate confidence
- 1 = high confidence, answer is fully supported

Respond with only a number between 0 and 1."""

        try:
            result = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_completion_tokens=max_tokens,
                top_p=1,
                stream=False
            )
            content = result.choices[0].message.content.strip()
            import re
            numbers = re.findall(r'0?\.\d+|[01]', content)
            if numbers:
                score = float(numbers[0])
                return max(0.0, min(1.0, score))
            return 0.5
        except Exception:
            return 0.5
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
