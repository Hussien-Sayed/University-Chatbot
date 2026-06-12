import os
import re
from typing import List, Optional

from .providers.groq_api import GroqProvider
from .providers.ollama_api import OllamaProvider


class LLMAPI:
    """Class for LLM API supporting multiple providers (Groq, Ollama)."""

    def __init__(
        self,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        """Initialize LLM API with specified provider.

        Args:
            provider: LLM provider to use ("groq" or "ollama"). Defaults to env var CHAT_LLM_PROVIDER or "groq".
            model_name: Model name to use. Provider-specific defaults if not provided.
            api_key: API key (only used for Groq, optional for Ollama).
        """
        self.chat_llm_model = os.getenv("CHAT_LLM_MODEL")
        self.provider_name = (provider or os.getenv("CHAT_LLM_PROVIDER", "groq")).lower()
        self.llm_call_count = 0

        if self.provider_name == "groq":
            self.provider = GroqProvider(model_name=model_name or self.chat_llm_model, api_key=api_key)
            self.model_name = self.provider.model_name
        elif self.provider_name == "ollama":
            self.provider = OllamaProvider(model_name=model_name or self.chat_llm_model)
            self.model_name = self.provider.model_name
        else:
            raise ValueError(f"Unknown provider: {self.provider_name}. Supported: groq, ollama")

    def reset_counters(self) -> None:
        """Reset the LLM API call counter."""
        self.llm_call_count = 0

    def _increment_counter(self) -> None:
        """Increment the LLM API call counter."""
        self.llm_call_count += 1

    def generate_response(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        if not prompt or not prompt.strip():
            raise ValueError("prompt cannot be empty")

        self._increment_counter()
        return self.provider._call_api(
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )

    def evaluate_chunk_relevance(self, query: str, chunk_content: str, max_tokens: int = 10) -> float:
        """Evaluate how relevant a chunk is to the query. Returns score 0-1."""
        self._increment_counter()
        prompt = f"""Rate how relevant this text is to answering the question.

Question: {query}

Text: {chunk_content}

Rate relevance from 0 to 1, where:
- 0 = completely irrelevant
- 0.5 = somewhat relevant
- 1 = highly relevant

Respond with only a number between 0 and 1."""

        try:
            content = self.provider._call_api(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=max_tokens
            )
            # Extract number from response
            numbers = re.findall(r'0?\.\d+|[01]', content)
            if numbers:
                score = float(numbers[0])
                return max(0.0, min(1.0, score))  # Clamp to [0, 1]
            return 0.5  # Default if parsing fails
        except Exception:
            return 0.5  # Default on error

    def evaluate_response_confidence(self, query: str, response: str, context: str, max_tokens: int = 10) -> float:
        """Evaluate confidence in the generated response. Returns score 0-1."""
        self._increment_counter()
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
            content = self.provider._call_api(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=max_tokens
            )
            numbers = re.findall(r'0?\.\d+|[01]', content)
            if numbers:
                score = float(numbers[0])
                return max(0.0, min(1.0, score))
            return 0.5
        except Exception:
            return 0.5
    def generate_query_variants(self, query: str, num_variants: int = 3) -> List[str]:
        """Generate semantically similar query variations for RAG-Fusion.

        Args:
            query: Original user query
            num_variants: Number of query variations to generate

        Returns:
            List of query strings including original + generated variants
        """
        self._increment_counter()
        if not query or not query.strip():
            return [query]

        prompt = f"""Generate {num_variants} semantically similar variations of the following query.
Each variation should ask the same thing in different words or from a different angle.

Original query: {query}

Return ONLY the query variations, one per line. Do not include numbering, bullets, or explanations.
Each variation should be a complete question."""

        try:
            content = self.provider._call_api(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=256
            )
            if not content:
                return [query]

            # Parse variants (one per line, clean up)
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            variants = [query]  # Always include original

            for line in lines:
                # Remove common prefixes like "1.", "-", "*"
                cleaned = line
                if cleaned and len(cleaned) > 10:  # Sanity check for valid query
                    # Avoid duplicates
                    if cleaned.lower() not in [v.lower() for v in variants]:
                        variants.append(cleaned)

                if len(variants) >= num_variants + 1:
                    break

            return variants[:num_variants + 1]  # Original + N variants
        except Exception:
            return [query]  # Fallback to original on error

    def generate_response_with_context(self, prompt: str, context: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        if not prompt or not prompt.strip():
            raise ValueError("prompt cannot be empty")

        self._increment_counter()
        system_message = f"Use the following context to answer the question:\n\n{context}"

        return self.provider._call_api(
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
