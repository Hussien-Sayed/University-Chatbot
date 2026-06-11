import os
import pytest
from unittest.mock import MagicMock, patch
from src.llm.llm_api import LLMAPI


@pytest.fixture
def llm_api():
    mock_groq = MagicMock()
    with patch.dict('sys.modules', {'groq': mock_groq}):
        with patch.dict(os.environ, {"GROQ_API_KEY": "test_api_key"}):
            api = LLMAPI()
            # Mock the provider's _call_api method
            api.provider._call_api = MagicMock()
            return api


class TestLLMAPI:
    def test_init_with_valid_key(self):
        mock_groq = MagicMock()
        with patch.dict('sys.modules', {'groq': mock_groq}):
            with patch.dict(os.environ, {"GROQ_API_KEY": "test_key"}):
                api = LLMAPI()
                assert api.provider.api_key == "test_key"

    def test_init_with_missing_key(self):
        mock_groq = MagicMock()
        with patch.dict('sys.modules', {'groq': mock_groq}):
            with patch.dict(os.environ, {}, clear=True):
                with pytest.raises(ValueError, match="api_key must be provided"):
                    LLMAPI()

    def test_init_with_explicit_key(self):
        mock_groq = MagicMock()
        with patch.dict('sys.modules', {'groq': mock_groq}):
            api = LLMAPI(api_key="explicit_key")
            assert api.provider.api_key == "explicit_key"

    def test_generate_response(self, llm_api):
        llm_api.provider._call_api.return_value = "This is a test response"

        result = llm_api.generate_response("Hello")

        assert result == "This is a test response"
        llm_api.provider._call_api.assert_called_once()

    def test_generate_response_empty_prompt(self, llm_api):
        with pytest.raises(ValueError, match="prompt cannot be empty"):
            llm_api.generate_response("")

    def test_generate_response_with_context(self, llm_api):
        llm_api.provider._call_api.return_value = "This is a test response"

        result = llm_api.generate_response_with_context(
            prompt="What is this?",
            context="This is a test document about AI."
        )

        assert result == "This is a test response"
        call_args = llm_api.provider._call_api.call_args
        assert call_args[1]['messages'][0]['role'] == 'system'
        assert 'test document about AI' in call_args[1]['messages'][0]['content']

    def test_generate_response_empty_llm_output(self, llm_api):
        llm_api.provider._call_api.side_effect = RuntimeError("LLM returned empty response")

        with pytest.raises(RuntimeError, match="LLM returned empty response"):
            llm_api.generate_response("Hello")

    def test_custom_model_name(self):
        mock_groq = MagicMock()
        with patch.dict('sys.modules', {'groq': mock_groq}):
            with patch.dict(os.environ, {"GROQ_API_KEY": "test_key"}):
                api = LLMAPI(model_name="custom-model")
                assert api.model_name == "custom-model"

    def test_evaluate_chunk_relevance(self, llm_api):
        llm_api.provider._call_api.return_value = "0.85"

        result = llm_api.evaluate_chunk_relevance("What is AI?", "AI is artificial intelligence.")

        assert result == 0.85
        assert llm_api.provider._call_api.call_args[1]['temperature'] == 0.1
        assert llm_api.provider._call_api.call_args[1]['max_tokens'] == 10

    def test_evaluate_chunk_relevance_parsing_variants(self, llm_api):
        test_cases = [
            ("0.75", 0.75),
            ("0.5", 0.5),
            ("1", 1.0),
            ("0", 0.0),
            (".8", 0.8),
            ("The score is 0.65", 0.65),
        ]

        for content, expected in test_cases:
            llm_api.provider._call_api.return_value = content

            result = llm_api.evaluate_chunk_relevance("query", "chunk")
            assert result == expected, f"Failed for content: {content}"

    def test_evaluate_chunk_relevance_default_on_error(self, llm_api):
        llm_api.provider._call_api.side_effect = Exception("API error")

        result = llm_api.evaluate_chunk_relevance("query", "chunk")

        assert result == 0.5  # Default value on error

    def test_evaluate_response_confidence(self, llm_api):
        llm_api.provider._call_api.return_value = "0.92"

        result = llm_api.evaluate_response_confidence(
            "What is AI?",
            "AI is a technology.",
            "Context about AI"
        )

        assert result == 0.92
        assert llm_api.provider._call_api.call_args[1]['temperature'] == 0.1

    def test_evaluate_response_confidence_clamping(self, llm_api):
        llm_api.provider._call_api.return_value = "1.5"  # Above 1, should be clamped

        result = llm_api.evaluate_response_confidence("q", "r", "c")

        assert result == 1.0  # Clamped to 1.0

    def test_evaluate_response_confidence_default_on_error(self, llm_api):
        llm_api.provider._call_api.side_effect = Exception("API error")

        result = llm_api.evaluate_response_confidence("query", "response", "context")

        assert result == 0.5  # Default value on error

    def test_ollama_provider_selection(self):
        """Test that Ollama provider can be selected."""
        mock_ollama = MagicMock()
        with patch.dict('sys.modules', {'ollama': mock_ollama}):
            api = LLMAPI(provider="ollama", model_name="llama3.2")
            assert api.provider_name == "ollama"
            assert api.model_name == "llama3.2"
            assert api.provider.__class__.__name__ == "OllamaProvider"

    def test_groq_provider_default(self):
        """Test that Groq is the default provider."""
        mock_groq = MagicMock()
        with patch.dict('sys.modules', {'groq': mock_groq}):
            with patch.dict(os.environ, {"GROQ_API_KEY": "test_key"}):
                api = LLMAPI()
                assert api.provider_name == "groq"
                assert api.provider.__class__.__name__ == "GroqProvider"

    def test_unknown_provider_raises_error(self):
        """Test that unknown provider raises ValueError."""
        with pytest.raises(ValueError, match="Unknown provider"):
            LLMAPI(provider="unknown")

    def test_provider_from_env_var(self):
        """Test that provider can be set via LLM_PROVIDER env var."""
        mock_groq = MagicMock()
        mock_ollama = MagicMock()
        with patch.dict('sys.modules', {'groq': mock_groq, 'ollama': mock_ollama}):
            with patch.dict(os.environ, {"GROQ_API_KEY": "test_key", "LLM_PROVIDER": "ollama"}):
                api = LLMAPI()
                assert api.provider_name == "ollama"

    def test_provider_arg_overrides_env_var(self):
        """Test that provider argument overrides env var."""
        mock_groq = MagicMock()
        with patch.dict('sys.modules', {'groq': mock_groq}):
            with patch.dict(os.environ, {"GROQ_API_KEY": "test_key", "LLM_PROVIDER": "ollama"}):
                # Provider arg should override env var
                api = LLMAPI(provider="groq")
                assert api.provider_name == "groq"
