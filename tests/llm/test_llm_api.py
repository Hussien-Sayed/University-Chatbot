import os
import pytest
from unittest.mock import MagicMock, patch
from src.llm.llm_api import LLMAPI


@pytest.fixture
def mock_llm_response():
    mock_choice = MagicMock()
    mock_choice.message.content = "This is a test response"
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    return mock_response


@pytest.fixture
def llm_api():
    mock_groq = MagicMock()
    with patch.dict('sys.modules', {'groq': mock_groq}):
        with patch.dict(os.environ, {"GROQ_API_KEY": "test_api_key"}):
            api = LLMAPI()
            return api


class TestLLMAPI:
    def test_init_with_valid_key(self):
        mock_groq = MagicMock()
        with patch.dict('sys.modules', {'groq': mock_groq}):
            with patch.dict(os.environ, {"GROQ_API_KEY": "test_key"}):
                api = LLMAPI()
                assert api.api_key == "test_key"

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
            assert api.api_key == "explicit_key"

    def test_generate_response(self, llm_api, mock_llm_response):
        llm_api.client.chat.completions.create.return_value = mock_llm_response

        result = llm_api.generate_response("Hello")

        assert result == "This is a test response"
        llm_api.client.chat.completions.create.assert_called_once()

    def test_generate_response_empty_prompt(self, llm_api):
        with pytest.raises(ValueError, match="prompt cannot be empty"):
            llm_api.generate_response("")

    def test_generate_response_with_context(self, llm_api, mock_llm_response):
        llm_api.client.chat.completions.create.return_value = mock_llm_response

        result = llm_api.generate_response_with_context(
            prompt="What is this?",
            context="This is a test document about AI."
        )

        assert result == "This is a test response"
        call_args = llm_api.client.chat.completions.create.call_args
        assert call_args[1]['messages'][0]['role'] == 'system'
        assert 'test document about AI' in call_args[1]['messages'][0]['content']

    def test_generate_response_empty_llm_output(self, llm_api):
        mock_empty = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = None
        mock_empty.choices = [mock_choice]
        llm_api.client.chat.completions.create.return_value = mock_empty

        with pytest.raises(RuntimeError, match="LLM returned empty response"):
            llm_api.generate_response("Hello")

    def test_custom_model_name(self):
        mock_groq = MagicMock()
        with patch.dict('sys.modules', {'groq': mock_groq}):
            with patch.dict(os.environ, {"GROQ_API_KEY": "test_key"}):
                api = LLMAPI(model_name="custom-model")
                assert api.model_name == "custom-model"

    def test_evaluate_chunk_relevance(self, llm_api):
        mock_choice = MagicMock()
        mock_choice.message.content = "0.85"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        llm_api.client.chat.completions.create.return_value = mock_response

        result = llm_api.evaluate_chunk_relevance("What is AI?", "AI is artificial intelligence.")

        assert result == 0.85
        assert llm_api.client.chat.completions.create.call_args[1]['temperature'] == 0.1
        assert llm_api.client.chat.completions.create.call_args[1]['max_completion_tokens'] == 10

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
            mock_choice = MagicMock()
            mock_choice.message.content = content
            mock_response = MagicMock()
            mock_response.choices = [mock_choice]
            llm_api.client.chat.completions.create.return_value = mock_response

            result = llm_api.evaluate_chunk_relevance("query", "chunk")
            assert result == expected, f"Failed for content: {content}"

    def test_evaluate_chunk_relevance_default_on_error(self, llm_api):
        llm_api.client.chat.completions.create.side_effect = Exception("API error")

        result = llm_api.evaluate_chunk_relevance("query", "chunk")

        assert result == 0.5  # Default value on error

    def test_evaluate_response_confidence(self, llm_api):
        mock_choice = MagicMock()
        mock_choice.message.content = "0.92"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        llm_api.client.chat.completions.create.return_value = mock_response

        result = llm_api.evaluate_response_confidence(
            "What is AI?",
            "AI is a technology.",
            "Context about AI"
        )

        assert result == 0.92
        assert llm_api.client.chat.completions.create.call_args[1]['temperature'] == 0.1

    def test_evaluate_response_confidence_clamping(self, llm_api):
        mock_choice = MagicMock()
        mock_choice.message.content = "1.5"  # Above 1, should be clamped
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        llm_api.client.chat.completions.create.return_value = mock_response

        result = llm_api.evaluate_response_confidence("q", "r", "c")

        assert result == 1.0  # Clamped to 1.0

    def test_evaluate_response_confidence_default_on_error(self, llm_api):
        llm_api.client.chat.completions.create.side_effect = Exception("API error")

        result = llm_api.evaluate_response_confidence("query", "response", "context")

        assert result == 0.5  # Default value on error
