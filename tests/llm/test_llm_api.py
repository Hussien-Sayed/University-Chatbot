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
