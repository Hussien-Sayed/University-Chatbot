import os
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from src.llm.embedding_api import EmbeddingAPI


@pytest.fixture
def mock_embedding():
    return [0.1, 0.2, 0.3, 0.4, 0.5]


@pytest.fixture
def embedding_api():
    with patch.dict(os.environ, {"HUGGINGFACE_API_KEY": "test_api_key"}):
        with patch('huggingface_hub.InferenceClient') as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            api = EmbeddingAPI()
            api.client = mock_instance
            return api


class TestEmbeddingAPI:
    def test_init_with_valid_key(self):
        with patch.dict(os.environ, {"HUGGINGFACE_API_KEY": "test_key"}):
            with patch('huggingface_hub.InferenceClient'):
                api = EmbeddingAPI()
                assert api.api_key == "test_key"

    def test_init_with_missing_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="api_key must be provided"):
                EmbeddingAPI()

    def test_init_with_explicit_key(self):
        with patch('huggingface_hub.InferenceClient'):
            api = EmbeddingAPI(api_key="explicit_key")
            assert api.api_key == "explicit_key"

    def test_generate_embedding_single_text(self, embedding_api, mock_embedding):
        embedding_api.client.feature_extraction.return_value = np.array([mock_embedding])

        result = embedding_api.generate_embedding("Hello world")

        assert result == mock_embedding
        embedding_api.client.feature_extraction.assert_called_once_with("Hello world")

    def test_generate_embedding_empty_text(self, embedding_api):
        with pytest.raises(ValueError, match="text cannot be empty"):
            embedding_api.generate_embedding("")

    def test_generate_embeddings_multiple_texts(self, embedding_api, mock_embedding):
        embedding_api.client.feature_extraction.return_value = np.array([mock_embedding])

        texts = ["Hello", "World", "Test"]
        result = embedding_api.generate_embeddings(texts)

        assert len(result) == 3
        assert result[0] == mock_embedding
        assert embedding_api.client.feature_extraction.call_count == 3

    def test_generate_embeddings_empty_list(self, embedding_api):
        with pytest.raises(ValueError, match="texts list cannot be empty"):
            embedding_api.generate_embeddings([])

    def test_custom_model_name(self):
        with patch.dict(os.environ, {"HUGGINGFACE_API_KEY": "test_key"}):
            with patch('src.llm.embedding_api.InferenceClient') as mock_client:
                api = EmbeddingAPI(model_name="custom-model")
                assert api.model_name == "custom-model"
                mock_client.assert_called_once()
                call_kwargs = mock_client.call_args[1]
                assert call_kwargs['model'] == "custom-model"

    def test_list_embedding_response(self, embedding_api):
        mock_list = [[0.1, 0.2, 0.3]]
        embedding_api.client.feature_extraction.return_value = mock_list

        result = embedding_api.generate_embedding("test")

        assert result == [0.1, 0.2, 0.3]
