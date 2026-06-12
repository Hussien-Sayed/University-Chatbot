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
    mock_hf = MagicMock()
    with patch.dict(os.environ, {"HUGGINGFACE_API_KEY": "test_api_key", "EMBEDDING_PROVIDER": "cloud"}):
        with patch.dict('sys.modules', {'huggingface_hub': mock_hf}):
            api = EmbeddingAPI()
            api.provider._generate_embedding = MagicMock()
            return api


class TestEmbeddingAPI:
    def test_init_with_valid_key(self):
        mock_hf = MagicMock()
        with patch.dict(os.environ, {"HUGGINGFACE_API_KEY": "test_key", "EMBEDDING_PROVIDER": "cloud"}):
            with patch.dict('sys.modules', {'huggingface_hub': mock_hf}):
                api = EmbeddingAPI()
                assert api.api_key == "test_key"
                assert api.provider_name == "cloud"

    def test_init_with_missing_key(self):
        with patch.dict(os.environ, {"EMBEDDING_PROVIDER": "cloud"}, clear=True):
            with patch.dict(os.environ, {"EMBEDDING_PROVIDER": "cloud"}):
                mock_hf = MagicMock()
                mock_hf.InferenceClient.side_effect = None
                with patch.dict('sys.modules', {'huggingface_hub': mock_hf}):
                    with pytest.raises(ValueError, match="api_key must be provided"):
                        EmbeddingAPI(provider="cloud")

    def test_init_with_explicit_key(self):
        mock_hf = MagicMock()
        with patch.dict('sys.modules', {'huggingface_hub': mock_hf}):
            api = EmbeddingAPI(provider="cloud", api_key="explicit_key")
            assert api.api_key == "explicit_key"

    def test_generate_embedding_single_text(self, embedding_api, mock_embedding):
        embedding_api.provider._generate_embedding.return_value = mock_embedding

        result = embedding_api.generate_embedding("Hello world")

        assert result == mock_embedding
        embedding_api.provider._generate_embedding.assert_called_once_with("Hello world")

    def test_generate_embedding_empty_text(self, embedding_api):
        with pytest.raises(ValueError, match="text cannot be empty"):
            embedding_api.generate_embedding("")

    def test_generate_embeddings_multiple_texts(self, embedding_api, mock_embedding):
        embedding_api.provider._generate_embedding.return_value = mock_embedding

        texts = ["Hello", "World", "Test"]
        result = embedding_api.generate_embeddings(texts)

        assert len(result) == 3
        assert result[0] == mock_embedding
        assert embedding_api.provider._generate_embedding.call_count == 3

    def test_generate_embeddings_empty_list(self, embedding_api):
        with pytest.raises(ValueError, match="texts list cannot be empty"):
            embedding_api.generate_embeddings([])

    def test_custom_model_name(self):
        mock_hf = MagicMock()
        with patch.dict(os.environ, {"HUGGINGFACE_API_KEY": "test_key"}):
            with patch.dict('sys.modules', {'huggingface_hub': mock_hf}):
                api = EmbeddingAPI(provider="cloud", model_name="custom-model")
                assert api.model_name == "custom-model"

    def test_local_provider_selection(self):
        mock_transformers = MagicMock()
        with patch.dict('sys.modules', {'transformers': mock_transformers}):
            api = EmbeddingAPI(provider="local", model_name="sentence-transformers/all-MiniLM-L6-v2")
            assert api.provider_name == "local"
            assert api.model_name == "sentence-transformers/all-MiniLM-L6-v2"
            assert api.api_key is None

    def test_cloud_provider_default(self):
        mock_hf = MagicMock()
        with patch.dict(os.environ, {"HUGGINGFACE_API_KEY": "test_key"}):
            with patch.dict('sys.modules', {'huggingface_hub': mock_hf}):
                api = EmbeddingAPI()
                assert api.provider_name == "cloud"
                assert api.provider.__class__.__name__ == "HuggingFaceCloudProvider"

    def test_unknown_provider_raises_error(self):
        with pytest.raises(ValueError, match="Unknown embedding provider"):
            EmbeddingAPI(provider="unknown")

    def test_provider_from_env_var(self):
        mock_transformers = MagicMock()
        with patch.dict(os.environ, {"EMBEDDING_PROVIDER": "local"}):
            with patch.dict('sys.modules', {'transformers': mock_transformers}):
                api = EmbeddingAPI()
                assert api.provider_name == "local"
