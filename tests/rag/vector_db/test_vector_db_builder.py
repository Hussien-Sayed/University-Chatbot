import os
import json
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch


@pytest.fixture
def temp_data_file(tmp_path):
    json_file = tmp_path / "test_data.json"
    data = {
        "intents": [
            {
                "tag": "greeting",
                "patterns": ["Hi", "Hello"],
                "responses": ["Hello!"]
            }
        ]
    }
    json_file.write_text(json.dumps(data))
    return str(json_file)


@pytest.fixture
def mock_data_loader(temp_data_file):
    with patch('src.data_utils.data_loader.DataLoader') as mock_loader:
        mock_loader_instance = MagicMock()
        mock_loader_instance.load_content.return_value = [
            {
                'id': 0,
                'tag': 'greeting',
                'content': 'Hi Hello Hello!',
                'source': str(temp_data_file)
            }
        ]
        mock_loader.return_value = mock_loader_instance
        yield mock_loader_instance


@pytest.fixture
def mock_embedding_api():
    with patch('src.llm.embedding_api.EmbeddingAPI') as mock_api:
        mock_api_instance = MagicMock()
        mock_api_instance.generate_embeddings.return_value = [
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.2, 0.3, 0.4, 0.5, 0.6]
        ]
        mock_api.return_value = mock_api_instance
        yield mock_api_instance


@pytest.fixture
def mock_faiss_module():
    mock_faiss = MagicMock()
    mock_index = MagicMock()
    mock_faiss.IndexFlatIP.return_value = mock_index
    with patch.dict('sys.modules', {'faiss': mock_faiss}):
        yield mock_faiss


@pytest.fixture
def vector_db_builder(mock_data_loader, mock_embedding_api, tmp_path):
    from src.rag.vector_db.vector_db_builder import VectorDBBuilder

    vdb_path = str(tmp_path / "test_vdb")
    builder = VectorDBBuilder(
        data_loader=mock_data_loader,
        embedding_api=mock_embedding_api,
        vdb_save_path=vdb_path
    )
    return builder


class TestVectorDBBuilder:
    def test_init_with_defaults(self, mock_data_loader, mock_embedding_api, tmp_path):
        from src.rag.vector_db.vector_db_builder import VectorDBBuilder

        builder = VectorDBBuilder(
            data_loader=mock_data_loader,
            embedding_api=mock_embedding_api,
            vdb_save_path=str(tmp_path / "vdb")
        )

        assert builder.vdb_type == "faiss"
        assert builder.chunking_strategy == "fixed_size"
        assert builder.chunk_size == 500

    def test_load_data(self, vector_db_builder, mock_data_loader):
        result = vector_db_builder.load_data()
        mock_data_loader.load_content.assert_called_once()
        assert len(result) == 1

    def test_chunk_documents(self, vector_db_builder):
        documents = [
            {
                'id': 0,
                'tag': 'test',
                'content': 'word1 word2 word3 word4 word5',
                'source': 'test.txt'
            }
        ]

        vector_db_builder.chunk_size = 3
        chunks = vector_db_builder.chunk_documents(documents)

        assert len(chunks) == 2
        assert chunks[0]['chunk_id'] == 0
        assert 'word1 word2 word3' in chunks[0]['content']

    def test_generate_embeddings(self, vector_db_builder, mock_embedding_api):
        vector_db_builder.chunks = [
            {'chunk_id': 0, 'content': 'test chunk 1'},
            {'chunk_id': 1, 'content': 'test chunk 2'}
        ]

        embeddings = vector_db_builder.generate_embeddings()

        mock_embedding_api.generate_embeddings.assert_called_once()
        assert len(embeddings) == 2

    def test_normalize_embeddings(self, vector_db_builder):
        vector_db_builder.embeddings = [
            [3.0, 4.0],
            [1.0, 0.0]
        ]

        normalized = vector_db_builder.normalize_embeddings()

        assert normalized.shape == (2, 2)
        norm_0 = np.linalg.norm(normalized[0])
        assert abs(norm_0 - 1.0) < 1e-6

    def test_save_vector_db(self, vector_db_builder, tmp_path):
        import pickle

        vector_db_builder.chunks = [{'chunk_id': 0, 'content': 'test'}]
        vector_db_builder.embeddings = [[0.1, 0.2, 0.3]]

        mock_index = {'mock': True}
        vector_db_builder.index = mock_index

        vector_db_builder.save_vector_db()

        saved_file = Path(vector_db_builder.vdb_save_path) / 'vector_db.pkl'
        assert saved_file.exists()

        with open(saved_file, 'rb') as f:
            data = pickle.load(f)
        assert 'chunks' in data
        assert 'metadata' in data

    def test_build_index(self, vector_db_builder, mock_faiss_module):
        mock_index = MagicMock()
        mock_faiss_module.IndexFlatIP.return_value = mock_index

        embeddings = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        vector_db_builder.build_index(embeddings)

        mock_faiss_module.IndexFlatIP.assert_called_once_with(2)
        mock_index.add.assert_called_once()

    def test_build_vector_db(self, vector_db_builder, mock_embedding_api, mock_faiss_module):
        vector_db_builder.embedding_api.generate_embeddings.return_value = [
            [0.1, 0.2, 0.3, 0.4, 0.5]
        ]

        with patch.object(vector_db_builder, 'save_vector_db'):
            result = vector_db_builder.build_vector_db()

        assert 'num_chunks' in result
        assert 'save_path' in result
