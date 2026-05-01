import os
import pickle
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch


@pytest.fixture
def temp_vector_db(tmp_path):
    vdb_path = tmp_path / "test_vdb"
    vdb_path.mkdir()

    db_data = {
        'index': {'mock_index': True},
        'chunks': [
            {'chunk_id': 0, 'content': 'This is chunk one about AI.'},
            {'chunk_id': 1, 'content': 'This is chunk two about ML.'},
            {'chunk_id': 2, 'content': 'This is chunk three about DL.'}
        ],
        'metadata': {
            'vdb_type': 'faiss',
            'chunk_size': 500
        }
    }

    db_file = vdb_path / 'vector_db.pkl'
    with open(db_file, 'wb') as f:
        pickle.dump(db_data, f)

    return str(vdb_path)


@pytest.fixture
def mock_llm_api():
    with patch('src.llm.llm_api.LLMAPI') as mock_api:
        mock_api_instance = MagicMock()
        mock_api_instance.generate_response.return_value = "LLM response without context"
        mock_api_instance.generate_response_with_context.return_value = "LLM response with context"
        mock_api.return_value = mock_api_instance
        yield mock_api_instance


@pytest.fixture
def rag_retriever(temp_vector_db, mock_llm_api):
    from src.rag.retriever.rag_retriever import RAGRetriever

    retriever = RAGRetriever(
        vector_db_path=temp_vector_db,
        llm_api=mock_llm_api,
        num_chunks=3
    )
    return retriever


class TestRAGRetriever:
    def test_init_with_valid_path(self, temp_vector_db, mock_llm_api):
        from src.rag.retriever.rag_retriever import RAGRetriever

        retriever = RAGRetriever(
            vector_db_path=temp_vector_db,
            llm_api=mock_llm_api
        )

        assert retriever.num_chunks == 3
        assert retriever.retriever_type == "cosine_similarity"
        assert len(retriever.chunks) == 3

    def test_init_with_missing_db(self, tmp_path, mock_llm_api):
        from src.rag.retriever.rag_retriever import RAGRetriever

        with pytest.raises(FileNotFoundError):
            RAGRetriever(
                vector_db_path=str(tmp_path / "nonexistent"),
                llm_api=mock_llm_api
            )

    def test_retrieve_chunks(self, rag_retriever):
        rag_retriever.index = MagicMock()
        rag_retriever.index.search.return_value = (
            np.array([[0.95, 0.85, 0.75]]),
            np.array([[0, 1, 2]])
        )

        query_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]

        results = rag_retriever.retrieve_chunks(query_embedding)

        assert len(results) == 3
        assert 'chunk' in results[0]
        assert 'similarity_score' in results[0]
        assert results[0]['chunk']['chunk_id'] == 0

    def test_format_context(self, rag_retriever):
        retrieved = [
            {'chunk': {'content': 'First chunk'}, 'score': 0.9},
            {'chunk': {'content': 'Second chunk'}, 'score': 0.8}
        ]

        context = rag_retriever._format_context(retrieved)

        assert 'First chunk' in context
        assert 'Second chunk' in context

    def test_generate_response_with_context(self, rag_retriever, mock_llm_api):
        rag_retriever.index = MagicMock()
        rag_retriever.index.search.return_value = (
            np.array([[0.95, 0.85, 0.75]]),
            np.array([[0, 1, 2]])
        )

        query = "What is AI?"
        query_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]

        response = rag_retriever.generate_response(query, query_embedding)

        assert response == "LLM response with context"
        mock_llm_api.generate_response_with_context.assert_called_once()

    def test_generate_response_no_results(self, rag_retriever, mock_llm_api):
        rag_retriever.index = MagicMock()
        rag_retriever.index.search.return_value = (
            np.array([[0.1]]),
            np.array([[100]])
        )

        query = "What is this?"
        query_embedding = [0.1, 0.2, 0.3]

        response = rag_retriever.generate_response(query, query_embedding)

        mock_llm_api.generate_response.assert_called_once_with(query)

    def test_get_retrieval_info(self, rag_retriever):
        info = rag_retriever.get_retrieval_info()

        assert info['num_chunks_in_db'] == 3
        assert info['retriever_type'] == "cosine_similarity"
        assert info['num_chunks_to_retrieve'] == 3
        assert 'vdb_metadata' in info
