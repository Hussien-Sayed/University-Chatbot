"""Tests for the centralized RAG Pipeline."""
import os
import pickle
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.rag.retriever.rag_retriever import RAGRetriever


@pytest.fixture
def temp_vector_db(tmp_path):
    """Create a temporary vector database for testing."""
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
def mock_components():
    """Mock the pipeline components."""
    mock_embedding_api = MagicMock()
    mock_embedding_api.generate_embedding.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    mock_llm_api = MagicMock()
    mock_llm_api.generate_response.return_value = "Test response without context"
    mock_llm_api.generate_response_with_context.return_value = "Test response with context"
    mock_llm_api.evaluate_chunk_relevance.return_value = 0.8
    mock_llm_api.evaluate_response_confidence.return_value = 0.7
    
    return mock_embedding_api, mock_llm_api


class TestRAGPipeline:
    """Test cases for RAGPipeline class."""

    def test_pipeline_initialization(self, temp_vector_db, mock_components):
        """Test that pipeline initializes correctly."""
        from src.rag.pipeline import RAGPipeline
        
        mock_emb, mock_llm = mock_components
        
        pipeline = RAGPipeline(
            vector_db_path=temp_vector_db,
            llm_api=mock_llm,
            embedding_api=mock_emb
        )
        
        assert pipeline.vector_db_path == temp_vector_db
        assert pipeline.embedding_api == mock_emb
        assert pipeline.llm_api == mock_llm
        assert pipeline.retriever is not None

    def test_pipeline_run_returns_correct_structure(self, temp_vector_db, mock_components):
        """Test that pipeline.run() returns the expected dictionary structure."""
        from src.rag.pipeline import RAGPipeline
        
        mock_emb, mock_llm = mock_components
        
        # Patch the retriever's generate_response_with_self_eval
        with patch.object(RAGRetriever, 'generate_response_with_self_eval') as mock_gen:
            with patch.object(RAGRetriever, 'retrieve_chunks') as mock_retrieve:
                mock_gen.return_value = {
                    'response': 'Test answer',
                    'evaluation': {
                        'self_eval_enabled': True,
                        'used_context': True,
                        'chunks_retrieved': 3,
                        'chunks_used': 2,
                        'avg_llm_relevance': 0.85,
                        'response_confidence': 0.75,
                        'fallback_triggered': False
                    }
                }
                mock_retrieve.return_value = [
                    {'chunk': {'content': 'Chunk 1'}, 'similarity_score': 0.9},
                    {'chunk': {'content': 'Chunk 2'}, 'similarity_score': 0.8}
                ]
                
                pipeline = RAGPipeline(
                    vector_db_path=temp_vector_db,
                    llm_api=mock_llm,
                    embedding_api=mock_emb
                )
                
                result = pipeline.run("What is AI?")
                
                # Verify required keys exist
                assert 'response' in result
                assert 'query' in result
                assert 'retrieved_chunks' in result
                assert 'contexts' in result
                assert 'self_eval' in result
                assert 'retriever_type' in result
                assert 'query_time_seconds' in result
                
                # Verify response is the main answer
                assert result['response'] == 'Test answer'
                assert result['query'] == 'What is AI?'

    def test_pipeline_self_eval_structure(self, temp_vector_db, mock_components):
        """Test that self_eval dictionary has correct structure."""
        from src.rag.pipeline import RAGPipeline
        
        mock_emb, mock_llm = mock_components
        
        with patch.object(RAGRetriever, 'generate_response_with_self_eval') as mock_gen:
            with patch.object(RAGRetriever, 'retrieve_chunks'):
                mock_gen.return_value = {
                    'response': 'Test',
                    'evaluation': {
                        'self_eval_enabled': True,
                        'used_context': True,
                        'chunks_retrieved': 3,
                        'chunks_after_score_filter': 2,
                        'chunks_used': 2,
                        'avg_llm_relevance': 0.85,
                        'response_confidence': 0.75,
                        'fallback_triggered': False
                    }
                }
                
                pipeline = RAGPipeline(
                    vector_db_path=temp_vector_db,
                    llm_api=mock_llm,
                    embedding_api=mock_emb
                )
                
                result = pipeline.run("Test query")
                self_eval = result['self_eval']
                
                # Verify self_eval fields
                assert 'enabled' in self_eval
                assert 'used_context' in self_eval
                assert 'chunks_retrieved' in self_eval
                assert 'chunks_after_score_filter' in self_eval
                assert 'chunks_used' in self_eval
                assert 'avg_relevance' in self_eval
                assert 'confidence' in self_eval
                assert 'fallback_triggered' in self_eval

    def test_pipeline_with_disabled_self_eval(self, temp_vector_db, mock_components):
        """Test pipeline when self-evaluation is disabled."""
        from src.rag.pipeline import RAGPipeline
        
        mock_emb, mock_llm = mock_components
        
        with patch.dict(os.environ, {"ENABLE_SELF_EVAL": "false"}):
            with patch.object(RAGRetriever, 'generate_response_with_self_eval') as mock_gen:
                with patch.object(RAGRetriever, 'retrieve_chunks'):
                    mock_gen.return_value = {
                        'response': 'Test answer',
                        'evaluation': {
                            'self_eval_enabled': False,
                            'used_context': True
                        }
                    }
                    
                    pipeline = RAGPipeline(
                        vector_db_path=temp_vector_db,
                        llm_api=mock_llm,
                        embedding_api=mock_emb
                    )
                    
                    result = pipeline.run("Test query")
                    
                    assert result['self_eval']['enabled'] is False

    def test_pipeline_get_config(self, temp_vector_db, mock_components):
        """Test that get_config returns pipeline configuration."""
        from src.rag.pipeline import RAGPipeline
        
        mock_emb, mock_llm = mock_components
        
        with patch.dict(os.environ, {
            "RETRIEVAL_TYPE": "hybrid",
            "BM25_WEIGHT": "0.6",
            "ENABLE_SELF_EVAL": "true",
            "RELEVANCE_THRESHOLD": "0.4",
            "CONFIDENCE_THRESHOLD": "0.5"
        }):
            pipeline = RAGPipeline(
                vector_db_path=temp_vector_db,
                llm_api=mock_llm,
                embedding_api=mock_emb
            )
            
            config = pipeline.get_config()
            
            assert 'vector_db_path' in config
            assert 'retriever_type' in config
            assert 'bm25_weight' in config
            assert 'num_chunks' in config
            assert 'self_eval_enabled' in config
            assert 'relevance_threshold' in config
            assert 'confidence_threshold' in config

    def test_pipeline_error_handling(self, temp_vector_db, mock_components):
        """Test that pipeline handles errors gracefully."""
        from src.rag.pipeline import RAGPipeline
        
        mock_emb, mock_llm = mock_components
        mock_emb.generate_embedding.side_effect = Exception("Embedding error")
        
        pipeline = RAGPipeline(
            vector_db_path=temp_vector_db,
            llm_api=mock_llm,
            embedding_api=mock_emb
        )
        
        with pytest.raises(Exception, match="Embedding error"):
            pipeline.run("Test query")
