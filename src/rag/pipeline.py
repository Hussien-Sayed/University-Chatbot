"""Centralized RAG Pipeline for consistent query processing across app and evaluator."""
import os
import time
from typing import Dict, Any, List

from src.llm.embedding_api import EmbeddingAPI
from src.llm.llm_api import LLMAPI
from src.rag.retriever.rag_retriever import RAGRetriever


class RAGPipeline:
    """
    Centralized RAG pipeline that encapsulates the entire RAG workflow.
    
    This class ensures both the Streamlit app and the evaluation script
    use identical logic for query processing, retrieval, and response generation.
    """

    def __init__(
        self,
        vector_db_path: str = None,
        llm_api: LLMAPI = None,
        embedding_api: EmbeddingAPI = None
    ):
        """
        Initialize the RAG pipeline with all required components.
        
        Args:
            vector_db_path: Path to vector database (defaults to env var VDB_SAVE_PATH)
            llm_api: LLM API instance (creates new if None)
            embedding_api: Embedding API instance (creates new if None)
        """
        self.vector_db_path = vector_db_path or os.getenv("VDB_SAVE_PATH", "data/vector_db")
        self.embedding_api = embedding_api or EmbeddingAPI()
        self.llm_api = llm_api or LLMAPI()
        
        # Initialize retriever with all configuration from environment
        self.retriever = RAGRetriever(
            vector_db_path=self.vector_db_path,
            llm_api=self.llm_api,
            embedding_api=self.embedding_api
        )

    def run(self, query: str) -> Dict[str, Any]:
        """
        Run the complete RAG pipeline for a single query.

        Args:
            query: The user's question

        Returns:
            Dictionary containing:
                - response: The main answer string
                - query: The original query
                - retrieved_chunks: List of retrieved chunk objects
                - contexts: List of chunk content strings
                - self_eval: Dict with self-evaluation metadata
                - retriever_type: Type of retrieval used
                - query_time_seconds: Total processing time
                - api_calls: Dict with llm_calls and embedding_calls counts
        """
        # Reset API call counters at the start of each run
        self.llm_api.reset_counters()
        self.embedding_api.reset_counters()

        query_start_time = time.time()

        # Generate embedding for the query
        query_embedding = self.embedding_api.generate_embedding(query)
        
        # Run self-evaluation enabled response generation
        result = self.retriever.generate_response_with_self_eval(query, query_embedding)
        
        # Get retrieved chunks for context extraction
        retrieved_chunks = self.retriever.retrieve_chunks(query, query_embedding)
        contexts = [item['chunk'].get('content', '') for item in retrieved_chunks]

        # Get fusion variants if enabled
        fusion_variants = self.retriever._last_fusion_variants if self.retriever.enable_query_fusion else []
        
        query_time_seconds = time.time() - query_start_time
        
        # Extract self-evaluation metadata
        eval_metadata = result.get('evaluation', {})

        # Capture API call counts before resetting
        llm_calls = self.llm_api.llm_call_count
        embedding_calls = self.embedding_api.embedding_call_count

        # Build standardized response dictionary
        response_dict = {
            "response": result['response'],
            "query": query,
            "retrieved_chunks": retrieved_chunks,
            "contexts": contexts,
            "self_eval": {
                "enabled": eval_metadata.get('self_eval_enabled', False),
                "used_context": eval_metadata.get('used_context', False),
                "chunks_retrieved": eval_metadata.get('chunks_retrieved', 0),
                "chunks_after_score_filter": eval_metadata.get('chunks_after_score_filter', 0),
                "chunks_used": eval_metadata.get('chunks_used', 0),
                "avg_relevance": eval_metadata.get('avg_llm_relevance', None),
                "confidence": eval_metadata.get('response_confidence', None),
                "fallback_triggered": eval_metadata.get('fallback_triggered', False)
            },
            "fusion": {
                "enabled": self.retriever.enable_query_fusion,
                "num_variants": self.retriever.fusion_num_variants if self.retriever.enable_query_fusion else None,
                "k": self.retriever.fusion_k if self.retriever.enable_query_fusion else None,
                "top_k": self.retriever.fusion_top_k if self.retriever.enable_query_fusion else None,
                "query_variants": fusion_variants
            },
            "retriever_type": self.retriever.retriever_type,
            "query_time_seconds": query_time_seconds,
            "api_calls": {
                "llm_calls": llm_calls,
                "embedding_calls": embedding_calls
            }
        }

        # Reset counters after capturing for next run
        self.llm_api.reset_counters()
        self.embedding_api.reset_counters()

        return response_dict

    def get_config(self) -> Dict[str, Any]:
        """
        Get current pipeline configuration for debugging/logging.
        
        Returns:
            Dictionary with configuration details
        """
        return {
            "vector_db_path": self.vector_db_path,
            "retriever_type": self.retriever.retriever_type,
            "bm25_weight": self.retriever.bm25_weight,
            "num_chunks": self.retriever.num_chunks,
            "self_eval_enabled": self.retriever.enable_self_eval,
            "relevance_threshold": self.retriever.relevance_threshold,
            "confidence_threshold": self.retriever.confidence_threshold,
            "query_fusion_enabled": self.retriever.enable_query_fusion,
            "fusion_num_variants": self.retriever.fusion_num_variants,
            "fusion_k": self.retriever.fusion_k,
            "fusion_top_k": self.retriever.fusion_top_k,
            "retrieval_info": self.retriever.get_retrieval_info()
        }
