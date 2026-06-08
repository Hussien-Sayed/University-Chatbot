import os
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.llm.llm_api import LLMAPI


def normalize_text(text: str) -> str:
    """Normalize text by lowercasing and removing punctuation/quotes."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation and quotes
    return text


class RAGRetriever:
    """Class for retrieving chunks from vector DB and generating responses"""

    def __init__(
        self,
        vector_db_path: str,
        llm_api: LLMAPI,
        num_chunks: int = 3,
        retriever_type: Optional[str] = None
    ):
        self.vector_db_path = Path(vector_db_path)
        self.llm_api = llm_api
        self.num_chunks = num_chunks
        self.retriever_type = retriever_type or os.getenv("RETRIEVAL_TYPE", "vector")
        self.bm25_weight = float(os.getenv("BM25_WEIGHT", "0.5"))

        # Self-evaluation configuration
        self.enable_self_eval = os.getenv("ENABLE_SELF_EVAL", "false").lower() == "true"
        self.relevance_threshold = float(os.getenv("RELEVANCE_THRESHOLD", "0.3"))
        self.confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.4"))

        self.index = None
        self.chunks = []
        self.metadata = {}
        self.bm25_index = None

        self._load_vector_db()

    def _load_vector_db(self):
        db_file = self.vector_db_path / 'vector_db.pkl'
        if not db_file.exists():
            raise FileNotFoundError(f"Vector database not found: {db_file}")

        with open(db_file, 'rb') as f:
            db_data = pickle.load(f)

        self.index = db_data['index']
        self.chunks = db_data['chunks']
        self.metadata = db_data.get('metadata', {})
        self.bm25_index = db_data.get('bm25_index')

    def _retrieve_vector(self, query_embedding: List[float]) -> List[Dict[str, Any]]:
        query_array = np.array([query_embedding], dtype=np.float32)

        norms = np.linalg.norm(query_array)
        if norms > 0:
            query_array = query_array / norms

        scores, indices = self.index.search(query_array, self.num_chunks)

        retrieved = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.chunks):
                retrieved.append({
                    'chunk': self.chunks[idx],
                    'similarity_score': float(score),
                    'vector_score': float(score),
                    'bm25_score': 0.0
                })

        return retrieved

    def _retrieve_bm25(self, query: str) -> List[Dict[str, Any]]:
        if self.bm25_index is None:
            raise ValueError("BM25 index not available. Rebuild vector database with BM25 support.")

        tokenized_query = normalize_text(query).split()
        scores = self.bm25_index.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:self.num_chunks]

        retrieved = []
        for idx in top_indices:
            if idx < len(self.chunks):
                retrieved.append({
                    'chunk': self.chunks[idx],
                    'similarity_score': float(scores[idx]),
                    'vector_score': 0.0,
                    'bm25_score': float(scores[idx])
                })

        return retrieved

    def _retrieve_hybrid(self, query: str, query_embedding: List[float]) -> List[Dict[str, Any]]:
        if self.bm25_index is None:
            raise ValueError("BM25 index not available. Rebuild vector database with BM25 support.")

        # Get vector scores
        query_array = np.array([query_embedding], dtype=np.float32)
        norms = np.linalg.norm(query_array)
        if norms > 0:
            query_array = query_array / norms
        vector_scores, vector_indices = self.index.search(query_array, len(self.chunks))

        # Get BM25 scores
        tokenized_query = normalize_text(query).split()
        bm25_scores = self.bm25_index.get_scores(tokenized_query)

        # Normalize scores to [0, 1]
        vector_scores_flat = vector_scores[0]
        if vector_scores_flat.max() > 0:
            vector_scores_normalized = vector_scores_flat / vector_scores_flat.max()
        else:
            vector_scores_normalized = vector_scores_flat

        if bm25_scores.max() > 0:
            bm25_scores_normalized = bm25_scores / bm25_scores.max()
        else:
            bm25_scores_normalized = bm25_scores

        # Combine with weighted averaging
        combined_scores = (self.bm25_weight * bm25_scores_normalized) + \
                         ((1 - self.bm25_weight) * vector_scores_normalized)

        # Get top-k
        top_indices = np.argsort(combined_scores)[::-1][:self.num_chunks]

        retrieved = []
        for idx in top_indices:
            if idx < len(self.chunks):
                retrieved.append({
                    'chunk': self.chunks[idx],
                    'similarity_score': float(combined_scores[idx]),
                    'vector_score': float(vector_scores_normalized[idx]),
                    'bm25_score': float(bm25_scores_normalized[idx])
                })

        return retrieved

    def retrieve_chunks(self, query: str, query_embedding: List[float]) -> List[Dict[str, Any]]:
        if self.retriever_type == "vector":
            return self._retrieve_vector(query_embedding)
        elif self.retriever_type == "bm25":
            return self._retrieve_bm25(query)
        elif self.retriever_type == "hybrid":
            return self._retrieve_hybrid(query, query_embedding)
        else:
            raise ValueError(f"Unknown retriever type: {self.retriever_type}. Must be 'vector', 'bm25', or 'hybrid'.")

    def _format_context(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        context_parts = []
        for item in retrieved_chunks:
            content = item['chunk'].get('content', '')
            context_parts.append(content)

        return '\n\n'.join(context_parts)

    def generate_response(self, query: str, query_embedding: List[float]) -> str:
        retrieved = self.retrieve_chunks(query, query_embedding)

        if not retrieved:
            return self.llm_api.generate_response(query)

        context = self._format_context(retrieved)
        response = self.llm_api.generate_response_with_context(
            prompt=query,
            context=context
        )

        return response

    def get_retrieval_info(self) -> Dict[str, Any]:
        return {
            'num_chunks_in_db': len(self.chunks),
            'retriever_type': self.retriever_type,
            'num_chunks_to_retrieve': self.num_chunks,
            'vdb_metadata': self.metadata,
            'self_eval_enabled': self.enable_self_eval
        }

    def _filter_by_relevance(self, retrieved_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter chunks based on relevance score threshold. Adapts to retrieval type."""
        if not retrieved_chunks:
            return []

        if self.retriever_type == "vector":
            # Cosine similarity is already 0-1
            return [c for c in retrieved_chunks if c.get('similarity_score', 0) >= self.relevance_threshold]

        elif self.retriever_type == "bm25":
            # BM25 scores need normalization relative to max
            max_score = max(c.get('bm25_score', 0) for c in retrieved_chunks)
            if max_score == 0:
                return []
            return [c for c in retrieved_chunks if (c.get('bm25_score', 0) / max_score) >= self.relevance_threshold]

        elif self.retriever_type == "hybrid":
            # Combined score is already normalized in _retrieve_hybrid
            return [c for c in retrieved_chunks if c.get('similarity_score', 0) >= self.relevance_threshold]

        return retrieved_chunks

    def _evaluate_chunks_with_llm(self, query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Use LLM to evaluate relevance of each chunk. Returns chunks with llm_relevance score."""
        evaluated = []
        for chunk_item in chunks:
            content = chunk_item['chunk'].get('content', '')
            llm_score = self.llm_api.evaluate_chunk_relevance(query, content)
            chunk_item['llm_relevance'] = llm_score
            evaluated.append(chunk_item)
        return evaluated

    def _should_use_context(self, query: str, chunks: List[Dict[str, Any]]) -> bool:
        """Decide whether to use retrieved context based on scores."""
        if not chunks:
            return False

        # Check if any chunk passes the threshold after LLM evaluation
        if self.enable_self_eval:
            # Filter by LLM relevance score
            relevant_chunks = [c for c in chunks if c.get('llm_relevance', 0) >= self.relevance_threshold]
            return len(relevant_chunks) > 0
        else:
            # Use basic score filtering
            filtered = self._filter_by_relevance(chunks)
            return len(filtered) > 0

    def generate_response_with_self_eval(self, query: str, query_embedding: List[float]) -> Dict[str, Any]:
        """Generate response with self-evaluation. Returns dict with response and evaluation metadata."""
        # Retrieve chunks
        retrieved = self.retrieve_chunks(query, query_embedding)

        # Initialize evaluation metadata
        eval_metadata = {
            'retriever_type': self.retriever_type,
            'chunks_retrieved': len(retrieved),
            'self_eval_enabled': self.enable_self_eval
        }

        # If self-evaluation is disabled, use standard response
        if not self.enable_self_eval:
            if not retrieved:
                response = self.llm_api.generate_response(query)
                eval_metadata['used_context'] = False
            else:
                context = self._format_context(retrieved)
                response = self.llm_api.generate_response_with_context(query, context)
                eval_metadata['used_context'] = True

            return {
                'response': response,
                'evaluation': eval_metadata
            }

        # Self-evaluation enabled
        # Step 1: Filter by relevance threshold
        filtered_chunks = self._filter_by_relevance(retrieved)
        eval_metadata['chunks_after_score_filter'] = len(filtered_chunks)

        # Step 2: LLM evaluation of chunk relevance
        if filtered_chunks:
            evaluated_chunks = self._evaluate_chunks_with_llm(query, filtered_chunks)
            eval_metadata['chunks_evaluated'] = len(evaluated_chunks)

            # Check if we should use context
            if self._should_use_context(query, evaluated_chunks):
                # Use only high-relevance chunks
                high_relevance_chunks = [c for c in evaluated_chunks if c.get('llm_relevance', 0) >= self.relevance_threshold]
                context = self._format_context(high_relevance_chunks)
                response = self.llm_api.generate_response_with_context(query, context)
                eval_metadata['used_context'] = True
                eval_metadata['chunks_used'] = len(high_relevance_chunks)
                eval_metadata['avg_llm_relevance'] = sum(c.get('llm_relevance', 0) for c in high_relevance_chunks) / len(high_relevance_chunks)

                # Step 3: Evaluate confidence in the response
                confidence = self.llm_api.evaluate_response_confidence(query, response, context)
                eval_metadata['response_confidence'] = confidence

                # Step 4: Check if confidence is too low
                if confidence < self.confidence_threshold:
                    response = "I don't have enough reliable information to answer this question accurately."
                    eval_metadata['fallback_triggered'] = True
                else:
                    eval_metadata['fallback_triggered'] = False
            else:
                # No relevant chunks found
                response = self.llm_api.generate_response(query)
                eval_metadata['used_context'] = False
                eval_metadata['fallback_triggered'] = False
        else:
            # No chunks passed score filter
            response = self.llm_api.generate_response(query)
            eval_metadata['used_context'] = False
            eval_metadata['fallback_triggered'] = False

        return {
            'response': response,
            'evaluation': eval_metadata
        }
