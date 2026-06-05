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
        print(f"[DEBUG BM25] Query: '{query}'")
        print(f"[DEBUG BM25] Normalized query: '{normalize_text(query)}'")
        print(f"[DEBUG BM25] Tokenized query: {tokenized_query}")

        scores = self.bm25_index.get_scores(tokenized_query)
        print(f"[DEBUG BM25] Raw scores shape: {scores.shape}")
        print(f"[DEBUG BM25] Max score: {scores.max():.4f}")
        print(f"[DEBUG BM25] Min score: {scores.min():.4f}")
        print(f"[DEBUG BM25] Non-zero scores: {np.count_nonzero(scores)}")

        # Show chunks with highest scores
        top_5_indices = np.argsort(scores)[::-1][:5]
        print(f"[DEBUG BM25] Top 5 chunks by score:")
        for idx in top_5_indices:
            if idx < len(self.chunks):
                chunk_preview = self.chunks[idx]['content'][:100]
                print(f"  [{idx}] Score: {scores[idx]:.4f} | Preview: '{chunk_preview}...'")

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
            'vdb_metadata': self.metadata
        }
