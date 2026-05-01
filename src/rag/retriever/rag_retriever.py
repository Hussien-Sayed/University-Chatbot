import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.llm.llm_api import LLMAPI


class RAGRetriever:
    """Class for retrieving chunks from vector DB and generating responses"""

    def __init__(
        self,
        vector_db_path: str,
        llm_api: LLMAPI,
        num_chunks: int = 3,
        retriever_type: str = "cosine_similarity"
    ):
        self.vector_db_path = Path(vector_db_path)
        self.llm_api = llm_api
        self.num_chunks = num_chunks
        self.retriever_type = retriever_type

        self.index = None
        self.chunks = []
        self.metadata = {}

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

    def retrieve_chunks(self, query_embedding: List[float]) -> List[Dict[str, Any]]:
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
                    'similarity_score': float(score)
                })

        return retrieved

    def _format_context(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        context_parts = []
        for item in retrieved_chunks:
            content = item['chunk'].get('content', '')
            context_parts.append(content)

        return '\n\n'.join(context_parts)

    def generate_response(self, query: str, query_embedding: List[float]) -> str:
        retrieved = self.retrieve_chunks(query_embedding)

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
