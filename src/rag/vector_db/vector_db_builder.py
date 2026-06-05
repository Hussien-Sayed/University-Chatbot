import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.data_utils.data_loader import DataLoader
from src.llm.embedding_api import EmbeddingAPI


class VectorDBBuilder:
    """Class for constructing vector database from embeddings"""

    def __init__(
        self,
        data_loader: DataLoader,
        embedding_api: EmbeddingAPI,
        vdb_type: str = "faiss",
        vdb_save_path: Optional[str] = None
    ):
        self.data_loader = data_loader
        self.embedding_api = embedding_api
        self.vdb_type = vdb_type
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "500"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "0"))
        self.vdb_save_path = vdb_save_path or os.getenv("VDB_SAVE_PATH", "data/vector_db")
        self.document_structure_mode = os.getenv("DOCUMENT_STRUCTURE_MODE", "structural")

        self.chunks = []
        self.embeddings = []
        self.metadata = []
        self.index = None

    def load_data(self) -> List[Dict[str, Any]]:
        return self.data_loader.load_content()

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        self.chunks = []
        chunk_id = 0

        step = self.chunk_size - self.chunk_overlap
        if step <= 0:
            step = self.chunk_size

        for doc in documents:
            content = doc.get('content', '')
            words = content.split()

            for i in range(0, len(words), step):
                chunk_words = words[i:i + self.chunk_size]
                chunk_text = ' '.join(chunk_words)

                if chunk_text.strip():
                    self.chunks.append({
                        'chunk_id': chunk_id,
                        'doc_id': doc.get('id'),
                        'content': chunk_text,
                        'tag': doc.get('tag'),
                        'source': doc.get('source')
                    })
                    chunk_id += 1

        return self.chunks

    def generate_embeddings(self) -> List[List[float]]:
        texts = [chunk['content'] for chunk in self.chunks]
        self.embeddings = self.embedding_api.generate_embeddings(texts)
        return self.embeddings

    def normalize_embeddings(self) -> np.ndarray:
        embeddings_array = np.array(self.embeddings, dtype=np.float32)

        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        embeddings_normalized = embeddings_array / norms

        return embeddings_normalized

    def build_index(self, embeddings_normalized: np.ndarray):
        try:
            import faiss
        except ImportError:
            raise ImportError("faiss package is required. Install it with: pip install faiss-cpu")

        dimension = embeddings_normalized.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings_normalized)

    def save_vector_db(self):
        save_path = Path(self.vdb_save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        db_data = {
            'index': self.index,
            'chunks': self.chunks,
            'embeddings': self.embeddings,
            'metadata': {
                'vdb_type': self.vdb_type,
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap,
                'document_structure_mode': self.document_structure_mode,
                'num_chunks': len(self.chunks),
                'embedding_dimension': len(self.embeddings[0]) if self.embeddings else 0
            }
        }

        index_path = save_path / 'vector_db.pkl'
        with open(index_path, 'wb') as f:
            pickle.dump(db_data, f)

    def build_vector_db(self):
        documents = self.load_data()
        self.chunk_documents(documents)
        self.generate_embeddings()
        embeddings_normalized = self.normalize_embeddings()
        self.build_index(embeddings_normalized)
        self.save_vector_db()

        return {
            'num_chunks': len(self.chunks),
            'embedding_dimension': len(self.embeddings[0]) if self.embeddings else 0,
            'save_path': self.vdb_save_path
        }
