import os
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.data_utils.data_loader import DataLoader
from src.llm.embedding_api import EmbeddingAPI

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False


def normalize_text(text: str) -> str:
    """Normalize text by lowercasing and removing punctuation/quotes."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation and quotes
    return text


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

        # FAISS index configuration
        self.faiss_index_type = os.getenv("FAISS_INDEX_TYPE", "flat_ip").lower()

        # IVF parameters
        self.faiss_ivf_nlist = int(os.getenv("FAISS_IVF_NLIST", "100"))
        self.faiss_ivf_nprobe = int(os.getenv("FAISS_IVF_NPROBE", "10"))

        # HNSW parameters
        self.faiss_hnsw_m = int(os.getenv("FAISS_HNSW_M", "16"))
        self.faiss_hnsw_ef_construction = int(os.getenv("FAISS_HNSW_EF_CONSTRUCTION", "200"))
        self.faiss_hnsw_ef_search = int(os.getenv("FAISS_HNSW_EF_SEARCH", "128"))

        # PQ parameters
        self.faiss_pq_m = int(os.getenv("FAISS_PQ_M", "8"))
        self.faiss_pq_nbits = int(os.getenv("FAISS_PQ_NBITS", "8"))

        self.chunks = []
        self.embeddings = []
        self.metadata = []
        self.index = None
        self.bm25_index = None

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

    def build_faiss_index(self, embeddings_normalized: np.ndarray):
        """Build FAISS index based on configured type. Supports: flat_ip, ivf_flat, hnsw, pq."""
        try:
            import faiss
        except ImportError:
            raise ImportError("faiss package is required. Install it with: pip install faiss-cpu")

        dimension = embeddings_normalized.shape[1]
        num_vectors = embeddings_normalized.shape[0]

        if self.faiss_index_type == "flat_ip":
            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(embeddings_normalized)

        elif self.faiss_index_type == "ivf_flat":
            # Adjust nlist if we have fewer vectors than clusters
            nlist = min(self.faiss_ivf_nlist, max(1, num_vectors // 2))
            quantizer = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
            # IVF requires training on representative vectors
            if num_vectors >= nlist:
                self.index.train(embeddings_normalized)
            else:
                # Fall back to flat if not enough vectors for meaningful clustering
                self.index = faiss.IndexFlatIP(dimension)
            self.index.add(embeddings_normalized)

        elif self.faiss_index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(dimension, self.faiss_hnsw_m)
            self.index.hnsw.efConstruction = self.faiss_hnsw_ef_construction
            self.index.add(embeddings_normalized)

        elif self.faiss_index_type == "pq":
            # Check if dimension is divisible by M
            if dimension % self.faiss_pq_m != 0:
                # Adjust M to be a divisor of dimension
                for m in range(self.faiss_pq_m, 0, -1):
                    if dimension % m == 0:
                        self.faiss_pq_m = m
                        break
            quantizer = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIVFPQ(quantizer, dimension, self.faiss_ivf_nlist, self.faiss_pq_m, self.faiss_pq_nbits)
            # PQ requires training
            if num_vectors >= self.faiss_ivf_nlist:
                self.index.train(embeddings_normalized)
            else:
                # Fall back to flat if not enough vectors
                self.index = faiss.IndexFlatIP(dimension)
            self.index.add(embeddings_normalized)

        else:
            # Unknown index type, fall back to flat_ip
            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(embeddings_normalized)

    def build_index(self, embeddings_normalized: np.ndarray):
        """Build FAISS index using configured index type."""
        self.build_faiss_index(embeddings_normalized)

    def build_bm25_index(self):
        if not BM25_AVAILABLE:
            raise ImportError("rank-bm25 package is required. Install it with: pip install rank-bm25")

        tokenized_chunks = [normalize_text(chunk['content']).split() for chunk in self.chunks]
        self.bm25_index = BM25Okapi(tokenized_chunks)

    def save_vector_db(self):
        save_path = Path(self.vdb_save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        db_data = {
            'index': self.index,
            'chunks': self.chunks,
            'embeddings': self.embeddings,
            'bm25_index': self.bm25_index,
            'metadata': {
                'vdb_type': self.vdb_type,
                'faiss_index_type': self.faiss_index_type,
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap,
                'document_structure_mode': self.document_structure_mode,
                'num_chunks': len(self.chunks),
                'embedding_dimension': len(self.embeddings[0]) if self.embeddings else 0,
                # Store FAISS parameters for retrieval-time configuration
                'faiss_ivf_nlist': self.faiss_ivf_nlist,
                'faiss_ivf_nprobe': self.faiss_ivf_nprobe,
                'faiss_hnsw_m': self.faiss_hnsw_m,
                'faiss_hnsw_ef_construction': self.faiss_hnsw_ef_construction,
                'faiss_hnsw_ef_search': self.faiss_hnsw_ef_search,
                'faiss_pq_m': self.faiss_pq_m,
                'faiss_pq_nbits': self.faiss_pq_nbits
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
        self.build_bm25_index()
        self.save_vector_db()

        return {
            'num_chunks': len(self.chunks),
            'embedding_dimension': len(self.embeddings[0]) if self.embeddings else 0,
            'save_path': self.vdb_save_path
        }
