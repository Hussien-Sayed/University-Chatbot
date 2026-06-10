"""Shared utilities for the Streamlit app."""
import os
import pickle
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np

from ..data_utils.data_loader import DataLoader
from ..llm.embedding_api import EmbeddingAPI
from ..llm.llm_api import LLMAPI
from ..rag.vector_db.vector_db_builder import VectorDBBuilder
from ..rag.pipeline import RAGPipeline


def build_vector_db():
    """Build or rebuild the vector database."""
    base_dir = Path(__file__).parent.parent.parent
    data_path = base_dir / os.getenv("DATA_SOURCE_PATH", "data/intents-v2.json")
    vdb_path = base_dir / os.getenv("VDB_SAVE_PATH", "data/vector_db")

    data_loader = DataLoader()
    embedding_api = EmbeddingAPI()

    builder = VectorDBBuilder(
        data_loader=data_loader,
        embedding_api=embedding_api,
        vdb_save_path=str(vdb_path),
    )

    result = builder.build_vector_db()
    return result


def load_retriever():
    """Load retriever (not cached to pick up settings changes)."""
    from ..rag.retriever.rag_retriever import RAGRetriever

    llm_api = LLMAPI()
    retriever = RAGRetriever(
        vector_db_path=os.getenv("VDB_SAVE_PATH", "data/vector_db"),
        llm_api=llm_api
        # num_chunks uses NUM_CHUNKS env var (defaults to 3)
    )
    return retriever


def load_pipeline():
    """Load the centralized RAG pipeline (not cached to pick up settings changes)."""
    return RAGPipeline()


def load_vector_db() -> Optional[Dict[str, Any]]:
    """Load vector database from disk."""
    vector_db_path = Path(os.getenv("VDB_SAVE_PATH", "data/vector_db")) / "vector_db.pkl"
    if not vector_db_path.exists():
        return None
    with open(vector_db_path, "rb") as f:
        return pickle.load(f)


def reduce_embeddings_pca(embeddings: np.ndarray) -> np.ndarray:
    """Reduce embeddings to 2D using PCA."""
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings array, got shape: {embeddings.shape}")

    if embeddings.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float32)

    if embeddings.shape[0] == 1:
        return np.array([[0.0, 0.0]], dtype=np.float32)

    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)

    components = vt[:2]
    points = centered @ components.T

    if points.shape[1] == 1:
        points = np.column_stack([points[:, 0], np.zeros(points.shape[0])])

    return points.astype(np.float32)
