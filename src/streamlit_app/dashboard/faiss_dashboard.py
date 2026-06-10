"""Main FAISS dashboard component."""
import numpy as np
import streamlit as st
from typing import Dict, Any, List, Optional
import pickle
from pathlib import Path

from .utils import reduce_embeddings_pca
from .visualizations.ivf_viz import IVFVisualizer
from .visualizations.hnsw_viz import HNSWVisualizer
from .visualizations.pq_viz import PQVisualizer


class FAISSDashboard:
    """Dashboard for visualizing FAISS index structure and retrieval flow."""

    def __init__(self, vector_db_path: str):
        """Initialize dashboard with vector database.

        Args:
            vector_db_path: Path to directory containing vector_db.pkl
        """
        self.vector_db_path = Path(vector_db_path)
        self.db_data = None
        self.chunks = []
        self.embeddings = None
        self.metadata = {}
        self.index = None
        self.faiss_index_type = 'flat_ip'

        self._load_vector_db()

    def _load_vector_db(self):
        """Load vector database from disk."""
        db_file = self.vector_db_path / 'vector_db.pkl'
        if not db_file.exists():
            st.error(f"Vector database not found: {db_file}")
            return

        try:
            with open(db_file, 'rb') as f:
                self.db_data = pickle.load(f)

            self.chunks = self.db_data.get('chunks', [])
            self.embeddings = np.array(self.db_data.get('embeddings', []), dtype=np.float32)
            self.metadata = self.db_data.get('metadata', {})
            self.index = self.db_data.get('index')
            self.faiss_index_type = self.metadata.get('faiss_index_type', 'flat_ip')

        except Exception as e:
            st.error(f"Error loading vector database: {e}")

    def render(self, query_data: Optional[Dict[str, Any]] = None):
        """Render the dashboard.

        Args:
            query_data: Optional dict with 'query', 'query_embedding', 'retrieved_chunks'
                       If provided, shows query-aware visualizations
        """
        if self.db_data is None:
            return

        if len(self.chunks) == 0 or self.embeddings is None or self.embeddings.size == 0:
            st.error("No chunks or embeddings found in the vector database.")
            return

        if len(self.chunks) != self.embeddings.shape[0]:
            st.error(f"Chunk count ({len(self.chunks)}) does not match embedding count ({self.embeddings.shape[0]}).")
            return

        # Render sidebar info
        self._render_sidebar()

        # Render main visualizations based on index type
        if self.faiss_index_type == 'ivf_flat':
            self._render_ivf(query_data)
        elif self.faiss_index_type == 'hnsw':
            self._render_hnsw(query_data)
        elif self.faiss_index_type == 'pq':
            self._render_pq(query_data)

    def _render_sidebar(self):
        """Render sidebar with index information."""
        # Get FAISS index info from metadata
        faiss_index_type = self.faiss_index_type
        embedding_dim = self.embeddings.shape[1]
        num_chunks = len(self.chunks)

        with st.sidebar:
            st.header("Vector DB Info")
            st.write(f"Chunks: `{num_chunks}`")
            st.write(f"Embedding shape: `{self.embeddings.shape}`")
            st.write(f"Document structure mode: `{self.metadata.get('document_structure_mode', 'unknown')}`")

            # FAISS Index Information Section
            st.divider()
            st.header("🔍 FAISS Index Info")

            # Index type with icon
            index_type_icons = {
                'flat_ip': '🔍',
                'ivf_flat': '📦',
                'hnsw': '🕸️',
                'pq': '🗜️'
            }
            index_type_labels = {
                'flat_ip': 'FlatIP (Exact)',
                'ivf_flat': 'IVF-Flat (Clustering)',
                'hnsw': 'HNSW (Graph)',
                'pq': 'PQ (Quantized)'
            }
            icon = index_type_icons.get(faiss_index_type, '🔍')
            label = index_type_labels.get(faiss_index_type, faiss_index_type.upper())
            st.write(f"**Index Type:** {icon} {label}")

            # Index-specific parameters
            if faiss_index_type == 'ivf_flat':
                nlist = self.metadata.get('faiss_ivf_nlist', 100)
                nprobe = self.metadata.get('faiss_ivf_nprobe', 10)
                st.write(f"**Clusters (nlist):** `{nlist}`")
                st.write(f"**Search clusters (nprobe):** `{nprobe}`")
                st.progress(min(nprobe / nlist, 1.0), text=f"Search coverage: {nprobe}/{nlist} clusters")
                st.caption("💡 Higher nprobe = more accurate but slower")

            elif faiss_index_type == 'hnsw':
                m = self.metadata.get('faiss_hnsw_m', 16)
                ef_construction = self.metadata.get('faiss_hnsw_ef_construction', 200)
                ef_search = self.metadata.get('faiss_hnsw_ef_search', 128)
                st.write(f"**Connections (M):** `{m}`")
                st.write(f"**Build depth:** `{ef_construction}`")
                st.write(f"**Search depth:** `{ef_search}`")
                st.caption("💡 Higher M/ef = more accurate but more memory/slower")

            elif faiss_index_type == 'pq':
                m = self.metadata.get('faiss_pq_m', 8)
                nbits = self.metadata.get('faiss_pq_nbits', 8)
                original_size = num_chunks * embedding_dim * 4  # 4 bytes per float
                compressed_size = num_chunks * m * (nbits // 8)
                compression_ratio = original_size / max(compressed_size, 1)
                st.write(f"**Subquantizers (M):** `{m}`")
                st.write(f"**Bits per code:** `{nbits}`")
                st.write(f"**Compression ratio:** `{compression_ratio:.1f}x`")
                st.progress(min(compression_ratio / 50, 1.0), text=f"Memory saved: {compression_ratio:.1f}x")
                st.caption("💡 Higher compression = less memory but lower accuracy")

            else:  # flat_ip
                st.write("**Search method:** Exact (brute force)")
                st.write("**Accuracy:** 100%")
                st.caption("💡 Best for small datasets (<10K chunks)")

            # Memory estimation
            st.divider()
            st.header("💾 Memory Estimate")
            if faiss_index_type == 'flat_ip':
                index_size_mb = (num_chunks * embedding_dim * 4) / (1024 * 1024)
                st.write(f"**Index size:** `{index_size_mb:.1f} MB`")
            elif faiss_index_type == 'ivf_flat':
                index_size_mb = (num_chunks * embedding_dim * 4) / (1024 * 1024)
                overhead_mb = (self.metadata.get('faiss_ivf_nlist', 100) * embedding_dim * 4) / (1024 * 1024)
                st.write(f"**Index size:** `{index_size_mb + overhead_mb:.1f} MB`")
                st.caption(f"Includes {overhead_mb:.1f} MB cluster overhead")
            elif faiss_index_type == 'hnsw':
                base_size_mb = (num_chunks * embedding_dim * 4) / (1024 * 1024)
                m = self.metadata.get('faiss_hnsw_m', 16)
                graph_overhead_mb = (num_chunks * m * 2 * 4) / (1024 * 1024)  # 2 connections per M on avg
                st.write(f"**Index size:** `{base_size_mb + graph_overhead_mb:.1f} MB`")
                st.caption(f"Includes {graph_overhead_mb:.1f} MB graph overhead")
            elif faiss_index_type == 'pq':
                m = self.metadata.get('faiss_pq_m', 8)
                nbits = self.metadata.get('faiss_pq_nbits', 8)
                compressed_size_mb = (num_chunks * m * (nbits // 8)) / (1024 * 1024)
                codebook_size_mb = (self.metadata.get('faiss_ivf_nlist', 100) * embedding_dim * 4) / (1024 * 1024)
                st.write(f"**Index size:** `{compressed_size_mb + codebook_size_mb:.2f} MB`")
                st.caption("Highly compressed representation")

            if self.metadata:
                st.divider()
                st.subheader("Full Metadata")
                st.json(self.metadata)

    def _render_ivf(self, query_data: Optional[Dict[str, Any]]):
        """Render IVF-specific visualizations."""
        visualizer = IVFVisualizer(self.index, self.embeddings, self.chunks, self.metadata)

        if query_data:
            query_embedding = query_data.get('query_embedding')
            retrieved_chunks = query_data.get('retrieved_chunks', [])
            if query_embedding is not None:
                visualizer.render_with_query(query_embedding, retrieved_chunks)
        else:
            visualizer.render_static()

    def _render_hnsw(self, query_data: Optional[Dict[str, Any]]):
        """Render HNSW-specific visualizations."""
        visualizer = HNSWVisualizer(self.index, self.embeddings, self.chunks, self.metadata)

        if query_data:
            query_embedding = query_data.get('query_embedding')
            retrieved_chunks = query_data.get('retrieved_chunks', [])
            if query_embedding is not None:
                visualizer.render_with_query(query_embedding, retrieved_chunks)
        else:
            visualizer.render_static()

    def _render_pq(self, query_data: Optional[Dict[str, Any]]):
        """Render PQ-specific visualizations showing compression effect."""
        visualizer = PQVisualizer(self.index, self.embeddings, self.chunks, self.metadata)

        if query_data:
            query_embedding = query_data.get('query_embedding')
            retrieved_chunks = query_data.get('retrieved_chunks', [])
            if query_embedding is not None:
                visualizer.render_with_query(query_embedding, retrieved_chunks)
        else:
            visualizer.render_static()

