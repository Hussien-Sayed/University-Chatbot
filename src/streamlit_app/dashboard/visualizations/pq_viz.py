"""PQ (Product Quantization) index visualizations."""
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional

from ..utils import reduce_embeddings_pca, get_pq_reconstructed_vectors, _get_pq_index


class PQVisualizer:
    """Visualizations for PQ compression effect: original vs reconstructed vectors."""

    def __init__(self, index, embeddings: np.ndarray, chunks: List[Dict], metadata: Dict[str, Any]):
        """Initialize with PQ index data.

        Args:
            index: FAISS IndexPQ or IndexIVFPQ
            embeddings: Array of shape (n_chunks, dimension) - original vectors
            chunks: List of chunk dictionaries
            metadata: Vector DB metadata dict
        """
        self.index = index
        self.embeddings = embeddings
        self.chunks = chunks
        self.metadata = metadata

        # Get PQ parameters
        self.m = metadata.get('faiss_pq_m', 8)
        self.nbits = metadata.get('faiss_pq_nbits', 8)
        self.dim = embeddings.shape[1]

        # Reconstruct vectors from PQ codes
        self.reconstructed = get_pq_reconstructed_vectors(index, embeddings)

        # Compute PCA: fit on original, apply to both
        self.points_2d_original = None
        self.points_2d_reconstructed = None
        if self.reconstructed is not None:
            self.points_2d_original = reduce_embeddings_pca(embeddings, n_components=2)
            # For reconstructed, we need to use the same PCA transformation
            # Compute PCA components from original and apply to reconstructed
            centered_orig = embeddings - embeddings.mean(axis=0, keepdims=True)
            _, _, vt = np.linalg.svd(centered_orig, full_matrices=False)
            components = vt[:2]
            centered_recon = self.reconstructed - embeddings.mean(axis=0, keepdims=True)
            self.points_2d_reconstructed = (centered_recon @ components.T).astype(np.float32)

        # Generate shared colors for all chunks (used in both plots)
        self.colors = self._generate_chunk_colors()

    def render_static(self):
        """Render static PQ comparison (no query)."""
        # Check if index is actually a PQ type
        pq_index = _get_pq_index(self.index)
        if pq_index is None:
            st.warning("🔢 PQ Visualization Unavailable")
            st.info(
                "The index was built without PQ compression (likely due to insufficient training data "
                "or settings incompatible with your dataset size). "
                "Try reducing FAISS_PQ_NBITS in Settings (e.g., from 8 to 4) and rebuild the vector DB."
            )
            return

        if self.reconstructed is None:
            st.warning("Could not reconstruct vectors from PQ codes.")
            return

        if self.points_2d_original is None or self.points_2d_reconstructed is None:
            st.warning("Could not compute PCA for visualization.")
            return

        with st.expander("🔢 PQ Compression Effect", expanded=True):
            self._plot_side_by_side_comparison()

            # Compression metrics
            self._show_compression_metrics()

    def render_with_query(self, query_embedding: np.ndarray, retrieved_chunks: List[Dict]):
        """Render PQ comparison with query overlay."""
        # Check if index is actually a PQ type
        pq_index = _get_pq_index(self.index)
        if pq_index is None:
            st.warning("🔢 PQ Visualization Unavailable")
            st.info(
                "The index was built without PQ compression (likely due to insufficient training data "
                "or settings incompatible with your dataset size). "
                "Try reducing FAISS_PQ_NBITS in Settings (e.g., from 8 to 4) and rebuild the vector DB."
            )
            return

        if self.reconstructed is None:
            st.warning("Could not reconstruct vectors from PQ codes.")
            return

        if self.points_2d_original is None or self.points_2d_reconstructed is None:
            st.warning("Could not compute PCA for visualization.")
            return

        with st.expander("🔢 PQ Compression Effect (with Query)", expanded=True):
            self._plot_side_by_side_comparison_with_query(query_embedding, retrieved_chunks)

            # Compression metrics
            self._show_compression_metrics()

    def _generate_chunk_colors(self) -> List[str]:
        """Generate unique color for each chunk using vibrant color palette."""
        import plotly.express as px
        n_chunks = len(self.chunks)

        # Use vibrant continuous color scale - each chunk gets unique color
        colors = px.colors.sample_colorscale(
            px.colors.sequential.Plasma,
            [i / max(1, n_chunks - 1) for i in range(n_chunks)]
        )
        return colors

    def _get_colors_by_tag(self) -> List[str]:
        """Return pre-generated shared colors for consistency across plots."""
        return self.colors

    def _plot_side_by_side_comparison(self):
        """Plot original vs reconstructed vectors side by side."""
        colors = self._get_colors_by_tag()
        n_chunks = len(self.chunks)

        col1, col2 = st.columns(2)

        with col1:
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=self.points_2d_original[:, 0],
                y=self.points_2d_original[:, 1],
                mode='markers',
                marker=dict(
                    size=8,
                    color=colors,
                    opacity=0.7,
                    line=dict(width=1, color='white')
                ),
                name='Original',
                hovertemplate='Chunk %{text}<br>Tag: %{customdata}<extra></extra>',
                text=[f"#{i}" for i in range(n_chunks)],
                customdata=[chunk.get('tag', 'unknown') for chunk in self.chunks]
            ))
            fig1.update_layout(
                title='Original Vectors (PCA)',
                xaxis_title='PC1',
                yaxis_title='PC2',
                height=500,
                showlegend=False
            )
            st.plotly_chart(fig1, use_container_width=True)
            st.caption(f"Original: {n_chunks} vectors, {self.dim}D")

        with col2:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=self.points_2d_reconstructed[:, 0],
                y=self.points_2d_reconstructed[:, 1],
                mode='markers',
                marker=dict(
                    size=8,
                    color=colors,
                    opacity=0.7,
                    line=dict(width=1, color='white')
                ),
                name='PQ Reconstructed',
                hovertemplate='Chunk %{text}<br>Tag: %{customdata}<extra></extra>',
                text=[f"#{i}" for i in range(n_chunks)],
                customdata=[chunk.get('tag', 'unknown') for chunk in self.chunks]
            ))
            fig2.update_layout(
                title='PQ-Reconstructed Vectors (Same PCA)',
                xaxis_title='PC1',
                yaxis_title='PC2',
                height=500,
                showlegend=False
            )
            st.plotly_chart(fig2, use_container_width=True)
            st.caption(f"PQ: M={self.m}, nbits={self.nbits}")

        st.caption(
            "Each chunk has a unique color. Notice how PQ reconstruction 'snaps' vectors to discrete grid positions. "
            "PCA fitted on originals and applied to both for direct comparison."
        )

        # Color map legend showing chunk to color mapping
        self._show_color_map()

    def _plot_side_by_side_comparison_with_query(self, query_embedding: np.ndarray, retrieved_chunks: List[Dict]):
        """Plot comparison with query overlay."""
        colors = self._get_colors_by_tag()
        n_chunks = len(self.chunks)

        # Project query using same PCA
        query_arr = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        centered_query = query_arr - self.embeddings.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(self.embeddings - self.embeddings.mean(axis=0, keepdims=True), full_matrices=False)
        components = vt[:2]
        query_2d = (centered_query @ components.T).astype(np.float32).flatten()

        # Get retrieved chunk indices
        retrieved_indices = []
        for chunk_data in retrieved_chunks:
            chunk = chunk_data.get('chunk', chunk_data)
            chunk_id = chunk.get('chunk_id')
            if chunk_id is not None:
                for i, c in enumerate(self.chunks):
                    if c.get('chunk_id') == chunk_id:
                        retrieved_indices.append(i)
                        break

        col1, col2 = st.columns(2)

        with col1:
            fig1 = go.Figure()
            # All chunks
            fig1.add_trace(go.Scatter(
                x=self.points_2d_original[:, 0],
                y=self.points_2d_original[:, 1],
                mode='markers',
                marker=dict(
                    size=8,
                    color=colors,
                    opacity=0.5,
                    line=dict(width=1, color='white')
                ),
                name='Original',
                hovertemplate='Chunk %{text}<br>Tag: %{customdata}<extra></extra>',
                text=[f"#{i}" for i in range(n_chunks)],
                customdata=[chunk.get('tag', 'unknown') for chunk in self.chunks]
            ))
            # Retrieved chunks (highlighted)
            if retrieved_indices:
                fig1.add_trace(go.Scatter(
                    x=self.points_2d_original[retrieved_indices, 0],
                    y=self.points_2d_original[retrieved_indices, 1],
                    mode='markers',
                    marker=dict(
                        size=12,
                        color='green',
                        symbol='circle',
                        line=dict(color='darkgreen', width=2)
                    ),
                    name='Retrieved',
                    hovertemplate='Retrieved Chunk %{text}<extra></extra>',
                    text=[f"#{i}" for i in retrieved_indices]
                ))
            # Query
            fig1.add_trace(go.Scatter(
                x=[query_2d[0]],
                y=[query_2d[1]],
                mode='markers+text',
                marker=dict(size=20, color='gold', symbol='star', line=dict(color='black', width=2)),
                text=['QUERY'],
                textposition='top center',
                name='Query',
                hovertemplate='Query<extra></extra>'
            ))
            fig1.update_layout(
                title='Original Vectors (PCA)',
                xaxis_title='PC1',
                yaxis_title='PC2',
                height=500,
                showlegend=True
            )
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            fig2 = go.Figure()
            # All chunks (reconstructed)
            fig2.add_trace(go.Scatter(
                x=self.points_2d_reconstructed[:, 0],
                y=self.points_2d_reconstructed[:, 1],
                mode='markers',
                marker=dict(
                    size=8,
                    color=colors,
                    opacity=0.5,
                    line=dict(width=1, color='white')
                ),
                name='PQ Reconstructed',
                hovertemplate='Chunk %{text}<br>Tag: %{customdata}<extra></extra>',
                text=[f"#{i}" for i in range(n_chunks)],
                customdata=[chunk.get('tag', 'unknown') for chunk in self.chunks]
            ))
            # Retrieved chunks (highlighted)
            if retrieved_indices:
                valid_retrieved = [i for i in retrieved_indices if i < len(self.points_2d_reconstructed)]
                if valid_retrieved:
                    fig2.add_trace(go.Scatter(
                        x=self.points_2d_reconstructed[valid_retrieved, 0],
                        y=self.points_2d_reconstructed[valid_retrieved, 1],
                        mode='markers',
                        marker=dict(
                            size=12,
                            color='green',
                            symbol='circle',
                            line=dict(color='darkgreen', width=2)
                        ),
                        name='Retrieved',
                        hovertemplate='Retrieved Chunk %{text}<extra></extra>',
                        text=[f"#{i}" for i in valid_retrieved]
                    ))
            # Query (reconstructed position)
            fig2.add_trace(go.Scatter(
                x=[query_2d[0]],
                y=[query_2d[1]],
                mode='markers+text',
                marker=dict(size=20, color='gold', symbol='star', line=dict(color='black', width=2)),
                text=['QUERY'],
                textposition='top center',
                name='Query',
                hovertemplate='Query<extra></extra>'
            ))
            fig2.update_layout(
                title='PQ-Reconstructed Vectors (Same PCA)',
                xaxis_title='PC1',
                yaxis_title='PC2',
                height=500,
                showlegend=True
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.caption(
            "Each chunk has a unique color. Green circles = retrieved chunks. "
            "Gold star = query position. Notice PQ's 'snapping' effect on the right."
        )

        # Color map legend showing chunk to color mapping
        self._show_color_map()

    def _show_color_map(self):
        """Display a color legend mapping chunk indices to their colors."""
        n_chunks = len(self.chunks)
        if n_chunks > 50:
            # Too many chunks, show a simplified gradient bar
            st.write("**Color Map:** Chunks colored by Plasma scale (purple → yellow)")
            return

        # Create color map table
        st.write("**Chunk Color Map:**")

        # Display in columns (10 items per column)
        items_per_col = 10
        n_cols = (n_chunks + items_per_col - 1) // items_per_col
        cols = st.columns(min(n_cols, 4))

        for i in range(n_chunks):
            col_idx = i // items_per_col
            if col_idx < len(cols):
                color = self.colors[i]
                chunk = self.chunks[i]
                chunk_id = chunk.get('chunk_id', i)
                tag = chunk.get('tag', 'unknown')
                with cols[col_idx]:
                    st.markdown(
                        f"<div style='display:flex;align-items:center;margin:2px 0;'>"
                        f"<div style='width:12px;height:12px;background:{color};margin-right:8px;border-radius:2px;'></div>"
                        f"<span style='font-size:12px;'>#{chunk_id} ({tag})</span>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

    def _show_compression_metrics(self):
        """Display compression ratio and error metrics."""
        if self.reconstructed is None:
            return

        n_chunks = len(self.chunks)
        original_bytes = n_chunks * self.dim * 4  # 4 bytes per float32
        pq_bytes = n_chunks * self.m  # 1 byte per code per subspace
        compression_ratio = original_bytes / pq_bytes if pq_bytes > 0 else 0

        # Quantization error
        errors = np.linalg.norm(self.embeddings - self.reconstructed, axis=1)
        mean_error = np.mean(errors)
        max_error = np.max(errors)

        col1, col2, col3 = st.columns(3)
        col1.metric("Original Size", f"{original_bytes / 1024:.1f} KB")
        col2.metric("PQ Size", f"{pq_bytes / 1024:.1f} KB")
        col3.metric("Compression", f"{compression_ratio:.1f}x")

        col4, col5 = st.columns(2)
        col4.metric("Mean Error", f"{mean_error:.3f}")
        col5.metric("Max Error", f"{max_error:.3f}")
