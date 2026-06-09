"""IVF (Inverted File) index visualizations."""
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional, Tuple

from ..utils import reduce_embeddings_pca, get_ivf_centroids, compute_cluster_assignments, get_cluster_sizes


class IVFVisualizer:
    """Visualizations for IVF index structure and retrieval flow."""

    def __init__(self, index, embeddings: np.ndarray, chunks: List[Dict], metadata: Dict[str, Any]):
        """Initialize with IVF index data.

        Args:
            index: FAISS IndexIVFFlat or IndexIVFPQ
            embeddings: Array of shape (n_chunks, dimension)
            chunks: List of chunk dictionaries
            metadata: Vector DB metadata dict
        """
        self.index = index
        self.embeddings = embeddings
        self.chunks = chunks
        self.metadata = metadata

        # Get IVF parameters
        self.nlist = metadata.get('faiss_ivf_nlist', 100)
        self.nprobe = metadata.get('faiss_ivf_nprobe', 10)

        # Extract centroids
        self.centroids = get_ivf_centroids(index)

        # Compute assignments if we have centroids
        self.assignments = None
        if self.centroids is not None:
            self.assignments = compute_cluster_assignments(embeddings, self.centroids)

        # Reduce to 2D for visualization
        self.points_2d = reduce_embeddings_pca(embeddings, n_components=2)

    def render_static(self):
        """Render static structure visualizations (no query)."""
        if self.centroids is None:
            st.warning("Could not extract IVF centroids from index.")
            return

        with st.expander("📦 IVF Cluster Structure", expanded=True):
            # 1. Cluster Assignment Map
            self._plot_cluster_assignment_map()

            # 2. Cluster Size Distribution
            self._plot_cluster_size_distribution()

            # 3. Cluster Hierarchy (selectable)
            self._plot_cluster_hierarchy()

    def render_with_query(self, query_embedding: np.ndarray, retrieved_chunks: List[Dict]):
        """Render query-aware visualizations showing retrieval flow.

        Args:
            query_embedding: Query vector
            retrieved_chunks: List of retrieved chunk dicts
        """
        if self.centroids is None:
            st.warning("Could not extract IVF centroids from index.")
            return

        with st.expander("📦 IVF Retrieval Flow", expanded=True):
            # Convert query to numpy array if needed
            query_embedding_arr = np.array(query_embedding, dtype=np.float32)

            # Find nearest centroids to query (the nprobe clusters)
            query_distances = query_embedding_arr @ self.centroids.T
            nearest_centroid_ids = np.argsort(query_distances)[::-1][:self.nprobe]

            # Show retrieval flow
            self._plot_retrieval_flow(query_embedding, nearest_centroid_ids, retrieved_chunks)

            # Project query to 2D using same PCA as embeddings
            from ..utils import reduce_embeddings_pca
            combined = np.vstack([self.embeddings, query_embedding_arr.reshape(1, -1)])
            combined_2d = reduce_embeddings_pca(combined, n_components=2)
            query_2d = combined_2d[-1]  # Take the last row (query)

            # Show cluster assignment map with query highlighted
            self._plot_cluster_assignment_map(
                query_point=query_2d,
                query_embedding=query_embedding,
                highlighted_clusters=nearest_centroid_ids
            )

            # Also show cluster size distribution (works with or without query)
            self._plot_cluster_size_distribution()

            # Also show cluster explorer (works with or without query)
            self._plot_cluster_hierarchy()

    def _plot_cluster_assignment_map(
        self,
        query_point: Optional[np.ndarray] = None,
        query_embedding: Optional[np.ndarray] = None,
        highlighted_clusters: Optional[np.ndarray] = None
    ):
        """Plot 2D PCA map with chunks colored by cluster assignment."""
        st.write("**Cluster Assignment Map** (PCA visualization)")

        if self.assignments is None:
            st.warning("Cluster assignments not available.")
            return

        # Build Plotly figure with different symbols for chunks vs query
        fig = go.Figure()

        # Get unique clusters
        unique_clusters = sorted(set(self.assignments))

        # Add traces for each cluster (circles)
        for cluster_id in unique_clusters:
            mask = self.assignments == cluster_id
            cluster_points = self.points_2d[mask]

            fig.add_trace(go.Scatter(
                x=cluster_points[:, 0],
                y=cluster_points[:, 1],
                mode='markers',
                name=f'Cluster {cluster_id}',
                marker=dict(
                    symbol='circle',
                    size=8,
                    opacity=0.7
                ),
                hovertemplate=f'Cluster {cluster_id}<br>x: %{{x:.2f}}<br>y: %{{y:.2f}}<extra></extra>'
            ))

        # Add query point if provided (star symbol, larger)
        if query_point is not None:
            fig.add_trace(go.Scatter(
                x=[query_point[0]],
                y=[query_point[1]],
                mode='markers',
                name='QUERY',
                marker=dict(
                    symbol='star',
                    size=10,
                    color='green',
                    line=dict(width=2, color='darkgreen')
                ),
                hovertemplate='QUERY<br>x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>'
            ))

        fig.update_layout(
            title='Cluster Assignments' + (' with Query' if query_point is not None else ''),
            xaxis_title='PCA Component 1',
            yaxis_title='PCA Component 2',
            hovermode='closest',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Show highlighted clusters info
        if highlighted_clusters is not None and len(highlighted_clusters) > 0:
            st.write(f"**Searched clusters** (nprobe={len(highlighted_clusters)}): " +
                     ", ".join([f"C{cid}" for cid in highlighted_clusters[:5]]) +
                     ("..." if len(highlighted_clusters) > 5 else ""))

        # Explanation
        st.caption(
            f"Each color represents a cluster (nlist={self.nlist}). "
            f"Chunks are assigned to their nearest centroid. "
            f"At query time, only nprobe={self.nprobe} nearest clusters are searched."
        )

    def _plot_cluster_size_distribution(self):
        """Plot bar chart of chunks per cluster."""
        if self.assignments is None:
            return

        st.write("**Cluster Size Distribution**")

        sizes = get_cluster_sizes(self.assignments, self.nlist)

        # Create bar chart data
        chart_data = [{'Cluster': f'C{i}', 'Chunks': int(size)} for i, size in enumerate(sizes)]

        # Show top 20 clusters by size
        top_indices = np.argsort(sizes)[::-1][:20]
        top_data = [chart_data[i] for i in top_indices]

        st.bar_chart(top_data, x='Cluster', y='Chunks')

        # Stats
        col1, col2, col3 = st.columns(3)
        col1.metric("Min cluster size", int(sizes.min()))
        col2.metric("Max cluster size", int(sizes.max()))
        col3.metric("Avg cluster size", f"{sizes.mean():.1f}")

        # Balance check
        if sizes.max() > sizes.mean() * 3:
            st.warning("⚠️ Some clusters are much larger than average - may affect retrieval quality")

    def _plot_cluster_hierarchy(self):
        """Show interactive cluster-to-chunks hierarchy."""
        if self.assignments is None:
            return

        st.write("**Cluster Explorer**")

        # Let user select a cluster
        cluster_options = [f"Cluster {i}" for i in range(self.nlist)]
        selected = st.selectbox("Select a cluster to explore", cluster_options)
        selected_cluster = int(selected.split()[1])

        # Show chunks in this cluster
        cluster_chunk_indices = np.where(self.assignments == selected_cluster)[0]

        st.write(f"**Chunks in {selected}** ({len(cluster_chunk_indices)} total):")

        # Show as a dataframe
        cluster_chunks = []
        for idx in cluster_chunk_indices[:20]:  # Limit to 20 for performance
            chunk = self.chunks[idx]
            cluster_chunks.append({
                'chunk_id': chunk.get('chunk_id', idx),
                'doc_id': chunk.get('doc_id', 'unknown'),
                'tag': chunk.get('tag', 'unknown'),
                'preview': chunk.get('content', '')[:100] + '...'
            })

        if cluster_chunks:
            st.dataframe(cluster_chunks, use_container_width=True, hide_index=True)

        if len(cluster_chunk_indices) > 20:
            st.caption(f"... and {len(cluster_chunk_indices) - 20} more chunks")

    def _plot_retrieval_flow(
        self,
        query_embedding: np.ndarray,
        nearest_centroid_ids: np.ndarray,
        retrieved_chunks: List[Dict]
    ):
        """Visualize the IVF retrieval process step by step."""
        st.write("**Retrieval Flow**")

        # Step 1: Query
        st.write("**Step 1**: Query enters the system")
        st.progress(0.25, text="Query vector received")

        # Step 2: Find nearest centroids
        st.write(f"**Step 2**: Find {self.nprobe} nearest centroids (clusters)")
        centroid_list = ", ".join([f"C{cid}" for cid in nearest_centroid_ids[:5]])
        if len(nearest_centroid_ids) > 5:
            centroid_list += f" ... ({len(nearest_centroid_ids) - 5} more)"
        st.progress(0.50, text=f"Nearest centroids: {centroid_list}")

        # Step 3: Search within clusters
        total_clusters = self.nlist
        searched_clusters = len(nearest_centroid_ids)
        skipped_clusters = total_clusters - searched_clusters

        st.write(f"**Step 3**: Search only within {searched_clusters} clusters (skip {skipped_clusters})")
        st.progress(0.75, text=f"Searching {searched_clusters}/{total_clusters} clusters")

        # Step 4: Rank and return
        st.write(f"**Step 4**: Rank candidates and return top-{len(retrieved_chunks)} results")
        st.progress(1.0, text=f"Retrieved {len(retrieved_chunks)} chunks")

        # Show which clusters were searched
        st.write("**Searched clusters**:")
        searched_data = [{'Cluster': f'C{cid}', 'Status': 'Searched'} for cid in nearest_centroid_ids]
        st.dataframe(searched_data, use_container_width=True, hide_index=True, column_order=['Cluster', 'Status'])

        # Efficiency gain
        efficiency = (skipped_clusters / total_clusters) * 100
        st.success(f"⚡ Efficiency gain: Skipped {efficiency:.1f}% of clusters ({skipped_clusters}/{total_clusters})")
