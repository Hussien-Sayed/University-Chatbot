"""HNSW (Hierarchical Navigable Small World) index visualizations."""
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional, Tuple

from ..utils import reduce_embeddings_pca, get_hnsw_graph_structure, get_hnsw_neighbors_sample


class HNSWVisualizer:
    """Visualizations for HNSW graph structure and retrieval flow."""

    def __init__(self, index, embeddings: np.ndarray, chunks: List[Dict], metadata: Dict[str, Any]):
        """Initialize with HNSW index data.

        Args:
            index: FAISS IndexHNSWFlat
            embeddings: Array of shape (n_chunks, dimension)
            chunks: List of chunk dictionaries
            metadata: Vector DB metadata dict
        """
        self.index = index
        self.embeddings = embeddings
        self.chunks = chunks
        self.metadata = metadata

        # Get HNSW parameters
        self.m = metadata.get('faiss_hnsw_m', 16)
        self.ef_construction = metadata.get('faiss_hnsw_ef_construction', 200)
        self.ef_search = metadata.get('faiss_hnsw_ef_search', 128)

        # Extract graph structure
        self.graph_data = get_hnsw_graph_structure(index)

        # Reduce to 2D for visualization
        self.points_2d = reduce_embeddings_pca(embeddings, n_components=2)

        # Get layer assignments
        if self.graph_data:
            self.levels = self.graph_data['levels']
            self.max_level = self.graph_data['max_level']
            self.entry_point = self.graph_data['entry_point']
        else:
            self.levels = np.zeros(len(chunks), dtype=int)
            self.max_level = 0
            self.entry_point = 0

    def render_static(self):
        """Render static structure visualizations (no query)."""
        if self.graph_data is None:
            st.warning("Could not extract HNSW graph structure from index.")
            st.info("Debug info: Check console logs for detailed error messages.")
            # Show what we know about the index
            st.write(f"Index type: {type(self.index)}")
            st.write(f"Has 'hnsw' attr: {hasattr(self.index, 'hnsw')}")
            if hasattr(self.index, 'index'):
                st.write(f"Wrapped index type: {type(self.index.index)}")
                st.write(f"Wrapped has 'hnsw' attr: {hasattr(self.index.index, 'hnsw')}")
            return

        with st.expander("🕸️ HNSW Graph Structure", expanded=True):
            # 1. Layer Distribution
            self._plot_layer_distribution()

            # 2. Graph Structure Map (PCA)
            self._plot_graph_structure_map()


    def render_with_query(self, query_embedding: np.ndarray, retrieved_chunks: List[Dict]):
        """Render query-aware visualizations showing retrieval flow.

        Args:
            query_embedding: Query vector
            retrieved_chunks: List of retrieved chunk dicts
        """
        if self.graph_data is None:
            st.warning("Could not extract HNSW graph structure from index.")
            return

        with st.expander("🕸️ HNSW Retrieval Flow", expanded=True):
            # Convert query to numpy array if needed
            query_embedding_arr = np.array(query_embedding, dtype=np.float32)

            # Project query to 2D using same PCA as embeddings
            from ..utils import reduce_embeddings_pca
            combined = np.vstack([self.embeddings, query_embedding_arr.reshape(1, -1)])
            combined_2d = reduce_embeddings_pca(combined, n_components=2)
            query_2d = combined_2d[-1]  # Take the last row (query)

            # 1. Query Traversal Path
            self._plot_query_path(query_2d, retrieved_chunks)

            # 2. Layer Distribution (also show)
            self._plot_layer_distribution()

            # 3. Graph Structure Map with Query
            self._plot_graph_structure_map(query_point=query_2d)

    def _plot_layer_distribution(self):
        """Plot bar chart showing nodes per layer."""
        st.write("**Layer Distribution**")

        if self.graph_data is None:
            st.warning("Graph data not available.")
            return

        nodes_per_level = self.graph_data['nodes_per_level']
        max_level = self.graph_data['max_level']

        # Create bar chart data
        chart_data = [{'Layer': f'L{i}', 'Nodes': int(count)} for i, count in enumerate(nodes_per_level)]

        st.bar_chart(chart_data, x='Layer', y='Nodes')

        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Nodes", self.graph_data['num_elements'])
        col2.metric("Max Layer", max_level)
        col3.metric("Entry Point", f"#{self.entry_point}")
        col4.metric("M (Connections)", self.m)

        st.caption(
            f"HNSW uses a multi-layer graph. Upper layers have fewer nodes for fast traversal, "
            f"lower layers have all nodes for precise search. "
            f"Each node connects to ~{self.m} neighbors per layer."
        )

    def _plot_graph_structure_map(self, query_point: Optional[np.ndarray] = None):
        """Plot 2D PCA map with nodes colored by layer."""
        st.write("**Graph Structure Map** (PCA visualization)")

        # Build Plotly figure
        fig = go.Figure()

        # Create a color scale for layers (higher layers = darker/brighter)
        max_level = self.max_level

        # Add nodes for each layer (show each node only at its MAX level)
        plotted = np.zeros(len(self.chunks), dtype=bool)  # Track which nodes already plotted
        for level in range(max_level, -1, -1):  # Plot from highest layer first
            mask = (self.levels == level) & ~plotted  # Only unplotted nodes at this exact level
            if not np.any(mask):
                continue

            plotted[mask] = True  # Mark these as plotted
            layer_points = self.points_2d[mask]
            node_indices = np.where(mask)[0]

            # Color intensity based on layer
            opacity = 0.3 + 0.7 * (level / max(max_level, 1))
            size = 6 + level * 2  # Larger nodes for higher layers

            fig.add_trace(go.Scatter(
                x=layer_points[:, 0],
                y=layer_points[:, 1],
                mode='markers',
                name=f'Max Level {level} ({np.sum(mask)} nodes)',
                marker=dict(
                    size=size,
                    opacity=opacity,
                    line=dict(width=1, color='white')
                ),
                hovertemplate=f'Level {level}<br>Node: %{{text}}<br>x: %{{x:.2f}}<br>y: %{{y:.2f}}<extra></extra>',
                text=[f"#{idx}" + (" (ENTRY)" if idx == self.entry_point else "") for idx in node_indices]
            ))

        # Add query point if provided (green star)
        if query_point is not None:
            fig.add_trace(go.Scatter(
                x=[query_point[0]],
                y=[query_point[1]],
                mode='markers',
                name='QUERY',
                marker=dict(
                    symbol='star',
                    size=12,
                    color='green',
                    line=dict(width=2, color='darkgreen')
                ),
                hovertemplate='QUERY<br>x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>'
            ))

        fig.update_layout(
            title='HNSW Graph Structure' + (' with Query' if query_point is not None else ''),
            xaxis_title='PCA Component 1',
            yaxis_title='PCA Component 2',
            hovermode='closest',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )

        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            f"Nodes shown at their maximum layer level (higher layers have fewer nodes). "
            f"Larger markers = higher layer. Entry point (node #{self.entry_point}) is where all searches start. "
            f"Note: In HNSW, nodes appear in all layers from 0 up to their max level, but we show each node only once at its highest layer for clarity."
        )




    def _plot_query_path(self, query_point: np.ndarray, retrieved_chunks: List[Dict]):
        """Plot the query traversal path through the graph."""
        st.write("**Query Traversal**")

        # Get retrieved chunk indices
        retrieved_indices = []
        for chunk_data in retrieved_chunks:
            chunk = chunk_data.get('chunk', chunk_data)
            chunk_id = chunk.get('chunk_id')
            if chunk_id is not None:
                try:
                    # Find index in our chunks list
                    for i, c in enumerate(self.chunks):
                        if c.get('chunk_id') == chunk_id:
                            retrieved_indices.append(i)
                            break
                except:
                    pass

        # Show search parameters
        col1, col2, col3 = st.columns(3)
        col1.metric("Search Depth (ef)", self.ef_search)
        col2.metric("Entry Point", f"#{self.entry_point}")
        col3.metric("Results Found", len(retrieved_indices))

        # Show retrieved nodes
        if retrieved_indices:
            st.write(f"**Nearest Neighbors:** {retrieved_indices[:10]}")

        st.caption(
            f"Search starts at node #{self.entry_point} (top layer), "
            f"traverses down layers finding closer neighbors, "
            f"checks up to {self.ef_search} candidates in bottom layer. "
            f"Returns {len(retrieved_indices)} nearest neighbors to query."
        )
