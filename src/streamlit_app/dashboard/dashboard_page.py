"""Dashboard page for the Streamlit app."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

from . import FAISSDashboard
from .utils import reduce_embeddings_pca
from ..utils import load_vector_db


def dashboard_page():
    """Render the dashboard page."""
    st.set_page_config(page_title="Dashboard - University Chatbot", page_icon="📊", layout="wide")

    st.title("📊 Dashboard")
    st.caption("Vector DB visualization and analytics")

    # Check if vector DB exists
    import os
    vdb_path = os.getenv("VDB_SAVE_PATH", "data/vector_db")
    vdb_full_path = Path(vdb_path) / "vector_db.pkl"
    if not vdb_full_path.exists():
        st.error("Vector database not found. Please build it first from the Chat page.")
        return

    # Get query data from session state if available
    query_data = st.session_state.get("last_query_result", None)

    if query_data:
        st.info(f"ℹ️ Showing visualizations for last query: \"{query_data.get('query', 'Unknown')[:50]}...\"")

    # Initialize and render FAISS dashboard
    dashboard = FAISSDashboard(vdb_path)
    dashboard.render(query_data=query_data)

    # Additional: Show chunk explorer at the bottom
    with st.expander("🔍 Chunk Explorer", expanded=False):
        # Load data for chunk explorer
        db_data = load_vector_db()
        if db_data:
            chunks = db_data.get("chunks", [])
            embeddings = np.array(db_data.get("embeddings", []), dtype=np.float32)

            if len(chunks) > 0 and embeddings.size > 0:
                # Reduce to 2D for display
                points_2d = reduce_embeddings_pca(embeddings)

                # Get query embedding if available for projection
                query_point_2d = None
                if query_data and query_data.get("query_embedding") is not None:
                    query_emb = np.array(query_data["query_embedding"], dtype=np.float32)
                    combined = np.vstack([embeddings, query_emb.reshape(1, -1)])
                    combined_2d = reduce_embeddings_pca(combined, n_components=2)
                    query_point_2d = combined_2d[-1]  # Last row is query

                # Create rows for filtering
                rows = []
                for i, chunk in enumerate(chunks):
                    content = chunk.get("content", "")
                    rows.append({
                        "x": float(points_2d[i, 0]),
                        "y": float(points_2d[i, 1]),
                        "chunk_id": chunk.get("chunk_id", i),
                        "doc_id": chunk.get("doc_id", "unknown"),
                        "tag": chunk.get("tag") or "unknown",
                        "source": chunk.get("source") or "unknown",
                        "preview": content[:100] + "..." if len(content) > 100 else content,
                        "content": content,
                    })

                # Filter controls
                col1, col2 = st.columns(2)
                with col1:
                    search_text = st.text_input("Search chunks", key="chunk_search")
                with col2:
                    tags = sorted({row["tag"] for row in rows})
                    selected_tags = st.multiselect("Filter by tag", tags, key="chunk_tags")

                # Apply filters
                filtered_rows = rows
                if selected_tags:
                    filtered_rows = [row for row in filtered_rows if row["tag"] in selected_tags]

                if search_text.strip():
                    query = search_text.casefold().strip()
                    filtered_rows = [
                        row for row in filtered_rows
                        if query in str(row["content"]).casefold()
                        or query in str(row["tag"]).casefold()
                        or query in str(row["doc_id"]).casefold()
                        or query in str(row["chunk_id"]).casefold()
                    ]

                # Show 2D map with Plotly for consistency
                st.write(f"Showing `{len(filtered_rows)}` of `{len(rows)}` chunks:")
                if filtered_rows:
                    # Build Plotly figure
                    fig = go.Figure()

                    # Get unique tags
                    unique_tags = sorted({row["tag"] for row in filtered_rows})

                    # Add trace for each tag
                    for tag in unique_tags:
                        tag_rows = [row for row in filtered_rows if row["tag"] == tag]
                        fig.add_trace(go.Scatter(
                            x=[row["x"] for row in tag_rows],
                            y=[row["y"] for row in tag_rows],
                            mode='markers',
                            name=tag,
                            marker=dict(size=8, opacity=0.7),
                            hovertemplate=f'{tag}<br>chunk_id: %{{text}}<br>x: %{{x:.2f}}<br>y: %{{y:.2f}}<extra></extra>',
                            text=[row["chunk_id"] for row in tag_rows]
                        ))

                    # Add query point if available (green star, matching user's style)
                    if query_point_2d is not None:
                        fig.add_trace(go.Scatter(
                            x=[query_point_2d[0]],
                            y=[query_point_2d[1]],
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
                        title='Chunk Locations by Tag' + (' with Query' if query_point_2d is not None else ''),
                        xaxis_title='PCA Component 1',
                        yaxis_title='PCA Component 2',
                        hovermode='closest',
                        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                    )

                    st.plotly_chart(fig, use_container_width=True)

                # Show data table
                st.dataframe(
                    filtered_rows,
                    use_container_width=True,
                    column_order=["chunk_id", "doc_id", "tag", "source", "preview", "x", "y"],
                    hide_index=True,
                )

                # Chunk inspector
                st.subheader("Inspect a Chunk")
                if filtered_rows:
                    selected_chunk_id = st.selectbox(
                        "Choose chunk_id",
                        [row["chunk_id"] for row in filtered_rows],
                        key="inspect_chunk"
                    )
                    selected = next(row for row in filtered_rows if row["chunk_id"] == selected_chunk_id)
                    st.json({key: selected[key] for key in ["chunk_id", "doc_id", "tag", "source", "x", "y"]})
                    st.text_area("Content", selected["content"], height=250)
