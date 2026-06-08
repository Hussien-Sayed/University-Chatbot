import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import streamlit as st
from dotenv import load_dotenv

from src.data_utils.data_loader import DataLoader
from src.llm.embedding_api import EmbeddingAPI
from src.llm.llm_api import LLMAPI
from src.rag.vector_db.vector_db_builder import VectorDBBuilder
from src.rag.retriever.rag_retriever import RAGRetriever


load_dotenv()


@st.cache_resource
def build_vector_db():
    data_loader = DataLoader()
    embedding_api = EmbeddingAPI()

    builder = VectorDBBuilder(
        data_loader=data_loader,
        embedding_api=embedding_api,
        vdb_save_path=os.getenv("VDB_SAVE_PATH", "data/vector_db"),
    )

    result = builder.build_vector_db()
    return result


@st.cache_resource
def load_retriever():
    llm_api = LLMAPI()
    retriever = RAGRetriever(
        vector_db_path=os.getenv("VDB_SAVE_PATH", "data/vector_db"),
        llm_api=llm_api,
        num_chunks=3
    )
    return retriever


@st.cache_resource
def load_vector_db():
    vector_db_path = Path(os.getenv("VDB_SAVE_PATH", "data/vector_db")) / "vector_db.pkl"
    if not vector_db_path.exists():
        return None
    with open(vector_db_path, "rb") as f:
        return pickle.load(f)


def reduce_embeddings_pca(embeddings: np.ndarray) -> np.ndarray:
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


def chat_page():
    st.set_page_config(page_title="Chat - University Chatbot", page_icon="💬", layout="centered")

    st.title("💬 Chat")
    st.caption("Ask me anything about the university!")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "vector_db_built" not in st.session_state:
        st.session_state.vector_db_built = False

    with st.sidebar:
        st.header("Settings")

        if st.button("Build/Rebuild Vector DB"):
            with st.spinner("Building vector database..."):
                try:
                    result = build_vector_db()
                    st.session_state.vector_db_built = True
                    st.success(f"Vector DB built! {result['num_chunks']} chunks created.")
                except Exception as e:
                    st.error(f"Error building vector DB: {str(e)}")

        st.divider()
        st.info("Type your question in the chat below!")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "retrieved_chunks" in message:
                with st.expander("📄 Retrieved Chunks"):
                    for i, chunk in enumerate(message["retrieved_chunks"], 1):
                        st.markdown(f"**Chunk {i}** (Source: `{chunk.get('source', 'unknown')}`)")
                        st.text(chunk.get('content', ''))
                        st.divider()
            if "query_time" in message:
                st.caption(f"⏱️ Query time: {message['query_time']:.2f} seconds")

    if query := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            if not st.session_state.vector_db_built:
                st.warning("Please build the vector database first using the sidebar button.")
            else:
                with st.spinner("Thinking..."):
                    try:
                        retriever = load_retriever()
                        embedding_api = EmbeddingAPI()

                        query_start = time.time()
                        query_embedding = embedding_api.generate_embedding(query)

                        # Use new self-evaluation method
                        result = retriever.generate_response_with_self_eval(query, query_embedding)
                        response = result['response']
                        eval_metadata = result.get('evaluation', {})

                        query_time = time.time() - query_start

                        st.markdown(response)

                        # Show self-evaluation info if enabled
                        if eval_metadata.get('self_eval_enabled'):
                            with st.expander("🧠 Self-Evaluation Details"):
                                st.write(f"**Retrieved:** {eval_metadata.get('chunks_retrieved', 0)} chunks")
                                st.write(f"**After score filter:** {eval_metadata.get('chunks_after_score_filter', 0)} chunks")
                                st.write(f"**Chunks used:** {eval_metadata.get('chunks_used', 0)} chunks")
                                if 'avg_llm_relevance' in eval_metadata:
                                    st.write(f"**Avg LLM relevance:** {eval_metadata['avg_llm_relevance']:.2f}")
                                if 'response_confidence' in eval_metadata:
                                    confidence = eval_metadata['response_confidence']
                                    color = "🟢" if confidence >= 0.7 else "🟡" if confidence >= 0.4 else "🔴"
                                    st.write(f"**Response confidence:** {color} {confidence:.2f}")
                                if eval_metadata.get('fallback_triggered'):
                                    st.warning("⚠️ Low confidence - fallback response triggered")
                                if not eval_metadata.get('used_context'):
                                    st.info("ℹ️ No relevant context found - answered without retrieval")

                        with st.expander("📄 Retrieved Chunks"):
                            # Re-retrieve to show chunks (if self-eval filtered them)
                            retrieved_chunks = retriever.retrieve_chunks(query, query_embedding)
                            retrieval_type = retriever.retriever_type
                            for i, chunk in enumerate(retrieved_chunks, 1):
                                chunk_data = chunk.get('chunk', {})
                                st.markdown(f"**Chunk {i}** (Source: `{chunk_data.get('source', 'unknown')}`)")

                                # Display scores based on retrieval type
                                if retrieval_type == "vector":
                                    st.caption(f"Vector Similarity: {chunk.get('vector_score', 0):.4f}")
                                elif retrieval_type == "bm25":
                                    st.caption(f"BM25 Score: {chunk.get('bm25_score', 0):.4f}")
                                elif retrieval_type == "hybrid":
                                    st.caption(f"Combined Score: {chunk.get('similarity_score', 0):.4f}")
                                    st.caption(f"  Vector: {chunk.get('vector_score', 0):.4f} | BM25: {chunk.get('bm25_score', 0):.4f}")

                                st.text(chunk_data.get('content', ''))
                                st.divider()

                        st.caption(f"⏱️ Query time: {query_time:.2f} seconds")

                        message_data = {
                            "role": "assistant",
                            "content": response,
                            "retrieved_chunks": [item.get('chunk', {}) for item in retrieved_chunks],
                            "query_time": query_time,
                            "evaluation": eval_metadata
                        }
                        st.session_state.messages.append(message_data)
                    except Exception as e:
                        error_msg = f"Sorry, I encountered an error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})


def dashboard_page():
    st.set_page_config(page_title="Dashboard - University Chatbot", page_icon="📊", layout="wide")

    st.title("📊 Dashboard")
    st.caption("Vector DB visualization and analytics")

    db_data = load_vector_db()

    if db_data is None:
        st.error("Vector database not found. Please build it first from the Chat page.")
        return

    chunks = db_data.get("chunks", [])
    embeddings = np.array(db_data.get("embeddings", []), dtype=np.float32)
    metadata = db_data.get("metadata", {})

    if not chunks or embeddings.size == 0:
        st.error("No chunks or embeddings found in the vector database.")
        return

    if len(chunks) != embeddings.shape[0]:
        st.error(f"Chunk count ({len(chunks)}) does not match embedding count ({embeddings.shape[0]}).")
        return

    points = reduce_embeddings_pca(embeddings)

    rows = []
    for index, chunk in enumerate(chunks):
        content = chunk.get("content", "")
        rows.append({
            "x": float(points[index, 0]),
            "y": float(points[index, 1]),
            "chunk_id": chunk.get("chunk_id"),
            "doc_id": chunk.get("doc_id"),
            "tag": chunk.get("tag") or "unknown",
            "source": chunk.get("source") or "unknown",
            "preview": content[:250],
            "content": content,
        })

    tags = sorted({row["tag"] for row in rows})

    with st.sidebar:
        st.header("Vector DB Info")
        st.write(f"Chunks: `{len(chunks)}`")
        st.write(f"Embedding shape: `{embeddings.shape}`")
        st.write(f"Document structure mode: `{metadata.get('document_structure_mode', 'unknown')}`")

        if metadata:
            st.subheader("Metadata")
            st.json(metadata)

        st.subheader("Filters")
        search_text = st.text_input("Search chunks")
        selected_tags = st.multiselect("Filter by tag", tags)

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

    st.subheader("2D PCA Map")
    st.write(f"Showing `{len(filtered_rows)}` of `{len(rows)}` chunks.")

    if filtered_rows:
        chart_rows = [
            {
                "x": row["x"],
                "y": row["y"],
                "tag": row["tag"],
                "chunk_id": row["chunk_id"],
            }
            for row in filtered_rows
        ]
        st.scatter_chart(chart_rows, x="x", y="y", color="tag", size=80)
    else:
        st.warning("No chunks match the selected filters.")

    st.subheader("Chunks")
    st.dataframe(
        filtered_rows,
        use_container_width=True,
        column_order=["chunk_id", "doc_id", "tag", "source", "preview", "x", "y"],
        hide_index=True,
    )

    st.subheader("Inspect a Chunk")
    if filtered_rows:
        selected_chunk_id = st.selectbox(
            "Choose chunk_id",
            [row["chunk_id"] for row in filtered_rows],
        )
        selected = next(row for row in filtered_rows if row["chunk_id"] == selected_chunk_id)
        st.json({key: selected[key] for key in ["chunk_id", "doc_id", "tag", "source", "x", "y"]})
        st.text_area("Content", selected["content"], height=250)


def main():
    pg = st.navigation([
        st.Page(chat_page, title="Chat", icon="💬"),
        st.Page(dashboard_page, title="Dashboard", icon="📊"),
    ])
    pg.run()


if __name__ == "__main__":
    main()
