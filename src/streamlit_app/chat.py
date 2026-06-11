"""Chat page for the Streamlit app."""
import os
from pathlib import Path

import streamlit as st

from ..llm.embedding_api import EmbeddingAPI
from ..rag.pipeline import RAGPipeline
from .utils import build_vector_db


def chat_page():
    """Render the chat page."""
    st.set_page_config(page_title="Chat - University Chatbot", page_icon="💬", layout="centered")

    st.title("💬 Chat")
    st.caption("Ask me anything about the university!")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Check if vector DB exists on disk
    vdb_path = Path(os.getenv("VDB_SAVE_PATH", "data/vector_db")) / "vector_db.pkl"
    vdb_exists = vdb_path.exists()

    if "vector_db_built" not in st.session_state:
        # Auto-set to True if VDB exists, otherwise False
        st.session_state.vector_db_built = vdb_exists

    with st.sidebar:
        st.header("Settings")

        if vdb_exists and st.session_state.vector_db_built:
            st.success("Vector DB loaded ✓")

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
            if "fusion" in message and message["fusion"].get('enabled') and message["fusion"].get('query_variants'):
                with st.expander("🔀 Query Fusion Variants"):
                    fusion = message["fusion"]
                    st.write(f"**RRF k={fusion.get('k', 60)}, Top {fusion.get('top_k', 5)} chunks**")
                    for i, variant in enumerate(fusion['query_variants'], 1):
                        st.write(f"{i}. {variant}")
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
                        # Use centralized RAG pipeline
                        pipeline = RAGPipeline()
                        result = pipeline.run(query)

                        # Generate query embedding for dashboard visualization
                        embedding_api = EmbeddingAPI()
                        query_embedding = embedding_api.generate_embedding(query)

                        # Store query data in session state for dashboard
                        st.session_state.last_query_result = {
                            "query": query,
                            "query_embedding": query_embedding,
                            "retrieved_chunks": result['retrieved_chunks'],
                            "response": result['response']
                        }

                        response = result['response']
                        self_eval = result['self_eval']
                        fusion = result.get('fusion', {})
                        retrieved_chunks = result['retrieved_chunks']
                        query_time = result['query_time_seconds']

                        st.markdown(response)

                        # Show self-evaluation info if enabled
                        if self_eval.get('enabled'):
                            with st.expander("🧠 Self-Evaluation Details"):
                                st.write(f"**Retrieved:** {self_eval.get('chunks_retrieved', 0)} chunks")
                                st.write(f"**After score filter:** {self_eval.get('chunks_after_score_filter', 0)} chunks")
                                st.write(f"**Chunks used:** {self_eval.get('chunks_used', 0)} chunks")
                                if self_eval.get('avg_relevance') is not None:
                                    st.write(f"**Avg LLM relevance:** {self_eval['avg_relevance']:.2f}")
                                if self_eval.get('confidence') is not None:
                                    confidence = self_eval['confidence']
                                    color = "🟢" if confidence >= 0.7 else "🟡" if confidence >= 0.4 else "🔴"
                                    st.write(f"**Response confidence:** {color} {confidence:.2f}")
                                if self_eval.get('fallback_triggered'):
                                    st.warning("⚠️ Low confidence - fallback response triggered")
                                if not self_eval.get('used_context'):
                                    st.info("ℹ️ No relevant context found - answered without retrieval")

                        # Show API call counts
                        api_calls = result.get('api_calls', {})
                        with st.expander("📊 API Call Statistics"):
                            st.write(f"**LLM API calls:** {api_calls.get('llm_calls', 0)}")
                            st.write(f"**Embedding API calls:** {api_calls.get('embedding_calls', 0)}")

                        # Show fusion query variants if enabled
                        if fusion.get('enabled') and fusion.get('query_variants'):
                            with st.expander("🔀 Query Fusion Variants"):
                                st.write(f"**RRF k={fusion.get('k', 60)}, Top {fusion.get('top_k', 5)} chunks**")
                                for i, variant in enumerate(fusion['query_variants'], 1):
                                    st.write(f"{i}. {variant}")

                        with st.expander("📄 Retrieved Chunks"):
                            retrieval_type = result['retriever_type']
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
                            "evaluation": self_eval,
                            "fusion": fusion
                        }
                        st.session_state.messages.append(message_data)
                    except Exception as e:
                        error_msg = f"Sorry, I encountered an error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
