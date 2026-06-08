import json
import logging
import os
import pickle
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List

# Suppress transformers verbose logging BEFORE any imports
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Configure logging to suppress transformers
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.configuration_utils").setLevel(logging.ERROR)

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", message=".*__path__.*", category=UserWarning)

import numpy as np
import streamlit as st
from dotenv import load_dotenv

from src.data_utils.data_loader import DataLoader
from src.data_utils.testset_utils import generate_testset, load_testset
from src.llm.embedding_api import EmbeddingAPI
from src.llm.llm_api import LLMAPI
from src.rag.vector_db.vector_db_builder import VectorDBBuilder
from src.rag.retriever.rag_retriever import RAGRetriever
from src.rag.pipeline import RAGPipeline
from src.rag.evaluation.rag_evaluator import RAGEvaluator


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
def load_pipeline():
    """Load the centralized RAG pipeline."""
    return RAGPipeline()


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
                        pipeline = load_pipeline()
                        result = pipeline.run(query)
                        
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


def evaluation_page():
    st.set_page_config(page_title="Evaluation - University Chatbot", page_icon="🧪", layout="wide")
    
    st.title("🧪 RAG Evaluation")
    st.caption("Run RAGAS evaluation and view results")
    
    # Read global configuration (from Settings page)
    data_source_path = os.getenv("DATA_SOURCE_PATH", "data/intents-v2.json")
    test_data_path = os.getenv("RAG_TEST_DATA_PATH", "data/rag_test_data.json")
    experiment_results_dir = os.getenv("RAG_EXPERIMENTS_DIR", "data/evaluation_results")
    
    # Display current global configuration (read-only from Settings)
    with st.expander("📋 Current Global Configuration", expanded=False):
        st.info("These settings are configured in the ⚙️ Settings page")
        config_cols = st.columns(4)
        config_cols[0].metric("Retriever", os.getenv("RETRIEVAL_TYPE", "vector"))
        config_cols[1].metric("Self-Eval", "Enabled" if os.getenv("ENABLE_SELF_EVAL", "false").lower() == "true" else "Disabled")
        config_cols[2].metric("Data Source", Path(data_source_path).name)
        config_cols[3].metric("Test Data", Path(test_data_path).name)
    
    # Only experiment name is configurable per-run
    st.subheader("Experiment Configuration")
    experiment_name = st.text_input(
        "Experiment Name",
        value=os.getenv("RAG_EXPERIMENT_NAME", "baseline"),
        help="Unique name for this evaluation run (results will be saved with this name)"
    )
    
    # Run evaluation button
    st.divider()
    
    if st.button("🚀 Run Evaluation", type="primary", use_container_width=True):
        # Set only experiment name for this run
        os.environ["RAG_EXPERIMENT_NAME"] = experiment_name
        
        # Create progress containers
        progress_container = st.container()
        with progress_container:
            status_text = st.empty()
            progress_bar = st.progress(0)
            log_output = st.empty()
            
        try:
            status_text.text("Initializing evaluator...")
            progress_bar.progress(10)
            
            # Initialize evaluator with centralized pipeline
            evaluator = RAGEvaluator(
                data_source_path=data_source_path,
                test_data_path=test_data_path,
                experiment_results_dir=experiment_results_dir
            )
            
            # Load or generate test samples
            test_path = Path(test_data_path)
            if test_path.exists():
                status_text.text("Loading test samples...")
                test_samples = load_testset(test_path)
                progress_bar.progress(20)
                st.success(f"✅ Loaded {len(test_samples)} test samples from {test_data_path}")
            else:
                status_text.text("Generating test samples...")
                test_samples = generate_testset(
                    data_source_path,
                    test_set_ratio=0.2,
                    test_data_path=test_path,
                    refined_data_path=None
                )
                progress_bar.progress(30)
                st.success(f"✅ Generated {len(test_samples)} test samples")
            
            # Run evaluation
            status_text.text(f"Running evaluation with {len(test_samples)} samples...")
            progress_bar.progress(40)
            
            results = evaluator.run_evaluation(test_samples, experiment_name=experiment_name)
            
            progress_bar.progress(100)
            status_text.text("Evaluation complete!")
            
            # Display results
            st.divider()
            st.subheader("📊 Evaluation Results")
            
            if "error" in results:
                st.error(f"Error: {results['error']}")
            else:
                # Summary metrics
                summary = results.get("summary", {})
                scores = summary.get("scores", {})
                
                st.success(f"✅ Evaluation completed: {summary.get('num_successful_samples', 0)} successful, {summary.get('num_failed_samples', 0)} failed")
                
                # Main metrics
                st.subheader("Overall Metrics")
                metric_cols = st.columns(4)
                metrics_to_show = [
                    ("faithfulness", "Faithfulness"),
                    ("answer_relevancy", "Answer Relevancy"),
                    ("context_precision", "Context Precision"),
                    ("context_recall", "Context Recall")
                ]
                
                for i, (key, label) in enumerate(metrics_to_show):
                    if key in scores:
                        value = scores[key]
                        metric_cols[i].metric(label, f"{value:.3f}" if isinstance(value, (int, float)) else "N/A")
                
                # Breakdown by exists_in_source
                if "by_exists_in_source" in scores:
                    st.subheader("Metrics by Question Type")
                    breakdown = scores["by_exists_in_source"]
                    
                    breakdown_cols = st.columns(2)
                    
                    with breakdown_cols[0]:
                        st.write("**In Source (Knowledge Base)**")
                        in_source = breakdown.get("in_source", {})
                        st.write(f"Count: {in_source.get('count', 0)}")
                        for key, label in metrics_to_show:
                            if key in in_source:
                                st.write(f"{label}: {in_source[key]:.3f}")
                    
                    with breakdown_cols[1]:
                        st.write("**Not In Source (Out-of-Domain)**")
                        not_in_source = breakdown.get("not_in_source", {})
                        st.write(f"Count: {not_in_source.get('count', 0)}")
                        for key, label in metrics_to_show:
                            if key in not_in_source:
                                st.write(f"{label}: {not_in_source[key]:.3f}")
                
                # Show file paths
                st.subheader("📁 Result Files")
                paths = evaluator._experiment_paths(experiment_name)
                st.write(f"- Summary: `{paths['summary']}`")
                st.write(f"- Samples (JSON): `{paths['samples_json']}`")
                st.write(f"- Samples (CSV): `{paths['samples_csv']}`")
                st.write(f"- Failures: `{paths['failures']}`")
                
                # Option to view samples
                if st.checkbox("View Sample Results"):
                    samples_path = paths['samples_json']
                    if samples_path.exists():
                        with open(samples_path, 'r') as f:
                            samples = json.load(f)
                        st.json(samples[:3])  # Show first 3 samples
                    
        except Exception as e:
            st.error(f"❌ Error during evaluation: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    
    # Previous results section
    st.divider()
    st.subheader("📂 Previous Results")
    
    results_dir = Path(experiment_results_dir)
    if results_dir.exists():
        # Find all experiment result files
        summary_files = list(results_dir.glob("*_summary.json"))
        if summary_files:
            selected_file = st.selectbox(
                "Select experiment to view",
                options=[f.stem.replace("_summary", "") for f in summary_files],
                format_func=lambda x: f"📊 {x}"
            )
            
            if selected_file:
                summary_path = results_dir / f"{selected_file}_summary.json"
                if summary_path.exists():
                    with open(summary_path, 'r') as f:
                        prev_results = json.load(f)
                    
                    st.json(prev_results)
        else:
            st.info("No previous evaluation results found.")
    else:
        st.info(f"Results directory does not exist: {results_dir}")


def settings_page():
    st.set_page_config(page_title="Settings - University Chatbot", page_icon="⚙️", layout="wide")
    
    st.title("⚙️ Global Settings")
    st.caption("Configure environment variables used by Chat and Evaluation")
    
    # Load current .env values
    env_path = Path(".env")
    current_env = {}
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    current_env[key] = value
    
    # API Keys Section
    with st.expander("🔑 API Keys", expanded=False):
        st.warning("API keys are sensitive. They will be saved to .env file.")
        
        groq_key = st.text_input(
            "GROQ_API_KEY",
            value=current_env.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", "")),
            type="password",
            help="Your Groq API key for LLM access"
        )
        
        hf_key = st.text_input(
            "HUGGINGFACE_API_KEY",
            value=current_env.get("HUGGINGFACE_API_KEY", os.getenv("HUGGINGFACE_API_KEY", "")),
            type="password",
            help="Your HuggingFace API key for embeddings"
        )
    
    # Model Settings
    with st.expander("🤖 Model Settings", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            llm_model = st.text_input(
                "LLM_MODEL",
                value=current_env.get("LLM_MODEL", os.getenv("LLM_MODEL", "llama-3.1-8b-instant")),
                help="Groq model name for response generation"
            )
        
        with col2:
            embedding_model = st.text_input(
                "EMBEDDING_MODEL",
                value=current_env.get("EMBEDDING_MODEL", os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")),
                help="HuggingFace model for embeddings"
            )
    
    # Paths Configuration
    with st.expander("📁 Paths", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            data_source = st.text_input(
                "DATA_SOURCE_PATH",
                value=current_env.get("DATA_SOURCE_PATH", os.getenv("DATA_SOURCE_PATH", "data/intents-v2.json")),
                help="Path to intents data file"
            )
            
            vdb_path = st.text_input(
                "VDB_SAVE_PATH",
                value=current_env.get("VDB_SAVE_PATH", os.getenv("VDB_SAVE_PATH", "data/vector_db")),
                help="Directory for vector database files"
            )
        
        with col2:
            test_data = st.text_input(
                "RAG_TEST_DATA_PATH",
                value=current_env.get("RAG_TEST_DATA_PATH", os.getenv("RAG_TEST_DATA_PATH", "data/rag_test_data.json")),
                help="Path to test data file"
            )
            
            results_dir = st.text_input(
                "RAG_EXPERIMENTS_DIR",
                value=current_env.get("RAG_EXPERIMENTS_DIR", os.getenv("RAG_EXPERIMENTS_DIR", "data/evaluation_results")),
                help="Directory for evaluation results"
            )
    
    # Retrieval Settings
    with st.expander("🔍 Retrieval Settings", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            retrieval_type = st.selectbox(
                "RETRIEVAL_TYPE",
                options=["vector", "bm25", "hybrid"],
                index=["vector", "bm25", "hybrid"].index(current_env.get("RETRIEVAL_TYPE", os.getenv("RETRIEVAL_TYPE", "vector"))),
                help="Default retrieval method"
            )
            
            bm25_weight = st.slider(
                "BM25_WEIGHT",
                min_value=0.0,
                max_value=1.0,
                value=float(current_env.get("BM25_WEIGHT", os.getenv("BM25_WEIGHT", "0.5"))),
                step=0.1,
                help="Weight for BM25 in hybrid mode (0=vector only, 1=BM25 only)"
            )
        
        with col2:
            chunk_size = st.number_input(
                "CHUNK_SIZE",
                min_value=100,
                max_value=2000,
                value=int(current_env.get("CHUNK_SIZE", os.getenv("CHUNK_SIZE", "500"))),
                step=50,
                help="Size of text chunks for vector DB"
            )
            
            chunk_overlap = st.number_input(
                "CHUNK_OVERLAP",
                min_value=0,
                max_value=500,
                value=int(current_env.get("CHUNK_OVERLAP", os.getenv("CHUNK_OVERLAP", "0"))),
                step=10,
                help="Overlap between chunks"
            )
    
    # Self-Evaluation Settings
    with st.expander("🧠 Self-Evaluation (Self-RAG)", expanded=True):
        enable_self_eval = st.checkbox(
            "ENABLE_SELF_EVAL",
            value=current_env.get("ENABLE_SELF_EVAL", os.getenv("ENABLE_SELF_EVAL", "false")).lower() == "true",
            help="Enable self-evaluation with confidence scoring"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            relevance_threshold = st.slider(
                "RELEVANCE_THRESHOLD",
                min_value=0.0,
                max_value=1.0,
                value=float(current_env.get("RELEVANCE_THRESHOLD", os.getenv("RELEVANCE_THRESHOLD", "0.3"))),
                step=0.05,
                help="Minimum similarity score to consider a chunk relevant",
                disabled=not enable_self_eval
            )
        
        with col2:
            confidence_threshold = st.slider(
                "CONFIDENCE_THRESHOLD",
                min_value=0.0,
                max_value=1.0,
                value=float(current_env.get("CONFIDENCE_THRESHOLD", os.getenv("CONFIDENCE_THRESHOLD", "0.4"))),
                step=0.05,
                help="Minimum confidence to use context (below this triggers fallback)",
                disabled=not enable_self_eval
            )
    
    # Query Fusion Settings (RAG-Fusion)
    with st.expander("🔀 Query Fusion (RAG-Fusion)", expanded=True):
        enable_query_fusion = st.checkbox(
            "ENABLE_QUERY_FUSION",
            value=current_env.get("ENABLE_QUERY_FUSION", os.getenv("ENABLE_QUERY_FUSION", "false")).lower() == "true",
            help="Enable RAG-Fusion with query variants and reciprocal rank fusion"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            fusion_num_variants = st.number_input(
                "FUSION_NUM_VARIANTS",
                min_value=1,
                max_value=10,
                value=int(current_env.get("FUSION_NUM_VARIANTS", os.getenv("FUSION_NUM_VARIANTS", "3"))),
                step=1,
                help="Number of query variations to generate",
                disabled=not enable_query_fusion
            )
            
            fusion_top_k = st.number_input(
                "FUSION_TOP_K",
                min_value=1,
                max_value=20,
                value=int(current_env.get("FUSION_TOP_K", os.getenv("FUSION_TOP_K", "5"))),
                step=1,
                help="Number of chunks to return after fusion",
                disabled=not enable_query_fusion
            )
        
        with col2:
            fusion_k = st.number_input(
                "FUSION_K",
                min_value=1,
                max_value=100,
                value=int(current_env.get("FUSION_K", os.getenv("FUSION_K", "60"))),
                step=5,
                help="RRF constant (default 60, higher = less rank discrimination)",
                disabled=not enable_query_fusion
            )
    
    # Document Structure Mode
    with st.expander("📄 Document Processing", expanded=True):
        doc_mode = st.selectbox(
            "DOCUMENT_STRUCTURE_MODE",
            options=["structural", "structural-formatted", "non_structural"],
            index=["structural", "structural-formatted", "non_structural"].index(
                current_env.get("DOCUMENT_STRUCTURE_MODE", os.getenv("DOCUMENT_STRUCTURE_MODE", "structural"))
            ),
            help="How to structure documents during chunking"
        )
    
    
    
    if st.button("💾 Save Settings to .env", type="primary", use_container_width=True):
        try:
            # Build new .env content
            env_lines = [
                f"GROQ_API_KEY={groq_key}",
                f"HUGGINGFACE_API_KEY={hf_key}",
                f"DATA_SOURCE_PATH={data_source}",
                f"VDB_SAVE_PATH={vdb_path}",
                f"LLM_MODEL={llm_model}",
                f"EMBEDDING_MODEL={embedding_model}",
                f"RAG_TEST_DATA_PATH={test_data}",
                f"CHUNK_SIZE={chunk_size}",
                f"CHUNK_OVERLAP={chunk_overlap}",
                f"BM25_WEIGHT={bm25_weight}",
                f"DOCUMENT_STRUCTURE_MODE={doc_mode}",
                f"RETRIEVAL_TYPE={retrieval_type}",
                f"ENABLE_SELF_EVAL={str(enable_self_eval).lower()}",
                f"RELEVANCE_THRESHOLD={relevance_threshold}",
                f"CONFIDENCE_THRESHOLD={confidence_threshold}",
                f"ENABLE_QUERY_FUSION={str(enable_query_fusion).lower()}",
                f"FUSION_NUM_VARIANTS={fusion_num_variants}",
                f"FUSION_K={fusion_k}",
                f"FUSION_TOP_K={fusion_top_k}",
                f"RAG_EXPERIMENTS_DIR={results_dir}",
            ]
            
            # Write to .env file
            with open(env_path, 'w') as f:
                f.write('\n'.join(env_lines) + '\n')
            
            # Update current environment
            os.environ["GROQ_API_KEY"] = groq_key
            os.environ["HUGGINGFACE_API_KEY"] = hf_key
            os.environ["DATA_SOURCE_PATH"] = data_source
            os.environ["VDB_SAVE_PATH"] = vdb_path
            os.environ["LLM_MODEL"] = llm_model
            os.environ["EMBEDDING_MODEL"] = embedding_model
            os.environ["RAG_TEST_DATA_PATH"] = test_data
            os.environ["CHUNK_SIZE"] = str(chunk_size)
            os.environ["CHUNK_OVERLAP"] = str(chunk_overlap)
            os.environ["BM25_WEIGHT"] = str(bm25_weight)
            os.environ["DOCUMENT_STRUCTURE_MODE"] = doc_mode
            os.environ["RETRIEVAL_TYPE"] = retrieval_type
            os.environ["ENABLE_SELF_EVAL"] = str(enable_self_eval).lower()
            os.environ["RELEVANCE_THRESHOLD"] = str(relevance_threshold)
            os.environ["CONFIDENCE_THRESHOLD"] = str(confidence_threshold)
            os.environ["ENABLE_QUERY_FUSION"] = str(enable_query_fusion).lower()
            os.environ["FUSION_NUM_VARIANTS"] = str(fusion_num_variants)
            os.environ["FUSION_K"] = str(fusion_k)
            os.environ["FUSION_TOP_K"] = str(fusion_top_k)
            os.environ["RAG_EXPERIMENTS_DIR"] = results_dir
            
            st.success("✅ Settings saved to .env and applied!")
            st.info("🔄 Refresh the page to ensure all components pick up the new settings.")
            
        except Exception as e:
            st.error(f"❌ Error saving settings: {str(e)}")


def main():
    pg = st.navigation([
        st.Page(chat_page, title="Chat", icon="💬"),
        st.Page(dashboard_page, title="Dashboard", icon="📊"),
        st.Page(evaluation_page, title="Evaluation", icon="🧪"),
        st.Page(settings_page, title="Settings", icon="⚙️"),
    ])
    pg.run()


if __name__ == "__main__":
    main()
