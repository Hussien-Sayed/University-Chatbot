"""Settings page for the Streamlit app."""
import os
from pathlib import Path

import streamlit as st


def settings_page():
    """Render the settings page."""
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
                    value = value.strip().strip('"\'')  # Strip whitespace and quotes
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
            llm_provider = st.selectbox(
                "LLM_PROVIDER",
                options=["groq", "ollama"],
                index=["groq", "ollama"].index(current_env.get("LLM_PROVIDER", os.getenv("LLM_PROVIDER", "groq"))),
                help="LLM provider to use for response generation"
            )

            llm_model = st.text_input(
                "LLM_MODEL",
                value=current_env.get("LLM_MODEL", os.getenv("LLM_MODEL", "llama-3.1-8b-instant")),
                help="Model name for response generation (provider-specific)"
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

            num_chunks = st.number_input(
                "NUM_CHUNKS",
                min_value=1,
                max_value=20,
                value=int(current_env.get("NUM_CHUNKS", os.getenv("NUM_CHUNKS", "3"))),
                step=1,
                help="Number of chunks to retrieve per query"
            )

    # FAISS Index Settings (only applies to vector/hybrid retrieval)
    with st.expander("🗂️  FAISS Index Settings", expanded=True):
        faiss_index_type = st.selectbox(
            "FAISS_INDEX_TYPE",
            options=["flat_ip", "ivf_flat", "hnsw", "pq"],
            index=["flat_ip", "ivf_flat", "hnsw", "pq"].index(
                current_env.get("FAISS_INDEX_TYPE", os.getenv("FAISS_INDEX_TYPE", "flat_ip"))
            ),
            help="FAISS index type (only used when RETRIEVAL_TYPE=vector or hybrid). Requires rebuild to change."
        )

        # Show description of selected index type
        index_descriptions = {
            "flat_ip": "Exact search, 100% accuracy. Best for small datasets (<10K chunks).",
            "ivf_flat": "Clustering-based, ~95% accuracy. Faster for medium datasets (10K-1M).",
            "hnsw": "Graph-based, ~98% accuracy. Fastest general purpose option.",
            "pq": "Product quantization, ~90% accuracy. Memory-efficient for large datasets (>1M)."
        }
        st.caption(f"**{faiss_index_type.upper()}**: {index_descriptions.get(faiss_index_type, '')}")

        # IVF parameters
        if faiss_index_type == "ivf_flat":
            col1, col2 = st.columns(2)
            with col1:
                faiss_ivf_nlist = st.number_input(
                    "FAISS_IVF_NLIST",
                    min_value=1,
                    max_value=10000,
                    value=int(current_env.get("FAISS_IVF_NLIST", os.getenv("FAISS_IVF_NLIST", "100"))),
                    step=10,
                    help="Number of clusters (lower=faster, higher=more accurate)"
                )
            with col2:
                faiss_ivf_nprobe = st.number_input(
                    "FAISS_IVF_NPROBE",
                    min_value=1,
                    max_value=1000,
                    value=int(current_env.get("FAISS_IVF_NPROBE", os.getenv("FAISS_IVF_NPROBE", "10"))),
                    step=1,
                    help="Clusters to search at query time (higher=more accurate, slower)"
                )
        else:
            faiss_ivf_nlist = int(current_env.get("FAISS_IVF_NLIST", os.getenv("FAISS_IVF_NLIST", "100")))
            faiss_ivf_nprobe = int(current_env.get("FAISS_IVF_NPROBE", os.getenv("FAISS_IVF_NPROBE", "10")))

        # HNSW parameters
        if faiss_index_type == "hnsw":
            col1, col2, col3 = st.columns(3)
            with col1:
                faiss_hnsw_m = st.number_input(
                    "FAISS_HNSW_M",
                    min_value=4,
                    max_value=64,
                    value=int(current_env.get("FAISS_HNSW_M", os.getenv("FAISS_HNSW_M", "16"))),
                    step=2,
                    help="Connections per node (higher=more accurate, more memory)"
                )
            with col2:
                faiss_hnsw_ef_construction = st.number_input(
                    "FAISS_HNSW_EF_CONSTRUCTION",
                    min_value=10,
                    max_value=1000,
                    value=int(current_env.get("FAISS_HNSW_EF_CONSTRUCTION", os.getenv("FAISS_HNSW_EF_CONSTRUCTION", "200"))),
                    step=10,
                    help="Build-time search depth (higher=better index, slower build)"
                )
            with col3:
                faiss_hnsw_ef_search = st.number_input(
                    "FAISS_HNSW_EF_SEARCH",
                    min_value=10,
                    max_value=1000,
                    value=int(current_env.get("FAISS_HNSW_EF_SEARCH", os.getenv("FAISS_HNSW_EF_SEARCH", "128"))),
                    step=10,
                    help="Query-time search depth (higher=more accurate, slower)"
                )
        else:
            faiss_hnsw_m = int(current_env.get("FAISS_HNSW_M", os.getenv("FAISS_HNSW_M", "16")))
            faiss_hnsw_ef_construction = int(current_env.get("FAISS_HNSW_EF_CONSTRUCTION", os.getenv("FAISS_HNSW_EF_CONSTRUCTION", "200")))
            faiss_hnsw_ef_search = int(current_env.get("FAISS_HNSW_EF_SEARCH", os.getenv("FAISS_HNSW_EF_SEARCH", "128")))

        # PQ parameters
        if faiss_index_type == "pq":
            col1, col2 = st.columns(2)
            with col1:
                faiss_pq_m = st.number_input(
                    "FAISS_PQ_M",
                    min_value=1,
                    max_value=64,
                    value=int(current_env.get("FAISS_PQ_M", os.getenv("FAISS_PQ_M", "8"))),
                    step=1,
                    help="Subquantizers (embedding dim must be divisible by this)"
                )
            with col2:
                faiss_pq_nbits = st.selectbox(
                    "FAISS_PQ_NBITS",
                    options=[4, 8, 16],
                    index=[4, 8, 16].index(int(current_env.get("FAISS_PQ_NBITS", os.getenv("FAISS_PQ_NBITS", "8")))),
                    help="Bits per code (higher=more accurate, more memory). Use 4 for small datasets (<100 chunks)."
                )
        else:
            faiss_pq_m = int(current_env.get("FAISS_PQ_M", os.getenv("FAISS_PQ_M", "8")))
            faiss_pq_nbits = int(current_env.get("FAISS_PQ_NBITS", os.getenv("FAISS_PQ_NBITS", "4")))

        st.info("⚠️ Changing FAISS index type requires rebuilding the vector database.")

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
            # Map of keys to new values
            new_values = {
                "GROQ_API_KEY": groq_key,
                "HUGGINGFACE_API_KEY": hf_key,
                "DATA_SOURCE_PATH": data_source,
                "VDB_SAVE_PATH": vdb_path,
                "LLM_PROVIDER": llm_provider,
                "LLM_MODEL": llm_model,
                "EMBEDDING_MODEL": embedding_model,
                "RAG_TEST_DATA_PATH": test_data,
                "CHUNK_SIZE": str(chunk_size),
                "CHUNK_OVERLAP": str(chunk_overlap),
                "NUM_CHUNKS": str(num_chunks),
                "BM25_WEIGHT": str(bm25_weight),
                "DOCUMENT_STRUCTURE_MODE": doc_mode,
                "RETRIEVAL_TYPE": retrieval_type,
                "ENABLE_SELF_EVAL": str(enable_self_eval).lower(),
                "RELEVANCE_THRESHOLD": str(relevance_threshold),
                "CONFIDENCE_THRESHOLD": str(confidence_threshold),
                "ENABLE_QUERY_FUSION": str(enable_query_fusion).lower(),
                "FUSION_NUM_VARIANTS": str(fusion_num_variants),
                "FUSION_K": str(fusion_k),
                "FUSION_TOP_K": str(fusion_top_k),
                "RAG_EXPERIMENTS_DIR": results_dir,
                # FAISS index settings
                "FAISS_INDEX_TYPE": faiss_index_type,
                "FAISS_IVF_NLIST": str(faiss_ivf_nlist),
                "FAISS_IVF_NPROBE": str(faiss_ivf_nprobe),
                "FAISS_HNSW_M": str(faiss_hnsw_m),
                "FAISS_HNSW_EF_CONSTRUCTION": str(faiss_hnsw_ef_construction),
                "FAISS_HNSW_EF_SEARCH": str(faiss_hnsw_ef_search),
                "FAISS_PQ_M": str(faiss_pq_m),
                "FAISS_PQ_NBITS": str(faiss_pq_nbits),
            }

            # Read existing file and update in-place
            if env_path.exists():
                with open(env_path, 'r') as f:
                    lines = f.readlines()

                updated_keys = set()
                for i, line in enumerate(lines):
                    stripped = line.strip()
                    # Skip comments and empty lines
                    if not stripped or stripped.startswith('#'):
                        continue
                    # Check if this is a KEY=VALUE line we need to update
                    if '=' in stripped:
                        key = stripped.split('=', 1)[0]
                        if key in new_values:
                            # Preserve indentation, replace value
                            prefix = line[:line.index(key)]
                            lines[i] = f"{prefix}{key}={new_values[key]}\n"
                            updated_keys.add(key)

                # Add any new keys that weren't in the file
                new_keys = set(new_values.keys()) - updated_keys
                if new_keys:
                    lines.append("\n# Added by Settings page\n")
                    for key in sorted(new_keys):
                        lines.append(f"{key}={new_values[key]}\n")

                # Write back preserving structure
                with open(env_path, 'w') as f:
                    f.writelines(lines)
            else:
                # Create new .env file if it doesn't exist
                with open(env_path, 'w') as f:
                    for key, value in new_values.items():
                        f.write(f"{key}={value}\n")

            # Update current environment
            os.environ["GROQ_API_KEY"] = groq_key
            os.environ["HUGGINGFACE_API_KEY"] = hf_key
            os.environ["DATA_SOURCE_PATH"] = data_source
            os.environ["VDB_SAVE_PATH"] = vdb_path
            os.environ["LLM_PROVIDER"] = llm_provider
            os.environ["LLM_MODEL"] = llm_model
            os.environ["EMBEDDING_MODEL"] = embedding_model
            os.environ["RAG_TEST_DATA_PATH"] = test_data
            os.environ["CHUNK_SIZE"] = str(chunk_size)
            os.environ["CHUNK_OVERLAP"] = str(chunk_overlap)
            os.environ["NUM_CHUNKS"] = str(num_chunks)
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
            # FAISS index settings
            os.environ["FAISS_INDEX_TYPE"] = faiss_index_type
            os.environ["FAISS_IVF_NLIST"] = str(faiss_ivf_nlist)
            os.environ["FAISS_IVF_NPROBE"] = str(faiss_ivf_nprobe)
            os.environ["FAISS_HNSW_M"] = str(faiss_hnsw_m)
            os.environ["FAISS_HNSW_EF_CONSTRUCTION"] = str(faiss_hnsw_ef_construction)
            os.environ["FAISS_HNSW_EF_SEARCH"] = str(faiss_hnsw_ef_search)
            os.environ["FAISS_PQ_M"] = str(faiss_pq_m)
            os.environ["FAISS_PQ_NBITS"] = str(faiss_pq_nbits)

            st.success("✅ Settings saved to .env and applied!")
            st.info("🔄 Refresh the page to ensure all components pick up the new settings.")

        except Exception as e:
            st.error(f"❌ Error saving settings: {str(e)}")
