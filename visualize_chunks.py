import argparse
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import streamlit as st


def load_vector_db(vector_db_file: Path) -> Dict[str, Any]:
    if not vector_db_file.exists():
        raise FileNotFoundError(f"Vector database file not found: {vector_db_file}")

    with open(vector_db_file, "rb") as f:
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


def build_rows(chunks: List[Dict[str, Any]], points: np.ndarray) -> List[Dict[str, Any]]:
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

    return rows


def filter_rows(rows: List[Dict[str, Any]], search_text: str, selected_tags: List[str]) -> List[Dict[str, Any]]:
    filtered = rows

    if selected_tags:
        filtered = [row for row in filtered if row["tag"] in selected_tags]

    if search_text.strip():
        query = search_text.casefold().strip()
        filtered = [
            row for row in filtered
            if query in str(row["content"]).casefold()
            or query in str(row["tag"]).casefold()
            or query in str(row["doc_id"]).casefold()
            or query in str(row["chunk_id"]).casefold()
        ]

    return filtered


def get_default_vector_db_path() -> Path:
    return Path("data") / "vector_db" / "vector_db.pkl"


def parse_args() -> Tuple[Path, bool]:
    parser = argparse.ArgumentParser(description="Visualize chunks stored in the local FAISS vector database.")
    parser.add_argument(
        "--vector-db",
        default=str(get_default_vector_db_path()),
        help="Path to vector_db.pkl. Default: data/vector_db/vector_db.pkl",
    )
    parser.add_argument(
        "--headless-check",
        action="store_true",
        help="Load the vector DB and print summary without starting Streamlit UI.",
    )
    args, _ = parser.parse_known_args()
    return Path(args.vector_db), args.headless_check


def render_app(vector_db_file: Path) -> None:
    st.set_page_config(page_title="Vector DB Chunk Visualizer", page_icon="🧩", layout="wide")
    st.title("🧩 Vector DB Chunk Visualizer")
    st.caption("Visualize chunks and embeddings saved in your local FAISS vector database.")

    db_data = load_vector_db(vector_db_file)
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
    rows = build_rows(chunks, points)
    tags = sorted({row["tag"] for row in rows})

    with st.sidebar:
        st.header("Vector DB")
        st.write(f"File: `{vector_db_file}`")
        st.write(f"Chunks: `{len(chunks)}`")
        st.write(f"Embedding shape: `{embeddings.shape}`")

        if metadata:
            st.subheader("Metadata")
            st.json(metadata)

        st.subheader("Filters")
        search_text = st.text_input("Search chunks")
        selected_tags = st.multiselect("Filter by tag", tags)

    filtered_rows = filter_rows(rows, search_text, selected_tags)

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


def main() -> None:
    vector_db_file, headless_check = parse_args()

    if headless_check:
        db_data = load_vector_db(vector_db_file)
        chunks = db_data.get("chunks", [])
        embeddings = np.array(db_data.get("embeddings", []), dtype=np.float32)
        metadata = db_data.get("metadata", {})
        print(f"Vector DB: {vector_db_file}")
        print(f"Chunks: {len(chunks)}")
        print(f"Embedding shape: {embeddings.shape}")
        print(f"Metadata: {metadata}")
        return

    render_app(vector_db_file)


if __name__ == "__main__":
    main()
