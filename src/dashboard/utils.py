"""Utilities for FAISS index introspection and visualization."""
import numpy as np
from typing import Tuple, Optional, List, Dict, Any


def reduce_embeddings_pca(embeddings: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Reduce embeddings to 2D using PCA for visualization."""
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings array, got shape: {embeddings.shape}")

    if embeddings.shape[0] == 0:
        return np.empty((0, n_components), dtype=np.float32)

    if embeddings.shape[0] == 1:
        return np.zeros((1, n_components), dtype=np.float32)

    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)

    components = vt[:n_components]
    points = centered @ components.T

    # Pad with zeros if we got fewer components than requested
    if points.shape[1] < n_components:
        padding = np.zeros((points.shape[0], n_components - points.shape[1]), dtype=np.float32)
        points = np.hstack([points, padding])

    return points.astype(np.float32)


def get_ivf_centroids(index) -> Optional[np.ndarray]:
    """Extract centroids from an IVF index.

    Args:
        index: FAISS IndexIVFFlat or IndexIVFPQ

    Returns:
        Array of centroids with shape (nlist, dimension), or None if not IVF
    """
    try:
        import faiss

        if isinstance(index, (faiss.IndexIVFFlat, faiss.IndexIVFPQ)):
            # Access the quantizer (coarse quantizer holds centroids)
            quantizer = index.quantizer

            # The quantizer is an IndexFlatIP that stores nlist vectors (centroids)
            # Use reconstruct_n to get all vectors from the quantizer
            nlist = index.nlist
            dimension = index.d

            # Reconstruct all centroids from the quantizer
            centroids = np.zeros((nlist, dimension), dtype=np.float32)
            for i in range(nlist):
                centroids[i] = quantizer.reconstruct(i)

            return centroids
        return None
    except Exception as e:
        print(f"Error extracting IVF centroids: {e}")
        return None


def get_ivf_assignments(index, num_vectors: int) -> Optional[np.ndarray]:
    """Get cluster assignments for all vectors in an IVF index.

    This computes assignments by finding nearest centroid for each vector.

    Args:
        index: FAISS IndexIVFFlat or IndexIVFPQ
        num_vectors: Number of vectors to get assignments for

    Returns:
        Array of cluster IDs with shape (num_vectors,), or None if not IVF
    """
    try:
        import faiss

        if not isinstance(index, (faiss.IndexIVFFlat, faiss.IndexIVFPQ)):
            return None

        # Get centroids
        centroids = get_ivf_centroids(index)
        if centroids is None:
            return None

        # For each vector, we need to find its nearest centroid
        # Since we don't have direct access to stored vectors in index.xb,
        # we'll need to compute this differently

        # Option 1: If index is trained, use index.quantizer to assign
        if hasattr(index, 'quantizer'):
            # Create a temporary flat index with centroids
            temp_index = faiss.IndexFlatIP(index.d)
            temp_index.add(centroids)

            # We need the actual vectors to assign them
            # This requires access to the stored vectors, which may not be directly accessible
            # For now, return None - we'll compute assignments differently in the visualizer
            return None

        return None
    except Exception as e:
        print(f"Error getting IVF assignments: {e}")
        return None


def compute_cluster_assignments(embeddings: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Compute cluster assignments by finding nearest centroid for each embedding.

    Args:
        embeddings: Array of shape (n, dimension)
        centroids: Array of shape (nlist, dimension)

    Returns:
        Array of cluster IDs with shape (n,)
    """
    # Compute distances to all centroids
    # embeddings: (n, d), centroids: (nlist, d)
    # distances: (n, nlist)
    distances = embeddings @ centroids.T

    # Find nearest centroid for each embedding
    assignments = np.argmax(distances, axis=1)

    return assignments


def get_cluster_sizes(assignments: np.ndarray, nlist: int) -> np.ndarray:
    """Get the number of chunks in each cluster.

    Args:
        assignments: Array of cluster IDs
        nlist: Total number of clusters

    Returns:
        Array of cluster sizes with shape (nlist,)
    """
    sizes = np.zeros(nlist, dtype=int)
    for cluster_id in range(nlist):
        sizes[cluster_id] = np.sum(assignments == cluster_id)
    return sizes


def get_hnsw_levels(index) -> Optional[Dict[str, Any]]:
    """Extract HNSW graph structure.

    Args:
        index: FAISS IndexHNSWFlat

    Returns:
        Dictionary with levels, neighbors, etc., or None if not HNSW
    """
    try:
        import faiss

        if not isinstance(index, faiss.IndexHNSWFlat):
            return None

        hnsw = index.hnsw

        # Get number of levels for each node
        num_elements = index.ntotal
        levels = []
        for i in range(num_elements):
            level = hnsw.levels[i] if i < len(hnsw.levels) else 0
            levels.append(level)

        return {
            'num_elements': num_elements,
            'levels': np.array(levels),
            'max_level': max(levels) if levels else 0
        }
    except Exception as e:
        print(f"Error extracting HNSW structure: {e}")
        return None


def get_pq_codebook(index) -> Optional[np.ndarray]:
    """Extract PQ subspace centroids.

    Args:
        index: FAISS IndexIVFPQ or IndexPQ

    Returns:
        Array of shape (M, 2^nbits, d/M), or None if not PQ
    """
    try:
        import faiss

        if isinstance(index, faiss.IndexIVFPQ):
            pq = index.pq
        elif isinstance(index, faiss.IndexPQ):
            pq = index.pq
        else:
            return None

        # centroids shape: (M * 2^nbits * d/M,) = (M * 2^nbits, d/M)
        centroids = faiss.vector_float_to_array(pq.centroids)
        M = pq.M
        nbits = pq.nbits
        d_sub = pq.dsub

        # Reshape to (M, 2^nbits, d_sub)
        centroids = centroids.reshape(M, 2**nbits, d_sub)

        return centroids
    except Exception as e:
        print(f"Error extracting PQ codebook: {e}")
        return None
