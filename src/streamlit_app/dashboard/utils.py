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
        index: FAISS IndexHNSWFlat (or wrapped variant)

    Returns:
        Dictionary with levels, neighbors, etc., or None if not HNSW
    """
    try:
        # Get the actual HNSW index (unwrap if needed)
        hnsw_index = _get_hnsw_index(index)
        if hnsw_index is None:
            return None

        hnsw = hnsw_index.hnsw

        # Get number of levels for each node
        num_elements = hnsw_index.ntotal
        # Convert Int32Vector to numpy array
        # NOTE: FAISS levels are 1-indexed! Subtract 1 to get 0-indexed
        levels = []
        if hasattr(hnsw, 'levels'):
            # Access Int32Vector elements directly
            for i in range(num_elements):
                try:
                    raw_level = hnsw.levels.at(i) if i < hnsw.levels.size() else 1
                    level = raw_level - 1  # Convert from 1-indexed to 0-indexed
                except:
                    level = 0
                levels.append(level)
        else:
            levels = [0] * num_elements

        return {
            'num_elements': num_elements,
            'levels': np.array(levels),
            'max_level': max(levels) if levels else 0
        }
    except Exception as e:
        print(f"Error extracting HNSW levels: {e}")
        return None


def _is_hnsw_index(index) -> bool:
    """Check if index is HNSW type (handles wrapped indexes)."""
    try:
        import faiss
        # Direct HNSW type
        if isinstance(index, faiss.IndexHNSWFlat):
            return True
        # Wrapped HNSW (e.g., IndexIDMap2 wrapping IndexHNSWFlat)
        if hasattr(index, 'index') and isinstance(index.index, faiss.IndexHNSWFlat):
            return True
        # Check for hnsw attribute
        if hasattr(index, 'hnsw'):
            return True
        if hasattr(index, 'index') and hasattr(index.index, 'hnsw'):
            return True
        return False
    except:
        return False


def _get_hnsw_index(index):
    """Get the actual HNSW index (unwrap if necessary)."""
    import faiss
    if isinstance(index, faiss.IndexHNSWFlat):
        return index
    if hasattr(index, 'index') and isinstance(index.index, faiss.IndexHNSWFlat):
        return index.index
    if hasattr(index, 'hnsw'):
        return index
    if hasattr(index, 'index') and hasattr(index.index, 'hnsw'):
        return index.index
    return None


def get_hnsw_graph_structure(index) -> Optional[Dict[str, Any]]:
    """Extract HNSW graph structure with neighbor connections.

    Args:
        index: FAISS IndexHNSWFlat (or wrapped variant)

    Returns:
        Dictionary with:
        - num_elements: int
        - levels: np.ndarray of shape (num_elements,)
        - max_level: int
        - entry_point: int
        - nodes_per_level: List[int]
        - neighbor_counts: List[int] (average connections per node per level)
    """
    try:
        import faiss

        # Debug: print what we received
        print(f"DEBUG get_hnsw_graph_structure: index type = {type(index)}")
        print(f"DEBUG index attributes: {dir(index)}")
        if hasattr(index, 'index'):
            print(f"DEBUG wrapped index.index type = {type(index.index)}")

        # Get the actual HNSW index (unwrap if needed)
        hnsw_index = _get_hnsw_index(index)
        if hnsw_index is None:
            print(f"DEBUG: _get_hnsw_index returned None for {type(index)}")
            return None

        print(f"DEBUG: hnsw_index type = {type(hnsw_index)}")

        hnsw = hnsw_index.hnsw
        num_elements = hnsw_index.ntotal
        print(f"DEBUG: num_elements = {num_elements}")
        print(f"DEBUG: hnsw has levels attr: {hasattr(hnsw, 'levels')}")

        # Get levels for all nodes
        # hnsw.levels is a FAISS Int32Vector - access via .at() method
        # NOTE: FAISS levels are 1-indexed! Subtract 1 to get 0-indexed
        levels = []
        if hasattr(hnsw, 'levels'):
            levels_size = hnsw.levels.size()
            print(f"DEBUG: hnsw.levels size = {levels_size}")
            for i in range(num_elements):
                try:
                    raw_level = hnsw.levels.at(i) if i < levels_size else 1
                    level = raw_level - 1  # Convert from 1-indexed to 0-indexed
                except:
                    level = 0
                levels.append(level)
        else:
            levels = [0] * num_elements
        levels = np.array(levels)
        max_level = int(max(levels)) if len(levels) > 0 else 0

        # Count nodes per level
        nodes_per_level = []
        for l in range(max_level + 1):
            count = np.sum(levels >= l)
            nodes_per_level.append(int(count))

        # Get entry point (stored in hnsw)
        entry_point = int(hnsw.entry_point) if hasattr(hnsw, 'entry_point') else 0

        # Estimate neighbor counts from M parameter
        # In HNSW, max neighbors per layer is M for layer 0, M/2 for layer 1, etc.
        # We approximate based on M and the level distribution
        neighbor_counts = []
        m = index.hnsw.M if hasattr(index.hnsw, 'M') else 16

        for l in range(max_level + 1):
            # Higher layers have fewer connections
            if l == 0:
                max_conn = m
            else:
                max_conn = m // 2
            neighbor_counts.append(max_conn)

        return {
            'num_elements': num_elements,
            'levels': levels,
            'max_level': max_level,
            'entry_point': entry_point,
            'nodes_per_level': nodes_per_level,
            'neighbor_counts': neighbor_counts,
            'M': m
        }
    except Exception as e:
        print(f"Error extracting HNSW graph structure: {e}")
        return None


def get_hnsw_neighbors_sample(index, num_samples: int = 100) -> Dict[int, List[int]]:
    """Get a sample of neighbor connections for visualization.

    Args:
        index: FAISS IndexHNSWFlat
        num_samples: Number of nodes to sample for neighbor extraction

    Returns:
        Dictionary mapping node_id to list of neighbor_ids
    """
    try:
        import faiss

        if not isinstance(index, faiss.IndexHNSWFlat):
            return {}

        hnsw = index.hnsw
        num_elements = index.ntotal

        if num_elements == 0:
            return {}

        # Sample nodes to avoid too many connections
        sample_size = min(num_samples, num_elements)
        sampled_nodes = np.random.choice(num_elements, size=sample_size, replace=False)

        neighbors = {}
        hnsw_graph = hnsw.neighbors

        for node_id in sampled_nodes:
            node_neighbors = []
            # Try to get neighbors from the graph structure
            # This is a simplified version - actual FAISS HNSW graph access
            # may require more complex handling
            try:
                # Access neighbor list through FAISS internals
                # Each node has a linked list of neighbors per level
                # NOTE: FAISS levels are 1-indexed! Subtract 1 to get 0-indexed
                raw_level = hnsw.levels.at(int(node_id)) if int(node_id) < hnsw.levels.size() else 1
                level = raw_level - 1  # Convert from 1-indexed to 0-indexed
                if level >= 0:
                    # Node exists in graph, try to get neighbors
                    # This is approximate - full graph extraction is complex
                    node_neighbors = []  # Placeholder for actual neighbor extraction
            except:
                pass
            neighbors[int(node_id)] = node_neighbors

        return neighbors
    except Exception as e:
        print(f"Error extracting HNSW neighbors: {e}")
        return {}


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


def _get_pq_index(index) -> Optional[Any]:
    """Get underlying PQ index, handling wrapped indices.

    Args:
        index: FAISS index (possibly wrapped in IndexIDMap)

    Returns:
        Unwrapped IndexPQ or IndexIVFPQ, or None if not found
    """
    try:
        import faiss

        # Direct match
        if isinstance(index, (faiss.IndexPQ, faiss.IndexIVFPQ)):
            return index

        # Handle IndexIDMap2 wrapping
        if hasattr(index, 'index') and isinstance(index.index, (faiss.IndexPQ, faiss.IndexIVFPQ)):
            return index.index

        return None
    except Exception:
        return None


def get_pq_reconstructed_vectors(index, embeddings: np.ndarray) -> Optional[np.ndarray]:
    """Reconstruct vectors from PQ codes.

    Args:
        index: FAISS IndexPQ or IndexIVFPQ (possibly wrapped)
        embeddings: Original embeddings array (n_vectors, dim)

    Returns:
        Reconstructed vectors array same shape as embeddings, or None if error
    """
    try:
        import faiss

        # Get underlying PQ index (handle wrapped indices)
        pq_index = _get_pq_index(index)
        if pq_index is None:
            print("Index is not a PQ type (may be flat or different type)")
            return None

        # Compute PQ codes for the embeddings
        codes = pq_index.pq.compute_codes(embeddings)

        # Decode codes back to vectors
        reconstructed = pq_index.pq.decode(codes)

        return reconstructed
    except Exception as e:
        print(f"Error reconstructing PQ vectors: {e}")
        return None
