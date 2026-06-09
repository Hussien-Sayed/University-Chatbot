"""Debug script to explore HNSW graph structure from existing vector DB."""
import os
import sys
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def log(msg: str, indent: int = 0):
    """Print with indentation."""
    print("  " * indent + msg)


def load_vector_db():
    """Load the vector database."""
    vdb_path = os.getenv("VDB_SAVE_PATH", "data/vector_db")
    db_file = Path(vdb_path) / "vector_db.pkl"
    
    log(f"Loading vector DB from: {db_file}")
    
    if not db_file.exists():
        log(f"ERROR: File not found: {db_file}")
        return None
    
    try:
        with open(db_file, "rb") as f:
            db_data = pickle.load(f)
        log(f"✓ Successfully loaded vector DB")
        return db_data
    except Exception as e:
        log(f"ERROR loading vector DB: {e}")
        return None


def explore_hnsw_structure(db_data: Dict[str, Any]):
    """Explore and log HNSW graph structure."""
    log("\n" + "="*60)
    log("HNSW GRAPH STRUCTURE EXPLORATION")
    log("="*60)
    
    # Basic metadata
    metadata = db_data.get("metadata", {})
    log("\n[1] VECTOR DB METADATA:")
    log(f"Index type: {metadata.get('faiss_index_type', 'unknown')}", 1)
    log(f"HNSW M (connections): {metadata.get('faiss_hnsw_m', 'N/A')}", 1)
    log(f"HNSW ef_construction: {metadata.get('faiss_hnsw_ef_construction', 'N/A')}", 1)
    log(f"HNSW ef_search: {metadata.get('faiss_hnsw_ef_search', 'N/A')}", 1)
    
    # Get index
    index = db_data.get("index")
    if index is None:
        log("\nERROR: No index found in vector DB!")
        return
    
    log(f"\n[2] INDEX INFORMATION:")
    log(f"Index type: {type(index)}", 1)
    log(f"Is IndexHNSWFlat: {'IndexHNSWFlat' in str(type(index))}", 1)
    
    # Check index type with faiss
    try:
        import faiss
        is_hnsw_flat = isinstance(index, faiss.IndexHNSWFlat)
        log(f"isinstance(IndexHNSWFlat): {is_hnsw_flat}", 1)
        if hasattr(index, 'index'):
            inner_is_hnsw = isinstance(index.index, faiss.IndexHNSWFlat)
            log(f"Inner index isinstance(IndexHNSWFlat): {inner_is_hnsw}", 1)
    except Exception as e:
        log(f"Type check error: {e}", 1)
    
    # Check if wrapped (e.g., IndexIDMap2 wrapping IndexHNSWFlat)
    if hasattr(index, 'index'):
        log(f"Wrapped inner index type: {type(index.index)}", 1)
        log(f"Inner has hnsw attr: {hasattr(index.index, 'hnsw')}", 1)
        hnsw_index = index.index
    else:
        hnsw_index = index
    
    if not hasattr(hnsw_index, 'hnsw'):
        log("\nERROR: Index does not have hnsw attribute!")
        return
    
    hnsw = hnsw_index.hnsw
    log(f"\n[3] HNSW OBJECT:")
    log(f"hnsw type: {type(hnsw)}", 1)
    
    # List all hnsw attributes
    log(f"\n[4] HNSW ATTRIBUTES:")
    hnsw_attrs = [attr for attr in dir(hnsw) if not attr.startswith('_')]
    for attr in sorted(hnsw_attrs):
        log(f"- {attr}", 1)
    
    # Basic HNSW properties
    log(f"\n[5] HNSW PROPERTIES:")
    
    if hasattr(hnsw, 'M'):
        log(f"M (max connections): {hnsw.M}", 1)
    if hasattr(hnsw, 'max_level'):
        log(f"max_level: {hnsw.max_level}", 1)
    if hasattr(hnsw, 'entry_point'):
        log(f"entry_point: {hnsw.entry_point}", 1)
    if hasattr(hnsw, 'nb'):
        log(f"nb (number of elements): {hnsw.nb}", 1)
    
    # Get number of elements
    num_elements = hnsw_index.ntotal
    log(f"ntotal: {num_elements}", 1)
    
    # Explore levels
    log(f"\n[6] NODE LEVELS:")
    if hasattr(hnsw, 'levels'):
        levels_size = hnsw.levels.size()
        log(f"levels size: {levels_size}", 1)
        
        # Get all levels using .at() method
        # NOTE: FAISS levels are 1-indexed! Subtract 1 to get 0-indexed level
        levels = []
        for i in range(num_elements):
            try:
                raw_level = hnsw.levels.at(i) if i < levels_size else 1
                level = raw_level - 1  # Convert from 1-indexed to 0-indexed
            except Exception as e:
                log(f"Error reading level at {i}: {e}", 2)
                level = 0
            levels.append(level)
        levels = np.array(levels)
        
        log(f"\nNOTE: FAISS levels are 1-indexed internally. Converted to 0-indexed:", 1)
        log(f"  Raw levels (first 5): {[hnsw.levels.at(i) for i in range(min(5, num_elements))]}", 2)
        log(f"  Converted levels (first 5): {levels[:5].tolist()}", 2)
        
        # Alternative: Try reading with faiss.vector_to_array if available
        log(f"\nAlternative level reading methods:", 1)
        try:
            import faiss
            # Try faiss.vector_int32_to_array (for Int32Vector)
            if hasattr(faiss, 'vector_int32_to_array'):
                levels_alt = faiss.vector_int32_to_array(hnsw.levels)
                log(f"  faiss.vector_int32_to_array: {levels_alt[:20].tolist()}", 2)
                log(f"  Unique values (alt): {sorted(set(levels_alt.tolist()))}", 2)
                if not np.array_equal(levels, levels_alt):
                    log(f"  ⚠ MISMATCH between .at() and vector_int32_to_array!", 2)
        except Exception as e:
            log(f"  Alternative method failed: {e}", 2)
        
        # Check for consistency
        computed_max = int(max(levels)) if len(levels) > 0 else 0
        hnsw_max = int(hnsw.max_level) if hasattr(hnsw, 'max_level') else 'N/A'
        log(f"\nConsistency check:", 1)
        log(f"  hnsw.max_level property: {hnsw_max}", 2)
        log(f"  Max from levels array: {computed_max}", 2)
        if hnsw_max != 'N/A' and computed_max != hnsw_max:
            log(f"  ⚠ MISMATCH! Property says {hnsw_max} but nodes go up to {computed_max}", 2)
        
        # Show raw levels array for debugging
        log(f"\nRaw levels array (first 20):", 1)
        log(f"  {levels[:20].tolist()}", 2)
        log(f"  Unique values: {sorted(set(levels.tolist()))}", 2)
        
        log(f"\nLevel distribution:", 1)
        max_level = computed_max
        for l in range(max_level + 1):
            count = np.sum(levels == l)
            log(f"  Max Level {l}: {count} nodes", 2)
        
        # Cumulative (nodes in each layer)
        log(f"\nNodes per layer (cumulative):", 1)
        for l in range(max_level + 1):
            count = np.sum(levels >= l)
            log(f"  Layer {l}: {count} nodes", 2)
    else:
        log("No 'levels' attribute found!", 1)
    
    # Explore neighbors (the graph connections)
    log(f"\n[7] NEIGHBOR STRUCTURE:")
    if hasattr(hnsw, 'neighbors'):
        log(f"Has 'neighbors' attribute: True", 1)
        log(f"neighbors type: {type(hnsw.neighbors)}", 1)
        
        # Try to access neighbors for a sample of nodes
        sample_size = min(5, num_elements)
        log(f"\nSample neighbor extraction for first {sample_size} nodes:", 1)
        
        for node_id in range(sample_size):
            level = int(levels[node_id]) if node_id < len(levels) else 0
            log(f"\nNode #{node_id} (level={level}):", 2)
            
            # Try different ways to access neighbors
            try:
                # Method 1: Direct access if neighbors is array-like
                if hasattr(hnsw.neighbors, 'at'):
                    # Try to get neighbor at this node
                    neighbor_val = hnsw.neighbors.at(node_id)
                    log(f"  neighbors.at({node_id}) = {neighbor_val}", 3)
            except Exception as e:
                log(f"  neighbors.at() error: {e}", 3)
            
            # Try to access neighbors for each layer
            if level > 0:
                for l in range(level + 1):
                    try:
                        # FAISS HNSW stores neighbors in a complex structure
                        # neighbors[node_id] might give us a list or we need to iterate
                        log(f"  Trying layer {l}...", 3)
                    except Exception as e:
                        log(f"  Layer {l} error: {e}", 3)
    else:
        log("No 'neighbors' attribute found!", 1)
    
    # Chunks info
    log(f"\n[8] CHUNKS TO NODES MAPPING:")
    chunks = db_data.get("chunks", [])
    log(f"Total chunks: {len(chunks)}", 1)
    log(f"Total nodes: {num_elements}", 1)
    
    if len(chunks) == num_elements:
        log("✓ Chunks and nodes count match!", 1)
    else:
        log(f"⚠ Mismatch: {len(chunks)} chunks vs {num_elements} nodes", 1)
    
    # Show sample chunk-node mapping
    log(f"\nSample chunk-node mapping (first 5):", 1)
    for i in range(min(5, len(chunks))):
        chunk = chunks[i]
        level = int(levels[i]) if i < len(levels) else 'N/A'
        log(f"Node #{i} -> Chunk: {chunk.get('chunk_id', 'N/A')}, "
            f"Tag: {chunk.get('tag', 'N/A')}, "
            f"Level: {level}", 2)
        
        # Show content preview
        content = chunk.get('content', '')
        preview = content[:80].replace('\n', ' ') + "..." if len(content) > 80 else content
        log(f"  Content: {preview}", 3)
    
    # Entry point chunk
    if hasattr(hnsw, 'entry_point'):
        entry = int(hnsw.entry_point)
        log(f"\n[9] ENTRY POINT:")
        log(f"Entry point node: #{entry}", 1)
        if entry < len(chunks):
            entry_chunk = chunks[entry]
            log(f"Entry chunk ID: {entry_chunk.get('chunk_id', 'N/A')}", 1)
            log(f"Entry chunk tag: {entry_chunk.get('tag', 'N/A')}", 1)
            log(f"Entry chunk content: {entry_chunk.get('content', '')[:100]}...", 1)
    
    log("\n" + "="*60)
    log("END OF HNSW GRAPH STRUCTURE EXPLORATION")
    log("="*60)


def main():
    """Main debug function."""
    log("HNSW DEBUG SCRIPT")
    log(f"Project root: {project_root}")
    log(f"Python version: {sys.version}")
    
    # Check if faiss is available
    try:
        import faiss
        log(f"FAISS version: {faiss.__version__}")
        # Check FAISS compile options
        log(f"FAISS has SWIG wrapper: {hasattr(faiss, 'swigfaiss')}")
    except ImportError:
        log("ERROR: FAISS not installed!")
        return
    
    # Load vector DB
    db_data = load_vector_db()
    if db_data is None:
        log("\nFailed to load vector DB. Exiting.")
        return
    
    # Explore HNSW structure
    explore_hnsw_structure(db_data)
    
    log("\n✓ Debug complete. Check output above for HNSW graph details.")


if __name__ == "__main__":
    main()
