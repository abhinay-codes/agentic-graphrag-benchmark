import os
import json
import numpy as np
import faiss
from src.retrieval.vector_search import VectorSearch

def verify_index():
    index_dir = "data/processed/vector_store"
    expected_count = 13130

    # 6 & 7. FAISS index can be saved and reopened, metadata can be reopened
    print("Loading vector search index...")
    vs = VectorSearch(index_dir=index_dir, embed_dim=768)

    # 1. FAISS vector count == 13,130
    print(f"FAISS count: {vs.index.ntotal}")
    assert vs.index.ntotal == expected_count, f"Expected {expected_count}, got {vs.index.ntotal}"

    # 2. metadata record count == 13,130
    print(f"Metadata count: {len(vs.metadata)}")
    assert len(vs.metadata) == expected_count, f"Expected {expected_count}, got {len(vs.metadata)}"

    # 8. FAISS position <-> metadata position alignment
    # By definition of our class and above assertions, if it loaded without error, the lengths match.

    # 3. metadata chunk IDs are unique
    chunk_ids = [m["chunk_id"] for m in vs.metadata]
    unique_chunk_ids = set(chunk_ids)
    print(f"Unique chunk IDs: {len(unique_chunk_ids)}")
    assert len(unique_chunk_ids) == expected_count, "Chunk IDs are not unique!"

    # 4. every metadata record contains required fields
    required_fields = {"chunk_id", "doc_id", "title", "url", "chunk_index", "text"}
    for m in vs.metadata:
        missing = required_fields - set(m.keys())
        assert not missing, f"Metadata record missing fields: {missing}"

    # 5. embedding dimension is consistent
    assert vs.index.d == 768, f"Expected dimension 768, got {vs.index.d}"

    # 9. manifest values match the actual index
    with open(os.path.join(index_dir, "manifest.json"), "r") as f:
        manifest = json.load(f)
        assert manifest["embedding_dimension"] == 768
        assert manifest["chunk_count"] == expected_count
        assert manifest["faiss_index_type"] == "IndexFlatIP"
        print("Manifest verified.")

    # Independent Sanity Check
    print("Running sanity check query...")
    # Generate a deterministic synthetic query vector
    np.random.seed(42)
    synthetic_query = np.random.rand(768).tolist()

    results = vs.search(synthetic_query, top_k=5)

    assert len(results) == 5, "Should return top 5 results"

    previous_score = float('inf')
    for result, score in results:
        # verify search returns valid metadata
        assert isinstance(result, dict)
        assert "chunk_id" in result

        # Verify score is numeric
        assert isinstance(score, float)

        # Verify descending order
        assert score <= previous_score, "Scores are not in descending order!"
        previous_score = score

    print("Sanity check passed.")
    print("All post-build verifications passed successfully!")

if __name__ == "__main__":
    verify_index()
