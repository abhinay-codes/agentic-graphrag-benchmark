import os
import json
import datetime
import numpy as np
import faiss
from typing import List, Dict, Any, Tuple, Optional

class VectorSearch:
    def __init__(self, index_dir: str = "data/processed/vector_store", embed_dim: int = 768):
        self.index_dir = index_dir
        self.embed_dim = embed_dim
        self.faiss_index_path = os.path.join(index_dir, "index.faiss")
        self.metadata_path = os.path.join(index_dir, "metadata.jsonl")
        self.manifest_path = os.path.join(index_dir, "manifest.json")

        os.makedirs(self.index_dir, exist_ok=True)

        # We use IndexFlatIP for Cosine Similarity (requires normalized vectors)
        self.index = faiss.IndexFlatIP(self.embed_dim)
        self.metadata: List[Dict[str, Any]] = []

        self.load()

    def normalize(self, vectors: np.ndarray) -> np.ndarray:
        """Normalizes vectors for inner-product to act as cosine similarity."""
        if len(vectors.shape) == 1:
            vectors = vectors.reshape(1, -1)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1e-10, norms)
        return vectors / norms

    def add_chunks(self, vectors: List[List[float]], metadata_list: List[Dict[str, Any]]):
        """Adds a batch of embeddings and parallel metadata."""
        if len(vectors) != len(metadata_list):
            raise ValueError("Number of vectors must match number of metadata entries.")
        if not vectors:
            return

        np_vectors = np.array(vectors, dtype=np.float32)

        if np_vectors.shape[1] != self.embed_dim:
            raise ValueError(f"Expected embedding dimension {self.embed_dim}, got {np_vectors.shape[1]}")

        normalized_vectors = self.normalize(np_vectors)
        self.index.add(normalized_vectors)
        self.metadata.extend(metadata_list)

        # Verify alignment
        if self.index.ntotal != len(self.metadata):
            raise RuntimeError(f"Alignment Error: FAISS has {self.index.ntotal} items but metadata has {len(self.metadata)} items.")

    def search(self, query_vector: List[float], top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """Searches the vector index for the top-k most similar chunks."""
        if self.index.ntotal == 0:
            return []

        q_vec = np.array([query_vector], dtype=np.float32)
        q_vec_norm = self.normalize(q_vec)

        # Search returns distances (scores) and indices
        scores, indices = self.index.search(q_vec_norm, top_k)

        results = []
        for j, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.metadata):
                results.append((self.metadata[idx], float(scores[0][j])))

        return results

    def save(self, model_name: str = "nomic-embed-text", chunk_file: str = "data/processed/chunks.jsonl"):
        """Saves the FAISS index, metadata, and manifest to disk."""
        # 1. Save FAISS index
        faiss.write_index(self.index, self.faiss_index_path)

        # 2. Save metadata as JSONL
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            for md in self.metadata:
                f.write(json.dumps(md) + '\n')

        # 3. Save manifest
        manifest = {
            "embedding_model": model_name,
            "embedding_dimension": self.embed_dim,
            "similarity_metric": "Cosine (Inner Product on Normalized Vectors)",
            "faiss_index_type": "IndexFlatIP",
            "chunk_count": len(self.metadata),
            "source_chunk_file": chunk_file,
            "build_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "version": "1.0.0"
        }
        with open(self.manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2)

    def load(self):
        """Loads the FAISS index and metadata if they exist."""
        if os.path.exists(self.faiss_index_path) and os.path.exists(self.metadata_path):
            self.index = faiss.read_index(self.faiss_index_path)

            self.metadata = []
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        self.metadata.append(json.loads(line))

            if self.index.ntotal != len(self.metadata):
                raise RuntimeError(f"Corrupt Index: FAISS count ({self.index.ntotal}) != metadata count ({len(self.metadata)})")
