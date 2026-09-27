import os
import json
import time
import shutil
import datetime
from typing import List, Dict, Any

from src.llm.ollama_client import OllamaClient
from src.retrieval.vector_search import VectorSearch

def build_index():
    input_file = "data/processed/chunks.jsonl"
    output_dir = "data/processed/vector_store"

    # Clean rebuild
    if os.path.exists(output_dir):
        print(f"Removing existing vector store at {output_dir}")
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    # Read chunks
    print(f"Reading chunks from {input_file}...")
    chunks: List[Dict[str, Any]] = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    expected_count = 13130
    if len(chunks) != expected_count:
        raise ValueError(f"Expected {expected_count} chunks, but found {len(chunks)} in file.")

    client = OllamaClient()
    if not client.check_health():
        raise RuntimeError(f"Ollama server is not healthy: {client.last_error}")

    # We need to know the dimension. We'll embed one first to get dimension.
    print("Testing embedding dimension...")
    test_emb = client.embed(["test"], model="nomic-embed-text")
    if not test_emb or len(test_emb) != 1:
        raise RuntimeError("Failed to get test embedding.")
    embed_dim = len(test_emb[0])
    print(f"Detected embedding dimension: {embed_dim}")

    vs = VectorSearch(index_dir=output_dir, embed_dim=embed_dim)

    batch_size = 100
    total_processed = 0
    start_time = time.time()

    print(f"Starting embedding and indexing in batches of {batch_size}...")
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        texts = [c["text"] for c in batch]

        # Keep track of retry logic if needed
        max_retries = 5
        embeddings = None
        for attempt in range(max_retries):
            try:
                embeddings = client.embed(texts, model="nomic-embed-text")
                break
            except Exception as e:
                print(f"Error on batch {i}-{i+len(batch)}: {e}. Retrying {attempt+1}/{max_retries}...")
                time.sleep(5)

        if not embeddings or len(embeddings) != len(batch):
            raise RuntimeError(f"Failed to embed batch {i} to {i+len(batch)}. Exiting to prevent malformed index.")

        vs.add_chunks(embeddings, batch)
        total_processed += len(batch)
        print(f"Processed {total_processed}/{len(chunks)} chunks...")

    end_time = time.time()
    duration = end_time - start_time
    print(f"Completed indexing {total_processed} chunks in {duration:.2f} seconds.")

    vs.save(model_name="nomic-embed-text", chunk_file=input_file)
    print("Vector index saved successfully.")

if __name__ == "__main__":
    build_index()
