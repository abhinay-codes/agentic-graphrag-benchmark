import json
import logging
import statistics
from collections import defaultdict
from pathlib import Path

from src.ingestion.corpus_loader import load_documents
from src.ingestion.chunker import Chunker

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main():
    base_dir = Path(__file__).resolve().parent.parent.parent
    corpus_path = base_dir / "data" / "corpus" / "corpus.jsonl"
    processed_dir = base_dir / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / "chunks.jsonl"

    chunk_size_words = 400
    overlap_words = 80
    chunker = Chunker(chunk_size_words=chunk_size_words, overlap_words=overlap_words)

    logger.info(f"Loading corpus from {corpus_path}")
    logger.info(f"Chunking with size={chunk_size_words}, overlap={overlap_words}")

    total_docs = 0
    total_chunks = 0
    docs_requiring_multiple_chunks = 0
    empty_chunks = 0

    chunks_per_doc = []
    chunk_lengths = []

    with open(out_path, 'w', encoding='utf-8') as out_f:
        for doc in load_documents(corpus_path):
            total_docs += 1
            doc_chunks = list(chunker.chunk_document(doc))

            num_chunks = len(doc_chunks)
            chunks_per_doc.append(num_chunks)

            if num_chunks > 1:
                docs_requiring_multiple_chunks += 1

            for chunk in doc_chunks:
                total_chunks += 1
                c_len = len(chunk["text"].split())
                chunk_lengths.append(c_len)
                if c_len == 0:
                    empty_chunks += 1

                out_f.write(json.dumps(chunk) + "\n")

    logger.info(f"Finished building chunks. Saved to {out_path}")

    # Calculate statistics
    if chunks_per_doc:
        min_chunks = min(chunks_per_doc)
        max_chunks = max(chunks_per_doc)
        mean_chunks = statistics.mean(chunks_per_doc)
    else:
        min_chunks = max_chunks = mean_chunks = 0

    if chunk_lengths:
        min_len = min(chunk_lengths)
        max_len = max(chunk_lengths)
        mean_len = statistics.mean(chunk_lengths)
    else:
        min_len = max_len = mean_len = 0

    print("\n--- CHUNKING STATISTICS ---")
    print(f"Total documents: {total_docs}")
    print(f"Total chunks: {total_chunks}")
    print(f"Min chunks/document: {min_chunks}")
    print(f"Max chunks/document: {max_chunks}")
    print(f"Mean chunks/document: {mean_chunks:.2f}")
    print(f"Documents requiring multiple chunks: {docs_requiring_multiple_chunks}")
    print(f"Chunks with empty text: {empty_chunks}")
    print(f"Chunk length (words) - Min: {min_len}, Max: {max_len}, Mean: {mean_len:.2f}")

if __name__ == "__main__":
    main()
