import hashlib
from typing import Any, Dict, Iterator, List

from src.ingestion.models import Document

class Chunker:
    def __init__(self, chunk_size_words: int = 400, overlap_words: int = 80):
        self.chunk_size_words = chunk_size_words
        self.overlap_words = overlap_words
        if self.overlap_words >= self.chunk_size_words:
            raise ValueError("Overlap must be strictly less than chunk size")

    def _generate_chunk_id(self, doc_id: str, chunk_index: int) -> str:
        """Generate a deterministic chunk ID."""
        raw_id = f"{doc_id}::chunk_{chunk_index}"
        # We can just use the raw_id if we want readability, or hash it.
        # Let's just use the readable version for now as it's deterministic and unique.
        return raw_id

    def chunk_document(self, doc: Document) -> Iterator[Dict[str, Any]]:
        """
        Split a document into chunks based on word count, preserving overlap and metadata.
        Yields dictionaries representing each chunk.
        """
        text = doc.text
        if not text.strip():
            # If text is entirely empty, maybe yield one empty chunk or zero.
            # Usually better to yield zero chunks for empty text to avoid noise.
            return

        words = text.split()
        if not words:
            return

        chunk_index = 0
        start_word = 0
        total_words = len(words)

        while start_word < total_words:
            end_word = min(start_word + self.chunk_size_words, total_words)
            chunk_words = words[start_word:end_word]
            chunk_text = " ".join(chunk_words)

            yield {
                "chunk_id": self._generate_chunk_id(doc.doc_id, chunk_index),
                "doc_id": doc.doc_id,
                "title": doc.title,
                "url": doc.url,
                "wikidata_qid": doc.wikidata_qid,
                "wikipedia_pageid": doc.wikipedia_pageid,
                "text": chunk_text,
                "chunk_index": chunk_index,
                "start_word": start_word,
                "end_word": end_word
            }

            # If we've reached the end of the document, break
            if end_word == total_words:
                break

            # Move forward by the chunk size minus the overlap
            step = self.chunk_size_words - self.overlap_words
            # Ensure we always make progress to avoid infinite loops
            start_word += max(1, step)
            chunk_index += 1
