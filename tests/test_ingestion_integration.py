import unittest
from pathlib import Path

from src.ingestion.corpus_loader import load_documents
from src.ingestion.chunker import Chunker

class TestCorpusIntegration(unittest.TestCase):
    def setUp(self):
        self.base_dir = Path(__file__).resolve().parent.parent
        self.corpus_path = self.base_dir / "data" / "corpus" / "corpus.jsonl"

    def test_corpus_integrity(self):
        """Test against the real corpus to verify 2951 documents and no duplicates."""
        if not self.corpus_path.exists():
            self.skipTest(f"Corpus file not found at {self.corpus_path}")

        seen_ids = set()
        doc_count = 0

        for doc in load_documents(self.corpus_path):
            self.assertNotIn(doc.doc_id, seen_ids, f"Duplicate doc_id found: {doc.doc_id}")
            seen_ids.add(doc.doc_id)
            doc_count += 1

        self.assertEqual(doc_count, 2951, "Expected exactly 2951 documents in the corpus.")

    def test_chunking_integrity(self):
        """Test that chunks map correctly back to valid doc_ids."""
        if not self.corpus_path.exists():
            self.skipTest("Corpus file not found")

        chunker = Chunker(chunk_size_words=400, overlap_words=80)

        # Test just the first 10 documents to keep the test fast
        doc_iter = iter(load_documents(self.corpus_path))

        for _ in range(10):
            try:
                doc = next(doc_iter)
            except StopIteration:
                break

            chunks = list(chunker.chunk_document(doc))
            if doc.text.strip():
                self.assertGreater(len(chunks), 0)

            for i, chunk in enumerate(chunks):
                self.assertEqual(chunk["doc_id"], doc.doc_id)
                self.assertEqual(chunk["chunk_id"], f"{doc.doc_id}::chunk_{i}")
                self.assertEqual(chunk["title"], doc.title)
                self.assertTrue(len(chunk["text"]) > 0)
