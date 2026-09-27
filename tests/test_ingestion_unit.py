import json
import unittest
import tempfile
from pathlib import Path

from src.ingestion.models import Document
from src.ingestion.corpus_loader import load_documents, _check_path_safety
from src.ingestion.chunker import Chunker

class TestCorpusLoader(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.corpus_path = self.temp_path / "corpus.jsonl"

        data = [
            '{"doc_id": "1", "title": "A", "url": "http", "wikidata_qid": "Q1", "wikipedia_pageid": 10, "approx_tokens": 5, "text": "Hello world"}',
            '{"doc_id": "2", "title": "B", "url": "http", "wikidata_qid": "Q2", "wikipedia_pageid": 20, "approx_tokens": 10, "text": "Valid doc 2"}',
            '{"doc_id": "2", "title": "C", "url": "http", "wikidata_qid": "Q3", "wikipedia_pageid": 30, "approx_tokens": 15, "text": "Duplicate doc_id"}',
            '{"doc_id": "3", "url": "http", "text": "Missing fields"}',
            '{"doc_id": "4", "title": "M", "url": "http", "wikidata_qid": "Q4", "wikipedia_pageid": 40, "approx_tokens": 2, "text": ""}', # Empty text
            'Malformed JSON {',
        ]
        self.corpus_path.write_text("\n".join(data))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_load_documents(self):
        docs = list(load_documents(self.corpus_path))
        # Should load doc_id "1", "2" (first one), and "4" (empty text is allowed by loader, chunker filters it later)
        self.assertEqual(len(docs), 3)
        self.assertEqual(docs[0].doc_id, "1")
        self.assertEqual(docs[1].doc_id, "2")
        self.assertEqual(docs[1].title, "B")
        self.assertEqual(docs[2].doc_id, "4")
        self.assertEqual(docs[2].text, "")

    def test_check_path_safety(self):
        with self.assertRaises(ValueError):
            _check_path_safety(Path("data/hidden/eval_hidden.jsonl"))

class TestChunker(unittest.TestCase):
    def setUp(self):
        self.doc = Document(
            doc_id="doc_100",
            title="Test Doc",
            url="http://test",
            wikidata_qid="Q100",
            wikipedia_pageid=123,
            approx_tokens=50,
            text="word1 word2 word3 word4 word5 word6 word7 word8 word9 word10"
        )

    def test_deterministic_chunking(self):
        chunker = Chunker(chunk_size_words=4, overlap_words=2)
        chunks = list(chunker.chunk_document(self.doc))

        self.assertEqual(len(chunks), 4)

        # Chunk 0
        self.assertEqual(chunks[0]["chunk_id"], "doc_100::chunk_0")
        self.assertEqual(chunks[0]["text"], "word1 word2 word3 word4")
        self.assertEqual(chunks[0]["start_word"], 0)
        self.assertEqual(chunks[0]["end_word"], 4)

        # Chunk 1 (overlap 2)
        self.assertEqual(chunks[1]["chunk_id"], "doc_100::chunk_1")
        self.assertEqual(chunks[1]["text"], "word3 word4 word5 word6")
        self.assertEqual(chunks[1]["start_word"], 2)
        self.assertEqual(chunks[1]["end_word"], 6)

        # Chunk 2
        self.assertEqual(chunks[2]["chunk_id"], "doc_100::chunk_2")
        self.assertEqual(chunks[2]["text"], "word5 word6 word7 word8")

        # Chunk 3
        self.assertEqual(chunks[3]["text"], "word7 word8 word9 word10")

        # Check metadata preservation
        for chunk in chunks:
            self.assertEqual(chunk["doc_id"], "doc_100")
            self.assertEqual(chunk["title"], "Test Doc")
            self.assertEqual(chunk["wikidata_qid"], "Q100")
            self.assertEqual(chunk["wikipedia_pageid"], 123)

    def test_empty_text(self):
        empty_doc = Document("doc_2", "T", "U", "Q", 1, 1, "   ")
        chunker = Chunker(4, 2)
        chunks = list(chunker.chunk_document(empty_doc))
        self.assertEqual(len(chunks), 0)

if __name__ == "__main__":
    unittest.main()
