import unittest
from pathlib import Path
import tempfile
import os

from src.ingestion.tigergraph_loader import TigerGraphLoader
from src.retrieval.graph_retriever import GraphRetriever

class TestGraphLoader(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        self.corpus_path = self.temp_path / "corpus.jsonl"
        self.corpus_path.write_text(
            '{"doc_id": "D1", "title": "T1", "url": "U1", "wikidata_qid": "Q1", "wikipedia_pageid": 10, "approx_tokens": 100, "text": "Some text for D1"}\n'
            '{"doc_id": "D2", "title": "T2", "url": "U2", "wikidata_qid": "Q2", "wikipedia_pageid": 20, "approx_tokens": 100, "text": "Some text for D2"}'
        )

        self.hidden_path = self.temp_path / "data" / "hidden" / "eval_hidden.jsonl"
        self.hidden_path.parent.mkdir(parents=True)
        self.hidden_path.write_text("{}")

        # Test paths for chunk rejection
        self.chunks_hidden = self.temp_path / "data" / "hidden" / "chunks.jsonl"

        os.environ["TIGERGRAPH_HOST"] = "http://mock"
        os.environ["TIGERGRAPH_GRAPH"] = "MockGraph"
        os.environ["TIGERGRAPH_SECRET"] = "mock_secret"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_missing_configuration(self):
        loader = TigerGraphLoader(host="", graphname="", secret="")
        connected = loader.connect()
        self.assertFalse(connected, "Should fail to connect without config")

    def test_hidden_data_rejection(self):
        loader = TigerGraphLoader()
        with self.assertRaises(ValueError) as context:
            loader.load_corpus(self.hidden_path)
        self.assertTrue("SECURITY ALERT" in str(context.exception))

        with self.assertRaises(ValueError):
            loader.load_corpus(self.corpus_path, chunks_path=self.chunks_hidden)

    def test_mock_loader_behavior(self):
        loader = TigerGraphLoader()
        stats = loader.load_corpus(self.corpus_path)

        # 2 Documents + 2 Entities + 2 Chunks = 6 vertices
        self.assertEqual(stats["vertices_loaded"], 6)

        # 2 ABOUT_ENTITY + 2 HAS_CHUNK = 4 edges
        self.assertEqual(stats["edges_loaded"], 4)

    def test_payload_generation(self):
        loader = TigerGraphLoader()

        captured_vertices = {}
        captured_edges = {}

        # Override upsert to capture payloads
        def mock_upsert(vertices, edges):
            captured_vertices.update(vertices)
            captured_edges.update(edges)

        loader._upsert_to_graph = mock_upsert
        loader.load_corpus(self.corpus_path)

        # Verify Document vertices
        self.assertIn("D1", captured_vertices["Document"])
        self.assertEqual(captured_vertices["Document"]["D1"]["title"], "T1")
        self.assertEqual(captured_vertices["Document"]["D1"]["url"], "U1")
        self.assertEqual(captured_vertices["Document"]["D1"]["wikipedia_pageid"], 10)

        # Verify Entity vertices
        self.assertIn("Q1", captured_vertices["Entity"])

        # Verify Chunk vertices
        self.assertIn("D1::chunk_0", captured_vertices["Chunk"])

        # Verify Edges
        self.assertIn(("D1", "D1::chunk_0", {}), captured_edges["HAS_CHUNK"])
        self.assertIn(("D1", "Q1", {}), captured_edges["ABOUT_ENTITY"])

    def test_no_hardcoded_credentials(self):
        loader = TigerGraphLoader()
        self.assertEqual(loader.host, "http://mock")
        self.assertEqual(loader.graphname, "MockGraph")
        self.assertEqual(loader.secret, "mock_secret")
