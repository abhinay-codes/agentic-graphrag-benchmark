import json
import unittest
import tempfile
from pathlib import Path
from src.ingestion.dataset_profiler import (
    check_path_safety,
    profile_corpus,
    profile_public_questions
)

class TestDatasetProfiler(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        self.corpus_path = self.temp_path / "corpus.jsonl"
        corpus_data = [
            '{"doc_id": "1", "text": "Hello world"}',
            '{"doc_id": "2", "text": "Test record two with more words"}',
            '{"doc_id": "2", "text": "Duplicate ID"}',
            '{"doc_id": "3", "text": ""}',
            '',
            '{"doc_id": "4", "text": "Malformed',
            '{"title": "No text field"}'
        ]
        self.corpus_path.write_text("\n".join(corpus_data))

        self.public_path = self.temp_path / "eval_public.jsonl"
        public_data = [
            '{"qid": "q1", "question": "What is it?", "qtype": "lookup", "answer": ["A"], "gold_doc_ids": ["1", "2"]}',
            '{"qid": "q2", "question": "When?", "qtype": "temporal", "answer": ["B"], "gold_doc_ids": ["2", "3"]}',
            '{"qid": "q3", "question": "How?", "qtype": "lookup", "answer": ["C"], "gold_doc_ids": ["1"]}',
            '{"qid": "q4", "question": "Malformed'
        ]
        self.public_path.write_text("\n".join(public_data))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_check_path_safety(self):
        with self.assertRaises(ValueError) as context:
            check_path_safety(Path("some/path/data/hidden/eval_hidden.jsonl"))
        self.assertTrue("SECURITY ALERT" in str(context.exception))

        with self.assertRaises(ValueError) as context:
            check_path_safety(Path("data/hidden/stuff.jsonl"))
        self.assertTrue("SECURITY ALERT" in str(context.exception))

        # Should not raise
        check_path_safety(Path("data/public/eval_public.jsonl"))

    def test_profile_corpus(self):
        stats = profile_corpus(self.corpus_path)
        self.assertEqual(stats["total_records"], 6)
        self.assertEqual(stats["valid_records"], 5)
        self.assertEqual(stats["malformed_records"], 1)
        self.assertEqual(stats["empty_records"], 1)
        self.assertEqual(stats["duplicate_ids_count"], 1)

        self.assertEqual(stats["fields_presence"]["doc_id"], 4)
        self.assertEqual(stats["fields_presence"]["text"], 4)
        self.assertEqual(stats["fields_presence"]["title"], 1)

        self.assertEqual(stats["text_length_statistics"]["count"], 3)
        self.assertEqual(stats["text_length_statistics"]["words"]["min"], 2)

    def test_profile_public_questions(self):
        stats = profile_public_questions(self.public_path)
        self.assertEqual(stats["total_records"], 4)
        self.assertEqual(stats["valid_records"], 3)
        self.assertEqual(stats["malformed_records"], 1)

        self.assertEqual(stats["question_types"]["lookup"], 2)
        self.assertEqual(stats["question_types"]["temporal"], 1)

        self.assertEqual(stats["unique_source_docs_referenced"], 3)

if __name__ == "__main__":
    unittest.main()
