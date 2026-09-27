import pytest
from unittest.mock import patch, MagicMock

def test_benchmark_runner_lazy_init():
    """Verify BenchmarkRunner initializes without creating pipeline instances."""
    from evaluation.benchmark_runner import BenchmarkRunner
    # We mock the classes so faiss/TG aren't imported or used
    with patch('evaluation.benchmark_runner.VectorRAGPipeline') as mock_rag:
        with patch('evaluation.benchmark_runner.GraphRAGPipeline') as mock_gr:
            with patch('evaluation.benchmark_runner.AgenticGraphRAGPipeline') as mock_ag:
                runner = BenchmarkRunner()
                assert len(runner._pipeline_instances) == 0
                mock_rag.assert_not_called()
                mock_gr.assert_not_called()
                mock_ag.assert_not_called()

def test_graphrag_init_failure_caught():
    """Verify that a pipeline initialization failure (like TG connection) becomes an explicit error result, not a crash."""
    from evaluation.benchmark_runner import BenchmarkRunner

    with patch('evaluation.benchmark_runner.GraphRAGPipeline', side_effect=Exception("TigerGraph Connection Refused")):
        runner = BenchmarkRunner()
        res = runner._execute_pipeline("GraphRAG", "q1", "What is X?")

        assert res.status == "error"
        assert "TigerGraph Connection Refused" in res.error
        assert res.total_tokens == 0
        assert res.answer == ""

def test_jsonl_writer_validity(tmp_path):
    """Verify the JSONL writer outputs valid JSON per line with correct newlines."""
    import json
    from evaluation.benchmark_runner import BenchmarkRunner

    runner = BenchmarkRunner(output_dir=str(tmp_path))

    mock_pipeline = MagicMock()
    mock_pipeline.answer.return_value = {
        "answer": "Test answer",
        "trace": {"prompt_eval_count": 10, "eval_count": 5}
    }

    runner._pipeline_classes = {"RAG": MagicMock(return_value=mock_pipeline)}

    # Run a mock benchmark over 2 questions
    with patch.object(runner, 'load_public_questions', return_value=[
        {"question_id": "q1", "question": "Q1"},
        {"question_id": "q2", "question": "Q2"}
    ]):
        runner.run_benchmark(overwrite=True)

    with open(runner.results_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Check physical newlines
    assert "\n" in content
    assert "\\n" not in content[:50] # ensure no double-escaped literal \n at end of line (it would be at end of json)

    # Check valid JSON
    lines = content.strip().split("\n")
    assert len(lines) == 2
    for line in lines:
        data = json.loads(line)
        assert data["status"] == "success"
        assert data["total_tokens"] == 15

def test_manifest_model_authoritative(tmp_path):
    """Verify the manifest explicitly uses qwen3:8b."""
    import json
    from evaluation.benchmark_runner import BenchmarkRunner

    runner = BenchmarkRunner(output_dir=str(tmp_path))
    runner.generate_manifest()

    with open(runner.manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["llm_model"] == "qwen3:8b"
