import os
import json
import pytest
from evaluation.benchmark_runner import BenchmarkRunner

def test_resume_semantics(tmp_path):
    output_dir = str(tmp_path / "reports")
    runner = BenchmarkRunner(output_dir=output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Create fake results_official_public.jsonl with various statuses
    records = [
        {"question_id": "q1", "pipeline": "RAG", "status": "success"},
        {"question_id": "q1", "pipeline": "GraphRAG", "status": "error"},
        {"question_id": "q2", "pipeline": "AgenticGraphRAG", "status": "success"},
        {"question_id": "q3", "pipeline": "RAG", "status": "error/telemetry_unavailable"},
        {"question_id": "q4", "pipeline": "GraphRAG"} # Missing status
    ]

    with open(runner.results_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    completed = runner.get_completed_keys()

    # A. success record -> key is completed
    assert ("q1", "RAG") in completed
    assert ("q2", "AgenticGraphRAG") in completed

    # B. error record -> key is NOT completed
    assert ("q1", "GraphRAG") not in completed
    assert ("q3", "RAG") not in completed

    # C. missing status -> key is NOT completed
    assert ("q4", "GraphRAG") not in completed

    # Exactly 2 keys should be completed
    assert len(completed) == 2

def test_resume_append_behavior(tmp_path):
    output_dir = str(tmp_path / "reports")
    runner = BenchmarkRunner(output_dir=output_dir)
    os.makedirs(output_dir, exist_ok=True)

    with open(runner.results_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"question_id": "q1", "pipeline": "RAG", "status": "success", "prompt_tokens": 10, "output_tokens": 5, "total_tokens": 15}) + "\n")
        f.write(json.dumps({"question_id": "q1", "pipeline": "GraphRAG", "status": "error"}) + "\n")

    # Mock load_public_questions and _execute_pipeline
    import unittest.mock as mock

    runner.load_public_questions = mock.Mock(return_value=[
        {"question_id": "q1", "question": "test q1"}
    ])

    from evaluation.schemas import BenchmarkResult
    def mock_exec_side_effect(p_name, q_id, q_text):
        return BenchmarkResult(
            question_id=q_id,
            pipeline=p_name,
            question=q_text,
            answer="Fixed",
            latency_s=1.0,
            prompt_tokens=10,
            output_tokens=10,
            total_tokens=20,
            retrieved_chunks=1,
            citations=[],
            status="success"
        )

    with mock.patch.object(runner, "_execute_pipeline", side_effect=mock_exec_side_effect) as mock_exec, \
         mock.patch.object(runner, "validate_benchmark") as mock_validate:
        runner.run_benchmark(overwrite=False)

        assert mock_exec.call_count == 2
        calls = [c[0][0] for c in mock_exec.call_args_list]
        assert "GraphRAG" in calls
        assert "AgenticGraphRAG" in calls
        assert "RAG" not in calls

    # Ensure append behavior worked and preserved original lines
    lines = []
    with open(runner.results_file, "r", encoding="utf-8") as f:
        for line in f:
            lines.append(json.loads(line))

    assert len(lines) == 4 # 2 original + 2 new
    assert lines[0]["status"] == "success"
    assert lines[0]["pipeline"] == "RAG"
    assert lines[1]["status"] == "error"
    assert lines[1]["pipeline"] == "GraphRAG"
    assert lines[2]["status"] == "success"
    assert lines[2]["pipeline"] == "GraphRAG"
    assert lines[3]["status"] == "success"
    assert lines[3]["pipeline"] == "AgenticGraphRAG"
