import pytest
from unittest.mock import patch, MagicMock
from evaluation.schemas import BenchmarkResult
from evaluation.metrics import (
    normalize_text, exact_match, numeric_match, answer_contains_gold,
    calculate_completeness, calculate_grounding
)


# ── Schema tests ──

def test_result_schema():
    res = BenchmarkResult(
        question_id="q1",
        pipeline="RAG",
        question="q",
        answer="a",
        latency_s=1.0,
        prompt_tokens=10,
        output_tokens=20,
        total_tokens=30,
        retrieved_chunks=5,
        citations=[]
    )
    assert res.question_id == "q1"
    assert res.pipeline == "RAG"
    assert res.status == "success"

def test_result_schema_error():
    res = BenchmarkResult(
        question_id="q1",
        pipeline="RAG",
        question="q",
        answer="",
        latency_s=0.5,
        prompt_tokens=0,
        output_tokens=0,
        total_tokens=0,
        retrieved_chunks=0,
        citations=[],
        error="some error",
        status="error"
    )
    assert res.error == "some error"
    assert res.status == "error"


# ── Accuracy metric tests ──

def test_normalized_answer_matching():
    assert normalize_text("Hello World!") == "hello world"
    assert exact_match("Hello World!", "hello world") == True
    assert exact_match("Hello World!", "hello") == False

def test_numeric_matching():
    assert numeric_match("I have 5 apples and 10 oranges", "10") == True
    assert numeric_match("I have 5 apples", "10") == False
    assert numeric_match("No numbers here", "10") == False

def test_answer_contains_gold():
    assert answer_contains_gold("The capital of France is Paris.", "Paris") == True
    assert answer_contains_gold("The capital of France is Paris.", "London") == False


# ── Grounding metric tests ──

def test_grounding_with_citations():
    citations = [
        {"doc_id": "d1", "chunk_id": "c1", "url": "http://example.com"},
        {"doc_id": "d2", "chunk_id": "c2", "url": "http://example2.com"},
    ]
    result = calculate_grounding(citations)
    assert result["has_citations"] == True
    assert result["citation_count"] == 2
    assert "d1" in result["citation_doc_ids"]

def test_grounding_without_citations():
    result = calculate_grounding([])
    assert result["has_citations"] == False
    assert result["citation_count"] == 0


# ── Import regression test ──

def test_benchmark_runner_imports_vectorragpipeline():
    """Regression: benchmark_runner must import VectorRAGPipeline, not RAGPipeline."""
    from evaluation import benchmark_runner
    # Verify the import resolved to VectorRAGPipeline
    from src.pipelines.rag import VectorRAGPipeline
    assert benchmark_runner.VectorRAGPipeline is VectorRAGPipeline


# ── Question loading + evaluator separation test ──

def test_question_loading_evaluator_separation():
    """Verify load_public_questions returns only question_id and question,
    not any evaluator-only fields like answer, gold_doc_ids, qtype."""
    from evaluation.benchmark_runner import BenchmarkRunner
    runner = BenchmarkRunner.__new__(BenchmarkRunner)  # skip __init__ to avoid pipeline construction
    runner.output_dir = "reports/phase10_public_benchmark"
    runner.results_file = "reports/phase10_public_benchmark/results_official_public.jsonl"
    runner.manifest_file = "reports/phase10_public_benchmark/manifest.json"
    questions = runner.load_public_questions()
    assert len(questions) == 100
    for q in questions:
        assert set(q.keys()) == {"question_id", "question"}
        # Evaluator-only fields must NOT be present
        assert "answer" not in q
        assert "gold_doc_ids" not in q
        assert "qtype" not in q
        assert "answer_named_in_question" not in q
        assert "guess_baseline" not in q
        assert "answer_verified" not in q


# ── Hidden-data path rejection test ──

def test_hidden_data_path_rejection():
    """Verify load_public_questions does NOT access data/hidden."""
    import os
    from evaluation.benchmark_runner import BenchmarkRunner
    runner = BenchmarkRunner.__new__(BenchmarkRunner)
    runner.output_dir = "reports/phase10_public_benchmark"
    runner.results_file = "reports/phase10_public_benchmark/results_official_public.jsonl"
    runner.manifest_file = "reports/phase10_public_benchmark/manifest.json"
    # The loader hardcodes data/public/eval_public.jsonl
    # Verify the path does not reference hidden
    import inspect
    source = inspect.getsource(runner.load_public_questions)
    assert "data/hidden" not in source
    assert "eval_hidden" not in source


# ── Resumability test ──

def test_resumability_skip_key(tmp_path):
    """Verify get_completed_keys correctly identifies (question_id, pipeline) tuples."""
    import json
    from evaluation.benchmark_runner import BenchmarkRunner
    runner = BenchmarkRunner.__new__(BenchmarkRunner)
    runner.output_dir = str(tmp_path)
    runner.results_file = str(tmp_path / "results_official_public.jsonl")
    runner.manifest_file = str(tmp_path / "manifest.json")

    # Write two sample records
    with open(runner.results_file, "w") as f:
        f.write(json.dumps({"question_id": "q1", "pipeline": "RAG", "status": "success"}) + "\n")
        f.write(json.dumps({"question_id": "q1", "pipeline": "GraphRAG", "status": "success"}) + "\n")

    completed = runner.get_completed_keys()
    assert ("q1", "RAG") in completed
    assert ("q1", "GraphRAG") in completed
    assert ("q1", "AgenticGraphRAG") not in completed


# ── Agentic trace preservation test ──

def test_agentic_trace_preservation():
    """Verify BenchmarkResult can store complete agentic traces."""
    steps = [
        {"step_number": 1, "action": "vector_search", "reason": "init", "parameters": {}, "result": {"chunks": [{"chunk_id": "c1"}]}, "status": "success"},
        {"step_number": 2, "action": "stop", "reason": "done", "parameters": {}, "result": {}, "status": "success"},
    ]
    res = BenchmarkResult(
        question_id="q1",
        pipeline="AgenticGraphRAG",
        question="test",
        answer="test answer",
        latency_s=5.0,
        prompt_tokens=100,
        output_tokens=50,
        total_tokens=150,
        retrieved_chunks=3,
        citations=[],
        steps=steps,
        action_sequence=["vector_search", "stop"],
        tools_used=["vector_search"],
        strategy_changes=[],
        controller_tokens=80,
        final_answer_tokens=70,
        stopping_reason="controller_stopped"
    )
    assert len(res.steps) == 2
    assert res.steps[0]["action"] == "vector_search"
    assert res.steps[0]["result"]["chunks"][0]["chunk_id"] == "c1"
    assert res.action_sequence == ["vector_search", "stop"]
    assert res.stopping_reason == "controller_stopped"
