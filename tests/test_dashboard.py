import os
from dashboard.app import process_results, RESULTS_FILE

def test_dashboard_data_source():
    assert "phase11_public_benchmark/results_public_100.jsonl" in RESULTS_FILE

def test_dashboard_unique_pairs():
    stats = process_results()
    assert stats["unique_successes"] == 300

def test_dashboard_no_hidden():
    with open("dashboard/app.py", "r", encoding="utf-8") as f:
        content = f.read()
    assert "eval_hidden.jsonl" not in content

def test_api_live_query_parsing():
    with open("dashboard/app.py", "r", encoding="utf-8") as f:
        code = f.read()
    assert "/api/live-query" in code
    assert "evaluations = {}" in code
    assert "VectorRAGPipeline(llm_client=llm)" in code

def test_configuration_environment_variables():
    with open("dashboard/app.py", "r", encoding="utf-8") as f:
        code = f.read()
    assert "TIGERGRAPH_SECRET" not in code

def test_no_cot_leakage():
    with open("dashboard/app.py", "r", encoding="utf-8") as f:
        code = f.read()
    assert "res.answer" in code
