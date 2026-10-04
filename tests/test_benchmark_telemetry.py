from unittest.mock import MagicMock, patch
from evaluation.benchmark_runner import BenchmarkRunner

def test_graphrag_benchmark_schema():
    runner = BenchmarkRunner()
    
    mock_pipeline = MagicMock()
    mock_pipeline.answer.return_value = {
        "answer": "Test Answer",
        "citations": [{"doc_id": "d1", "chunk_id": "c1", "title": "t1", "url": "u1"}],
        "trace": {
            "prompt_eval_count": 10,
            "eval_count": 5,
            "retrieved_chunk_count": 20,
            "graph_candidate_chunk_count": 15,
            "selected_chunk_count": 1,
            "related_document_count": 4,
            "entity_count": 3,
            "graph_provenance": [{"foo": "bar"}]
        }
    }
    
    with patch.object(runner, '_get_pipeline', return_value=mock_pipeline):
        res = runner._execute_pipeline("GraphRAG", "q1", "What is testing?")
        
        assert res.status == "success"
        assert res.prompt_tokens == 10
        assert res.output_tokens == 5
        assert res.total_tokens == 15
        assert res.retrieved_chunks == 20
        assert res.graph_candidate_chunks == 15
        assert res.selected_chunks == 1
        assert res.graph_documents == 4
        assert res.graph_entities == 3
        assert res.graph_provenance == [{"foo": "bar"}]


def test_agentic_benchmark_schema():
    runner = BenchmarkRunner()
    
    mock_pipeline = MagicMock()
    mock_pipeline.answer.return_value = {
        "answer": "Agentic Answer",
        "citations": [{"doc_id": "d2", "chunk_id": "c2"}],
        "trace": {
            "graph_candidate_chunk_count": 18,
            "evidence_history": {
                "selected_chunks": 2
            },
            "token_usage": {
                "controller_input_tokens": 100,
                "controller_output_tokens": 50,
                "evaluator_input_tokens": 40,
                "evaluator_output_tokens": 20,
                "final_answer_input_tokens": 200,
                "final_answer_output_tokens": 100,
                "total_tokens": 510
            }
        }
    }
    
    with patch.object(runner, '_get_pipeline', return_value=mock_pipeline):
        res = runner._execute_pipeline("AgenticGraphRAG", "q2", "How does it work?")
        
        assert res.status == "success"
        assert res.prompt_tokens == 340
        assert res.output_tokens == 170
        assert res.total_tokens == 510
        assert res.graph_candidate_chunks == 18
        assert res.selected_chunks == 2
        assert res.controller_tokens == 150
        assert res.final_answer_tokens == 300
