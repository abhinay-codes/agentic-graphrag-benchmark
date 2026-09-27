import pytest
from unittest.mock import MagicMock, patch
from src.pipelines.rag import VectorRAGPipeline

@pytest.fixture
def mock_dependencies():
    with patch('src.pipelines.rag.OllamaClient') as MockClient, \
         patch('src.pipelines.rag.VectorSearch') as MockSearch, \
         patch('os.path.exists') as mock_exists:

        # We need exists to return True for the index directory, but False for manifest to use defaults
        mock_exists.side_effect = lambda path: "manifest.json" not in path

        mock_client_instance = MockClient.return_value
        mock_client_instance.embed.return_value = [[0.1] * 768]
        mock_client_instance.generate.return_value = {
            "response": "Based on [Source 1], the answer is 42.",
            "prompt_eval_count": 50,
            "eval_count": 10
        }

        mock_search_instance = MockSearch.return_value
        # return tuple (metadata, score)
        mock_search_instance.search.return_value = [
            (
                {
                    "chunk_id": "chunk_001",
                    "doc_id": "doc_001",
                    "title": "Test Title",
                    "url": "http://test",
                    "text": "The meaning of life is 42."
                },
                0.95
            )
        ]

        yield mock_client_instance, mock_search_instance

def test_rag_pipeline_flow(mock_dependencies):
    mock_client, mock_search = mock_dependencies

    pipeline = VectorRAGPipeline(index_dir="dummy/dir", top_k=1)

    result = pipeline.answer("q_001", "What is the meaning of life?")

    # Verify inputs flow to tools
    mock_client.embed.assert_called_once_with("What is the meaning of life?", model="nomic-embed-text")
    mock_search.search.assert_called_once_with([0.1]*768, top_k=1)

    # Verify generation call includes constructed prompt
    args, kwargs = mock_client.generate.call_args
    prompt = kwargs["prompt"]
    assert "What is the meaning of life?" in prompt
    assert "[Source 1]" in prompt
    assert "The meaning of life is 42." in prompt

    # Verify structured output
    assert result["question_id"] == "q_001"
    assert result["answer"] == "Based on [Source 1], the answer is 42."
    assert result["retrieval_count"] == 1
    assert len(result["citations"]) == 1

    citation = result["citations"][0]
    assert citation["source_id"] == 1
    assert citation["chunk_id"] == "chunk_001"
    assert citation["doc_id"] == "doc_001"

    # Verify Trace
    trace = result["trace"]
    assert trace["prompt_eval_count"] == 50
    assert trace["eval_count"] == 10
    assert trace["total_tokens"] == 60
    assert "total_pipeline_duration_s" in trace

@patch('os.path.exists')
def test_missing_index_fails(mock_exists):
    mock_exists.return_value = False
    with pytest.raises(RuntimeError) as exc:
        VectorRAGPipeline(index_dir="does/not/exist")
    assert "Vector index not found" in str(exc.value)
