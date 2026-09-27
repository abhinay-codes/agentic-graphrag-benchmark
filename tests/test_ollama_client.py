import pytest
from unittest.mock import patch, MagicMock
from src.llm.ollama_client import OllamaClient

@patch("requests.post")
def test_ollama_generate_telemetry(mock_post):
    # Mocking the JSON response from Ollama API
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "response": "Hello World!",
        "prompt_eval_count": 15,
        "eval_count": 30,
        "total_duration": 450000000
    }
    mock_post.return_value = mock_response

    client = OllamaClient()
    result = client.generate("Test prompt")

    assert isinstance(result, dict)
    assert result["response"] == "Hello World!"
    assert result["prompt_eval_count"] == 15
    assert result["eval_count"] == 30
    assert result["total_duration_ns"] == 450000000
