import os
import json
import pytest
import numpy as np
from src.retrieval.vector_search import VectorSearch
from src.llm.ollama_client import OllamaClient

@pytest.fixture
def temp_vector_store(tmp_path):
    # tmp_path is a pytest fixture that provides a temporary directory unique to the test invocation
    store_dir = tmp_path / "vector_store"
    return str(store_dir)

def test_vector_search_alignment(temp_vector_store):
    vs = VectorSearch(index_dir=temp_vector_store, embed_dim=4)

    # Add dummy vectors and metadata
    vectors = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.5, 0.5, 0.5, 0.5]
    ]
    metadata = [
        {"chunk_id": "c1", "text": "foo"},
        {"chunk_id": "c2", "text": "bar"},
        {"chunk_id": "c3", "text": "baz"}
    ]

    vs.add_chunks(vectors, metadata)
    assert vs.index.ntotal == 3
    assert len(vs.metadata) == 3

def test_vector_search_normalization_and_retrieval(temp_vector_store):
    vs = VectorSearch(index_dir=temp_vector_store, embed_dim=2)
    # Vectors with different magnitudes but same direction should be treated as identical due to cosine similarity
    vectors = [
        [10.0, 0.0], # Direction X
        [0.0, 5.0]   # Direction Y
    ]
    metadata = [{"id": 1}, {"id": 2}]

    vs.add_chunks(vectors, metadata)

    # Query vector in direction X, length 1
    query = [1.0, 0.0]
    results = vs.search(query, top_k=2)

    assert len(results) == 2
    top_doc, top_score = results[0]
    second_doc, second_score = results[1]

    # Score for perfectly aligned normalized vectors is 1.0
    assert top_doc["id"] == 1
    assert np.isclose(top_score, 1.0)

    # Orthogonal vectors have score 0.0
    assert second_doc["id"] == 2
    assert np.isclose(second_score, 0.0)

def test_vector_search_save_load(temp_vector_store):
    vs = VectorSearch(index_dir=temp_vector_store, embed_dim=3)
    vectors = [[1.0, 0.0, 0.0]]
    metadata = [{"chunk_id": "test_chunk", "doc_id": "test_doc", "text": "Save me"}]

    vs.add_chunks(vectors, metadata)
    vs.save(model_name="test-model", chunk_file="test_chunks.jsonl")

    # Verify files exist
    assert os.path.exists(os.path.join(temp_vector_store, "index.faiss"))
    assert os.path.exists(os.path.join(temp_vector_store, "metadata.jsonl"))
    assert os.path.exists(os.path.join(temp_vector_store, "manifest.json"))

    # Load into a new instance
    vs_new = VectorSearch(index_dir=temp_vector_store, embed_dim=3)
    assert vs_new.index.ntotal == 1
    assert len(vs_new.metadata) == 1
    assert vs_new.metadata[0]["chunk_id"] == "test_chunk"

    with open(os.path.join(temp_vector_store, "manifest.json"), "r") as f:
        manifest = json.load(f)
        assert manifest["embedding_model"] == "test-model"
        assert manifest["chunk_count"] == 1
        assert manifest["source_chunk_file"] == "test_chunks.jsonl"

def test_mismatched_alignment_raises_error(temp_vector_store):
    vs = VectorSearch(index_dir=temp_vector_store, embed_dim=2)
    vectors = [[1.0, 0.0]]
    metadata = [{"id": 1}, {"id": 2}] # Mismatch
    with pytest.raises(ValueError):
        vs.add_chunks(vectors, metadata)

def test_mismatched_dimension_raises_error(temp_vector_store):
    vs = VectorSearch(index_dir=temp_vector_store, embed_dim=2)
    vectors = [[1.0, 0.0, 0.0]] # Mismatch
    metadata = [{"id": 1}]
    with pytest.raises(ValueError):
        vs.add_chunks(vectors, metadata)

def test_ollama_client_embed_batching(requests_mock):
    client = OllamaClient("http://mock-ollama:11434")

    # Mock the requests.post behavior for embedding
    mock_response = {
        "embeddings": [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]
        ]
    }
    requests_mock.post("http://mock-ollama:11434/api/embed", json=mock_response)

    embeddings = client.embed(["test1", "test2"])

    assert len(embeddings) == 2
    assert embeddings[0] == [0.1, 0.2, 0.3]
    assert embeddings[1] == [0.4, 0.5, 0.6]
    # Check that it sent a batch
    assert requests_mock.last_request.json()["input"] == ["test1", "test2"]
