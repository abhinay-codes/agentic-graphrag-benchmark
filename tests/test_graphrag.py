import pytest
from unittest.mock import MagicMock, patch
from src.pipelines.graphrag import GraphRAGPipeline

@pytest.fixture
def mock_pipeline():
    with patch("src.pipelines.graphrag.VectorSearch") as MockVS, \
         patch("src.pipelines.graphrag.GraphRetriever") as MockGR, \
         patch("src.pipelines.graphrag.OllamaClient") as MockLLM:

        pipeline = GraphRAGPipeline()
        pipeline.vector_search = MockVS.return_value
        pipeline.graph_search = MockGR.return_value
        pipeline.llm = MockLLM.return_value
        yield pipeline

def test_insufficient_graph_evidence(mock_pipeline):
    mock_pipeline.llm.embed.return_value = [[0.1] * 768]
    mock_pipeline.vector_search.search.return_value = [({"doc_id": "doc1"}, 0.9)]
    mock_pipeline.graph_search.get_graph_context.return_value = {"chunks": [], "provenance": [], "trace": {}}

    res = mock_pipeline.answer("q1", "What is X?")

    assert res["answer"] == "Insufficient evidence to answer the question."
    assert res["citations"] == []
    mock_pipeline.llm.generate.assert_not_called()

def test_evidence_selection_ranking_and_reduction(mock_pipeline):
    # Mock query embedding
    q_emb = [[1.0, 0.0, 0.0]]

    # 10 chunks
    graph_chunks = []
    graph_provenance = []
    for i in range(10):
        graph_chunks.append({
            "chunk_id": f"c{i}",
            "doc_id": f"doc{i}",
            "title": f"Title {i}",
            "url": f"url{i}",
            "text": f"text {i}",
            "chunk_index": i
        })
        graph_provenance.append({
            "chunk_id": f"c{i}",
            "doc_id": f"doc{i}",
            "graph_path": f"path {i}",
            "seed_documents": ["seed1"],
            "entities_used": ["e1"]
        })

    # We want chunk 5 to be the best (score 1.0), chunk 2 second (score 0.8), others 0.0
    def mock_embed(texts):
        if isinstance(texts, str):
            return q_emb
        if "What is X?" in texts: # in case it receives question
            return q_emb

        # for chunk texts
        embs = []
        for t in texts:
            if t == "text 5":
                embs.append([1.0, 0.0, 0.0])
            elif t == "text 2":
                embs.append([0.8, 0.0, 0.0])
            else:
                embs.append([0.0, 1.0, 0.0])
        return embs

    mock_pipeline.llm.embed.side_effect = mock_embed
    mock_pipeline.vector_search.search.return_value = [({"doc_id": "doc1"}, 0.9)]
    mock_pipeline.graph_search.get_graph_context.return_value = {
        "chunks": graph_chunks,
        "provenance": graph_provenance,
        "trace": {"entity_count": 1, "related_document_count": 10}
    }

    mock_pipeline.llm.generate.return_value = {"response": "The answer is X.", "prompt_eval_count": 10, "eval_count": 5}

    res = mock_pipeline.answer("q1", "What is X?", final_top_k=5)

    # 1. Candidate reduction (10 -> 5)
    assert len(res["citations"]) == 5

    # 2. Ranking check
    assert res["citations"][0]["chunk_id"] == "c5"
    assert res["citations"][1]["chunk_id"] == "c2"

    # 3. Provenance preservation
    trace = res["trace"]
    assert len(trace["graph_provenance"]) == 5
    assert trace["graph_provenance"][0]["chunk_id"] == "c5"
    assert trace["graph_provenance"][0]["graph_path"] == "path 5"
    assert trace["graph_provenance"][0]["seed_documents"] == ["seed1"]

    # LLM context verification
    prompt_sent = mock_pipeline.llm.generate.call_args[0][0]
    assert "[Graph Source c5]" in prompt_sent
    assert "[Graph Source c2]" in prompt_sent
    # Ensure c9 which was score 0 is not in context if we only took 5 (some 0 score will be there, but we only have 2 with positive score)
    # Wait, 3 others with score 0.0 will be included to make 5.

    # Verify trace metrics
    assert trace["graph_candidate_chunk_count"] == 10
    assert trace["selected_chunk_count"] == 5
    assert trace["final_top_k"] == 5
    assert len(trace["selected_chunk_ids"]) == 5
    assert "c5" in trace["selected_chunk_ids"]
    assert trace["selected_chunk_scores"][0] == 1.0
    assert trace["selected_chunk_scores"][1] == 0.8

def test_evidence_selection_edge_cases(mock_pipeline):
    # Testing fewer candidates than final_top_k
    q_emb = [[1.0, 0.0, 0.0]]

    graph_chunks = [{
        "chunk_id": "c1",
        "doc_id": "doc1",
        "title": "T1",
        "url": "U1",
        "text": "text 1",
        "chunk_index": 0
    }]
    graph_provenance = [{
        "chunk_id": "c1",
        "doc_id": "doc1",
        "graph_path": "path 1",
        "seed_documents": [],
        "entities_used": []
    }]

    mock_pipeline.llm.embed.return_value = q_emb
    mock_pipeline.vector_search.search.return_value = [({"doc_id": "doc1"}, 0.9)]
    mock_pipeline.graph_search.get_graph_context.return_value = {
        "chunks": graph_chunks,
        "provenance": graph_provenance,
        "trace": {}
    }

    mock_pipeline.llm.generate.return_value = {"response": "The answer is X.", "prompt_eval_count": 10, "eval_count": 5}

    res = mock_pipeline.answer("q2", "What is X?", final_top_k=5)

    assert len(res["citations"]) == 1
    assert res["citations"][0]["chunk_id"] == "c1"

    trace = res["trace"]
    assert trace["graph_candidate_chunk_count"] == 1
    assert trace["selected_chunk_count"] == 1
