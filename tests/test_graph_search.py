import pytest
from unittest.mock import MagicMock, patch
from src.retrieval.graph_search import GraphRetriever

@pytest.fixture
def mock_conn():
    conn = MagicMock()
    conn.graphname = 'TEST'
    return conn

@pytest.fixture
def retriever(mock_conn):
    return GraphRetriever(conn=mock_conn)

def test_empty_seeds(retriever):
    # 2. seed document extraction & 1. seed retrieval (empty case)
    res = retriever.get_graph_context([])
    assert res["chunks"] == []
    assert res["provenance"] == []
    assert res["trace"]["retrieved_chunk_count"] == 0

def test_tigergraph_traversal(retriever, mock_conn):
    # 3. TigerGraph traversal & 8. graph provenance
    mock_conn.runInterpretedQuery.return_value = [
        {"Entities": [
            {"v_id": "Q123", "attributes": {"Entities.@seed_docs": ["doc1"]}}
        ]},
        {"ExpandedDocs": [
            {"v_id": "doc1", "attributes": {"ExpandedDocs.title": "T1", "ExpandedDocs.@connected_entities": ["Q123"]}},
            {"v_id": "doc2", "attributes": {"ExpandedDocs.title": "T2", "ExpandedDocs.@connected_entities": ["Q123"]}}
        ]},
        {"Chunks": [
            {"v_id": "c1", "attributes": {"Chunks.text": "text1", "Chunks.chunk_index": 0, "Chunks.@parent_doc": "doc1"}},
            {"v_id": "c2", "attributes": {"Chunks.text": "text2", "Chunks.chunk_index": 0, "Chunks.@parent_doc": "doc2"}}
        ]}
    ]

    res = retriever.get_graph_context(["doc1"])
    chunks = res["chunks"]
    prov = res["provenance"]

    assert len(chunks) == 2
    doc_ids = [c["doc_id"] for c in chunks]
    assert "doc1" in doc_ids
    assert "doc2" in doc_ids

    assert len(prov) == 2
    # Verify provenance path
    for p in prov:
        if p["doc_id"] == "doc1":
            assert p["graph_path"] == "doc1 --HAS_CHUNK--> c1"
        else:
            assert p["graph_path"] == "doc1 --ABOUT_ENTITY--> Q123 <--ABOUT_ENTITY-- doc2 --HAS_CHUNK--> c2"

def test_deduplication_and_limits(retriever, mock_conn):
    # 4. entity deduplication, 5. document deduplication, 6. chunk deduplication, 7. evidence limits
    mock_conn.runInterpretedQuery.return_value = [
        {"Entities": [{"v_id": "Q1", "attributes": {"Entities.@seed_docs": ["doc1"]}}]},
        {"ExpandedDocs": [
            {"v_id": "doc1", "attributes": {"ExpandedDocs.title": "T1", "ExpandedDocs.@connected_entities": ["Q1"]}}
        ]},
        {"Chunks": [
            {"v_id": "c1", "attributes": {"Chunks.text": "text1", "Chunks.chunk_index": 0, "Chunks.@parent_doc": "doc1"}},
            {"v_id": "c1", "attributes": {"Chunks.text": "text1", "Chunks.chunk_index": 0, "Chunks.@parent_doc": "doc1"}},
            {"v_id": "c2", "attributes": {"Chunks.text": "text2", "Chunks.chunk_index": 1, "Chunks.@parent_doc": "doc1"}},
            {"v_id": "c3", "attributes": {"Chunks.text": "text3", "Chunks.chunk_index": 2, "Chunks.@parent_doc": "doc1"}}
        ]}
    ]

    # Test max_chunks_per_document = 2
    res = retriever.get_graph_context(["doc1"], max_chunks_per_document=2)
    chunks = res["chunks"]

    # Chunk c1 duplicated in mock, but should be deduped. Then max 2 applied.
    assert len(chunks) == 2
    chunk_ids = [c["chunk_id"] for c in chunks]
    assert "c1" in chunk_ids
    assert "c2" in chunk_ids
    assert "c3" not in chunk_ids

def test_tigergraph_failure(retriever, mock_conn):
    # 11. TigerGraph failure
    mock_conn.runInterpretedQuery.side_effect = Exception("TG Down")
    with pytest.raises(Exception, match="TG Down"):
        retriever.get_graph_context(["doc1"])

def test_malformed_graph_response(retriever, mock_conn):
    # 12. malformed graph response
    mock_conn.runInterpretedQuery.return_value = [{"WrongKey": []}]
    res = retriever.get_graph_context(["doc1"])
    assert res["chunks"] == []
    assert res["provenance"] == []
