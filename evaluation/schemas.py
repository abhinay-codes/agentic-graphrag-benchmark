from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class BenchmarkResult(BaseModel):
    question_id: str
    pipeline: str
    question: str
    answer: str
    latency_s: float
    prompt_tokens: int
    output_tokens: int
    total_tokens: int
    retrieved_chunks: int
    citations: List[Dict[str, Any]]
    error: Optional[str] = None
    status: str = "success"

    # GraphRAG specific
    graph_candidate_chunks: Optional[int] = None
    selected_chunks: Optional[int] = None
    graph_documents: Optional[int] = None
    graph_entities: Optional[int] = None
    graph_provenance: Optional[List[Dict[str, Any]]] = None

    # Agentic GraphRAG specific
    steps: Optional[List[Dict[str, Any]]] = None
    action_sequence: Optional[List[str]] = None
    tools_used: Optional[List[str]] = None
    strategy_changes: Optional[List[str]] = None
    controller_tokens: Optional[int] = None
    final_answer_tokens: Optional[int] = None
    stopping_reason: Optional[str] = None
