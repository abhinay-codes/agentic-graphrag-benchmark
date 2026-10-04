from dataclasses import dataclass, field
from typing import List, Dict, Any, Set

@dataclass
class AgentState:
    question_id: str
    question: str
    current_step: int = 0
    maximum_steps: int = 6

    candidate_evidence: List[Dict[str, Any]] = field(default_factory=list)
    candidate_provenance: List[Dict[str, Any]] = field(default_factory=list)
    collected_evidence: List[Dict[str, Any]] = field(default_factory=list)

    entities_discovered: List[str] = field(default_factory=list)
    documents_discovered: List[str] = field(default_factory=list)

    actions_taken: List[Dict[str, Any]] = field(default_factory=list)
    retrieval_history: List[Dict[str, Any]] = field(default_factory=list)
    evidence_evaluations: List[Dict[str, Any]] = field(default_factory=list)
    missing_information: str = "Initial search required."

    controller_input_tokens: int = 0
    controller_output_tokens: int = 0
    evaluator_input_tokens: int = 0
    evaluator_output_tokens: int = 0
    final_answer_input_tokens: int = 0
    final_answer_output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return (self.controller_input_tokens + self.controller_output_tokens +
                self.evaluator_input_tokens + self.evaluator_output_tokens +
                self.final_answer_input_tokens + self.final_answer_output_tokens)

    total_retrieval_duration_s: float = 0.0
    total_llm_duration_s: float = 0.0
    total_pipeline_duration_s: float = 0.0

    final_answer: str = ""
    stopping_reason: str = ""
    strategy_changes: List[str] = field(default_factory=list)

    def add_action(self, action: str, reason: str, parameters: dict, result: dict = None, status: str = "success"):
        self.actions_taken.append({
            "action": action,
            "reason": reason,
            "parameters": parameters,
            "result": result or {},
            "status": status
        })

    def update_controller_tokens(self, prompt_tokens: int, eval_tokens: int):
        self.controller_input_tokens += prompt_tokens
        self.controller_output_tokens += eval_tokens

    def update_evaluator_tokens(self, prompt_tokens: int, eval_tokens: int):
        self.evaluator_input_tokens += prompt_tokens
        self.evaluator_output_tokens += eval_tokens

    def update_final_answer_tokens(self, prompt_tokens: int, eval_tokens: int):
        self.final_answer_input_tokens += prompt_tokens
        self.final_answer_output_tokens += eval_tokens

    def _canonicalize_value(self, val: Any) -> Any:
        if isinstance(val, (int, float)):
            return str(val)
        if isinstance(val, str):
            return " ".join(val.lower().split())
        if isinstance(val, dict):
            return {k: self._canonicalize_value(v) for k, v in sorted(val.items())}
        if isinstance(val, list):
            return [self._canonicalize_value(v) for v in val]
        return val

    def has_repeated_action(self, action: str, parameters: dict) -> bool:
        canonical_params = self._canonicalize_value(parameters)

        # Prevent semantically equivalent duplicate calls
        retrieval_count = 0
        for act in self.actions_taken:
            if act["action"] == action:
                if self._canonicalize_value(act["parameters"]) == canonical_params:
                    return True
            if act["action"] in ["vector_search", "graph_expansion", "entity_linking", "document_retrieval", "multi_hop_reasoning"]:
                retrieval_count += 1

        # Protect against wasting the entire budget on retrievals
        if action in ["vector_search", "graph_expansion", "entity_linking", "document_retrieval", "multi_hop_reasoning"] and retrieval_count >= 3:
            return True

        return False

    def add_candidate_evidence(self, chunks: List[Dict[str, Any]], provenance: List[Dict[str, Any]]):
        if len(chunks) != len(provenance):
            raise ValueError(f"Chunk/provenance alignment error: {len(chunks)} chunks vs {len(provenance)} provenance records")

        existing_ids = {c["chunk_id"] for c in self.candidate_evidence if "chunk_id" in c}

        for i, c in enumerate(chunks):
            cid = c.get("chunk_id")
            if not cid or cid not in existing_ids:
                self.candidate_evidence.append(c)
                self.candidate_provenance.append(provenance[i])
                if cid:
                    existing_ids.add(cid)
