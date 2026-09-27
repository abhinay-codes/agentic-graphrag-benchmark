from dataclasses import dataclass, field
from typing import List, Dict, Any, Set

@dataclass
class AgentState:
    question_id: str
    question: str
    current_step: int = 0
    maximum_steps: int = 6

    candidate_evidence: List[Dict[str, Any]] = field(default_factory=list)
    collected_evidence: List[Dict[str, Any]] = field(default_factory=list)

    entities_discovered: List[str] = field(default_factory=list)
    documents_discovered: List[str] = field(default_factory=list)

    actions_taken: List[Dict[str, Any]] = field(default_factory=list)
    retrieval_history: List[Dict[str, Any]] = field(default_factory=list)
    evidence_evaluations: List[Dict[str, Any]] = field(default_factory=list)
    missing_information: str = "Initial search required."

    accumulated_prompt_tokens: int = 0
    accumulated_eval_tokens: int = 0
    accumulated_total_tokens: int = 0

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

    def update_tokens(self, prompt_tokens: int, eval_tokens: int):
        self.accumulated_prompt_tokens += prompt_tokens
        self.accumulated_eval_tokens += eval_tokens
        self.accumulated_total_tokens += (prompt_tokens + eval_tokens)

    def has_repeated_action(self, action: str, parameters: dict) -> bool:
        # Prevent exact duplicate calls
        for act in self.actions_taken:
            if act["action"] == action and act["parameters"] == parameters:
                return True
        return False
