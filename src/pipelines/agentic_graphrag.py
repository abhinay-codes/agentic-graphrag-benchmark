import time
from typing import Dict, Any
from src.agent.state import AgentState
from src.agent.orchestrator import AgentOrchestrator

class AgenticGraphRAGPipeline:
    def __init__(self, llm_client=None):
        self.orchestrator = AgentOrchestrator(llm_client=llm_client)
        self.llm_client = llm_client

    def answer(self, question_id: str, question: str, maximum_steps: int = 6) -> Dict[str, Any]:
        start_time = time.time()

        state = AgentState(
            question_id=question_id,
            question=question,
            maximum_steps=maximum_steps
        )

        state = self.orchestrator.run(state)

        final_answer_prompt_tokens = 0
        final_answer_eval_tokens = 0
        gen_time = 0.0
        citations = []

        final_evidence = state.collected_evidence if state.collected_evidence else state.candidate_evidence
        state.collected_evidence = final_evidence

        if not final_evidence:
            final_answer = "Insufficient evidence to answer the question."
        else:
            context_parts = []
            for c in final_evidence:
                context_parts.append(
                    f"[Graph Source {c.get('chunk_id')}]\n"
                    f"Title: {c.get('title')}\n"
                    f"Document ID: {c.get('doc_id')}\n"
                    f"URL: {c.get('url')}\n"
                    f"Text:\n{c.get('text')}\n"
                )
                citations.append({
                    "doc_id": c.get("doc_id"),
                    "chunk_id": c.get("chunk_id"),
                    "title": c.get("title"),
                    "url": c.get("url")
                })

            context_str = "\n".join(context_parts)

            prompt = (
                "You are an Agentic GraphRAG assistant. Answer the user's question using ONLY the provided corpus evidence. "
                "Do not use outside knowledge. If the answer is not present in the context, explicitly state "
                "'Insufficient evidence to answer the question.'. "
                "When answering, cite the Document Titles you used.\n\n"
                f"Context:\n{context_str}\n\n"
                f"Question: {question}\n\n"
                "Answer:"
            )

            t0 = time.time()
            res = self.orchestrator.tools.llm_client.generate(prompt, temperature=0.0, options={"num_predict": 2048})
            gen_time = time.time() - t0

            final_answer = res.get("response", "").strip()
            final_answer_prompt_tokens = res.get("prompt_eval_count", 0)
            final_answer_eval_tokens = res.get("eval_count", 0)

        for act in state.actions_taken:
            if act.get("status") == "error":
                raise RuntimeError(f"Agentic pipeline failed due to tool error: {act.get('result', {}).get('error', 'Unknown')}")

        state.final_answer = final_answer
        state.total_llm_duration_s += gen_time
        state.total_pipeline_duration_s = time.time() - start_time

        total_toks = state.accumulated_total_tokens + final_answer_prompt_tokens + final_answer_eval_tokens

        trace = {
            "question_id": state.question_id,
            "steps": [],
            "tools_used": list(set([a["action"] for a in state.actions_taken])),
            "evidence_history": {
                "candidate_chunks": len(state.candidate_evidence),
                "selected_chunks": len(state.collected_evidence)
            },
            "strategy_changes": state.strategy_changes,
            "token_usage": {
                "controller_prompt_tokens": state.accumulated_prompt_tokens,
                "controller_eval_tokens": state.accumulated_eval_tokens,
                "final_answer_prompt_tokens": final_answer_prompt_tokens,
                "final_answer_eval_tokens": final_answer_eval_tokens,
                "total_tokens": total_toks
            },
            "timing": {
                "retrieval_s": state.total_retrieval_duration_s,
                "llm_s": state.total_llm_duration_s,
                "total_s": state.total_pipeline_duration_s
            },
            "final_evidence_chunks": state.retrieval_history,
            "stopping_reason": state.stopping_reason
        }

        for i, act in enumerate(state.actions_taken):
            trace["steps"].append({
                "step_number": i + 1,
                "action": act["action"],
                "reason": act["reason"],
                "parameters": act["parameters"],
                "result": act["result"],
                "status": act["status"]
            })

        return {
            "question_id": state.question_id,
            "answer": final_answer,
            "citations": citations,
            "trace": trace
        }
