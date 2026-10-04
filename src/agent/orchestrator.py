import json
import time
from typing import Dict, Any, List
from src.agent.state import AgentState
from src.agent.tools import AgentTools

class AgentOrchestrator:
    def __init__(self, llm_client=None):
        self.tools = AgentTools(llm_client=llm_client)

    def run(self, state: AgentState) -> AgentState:
        """Execute the agentic loop until a stopping condition is met."""

        system_prompt_template = (
            "You are an evidence-retrieval orchestrator. Your goal is to find sufficient corpus evidence to answer the user's question.\n"
            "You may ONLY choose registered actions.\n"
            "You must use only corpus evidence.\n"
            "You must not answer factual questions during planning.\n"
            "You must not invent entities, relationships, documents, or facts.\n"
            "You must consider the evidence already collected.\n"
            "You must consider missing information.\n"
            "Choose the next action that is most useful.\n"
            "If evidence is insufficient or missing_information is non-empty, do not stop. Investigate further using an appropriate retrieval tool. Only stop when the available evidence is sufficient to answer the question or when the maximum investigation steps have been reached.\n"
            "Do not repeat identical actions unnecessarily.\n\n"
            "Registered Actions:\n"
            "1. vector_search: Semantic retrieval (Input: 'query', 'top_k')\n"
            "2. graph_expansion: TigerGraph traversal (Input: 'seed_doc_ids' (list of strings))\n"
            "3. select_evidence: Local semantic re-ranking of candidate chunks (Input: 'final_top_k' - MUST be a single positive integer, e.g. 5. The candidate chunks are implicitly read from state. NEVER pass candidate chunks into final_top_k.)\n"
            "4. evaluate_evidence: Use LLM to check if collected evidence is sufficient (Input: none)\n"
            "5. stop: Stop execution (Input: 'reason')\n\n"
            "You must return ONLY a JSON object matching this schema exactly:\n"
            "{\n"
            '  "action": "vector_search | graph_expansion | select_evidence | evaluate_evidence | stop",\n'
            '  "reason": "short explanation",\n'
            '  "parameters": {}\n'
            "}\n"
        )

        while state.current_step < state.maximum_steps:
            state.current_step += 1

            cand_count = len(state.candidate_evidence)
            coll_count = len(state.collected_evidence)

            feedback = state.strategy_changes[-1] if state.strategy_changes else "None"
            summary = (
                f"Question: {state.question}\n"
                f"Current Step: {state.current_step}/{state.maximum_steps}\n"
                f"Candidate Evidence Chunks Available: {cand_count}\n"
                f"Collected/Selected Evidence Chunks: {coll_count}\n"
                f"Documents Discovered: {state.documents_discovered}\n"
                f"Missing Information: {state.missing_information}\n"
                f"Feedback: {feedback}\n"
                f"Past Actions: {json.dumps([a['action'] for a in state.actions_taken])}\n\n"
                "Next Action JSON:"
            )

            prompt = system_prompt_template + "\n" + summary
            t0 = time.time()
            res = self.tools.llm_client.generate(
                prompt,
                temperature=0.0,
                options={"num_predict": 2048},
            )
            llm_duration = time.time() - t0
            state.total_llm_duration_s += llm_duration
            state.update_controller_tokens(res.get("prompt_eval_count", 0), res.get("eval_count", 0))

            text_resp = res.get("response", "").strip()

            try:
                start = text_resp.find('{')
                end = text_resp.rfind('}') + 1
                if start >= 0 and end > start:
                    action_data = json.loads(text_resp[start:end])
                else:
                    raise ValueError("No JSON object")
            except Exception as e:
                state.strategy_changes.append(f"JSON Parse Error at step {state.current_step}. Attempting recovery.")
                if state.current_step >= state.maximum_steps:
                    state.stopping_reason = "malformed_controller_output"
                    break
                continue

            action = action_data.get("action")
            reason = action_data.get("reason", "")
            params = action_data.get("parameters", {})

            if state.has_repeated_action(action, params):
                state.strategy_changes.append(f"Blocked repeated action: {action} {params}")
                if state.current_step >= state.maximum_steps:
                    state.stopping_reason = "repeated_action_limit"
                    break
                continue

            tool_status = "success"
            tool_result = {}

            if action == "stop":
                # Deterministic guard against premature stopping
                is_insufficient = False
                if state.evidence_evaluations and state.evidence_evaluations[-1].get("status") != "sufficient":
                    is_insufficient = True
                elif state.missing_information and state.missing_information not in ["No issues identified.", ""]:
                    is_insufficient = True

                if is_insufficient and state.current_step < state.maximum_steps:
                    state.strategy_changes.append("Stop was blocked because evidence is insufficient. Investigate further before stopping.")
                    continue

                state.stopping_reason = reason or params.get("reason", "controller_stopped")
                state.add_action(action, reason, params, tool_result, tool_status)
                break

            elif action == "vector_search":
                q = params.get("query", state.question)
                if not q or not isinstance(q, str) or q.strip() == "":
                    q = state.question
                tk = params.get("top_k", 5)
                out = self.tools.vector_search(q, tk)
                tool_result = out
                if "error" in out:
                    tool_status = "error"
                    state.missing_information = f"vector_search failed: {out['error']}"
                else:
                    chunks_to_add = []
                    provs_to_add = []
                    for c in out.get("chunks", []):
                        chunks_to_add.append({
                            "chunk_id": c.get("chunk_id"),
                            "doc_id": c.get("doc_id"),
                            "title": c.get("title"),
                            "url": c.get("url"),
                            "text": c.get("text") or ""
                        })
                        provs_to_add.append({
                            "chunk_id": c.get("chunk_id"),
                            "doc_id": c.get("doc_id"),
                            "graph_path": "vector_search",
                            "seed_documents": [],
                            "entities_used": []
                        })
                        if c.get("doc_id") not in state.documents_discovered:
                            state.documents_discovered.append(c.get("doc_id"))
                    state.add_candidate_evidence(chunks_to_add, provs_to_add)
                state.total_retrieval_duration_s += out.get("duration_s", 0)

            elif action == "graph_expansion":
                seed_docs = params.get("seed_doc_ids", [])
                if not seed_docs and state.documents_discovered:
                    seed_docs = state.documents_discovered[:5]

                out = self.tools.graph_expansion(seed_docs)
                tool_result = out
                if "error" in out:
                    tool_status = "error"
                    state.missing_information = f"graph_expansion failed: {out['error']}"
                else:
                    state.add_candidate_evidence(out.get("chunks", []), out.get("provenance", []))
                    for d in out.get("documents", []):
                        if d not in state.documents_discovered:
                            state.documents_discovered.append(d)
                    for e in out.get("entities", []):
                        if e not in state.entities_discovered:
                            state.entities_discovered.append(e)
                state.total_retrieval_duration_s += out.get("duration_s", 0)

            elif action == "select_evidence":
                fk = params.get("final_top_k", 5)
                out = self.tools.select_evidence(state.question, state.candidate_evidence, state.retrieval_history, fk)
                tool_result = out
                if "error" in out:
                    tool_status = "error"
                    state.missing_information = f"select_evidence failed: {out['error']}"
                else:
                    state.collected_evidence = out.get("selected_chunks", [])
                    # We do NOT overwrite retrieval history. We just align it.
                    state.retrieval_history = out.get("selected_provenance", [])
                state.total_retrieval_duration_s += out.get("duration_s", 0)

            elif action == "evaluate_evidence":
                # evaluate_evidence must evaluate candidate_evidence if select hasn't been called.
                ev = state.collected_evidence if state.collected_evidence else state.candidate_evidence
                out = self.tools.evaluate_evidence(state.question, ev)
                tool_result = out
                if "error" in out:
                    tool_status = "error"
                    state.missing_information = f"evaluate_evidence failed: {out['error']}"
                else:
                    eval_data = out.get("evaluation", {})
                    state.evidence_evaluations.append(eval_data)
                    state.missing_information = eval_data.get("missing_information", "No issues identified.")

                    if eval_data.get("status") == "sufficient":
                        state.strategy_changes.append("Evidence is sufficient. Will stop next step.")

                    state.update_evaluator_tokens(out.get("prompt_eval_count", 0), out.get("eval_count", 0))
                state.total_llm_duration_s += out.get("duration_s", 0)

            else:
                tool_status = "error"
                tool_result = {"error": f"Invalid tool: {action}"}
                state.strategy_changes.append(f"Invalid tool selected: {action}")

            state.add_action(action, reason, params, tool_result, tool_status)

        if state.current_step >= state.maximum_steps and not state.stopping_reason:
            state.stopping_reason = "maximum_steps_reached"

        return state
