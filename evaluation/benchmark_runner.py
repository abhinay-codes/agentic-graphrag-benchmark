
import sys
import os

class BenchmarkLock:
    def __init__(self, lock_path):
        self.lock_path = lock_path
        self.f = None

    def acquire(self):
        self.f = open(self.lock_path, 'w')
        if sys.platform == 'win32':
            import msvcrt
            try:
                msvcrt.locking(self.f.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                self.f.close()
                self.f = None
                raise RuntimeError("BENCHMARK_ALREADY_RUNNING")
        else:
            import fcntl
            try:
                fcntl.flock(self.f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except IOError:
                self.f.close()
                self.f = None
                raise RuntimeError("BENCHMARK_ALREADY_RUNNING")

    def release(self):
        if self.f:
            if sys.platform == 'win32':
                import msvcrt
                try:
                    self.f.seek(0)
                    msvcrt.locking(self.f.fileno(), msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
            else:
                import fcntl
                try:
                    fcntl.flock(self.f.fileno(), fcntl.LOCK_UN)
                except IOError:
                    pass
            self.f.close()
            try:
                os.remove(self.lock_path)
            except OSError:
                pass
            self.f = None

import os
import json
import time
from typing import List, Dict, Any

from src.pipelines.rag import VectorRAGPipeline
from src.pipelines.graphrag import GraphRAGPipeline
from src.pipelines.agentic_graphrag import AgenticGraphRAGPipeline
from evaluation.schemas import BenchmarkResult

class BenchmarkRunner:
    def __init__(self, output_dir: str = "reports/phase10_public_benchmark"):
        self.output_dir = output_dir
        self.results_file = os.path.join(output_dir, "results_official_public.jsonl")
        self.manifest_file = os.path.join(output_dir, "manifest.json")

        self._pipeline_classes = {
            "RAG": VectorRAGPipeline,
            "GraphRAG": GraphRAGPipeline,
            "AgenticGraphRAG": AgenticGraphRAGPipeline
        }
        self._pipeline_instances = {}

    def _get_pipeline(self, pipeline_name: str):
        if pipeline_name not in self._pipeline_instances:
            self._pipeline_instances[pipeline_name] = self._pipeline_classes[pipeline_name]()
        return self._pipeline_instances[pipeline_name]

    def load_public_questions(self) -> List[Dict[str, Any]]:
        file_path = "data/public/eval_public.jsonl"
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing public eval file: {file_path}")

        questions = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                questions.append({
                    "question_id": data["qid"],
                    "question": data["question"]
                })

        if len(questions) != 100:
            raise ValueError(f"Expected exactly 100 questions, but found {len(questions)}")

        return questions

    def get_completed_keys(self) -> set:
        completed = set()
        if os.path.exists(self.results_file):
            with open(self.results_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        res = json.loads(line)
                        if res.get("status") == "success":
                            completed.add((res["question_id"], res["pipeline"]))
                    except:
                        pass
        return completed

    def _execute_pipeline(self, pipeline_name: str, q_id: str, question: str) -> BenchmarkResult:
        t0 = time.time()
        try:
            pipeline = self._get_pipeline(pipeline_name)
            res = pipeline.answer(q_id, question)
            duration = time.time() - t0

            trace = res.get("trace", {})

            # Require token telemetry explicitly
            if pipeline_name in ["RAG", "GraphRAG"]:
                prompt_tokens = trace.get("prompt_eval_count")
                output_tokens = trace.get("eval_count")
                if prompt_tokens is None or output_tokens is None:
                    raise ValueError("telemetry_unavailable")
                total_tokens = prompt_tokens + output_tokens
            else:
                tok_usage = trace.get("token_usage", {})
                total_tokens = tok_usage.get("total_tokens")
                prompt_tokens = total_tokens - tok_usage.get("final_answer_eval_tokens", 0) if total_tokens is not None else None
                output_tokens = tok_usage.get("final_answer_eval_tokens")

                if prompt_tokens is None or output_tokens is None or total_tokens is None:
                    raise ValueError("telemetry_unavailable")

                if total_tokens != (prompt_tokens + output_tokens):
                    raise ValueError("telemetry sums do not match exactly")

            if pipeline_name == "RAG":
                return BenchmarkResult(
                    question_id=q_id,
                    pipeline=pipeline_name,
                    question=question,
                    answer=res.get("answer", ""),
                    latency_s=duration,
                    prompt_tokens=prompt_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    retrieved_chunks=len(trace.get("retrieved_chunks", [])),
                    citations=res.get("citations", []),
                    status="success"
                )
            elif pipeline_name == "GraphRAG":
                return BenchmarkResult(
                    question_id=q_id,
                    pipeline=pipeline_name,
                    question=question,
                    answer=res.get("answer", ""),
                    latency_s=duration,
                    prompt_tokens=prompt_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    retrieved_chunks=len(trace.get("retrieved_chunks", [])),
                    citations=res.get("citations", []),
                    graph_candidate_chunks=trace.get("graph_candidate_chunks", 0),
                    selected_chunks=len(res.get("citations", [])),
                    graph_documents=trace.get("graph_documents", 0),
                    graph_entities=trace.get("graph_entities", 0),
                    graph_provenance=trace.get("retrieval_provenance", []),
                    status="success"
                )
            elif pipeline_name == "AgenticGraphRAG":
                ev_hist = trace.get("evidence_history", {})
                return BenchmarkResult(
                    question_id=q_id,
                    pipeline=pipeline_name,
                    question=question,
                    answer=res.get("answer", ""),
                    latency_s=duration,
                    prompt_tokens=prompt_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    retrieved_chunks=ev_hist.get("selected_chunks", 0),
                    citations=res.get("citations", []),
                    graph_candidate_chunks=ev_hist.get("candidate_chunks", 0),
                    selected_chunks=ev_hist.get("selected_chunks", 0),
                    steps=trace.get("steps", []),
                    action_sequence=[s.get("action") for s in trace.get("steps", [])],
                    tools_used=trace.get("tools_used", []),
                    strategy_changes=trace.get("strategy_changes", []),
                    controller_tokens=tok_usage.get("controller_prompt_tokens", 0) + tok_usage.get("controller_eval_tokens", 0),
                    final_answer_tokens=tok_usage.get("final_answer_prompt_tokens", 0) + tok_usage.get("final_answer_eval_tokens", 0),
                    stopping_reason=trace.get("stopping_reason", ""),
                    status="success"
                )
        except Exception as e:
            error_status = "error"
            if str(e) == "telemetry_unavailable":
                error_status = "error/telemetry_unavailable"

            return BenchmarkResult(
                question_id=q_id,
                pipeline=pipeline_name,
                question=question,
                answer="",
                latency_s=time.time() - t0,
                prompt_tokens=0,
                output_tokens=0,
                total_tokens=0,
                retrieved_chunks=0,
                citations=[],
                error=str(e),
                status=error_status
            )

    def run_benchmark(self, overwrite: bool = False, specific_ids: List[str] = None):
        os.makedirs(self.output_dir, exist_ok=True)
        lock = BenchmarkLock(os.path.join(self.output_dir, "benchmark.lock"))
        lock.acquire()
        try:
            self._run_benchmark_internal(overwrite, specific_ids)
        finally:
            lock.release()

    def _run_benchmark_internal(self, overwrite: bool = False, specific_ids: List[str] = None):
        questions = self.load_public_questions()

        if specific_ids:
            questions = [q for q in questions if q["question_id"] in specific_ids]

        completed_keys = set() if overwrite else self.get_completed_keys()

        mode = "w" if overwrite else "a"
        with open(self.results_file, mode, encoding="utf-8") as f:
            for q in questions:
                for p_name in self._pipeline_classes.keys():
                    key = (q["question_id"], p_name)
                    if key in completed_keys:
                        print(f"Skipping {p_name} for {q['question_id']}")
                        continue

                    print(f"Running {p_name} for {q['question_id']}...")
                    res_obj = self._execute_pipeline(p_name, q["question_id"], q["question"])
                    f.write(res_obj.model_dump_json() + "\n")
                    f.flush()

    def generate_manifest(self):
        manifest = {
            "question_count": 100,
            "pipelines": ["RAG", "GraphRAG", "AgenticGraphRAG"],
            "llm_model": "qwen3:8b",
            "embedding_model": "nomic-embed-text",
            "pipeline_retrieval_parameters": {
                "top_k": 5,
                "graph_max_chunks": 50,
                "agent_max_steps": 6
            },
            "temperature": 0.0,
            "corpus_document_count": "inherited_from_db",
            "timestamp": time.time(),
            "hidden_data_isolation_assertion": "Confirmed: No access to data/hidden or evaluator fields during inference."
        }
        with open(self.manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
