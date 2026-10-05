import os
import sys
import json
import time

sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/..'))
from evaluation.benchmark_runner import BenchmarkRunner

class HiddenBenchmarkRunner(BenchmarkRunner):
    def __init__(self, output_dir: str = "reports/hidden_benchmark"):
        # Initialize parent
        super().__init__(output_dir)
        # Override paths for hidden benchmark
        self.results_file = os.path.join(output_dir, "results_hidden.jsonl")
        self.manifest_file = os.path.join(output_dir, "manifest_hidden.json")

    def load_public_questions(self):
        """
        Overrides the base method to load hidden questions instead of public ones.
        Keeps the same method name so the base class loop and validation naturally pick it up.
        """
        file_path = "data/hidden/eval_hidden.jsonl"
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing hidden eval file: {file_path}")

        questions = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                questions.append({
                    "question_id": data["qid"],
                    "question": data["question"]
                })

        if len(questions) != 50:
            raise ValueError(f"Expected exactly 50 hidden questions, but found {len(questions)}")

        return questions

    def generate_manifest(self):
        manifest = {
            "question_count": 50,
            "pipelines": ["RAG", "GraphRAG", "AgenticGraphRAG"],
            "llm_model": "qwen3:8b",
            "embedding_model": "nomic-embed-text",
            "pipeline_retrieval_parameters": {
                "top_k": 5,
                "graph_max_chunks": 50,
                "agent_max_steps": 6
            },
            "temperature": 0.0,
            "timestamp": time.time(),
            "hidden_data_isolation_assertion": "Confirmed: Evaluating hidden dataset explicitly."
        }
        os.makedirs(self.output_dir, exist_ok=True)
        with open(self.manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

def main():
    runner = HiddenBenchmarkRunner()
    os.makedirs(runner.output_dir, exist_ok=True)
    runner.generate_manifest()

    print("Starting hidden 50-question benchmark for RAG, GraphRAG, and AgenticGraphRAG...")
    # Resumable, so overwrite=False by default
    runner.run_benchmark(overwrite=False)
    print("Hidden benchmark complete.")

if __name__ == "__main__":
    main()

