import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/..'))
from evaluation.benchmark_runner import BenchmarkRunner

def main():
    runner = BenchmarkRunner()
    runner.generate_manifest()

    print("Starting full 100-question benchmark for RAG, GraphRAG, and AgenticGraphRAG...")
    # Resumable, so overwrite=False by default
    runner.run_benchmark(overwrite=False)
    print("Benchmark complete.")

if __name__ == "__main__":
    main()
