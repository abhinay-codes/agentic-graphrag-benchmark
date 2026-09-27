# Evaluation Methodology

This project utilizes a rigorous, multi-dimensional evaluation methodology designed to capture not just binary correctness, but the systemic costs associated with each architecture.

## Evaluation Dimensions

- **Correctness & Completeness:** Did the generated answer accurately satisfy the user's question?
- **Grounding:** Did the answer rely strictly on the provided corpus evidence, or did it hallucinate?
- **Latency:** Wall-clock execution time (in seconds).
- **Token Usage:** Broken down into prompt tokens, output/eval tokens, and total tokens.
- **Agentic Telemetry:** Step counts, action sequences, and the controller's explicit stopping reason.
- **Retrieval Information:** Count and provenance of the final chunks provided to the generator LLM.

## LLM-as-Judge & Reference-Based Evaluation

The benchmark employs an LLM-as-Judge pattern comparing the pipeline's generated answer against a known **Reference Answer**.
- If the pipeline output contains or derives the same factual content as the reference, it passes.
- If the pipeline output says "insufficient evidence" or correctly refuses to answer because it couldn't find the data, it **fails**. (A correct refusal is still a failure to answer the user's question).
- Accuracy is never fabricated; if no reference answer exists, strict factual verification is skipped in favor of grounding checks.

The benchmark compares all three pipelines under the exact same question, model (`qwen3:8b`), and configuration.

## Benchmark vs. Live Demo

- **Benchmark Metrics:** Derived from the frozen JSONL artifacts processed by the CLI benchmark runner.
- **Live Demo Results:** Ad-hoc requests processed through the dashboard API in real-time.
- **Hidden Evaluation Data:** A private 50-question set intentionally isolated and never exposed in the public repository to prevent model/prompt overfitting.

**IMPORTANT:** Internal chain-of-thought (e.g., `<think>` tags) is strictly scrubbed. It is not collected, displayed, or used in evaluation metrics. We evaluate the Agentic pipeline based entirely on its structured action trace and its final factual output.
