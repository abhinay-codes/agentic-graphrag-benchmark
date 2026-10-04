# Public Benchmark Report

This document details the exact audited metrics of the Phase 10 Public Benchmark.

## Dataset Facts

- **Corpus:** 2,951 documents
- **Chunks:** 13,130
- **Public Questions:** 100
- **Hidden Questions:** 50 (Not exposed)

**Question Categories:**
- Multi-hop: 28
- Temporal: 22
- Aggregation: 21
- Lookup: 19
- Superlative: 10

## Benchmark Results

The public benchmark executed 100 questions across all 3 pipelines.
After audit and deduplication, the public benchmark contains **300 unique successful pipeline/question pairs**.

| Metric | RAG | GraphRAG | Agentic GraphRAG |
|---|---|---|---|
| **Unique Successes** | 100 | 100 | 100 |
| **Mean Latency** | 78.91s | 95.87s | 326.26s |
| **Valid Agentic Traces** | N/A | N/A | 100/100 |
| **Mean Agentic Steps** | N/A | N/A | 4.36 |

*(Note: The raw JSONL artifacts contain historical retry/error records which are automatically filtered out by the dashboard metrics engine).*

### Observations
- RAG and GraphRAG exhibit significantly lower latency due to their single-pass, hardcoded execution nature.
- Agentic GraphRAG exhibits a high mean latency (326.26s) and higher total token usage. This reflects the adaptive, multi-step investigation loop (averaging 4.36 steps per question).
- These results are specific measurements for the evaluated local environment (`qwen3:8b` via Ollama) and are not universal performance claims across larger models or hosted endpoints.
