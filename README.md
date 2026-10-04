# RAG vs GraphRAG vs Agentic GraphRAG

Welcome to the **RAG vs GraphRAG vs Agentic GraphRAG** project.
## What is this project?
This project is an end-to-end framework designed to strictly evaluate and compare three distinct retrieval architectures:
1. **Standard RAG (Vector Search):** Semantic search using FAISS embeddings.
2. **GraphRAG (Vector + Graph Expansion):** Hybrid retrieval using vector seeds followed by single-pass TigerGraph traversal.
3. **Agentic GraphRAG (Adaptive Investigation):** An autonomous controller loop that adaptively combines semantic search, TigerGraph traversal, evidence evaluation, and explicit stopping conditions to gather sufficient evidence before answering.

## Why this comparison matters
The core problem with standard RAG is its inability to effectively resolve multi-hop, highly connected, or aggregated queries. While static GraphRAG improves this by injecting structural graph context, it still acts in a rigid, single-pass manner. **Agentic GraphRAG** introduces dynamic reasoning, allowing the system to inspect what it has found, recognize missing information, and take further action (like navigating deep graph relationships) until it reaches factual sufficiency.

## High-Level Architecture
- **Vector Store:** FAISS (using `nomic-embed-text`)
- **Graph Database:** TigerGraph Savanna 4.2.5 (Graph: `GRAPHRAG`)
- **LLM Engine:** Local Ollama (`qwen3:8b`)
- **Interface:** Live Interactive Dashboard

## Documentation
Dive deeper into the implementation, architecture, and methodology:
- [Architecture](docs/architecture.md)
- [Data Flow](docs/data-flow.md)
- [Pipeline Details](docs/pipelines.md)
- [Evaluation](docs/evaluation.md)
- [Benchmark](docs/benchmark.md)

## Live Dashboard & Live Demo Example
The project features a Live Interactive Dashboard for A/B/C testing the pipelines.

**Smoke-Test Example:**
> **Question:** "How many nations competed in Swimming at the 1988 Summer Olympics – Men's 200 metre freestyle?"
>
> **Reference Answer:** 41

**Final Validated Live Run:**
- **RAG:** SUCCESS | Latency: 37.03s | Tokens: 2,442 | Judge: 20/100 FAIL
- **GraphRAG:** SUCCESS | Latency: 37.99s | Tokens: 2,416 | Judge: 20/100 FAIL
- **Agentic GraphRAG:** SUCCESS | Latency: 258.42s | Tokens: 12,780 | **Judge: 100/100 PASS**

**Agentic Trace for the above question:**
`vector_search → graph_expansion → select_evidence → evaluate_evidence → stop`
*Stopping reason:* "Evidence is sufficient to answer the question"
*Agentic answer:* 41 nations

*(Note: This is a live smoke-test example rather than a universal benchmark performance claim).*

## Benchmark Summary
The Phase 10 Public Benchmark contains 100 public questions spanning multi-hop, temporal, aggregation, lookup, and superlative queries.
After audit and deduplication, the public benchmark contains **300 unique successful pipeline/question pairs** (100 for each pipeline).

**Mean Latency:**
- RAG: 78.91 seconds
- GraphRAG: 95.87 seconds
- Agentic GraphRAG: 326.26 seconds (Mean Agentic steps: 4.36)

*Valid Agentic traces: 100/100*

## Local Setup

### Prerequisites
- Python 3.13
- Ollama
- TigerGraph Savanna

### Installation
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

ollama pull qwen3:8b
ollama pull nomic-embed-text
```

### Environment Setup
Create a `.env` file (never commit actual secrets):
```env
LLM_PROVIDER=ollama
LLM_MODEL=qwen3:8b
OLLAMA_BASE_URL=http://localhost:11434

TIGERGRAPH_HOST=https://your-instance.i.tgcloud.io
TIGERGRAPH_GRAPH=GRAPHRAG
TIGERGRAPH_SECRET=your_gsql_secret
```

### Running the Dashboard
```powershell
$env:PYTHONPATH="."
python dashboard/app.py
```
Access the dashboard at: `http://localhost:8080`

## Security
- `.env`, hidden evaluation data, and processed data are ignored by git.
- Provider credentials and TigerGraph secrets remain strictly server-side.
- The browser/frontend never receives API keys or credentials.

## Testing & Reliability
Before the final Agentic reliability changes, the full test suite was validated at 74 passed, 4 warnings. After the final reliability changes, focused Agentic tests passed all tests, and the changes were additionally validated with the live dashboard smoke test.

Recent reliability work ensured:
1. Qwen3 `<think>` output does not interfere with JSON parsing.
2. Evaluator generation limits are optimized.
3. Insufficient evidence explicitly blocks premature stopping.
4. Controller feedback is preserved when a stop is blocked.

## Limitations
- Local LLM inference can be slow.
- Agentic investigation inherently consumes more tokens and latency to achieve higher accuracy.
- GraphRAG quality is highly dependent on dataset/graph coverage.
- Public benchmark results are point-in-time measurements, not universal claims.
- A full live comparison requires a populated local TigerGraph instance and Ollama.

## Demo & Submission
- [Demo Script](docs/demo-script.md)
- [Demo Checklist](docs/demo-checklist.md)
- [Reproduction Guide](docs/reproduction.md)
- [Submission Checklist](docs/submission-checklist.md)
- [Presentation Outline](docs/presentation-outline.md)
