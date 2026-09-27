# Agentic GraphRAG Benchmark

An empirical benchmark and interactive dashboard evaluating the tradeoffs between standard RAG, GraphRAG, and Agentic GraphRAG.

## Why This Project?

As retrieval systems grow more complex, a critical question emerges: **When does agentic reasoning actually help compared with simpler RAG or GraphRAG approaches, and is the additional reasoning and retrieval cost justified?**

This project aims to answer that by providing a rigorous side-by-side evaluation framework, an audited 100-question public benchmark, and a live comparison dashboard. It frames the debate not by claiming universal superiority of one method, but by measuring empirical tradeoffs in latency, token usage, and factual correctness across distinct architectures.

## RAG vs GraphRAG vs Agentic GraphRAG

- **RAG**: Fast, semantic similarity-based lookup. Struggles with multi-hop reasoning or global entity context.
- **GraphRAG**: Enriches semantic search by expanding seed documents through a graph database. Excellent for finding explicitly connected context, but follows a rigid, static retrieval pipeline.
- **Agentic GraphRAG**: Uses an autonomous LLM controller to adaptively investigate the corpus. It evaluates its own evidence, decides whether to search the graph or vector space, and loops until it determines the collected evidence is factually sufficient to answer the question.

## Architecture

```text
User Question
     |
     v
Live Dashboard
     |
     +-------------------+-------------------+
     |                   |                   |
     v                   v                   v
    RAG              GraphRAG        Agentic GraphRAG
     |                   |                   |
   FAISS              FAISS             Controller
                         |                   |
                    TigerGraph        Vector / Graph /
                         |             Evidence Tools
                         |                   |
                    Reranking          Evidence Eval
                         |                   |
                         +---------+---------+
                                   |
                                   v
                              Local / Hosted LLM
                                   |
                                   v
                                Answer
                                   |
                                   v
                              Evaluation
```

## Three Pipelines

### RAG
1. **Question** → vector embedding
2. **FAISS retrieval** → retrieves top-k chunks
3. **LLM answer** generation based strictly on retrieved chunks

### GraphRAG
1. **Question** → FAISS seed retrieval
2. **Seed documents** → TigerGraph entity expansion
3. **Related documents** retrieved from graph neighborhood
4. **Candidate chunks** aggregated
5. **Embedding reranking**
6. **LLM answer** generation

### Agentic GraphRAG
1. **Question** → adaptive investigation loop
2. **Controller** intelligently selects actions: `vector_search`, `graph_expansion`, `select_evidence`, `evaluate_evidence`
3. **Evidence Evaluation** informs the controller of missing information
4. **Stop** action triggered only when evidence is strictly sufficient
5. **LLM answer** generation

## Live Comparison Dashboard

The project includes a web-based Live Comparison Dashboard that allows users to type a query, provide an optional reference answer, and watch all three pipelines execute side-by-side. It renders trace steps, latency, token usage, citations, and an automated Judge score in real-time.

## Example Investigation

**Question:** "How many nations competed in Swimming at the 1988 Summer Olympics – Men's 200 metre freestyle?"
**Reference Answer:** 41

In our live smoke tests using a local `qwen3:8b` model, the results clearly demonstrate the tradeoffs:

- **RAG**:
  - Latency: 37.03s | Tokens: 2,442
  - Judge: 20/100 FAIL (Insufficient evidence retrieved)
- **GraphRAG**:
  - Latency: 37.99s | Tokens: 2,416
  - Judge: 20/100 FAIL (Static expansion missed the specific fact)
- **Agentic GraphRAG**:
  - Latency: 258.42s | Tokens: 12,780
  - Trace: `vector_search → graph_expansion → select_evidence → evaluate_evidence → stop`
  - Reason: "Evidence is sufficient to answer the question"
  - Answer: 41 nations
  - Judge: 100/100 PASS

*(Note: These are specific configuration/model measurements, not universal performance claims).*

## Dataset

The evaluation corpus consists of complex documents requiring various forms of retrieval.
- **Documents**: 2,951
- **Chunks**: 13,130
- **Public Evaluation Questions**: 100
- **Hidden Evaluation Questions**: 50 (Used for private validation, strictly unexposed)

The 100 public questions span five categories: `multi_hop`, `temporal`, `aggregation`, `lookup`, and `superlative`.

## Public Benchmark

The public Phase 11 benchmark rigorously evaluates 100 questions across all three pipelines, yielding **300 unique successful pipeline/question evaluations**.

**Audited Pipeline Successes:**
- RAG: 100
- GraphRAG: 100
- Agentic GraphRAG: 100
- Valid Agentic Traces: 100/100

**Performance Metrics (Local Configuration):**
- **Mean Latency**: RAG (78.91s), GraphRAG (95.87s), Agentic GraphRAG (326.26s)
- **Mean Agentic Steps**: 4.36

Public artifacts are located in:
- `reports/phase11_public_benchmark/results_official_public.jsonl`
- `reports/phase11_public_benchmark/results_public_100.jsonl`

## Evaluation

Outputs are scored by an LLM-as-a-Judge prompt prioritizing factual correctness over formatting. A correct answer scores highly if it derives the same factual content as the reference. A grounded refusal ("insufficient evidence") is correctly scored as a failure if it misses the answer.

## TigerGraph

The backend graph relies on **TigerGraph Savanna (Version 4.2.5)**.

- **Graph Name**: `GRAPHRAG`
- **Validated Graph Counts**:
  - Documents: 2,951
  - Chunks: 13,130
  - Entities: 2,951
- **Edges**:
  - `HAS_CHUNK`: 13,130
  - `ABOUT_ENTITY`: 2,951

Our GraphRAG implementation is a vector-seeded hybrid model utilizing a strict Schema (`Document ──HAS_CHUNK──> Chunk`, `Document ──ABOUT_ENTITY──> Entity`). It does not rely on arbitrary or hallucinated semantic entity-to-entity relationships.

## Local Setup

By default, the benchmark runs entirely locally via **Ollama**.

- **Default Provider**: Ollama
- **Default Model**: `qwen3:8b`
- **Ollama URL**: `http://localhost:11434`
- **Embedding Model**: `nomic-embed-text`
- **Vector Index**: FAISS

The system architecture cleanly supports drop-in provider replacements for OpenAI, Google Gemini, and Anthropic Claude.

## Environment Variables

Copy `.env.example` to `.env` and configure your settings. **Never commit `.env` to version control.**

```env
LLM_PROVIDER=ollama
LLM_MODEL=qwen3:8b
OLLAMA_BASE_URL=http://localhost:11434

TIGERGRAPH_HOST=https://your-instance.i.tgcloud.io
TIGERGRAPH_GRAPH=GRAPHRAG
TIGERGRAPH_SECRET=your_gsql_secret
```

## Running the Dashboard

To start the interactive dashboard locally:

```bash
python dashboard/app.py
```
Navigate to `http://localhost:8080` in your browser.

## Repository Structure

- `/dashboard`: Interactive web UI for side-by-side pipeline testing and benchmark visualization.
- `/data`: Unprocessed and processed dataset files.
- `/evaluation`: Scripts and datasets for the 100-question public benchmark and hidden sets.
- `/reports`: Audited benchmark JSONL artifacts and legacy report tracking.
- `/src/agent`: Adaptive controller, tool registry, and state management for Agentic GraphRAG.
- `/src/ingestion`: Scripts to chunk, embed, and upsert documents into FAISS and TigerGraph.
- `/src/pipelines`: Core logic for `VectorRAGPipeline`, `GraphRAGPipeline`, and `AgenticGraphRAGPipeline`.
- `/src/retrieval`: FAISS and TigerGraph integration logic.
- `/tests`: Pytest suite for unit and integration testing.

## Testing

The project uses `pytest`. Before the final Agentic reliability patches, the full suite was validated at **71 passed, 8 warnings**. 

Following the final Agentic fixes, a focused suite of 9 Agentic unit tests successfully verified:
- Evaluator generation token limits
- Insufficient-evidence stop blocking
- Sufficient-evidence stopping
- Agentic action generation bounds

## Agentic Reliability

Recent reliability improvements ensure structured robustness when utilizing reasoning models (like `qwen3:8b`):
1. Safely stripping internal `<think>` output to prevent structured JSON parsing failures.
2. Hardening evidence evaluator generation limits (num_predict budget) to prevent payload truncation.
3. Establishing strict stop guards to block the controller from halting when evidence is insufficient.
4. Feeding blocked-action feedback explicitly back to the controller to prevent infinite action loops.

## Security

Our `.gitignore` aggressively excludes sensitive environments, unreleased datasets, and large generated files, including:
- `.env` / `.env.*`
- `data/hidden/`
- `data/processed/`
- `scratch/`
- `__pycache__/` / `.venv/`

## Limitations

- **Latency**: Agentic multi-step reasoning is significantly slower than standard RAG, often taking several minutes locally on consumer hardware.
- **Cost**: The expanded token usage required for adaptive looping limits viability for high-throughput, low-latency applications.
- **Empirical Scope**: Benchmark results reflect this specific corpus, query set, and graph schema. They do not constitute universal claims of pipeline superiority.

## Hackathon Deliverables

- Complete application codebase.
- Three fully implemented retrieval pipelines.
- Interactive local dashboard.
- 100-question audited public benchmark artifacts.

## License

MIT License
