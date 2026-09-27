# Demo Script

This document provides a polished 2–3 minute live demo script for the RAG vs GraphRAG vs Agentic GraphRAG project.

## 0:00–0:20 — Problem
"Not every question needs an agent."
Some questions can be answered through direct vector retrieval.
Some benefit from graph structure.
Some require iterative investigation.
This project measures that difference.

## 0:20–0:45 — Three pipelines
We compare three distinct approaches:
- **RAG:** `vector retrieval → answer`
- **GraphRAG:** `vector seed → TigerGraph expansion → evidence → answer`
- **Agentic GraphRAG:** `vector search → graph expansion → evidence selection → evidence evaluation → adaptive stopping → answer`

## 0:45–1:00 — Architecture
(Show the architecture diagram)
- **FAISS** handles vector retrieval.
- **TigerGraph** provides graph structure.
- **Ollama/Qwen3** provides local LLM inference.
- **Agentic GraphRAG** has a controller that decides what to investigate next.

## 1:00–2:00 — Live Question
Use this validated question:
> **Question:** "How many nations competed in Swimming at the 1988 Summer Olympics – Men's 200 metre freestyle?"
> **Reference answer:** 41

Present the observed result:

**RAG:**
SUCCESS
37.03s
2,442 tokens
Judge: 20/100 FAIL

**GraphRAG:**
SUCCESS
37.99s
2,416 tokens
Judge: 20/100 FAIL

**Agentic GraphRAG:**
SUCCESS
258.42s
12,780 tokens
Judge: 100/100 PASS

**Agentic trace:**
`vector_search → graph_expansion → select_evidence → evaluate_evidence → stop`

**Stopping reason:**
"Evidence is sufficient to answer the question"

**Answer:**
41 nations

*(Note: These are validated live smoke-test results. This illustrates the tradeoff for this specific question, rather than universally proving general superiority).*

## 2:00–2:30 — Benchmark
(Show the benchmark dashboard)

100 public questions × 3 pipelines = **300 unique successful pipeline/question evaluations**.

**Current benchmark summary:**
- RAG mean latency: 78.91s
- GraphRAG mean latency: 95.87s
- Agentic GraphRAG mean latency: 326.26s
- Agentic mean steps: 4.36
- Valid Agentic traces: 100/100

"The important result is not simply which pipeline is has the lowest latency. The benchmark measures whether the additional investigation is justified by observed evidence and answer quality."

## 2:30–3:00 — Close
"RAG retrieves.
GraphRAG connects.
Agentic GraphRAG investigates.

The benchmark tells us when that additional investigation is worth the cost."
