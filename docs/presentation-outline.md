# Presentation Outline

A concise 7-slide outline for the project presentation.

## Slide 1 — Title
**"RAG vs GraphRAG vs Agentic GraphRAG"**
- *Speaker Notes:* Introduce the project and the overarching theme of comparing retrieval architectures.

## Slide 2 — Problem
**"When does agentic reasoning actually help?"**
- *Speaker Notes:* Not every question needs an agent. Some questions can be answered with simple vectors, some with static graphs. This project measures the tradeoff between simplicity and deep investigation.

## Slide 3 — Three approaches
**RAG / GraphRAG / Agentic GraphRAG**
- *Speaker Notes:* Outline the three pipelines. RAG uses vector retrieval. GraphRAG uses a vector seed followed by a TigerGraph expansion. Agentic GraphRAG uses a controller loop to adaptively choose tools and evaluate evidence.

## Slide 4 — Architecture
**TigerGraph + FAISS + Ollama + Agent Controller**
- *Speaker Notes:* Briefly show the high-level architecture. FAISS handles vectors, TigerGraph handles relationships, Ollama/Qwen3 handles local inference, and the Agent Controller drives the logic.

## Slide 5 — Live investigation
**1988 Olympics question & Agentic trace**
- *Speaker Notes:* Present the validated live smoke-test example. Show the latency and token cost of RAG and GraphRAG (both failed this complex query) versus Agentic GraphRAG (which passed). Walk through the trace: `vector_search → graph_expansion → select_evidence → evaluate_evidence → stop`.

## Slide 6 — Benchmark
**100 questions | 300 unique successful evaluations | Latency/token/step metrics**
- *Speaker Notes:* Share the benchmark summary. 78.91s (RAG) vs 95.87s (GraphRAG) vs 326.26s (Agentic). Explain that the benchmark measures whether the additional latency and 4.36 mean steps are justified by higher evidence quality.

## Slide 7 — Takeaway
**"RAG retrieves. GraphRAG connects. Agentic GraphRAG investigates."**
- *Speaker Notes:* Conclude with the main thesis. The benchmark tells us when that additional investigation is worth the cost.
