# Pipeline Details

This document provides a side-by-side technical explanation of the three implemented architectures.

## Comparison

| Aspect | RAG | GraphRAG | Agentic GraphRAG |
|---|---|---|---|
| Vector retrieval | Yes | Yes | Yes |
| TigerGraph | No | Yes | Yes |
| Graph expansion | No | Yes | Adaptive |
| Evidence evaluation | Basic retrieval | Reranking | Explicit |
| Multiple investigation steps | No | Fixed | Adaptive |
| Stopping decision | Single pass | Single pass | Explicit |
| Telemetry | Retrieval/generation | Retrieval/graph/generation | Full action trace |

*Note: GraphRAG in this project represents a vector-seeded hybrid implementation.*

---

## 1. Standard RAG

**Flow:** `Question → Embedding → FAISS top-k → Context → LLM → Answer`

The baseline Retrieval-Augmented Generation approach relies entirely on semantic similarity. It excels at fast, localized lookups but often fails on multi-hop questions where the answer is scattered across disconnected documents that don't all share semantic similarity with the immediate question.

## 2. GraphRAG

**Flow:** `Question → Vector Seed Retrieval → Seed Documents → TigerGraph Entity Expansion → Related Documents → Candidate Chunks → Local Embedding Reranking → LLM`

GraphRAG bridges the gap by injecting structural context. 
It queries the FAISS index to find highly relevant "seed" documents. It then traverses the graph to find entities mentioned in those documents, and pulls in *other* documents related to those same entities. 

**TigerGraph Schema (`GRAPHRAG`):**
- `Document ──HAS_CHUNK──> Chunk` (13,130 edges)
- `Document ──ABOUT_ENTITY──> Entity` (2,951 edges)

*(Note: The implementation reflects the actual verified schema; it does not invent unsupported Entity-to-Entity relationships).*

## 3. Agentic GraphRAG

**Flow:** `Question → Controller → Inspect State → Choose Action → Execute Tool → Evaluate Evidence → Decide Next Action → Stop`

Agentic GraphRAG shifts control from hardcoded Python scripts to an LLM-driven orchestration loop. 

**Available Tools:**
- `vector_search`: Semantic retrieval.
- `graph_expansion`: TigerGraph traversal based on discovered seed documents.
- `select_evidence`: Local semantic re-ranking of candidate chunks.
- `evaluate_evidence`: Explicit LLM-based verification of whether the collected evidence satisfies the user's question.
- `entity_linking`: Specialized graph entry via entities.
- `document_retrieval`: Specialized wrapper for semantic retrieval.
- `multi_hop_reasoning`: Depth-first graph traversals via entities.
- `aggregate_evidence`: Specialized semantic synthesis over chunks.

**Stop Guard:**
The Agentic controller utilizes a programmatic stop guard. If the `evaluate_evidence` tool explicitly reports missing information, the orchestrator overrides any LLM attempts to prematurely stop, forcing further graph or vector investigation until the maximum step limit is reached or the evidence is verified as sufficient.

**Representative Trace:**
`vector_search → graph_expansion → select_evidence → evaluate_evidence → stop`

The system records structured telemetry (the exact actions, durations, and tool outputs) rather than fragile free-text chain-of-thought, ensuring reliable auditing and benchmarking.
