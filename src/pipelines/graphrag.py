import logging
import time
from typing import Dict, Any, List, Tuple

from src.retrieval.vector_search import VectorSearch
from src.retrieval.graph_search import GraphRetriever
from src.llm.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

class GraphRAGPipeline:
    def __init__(self, llm_client=None):
        self.vector_search = VectorSearch()
        self.graph_search = GraphRetriever()
        self.llm = llm_client or OllamaClient()

    def _select_evidence(self, query_embedding: List[float], graph_chunks: List[Dict[str, Any]], graph_provenance: List[Dict[str, Any]], final_top_k: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[float]]:
        if not graph_chunks:
            return [], [], []

        texts = [c.get("text") or "" for c in graph_chunks]
        try:
            chunk_embeddings = self.llm.embed(texts)
        except Exception as e:
            logger.error(f"Failed to embed chunks for evidence selection: {e}")
            return graph_chunks[:final_top_k], graph_provenance[:final_top_k], [0.0] * min(len(graph_chunks), final_top_k)

        if len(graph_chunks) != len(graph_provenance):
            raise ValueError(f"Chunk/provenance alignment error: {len(graph_chunks)} chunks vs {len(graph_provenance)} provenance records")

        scored_candidates = []
        for i, (chunk, prov) in enumerate(zip(graph_chunks, graph_provenance)):
            c_emb = chunk_embeddings[i]
            score = sum(x * y for x, y in zip(query_embedding, c_emb))
            scored_candidates.append({
                "score": score,
                "chunk": chunk,
                "prov": prov
            })

        # Sort descending by cosine similarity
        scored_candidates.sort(key=lambda x: x["score"], reverse=True)

        # Take top K
        selected = scored_candidates[:final_top_k]

        final_chunks = [s["chunk"] for s in selected]
        final_prov = [s["prov"] for s in selected]
        final_scores = [s["score"] for s in selected]

        return final_chunks, final_prov, final_scores

    def answer(
        self,
        question_id: str,
        question: str,
        seed_top_k: int = 5,
        max_total_chunks: int = 50,
        max_chunks_per_document: int = 15,
        final_top_k: int = 5
    ) -> Dict[str, Any]:
        start_time = time.time()
        trace = {
            "embedding_duration": 0.0,
            "seed_retrieval_duration": 0.0,
            "graph_retrieval_duration": 0.0,
            "evidence_selection_duration_s": 0.0,
            "generation_duration": 0.0,
            "total_pipeline_duration": 0.0,
            "seed_top_k": seed_top_k,
            "final_top_k": final_top_k,
            "seed_document_count": 0,
            "entity_count": 0,
            "related_document_count": 0,
            "retrieved_chunk_count": 0,
            "graph_candidate_chunk_count": 0,
            "selected_chunk_count": 0,
            "seed_similarity_scores": [],
            "selected_chunk_ids": [],
            "selected_chunk_scores": [],
            "graph_provenance": [],
            "prompt_eval_count": 0,
            "eval_count": 0,
            "total_tokens": 0
        }

        # 1. Embed Question
        embed_start = time.time()
        query_embedding_batch = self.llm.embed(question)
        query_embedding = query_embedding_batch[0] if query_embedding_batch else []
        trace["embedding_duration"] = time.time() - embed_start

        # 2. Vector Seed Retrieval
        seed_start = time.time()
        seed_chunks = self.vector_search.search(query_embedding, top_k=seed_top_k)
        trace["seed_retrieval_duration"] = time.time() - seed_start

        seed_doc_ids = []
        seed_scores = []
        for chunk_data in seed_chunks:
            metadata, score = chunk_data
            if metadata["doc_id"] not in seed_doc_ids:
                seed_doc_ids.append(metadata["doc_id"])
            seed_scores.append(score)

        trace["seed_document_count"] = len(seed_doc_ids)
        trace["seed_similarity_scores"] = seed_scores

        # 3. Graph Traversal
        graph_start = time.time()
        if not seed_doc_ids:
            graph_res = {"chunks": [], "provenance": [], "trace": {}}
        else:
            graph_res = self.graph_search.get_graph_context(
                seed_doc_ids,
                max_total_chunks=max_total_chunks,
                max_chunks_per_document=max_chunks_per_document
            )
        trace["graph_retrieval_duration"] = time.time() - graph_start

        graph_chunks = graph_res.get("chunks", [])
        graph_provenance = graph_res.get("provenance", [])

        trace["entity_count"] = graph_res.get("trace", {}).get("entity_count", 0)
        trace["related_document_count"] = graph_res.get("trace", {}).get("related_document_count", 0)
        trace["retrieved_chunk_count"] = len(graph_chunks)
        trace["graph_candidate_chunk_count"] = len(graph_chunks)

        # 4. Evidence Selection
        selection_start = time.time()
        selected_chunks, selected_prov, selected_scores = self._select_evidence(
            query_embedding,
            graph_chunks,
            graph_provenance,
            final_top_k
        )
        trace["evidence_selection_duration_s"] = time.time() - selection_start

        trace["selected_chunk_count"] = len(selected_chunks)
        trace["selected_chunk_ids"] = [c["chunk_id"] for c in selected_chunks]
        trace["selected_chunk_scores"] = selected_scores
        trace["graph_provenance"] = selected_prov

        citations = []

        if not selected_chunks:
            answer = "Insufficient evidence to answer the question."
            trace["generation_duration"] = 0.0
            trace["total_pipeline_duration"] = time.time() - start_time
            return {
                "question_id": question_id,
                "answer": answer,
                "citations": citations,
                "trace": trace
            }

        # 5. Context Construction
        context_parts = []
        for c in selected_chunks:
            context_parts.append(
                f"[Graph Source {c['chunk_id']}]\n"
                f"Title: {c['title']}\n"
                f"Document ID: {c['doc_id']}\n"
                f"URL: {c['url']}\n"
                f"Text:\n{c['text']}\n"
            )
            citations.append({
                "doc_id": c["doc_id"],
                "chunk_id": c["chunk_id"],
                "title": c["title"],
                "url": c["url"]
            })

        context_str = "\n".join(context_parts)

        prompt = (
            "You are a GraphRAG assistant. Answer the user's question using ONLY the provided graph context. "
            "Do not use outside knowledge. If the answer is not present in the context, explicitly state "
            "'Insufficient evidence to answer the question.'. "
            "When answering, cite the Document Titles you used.\n\n"
            f"Context:\n{context_str}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

                # 6. LLM Generation
        gen_start = time.time()
        response = self.llm.generate(
            prompt,
            temperature=0.0,
            options={"num_predict": 2048},
        )
        trace["generation_duration"] = time.time() - gen_start

        answer = response.get("response", "")
        trace["prompt_eval_count"] = response.get("prompt_eval_count", 0)
        trace["eval_count"] = response.get("eval_count", 0)
        trace["total_tokens"] = trace["prompt_eval_count"] + trace["eval_count"]

        trace["total_pipeline_duration"] = time.time() - start_time

        return {
            "question_id": question_id,
            "answer": answer.strip(),
            "citations": citations,
            "trace": trace
        }
