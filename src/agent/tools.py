import re
import time
import json
from typing import Dict, Any, List, Tuple
from src.retrieval.vector_search import VectorSearch
from src.retrieval.graph_search import GraphRetriever
from src.llm.ollama_client import OllamaClient

def normalize_top_k(value) -> int:
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"Expected exactly one element, got {len(value)}")
        value = value[0]
    if value is None:
        raise ValueError("Cannot be None")
    try:
        val_int = int(value)
    except (ValueError, TypeError):
        raise ValueError(f"Cannot convert {value} to int")
    if val_int <= 0:
        raise ValueError("Must be positive")
    return val_int

class AgentTools:
    def __init__(self, llm_client=None):
        self.vector_search_client = VectorSearch()
        self.graph_search_client = GraphRetriever()
        self.llm_client = llm_client or OllamaClient()

    def vector_search(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Perform semantic retrieval using VectorSearch."""
        start_time = time.time()
        try:
            # We embed query locally first
            if not query or not isinstance(query, str):
                return {"error": "Invalid query string provided to vector_search", "duration_s": time.time() - start_time}
            print(f"TOOLS.PY QUERY TYPE: {type(query)} VALUE: {repr(query)}"); embed_res = self.llm_client.embed(query)
            q_emb = embed_res[0] if embed_res else []
            chunks = self.vector_search_client.search(q_emb, top_k=top_k)
            duration = time.time() - start_time

            doc_ids = []
            parsed_chunks = []
            for c in chunks:
                metadata, score = c
                if metadata["doc_id"] not in doc_ids:
                    doc_ids.append(metadata["doc_id"])
                chunk_dict = dict(metadata)
                chunk_dict["score"] = score
                parsed_chunks.append(chunk_dict)

            return {
                "chunks": parsed_chunks,
                "doc_ids": doc_ids,
                "duration_s": duration
            }
        except Exception as e:
            return {"error": str(e), "duration_s": time.time() - start_time}

    def graph_expansion(self, seed_doc_ids: List[str], max_total_chunks: int = 50, max_chunks_per_document: int = 15) -> Dict[str, Any]:
        """Perform TigerGraph traversal."""
        start_time = time.time()
        try:
            if not seed_doc_ids:
                return {"chunks": [], "entities": [], "documents": [], "provenance": [], "duration_s": 0.0}

            res = self.graph_search_client.get_graph_context(
                seed_doc_ids,
                max_total_chunks=max_total_chunks,
                max_chunks_per_document=max_chunks_per_document
            )
            duration = time.time() - start_time

            chunks = res.get("chunks", [])
            provenance = res.get("provenance", [])

            docs = set()
            entities = set()
            for p in provenance:
                docs.add(p.get("doc_id", ""))
                for e in p.get("entities_used", []):
                    entities.add(e)

            return {
                "chunks": chunks,
                "provenance": provenance,
                "entities": list(entities),
                "documents": list(docs),
                "duration_s": duration
            }
        except Exception as e:
            return {"error": str(e), "duration_s": time.time() - start_time}

    def select_evidence(self, query: str, candidate_chunks: List[Dict[str, Any]], candidate_provenance: List[Dict[str, Any]], final_top_k: int = 5) -> Dict[str, Any]:
        """Reuse the GraphRAG local semantic evidence selection."""
        start_time = time.time()
        if not candidate_chunks:
            return {"selected_chunks": [], "selected_provenance": [], "scores": [], "duration_s": 0.0}

        try:
            # 1. Embed query
            q_emb_res = self.llm_client.embed(query)
            q_emb = q_emb_res[0] if q_emb_res else []

            # 2. Embed candidates
            texts = [c.get("text") or "" for c in candidate_chunks]
            c_embs = self.llm_client.embed(texts)

            # 3. Score and sort
            scored_candidates = []
            for i, (chunk, prov) in enumerate(zip(candidate_chunks, candidate_provenance)):
                c_emb = c_embs[i]
                # Dot product
                score = sum(x * y for x, y in zip(q_emb, c_emb))
                scored_candidates.append({
                    "score": score,
                    "chunk": chunk,
                    "prov": prov
                })

            scored_candidates.sort(key=lambda x: x["score"], reverse=True)
            selected = scored_candidates[:normalize_top_k(final_top_k)]

            duration = time.time() - start_time
            return {
                "selected_chunks": [s["chunk"] for s in selected],
                "selected_provenance": [s["prov"] for s in selected],
                "scores": [s["score"] for s in selected],
                "duration_s": duration
            }
        except Exception as e:
            return {"error": str(e), "duration_s": time.time() - start_time}

    def evaluate_evidence(self, question: str, collected_evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate if current collected evidence is sufficient to answer the question."""
        start_time = time.time()
        try:
            context_parts = []
            for c in collected_evidence:
                context_parts.append(
                    f"[Chunk {c.get('chunk_id')}]\n"
                    f"Title: {c.get('title')}\n"
                    f"Text: {c.get('text')}\n"
                )
            context_str = "\n".join(context_parts)

            prompt = (
                "You are an evidence evaluator. Your task is to determine if the provided context is sufficient "
                "to completely answer the question.\n"
                "Respond ONLY with valid JSON matching this schema:\n"
                "{\n"
                '  "status": "sufficient | insufficient | contradictory | irrelevant",\n'
                '  "missing_information": "Briefly state what is missing if not sufficient, else empty string",\n'
                '  "reason": "Brief explanation"\n'
                "}\n"
                "Do NOT use outside knowledge.\n\n"
                f"Context:\n{context_str}\n\n"
                f"Question: {question}\n\n"
                "JSON Output:"
            )

            res = self.llm_client.generate(prompt, temperature=0.0, options={"num_predict": 2048})
            text_resp = res.get("response", "")

            # Remove <think> blocks before parsing
            text_resp = re.sub(r'<think>.*?</think>', '', text_resp, flags=re.DOTALL).strip()

            # Basic parsing of JSON
            try:
                # Find boundaries to ignore any markdown fences
                start = text_resp.find('{')
                end = text_resp.rfind('}') + 1
                if start >= 0 and end > start:
                    json_str = text_resp[start:end]
                    parsed = json.loads(json_str)
                else:
                    raise ValueError("No JSON object found")
            except Exception:
                # Safe recovery fallback
                parsed = {
                    "status": "insufficient",
                    "missing_information": "Failed to parse evaluator JSON output.",
                    "reason": "parse_error"
                }

            return {
                "evaluation": parsed,
                "prompt_eval_count": res.get("prompt_eval_count", 0),
                "eval_count": res.get("eval_count", 0),
                "duration_s": time.time() - start_time
            }
        except Exception as e:
            return {"error": str(e), "duration_s": time.time() - start_time}
