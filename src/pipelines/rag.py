import time
import os
import json
from typing import Dict, Any, List

from src.llm.ollama_client import OllamaClient
from src.retrieval.vector_search import VectorSearch

class VectorRAGPipeline:
    def __init__(self,
                 index_dir: str = "data/processed/vector_store",
                 embedding_model: str = "nomic-embed-text",
                 generation_model: str = "qwen3:8b",
                 top_k: int = 5,
                 temperature: float = 0.0,
                 llm_client = None):

        self.embedding_model = embedding_model
        self.generation_model = generation_model
        self.top_k = top_k
        self.temperature = temperature

        self.llm_client = llm_client or OllamaClient()

        # Load Vector Store
        if not os.path.exists(index_dir):
            raise RuntimeError(f"Vector index not found at {index_dir}. Please run the indexing step first.")

        try:
            # Dimension is determined from manifest or defaults to 768
            manifest_path = os.path.join(index_dir, "manifest.json")
            if os.path.exists(manifest_path):
                with open(manifest_path, 'r') as f:
                    manifest = json.load(f)
                    embed_dim = manifest.get("embedding_dimension", 768)
            else:
                embed_dim = 768

            self.vector_search = VectorSearch(index_dir=index_dir, embed_dim=embed_dim)
        except Exception as e:
            raise RuntimeError(f"Failed to load vector index: {e}")

    def _construct_context(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Constructs a strict context string from retrieved chunks."""
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks):
            source_id = i + 1
            metadata = chunk["metadata"]
            part = (
                f"[Source {source_id}]\n"
                f"Title: {metadata.get('title', 'Unknown')}\n"
                f"Document ID: {metadata.get('doc_id', 'Unknown')}\n"
                f"Chunk ID: {metadata.get('chunk_id', 'Unknown')}\n"
                f"URL: {metadata.get('url', 'Unknown')}\n"
                f"Text:\n{metadata.get('text', '')}\n"
            )
            context_parts.append(part)
        return "\n".join(context_parts)

    def _construct_prompt(self, question: str, context: str) -> str:
        prompt = (
            "You are a helpful and precise assistant.\n"
            "Answer the user's question using ONLY the provided context.\n"
            "Do NOT use any outside knowledge.\n"
            "If the provided context does not contain sufficient information to answer the question, explicitly state that the evidence is insufficient.\n"
            "Do NOT invent facts or hallucinate.\n"
            "You must cite the exact sources you used to form your answer by referencing the [Source X] blocks.\n\n"
            "Context:\n"
            "---------------------\n"
            f"{context}\n"
            "---------------------\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )
        return prompt

    def answer(self, question_id: str, question: str) -> Dict[str, Any]:
        """Runs the full RAG pipeline and returns a structured output."""
        trace = {}
        t0 = time.time()

        # 1. Embed the question
        embed_start = time.time()
        question_emb = self.llm_client.embed(question, model=self.embedding_model)
        if not question_emb or len(question_emb) == 0:
            raise RuntimeError("Failed to generate question embedding.")
        embed_end = time.time()
        trace["embedding_duration_s"] = embed_end - embed_start

        # 2. Retrieve chunks
        retrieval_start = time.time()
        raw_results = self.vector_search.search(question_emb[0], top_k=self.top_k)
        retrieval_end = time.time()
        trace["retrieval_duration_s"] = retrieval_end - retrieval_start

        retrieved_chunks = []
        citations = []
        for i, (metadata, score) in enumerate(raw_results):
            chunk_data = {
                "rank": i + 1,
                "similarity_score": score,
                "metadata": metadata
            }
            retrieved_chunks.append(chunk_data)
            citations.append({
                "source_id": i + 1,
                "chunk_id": metadata.get("chunk_id"),
                "doc_id": metadata.get("doc_id"),
                "title": metadata.get("title"),
                "url": metadata.get("url")
            })

        # 3. Construct prompt
        context = self._construct_context(retrieved_chunks)
        prompt = self._construct_prompt(question, context)

        # 4. Generate Answer
        gen_start = time.time()
        gen_options = {
            "temperature": self.temperature,
            "num_predict": 1024 # Conservative max tokens
        }

        llm_response = self.llm_client.generate(prompt=prompt, model=self.generation_model, options=gen_options)
        gen_end = time.time()
        trace["generation_duration_s"] = gen_end - gen_start

        # 5. Build structured result
        total_time = time.time() - t0
        trace["total_pipeline_duration_s"] = total_time
        trace["prompt_eval_count"] = llm_response.get("prompt_eval_count")
        trace["eval_count"] = llm_response.get("eval_count")

        # Total tokens if available
        if trace["prompt_eval_count"] is not None and trace["eval_count"] is not None:
            trace["total_tokens"] = trace["prompt_eval_count"] + trace["eval_count"]
        else:
            trace["total_tokens"] = None

        result = {
            "question_id": question_id,
            "question": question,
            "answer": llm_response["response"],
            "retrieved_chunks": retrieved_chunks,
            "citations": citations,
            "retrieval_count": len(retrieved_chunks),
            "generation_model": self.generation_model,
            "embedding_model": self.embedding_model,
            "trace": trace
        }

        return result
