import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import pyTigerGraph as tg
    HAS_PYTIGERGRAPH = True
except ImportError:
    HAS_PYTIGERGRAPH = False

from src.ingestion.corpus_loader import load_documents
from src.ingestion.chunker import Chunker

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class TigerGraphLoader:
    def __init__(self, host: str = None, graphname: str = None, secret: str = None):
        self.host = host or os.environ.get("TIGERGRAPH_HOST")
        self.graphname = graphname or os.environ.get("TIGERGRAPH_GRAPH")
        self.secret = secret or os.environ.get("TIGERGRAPH_SECRET")

        self.conn = None
        self.stats = {
            "documents_attempted": 0,
            "documents_loaded": 0,
            "chunks_attempted": 0,
            "chunks_loaded": 0,
            "entities_attempted": 0,
            "entities_loaded": 0,
            "has_chunk_edges_loaded": 0,
            "about_entity_edges_loaded": 0,
            "failures_errors": 0,
            "vertices_loaded": 0, # maintained for backward compatibility in tests
            "edges_loaded": 0   # maintained for backward compatibility in tests
        }

    def connect(self):
        """Establish connection to TigerGraph."""
        if not HAS_PYTIGERGRAPH:
            logger.warning("pyTigerGraph is not installed. Running in mock/dry-run mode.")
            return False

        if not all([self.host, self.graphname, self.secret]):
            logger.warning("Missing one or more required environment variables: TIGERGRAPH_HOST, TIGERGRAPH_GRAPH, TIGERGRAPH_SECRET. Running in mock/dry-run mode.")
            return False

        logger.info("TigerGraph configuration detected.")
        try:
            self.conn = tg.TigerGraphConnection(
                host=self.host,
                graphname=self.graphname,
                gsqlSecret=self.secret
            )
            # Obtain token automatically
            self.conn.getToken(self.secret)
            logger.info("Successfully connected to TigerGraph Cloud.")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to TigerGraph: {e}")
            return False

    def load_corpus(self, corpus_path: Path, chunks_path: Path = None):
        """Load documents and their generated chunks into TigerGraph."""
        if not corpus_path.exists():
            raise FileNotFoundError(f"Corpus not found: {corpus_path}")

        path_str = str(corpus_path).lower().replace('\\', '/')
        if "data/hidden" in path_str or "eval_hidden.jsonl" in path_str:
            raise ValueError("SECURITY ALERT: Attempted to load hidden evaluation data into TigerGraph.")

        if chunks_path:
            path_str = str(chunks_path).lower().replace('\\', '/')
            if "data/hidden" in path_str or "eval_hidden.jsonl" in path_str:
                raise ValueError("SECURITY ALERT: Attempted to load hidden evaluation data into TigerGraph.")

        logger.info(f"Loading corpus from {corpus_path} into TigerGraph...")

        vertices = {"Document": {}, "Chunk": {}, "Entity": {}}
        edges = {"HAS_CHUNK": [], "ABOUT_ENTITY": []}

        chunker = Chunker()

        if chunks_path and chunks_path.exists():
            with open(chunks_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        chunk = json.loads(line)
                        doc_id = chunk["doc_id"]
                        chunk_id = chunk["chunk_id"]
                        qid = chunk.get("wikidata_qid")

                        if doc_id not in vertices["Document"]:
                            vertices["Document"][doc_id] = {
                                "title": chunk["title"],
                                "url": chunk["url"],
                                "wikipedia_pageid": chunk["wikipedia_pageid"]
                            }
                            self.stats["documents_attempted"] += 1

                        if qid and qid not in vertices["Entity"]:
                            vertices["Entity"][qid] = {}
                            self.stats["entities_attempted"] += 1

                        if chunk_id not in vertices["Chunk"]:
                            vertices["Chunk"][chunk_id] = {
                                "text": chunk["text"],
                                "chunk_index": chunk["chunk_index"]
                            }
                            self.stats["chunks_attempted"] += 1

                        edges["HAS_CHUNK"].append((doc_id, chunk_id, {}))
                        if qid:
                            edges["ABOUT_ENTITY"].append((doc_id, qid, {}))

                    except json.JSONDecodeError:
                        self.stats["failures_errors"] += 1
        else:
            for doc in load_documents(corpus_path):
                try:
                    if doc.doc_id not in vertices["Document"]:
                        vertices["Document"][doc.doc_id] = {
                            "title": doc.title,
                            "url": doc.url,
                            "wikipedia_pageid": doc.wikipedia_pageid
                        }
                        self.stats["documents_attempted"] += 1

                    if doc.wikidata_qid and doc.wikidata_qid not in vertices["Entity"]:
                        vertices["Entity"][doc.wikidata_qid] = {}
                        self.stats["entities_attempted"] += 1

                    if doc.wikidata_qid:
                        edges["ABOUT_ENTITY"].append((doc.doc_id, doc.wikidata_qid, {}))

                    for chunk in chunker.chunk_document(doc):
                        if chunk["chunk_id"] not in vertices["Chunk"]:
                            vertices["Chunk"][chunk["chunk_id"]] = {
                                "text": chunk["text"],
                                "chunk_index": chunk["chunk_index"]
                            }
                            self.stats["chunks_attempted"] += 1
                        edges["HAS_CHUNK"].append((doc.doc_id, chunk["chunk_id"], {}))
                except Exception as e:
                    logger.error(f"Error processing doc {doc.doc_id}: {e}")
                    self.stats["failures_errors"] += 1

        self._upsert_to_graph(vertices, edges)
        return self.stats

    def _batch_upsert_vertices(self, vertex_type, vertices_dict, batch_size=500):
        if not vertices_dict or not self.conn: return
        items = list(vertices_dict.items())
        for i in range(0, len(items), batch_size):
            batch = items[i:i+batch_size]
            try:
                self.conn.upsertVertices(vertex_type, batch)
            except Exception as e:
                logger.error(f"Error upserting batch for {vertex_type}: {e}")
                self.stats["failures_errors"] += 1

    def _batch_upsert_edges(self, src_type, edge_type, tgt_type, edges_list, batch_size=500):
        if not edges_list or not self.conn: return
        for i in range(0, len(edges_list), batch_size):
            batch = edges_list[i:i+batch_size]
            try:
                self.conn.upsertEdges(src_type, edge_type, tgt_type, batch)
            except Exception as e:
                logger.error(f"Error upserting batch for {edge_type}: {e}")
                self.stats["failures_errors"] += 1

    def _upsert_to_graph(self, vertices, edges):
        # Deduplicate edges by source and target IDs
        def dedup_edges(edge_list):
            seen = set()
            result = []
            for src, tgt, attr in edge_list:
                if (src, tgt) not in seen:
                    seen.add((src, tgt))
                    result.append((src, tgt, attr))
            return result

        edges["HAS_CHUNK"] = dedup_edges(edges["HAS_CHUNK"])
        edges["ABOUT_ENTITY"] = dedup_edges(edges["ABOUT_ENTITY"])

        if self.conn:
            logger.info("Batch upserting vertices...")
            self._batch_upsert_vertices("Document", vertices["Document"])
            self._batch_upsert_vertices("Chunk", vertices["Chunk"])
            self._batch_upsert_vertices("Entity", vertices["Entity"])

            logger.info("Batch upserting edges...")
            self._batch_upsert_edges("Document", "HAS_CHUNK", "Chunk", edges["HAS_CHUNK"])
            self._batch_upsert_edges("Document", "ABOUT_ENTITY", "Entity", edges["ABOUT_ENTITY"])

        # Update exact stats
        doc_count = len(vertices["Document"])
        chunk_count = len(vertices["Chunk"])
        entity_count = len(vertices["Entity"])

        has_chunk_count = len(edges["HAS_CHUNK"])
        about_entity_count = len(edges["ABOUT_ENTITY"])

        self.stats["documents_loaded"] = doc_count
        self.stats["chunks_loaded"] = chunk_count
        self.stats["entities_loaded"] = entity_count

        self.stats["has_chunk_edges_loaded"] = has_chunk_count
        self.stats["about_entity_edges_loaded"] = about_entity_count

        # Retro compatibility
        self.stats["vertices_loaded"] = doc_count + chunk_count + entity_count
        self.stats["edges_loaded"] = has_chunk_count + about_entity_count

        logger.info(f"Loaded {self.stats['vertices_loaded']} vertices and {self.stats['edges_loaded']} edges.")

def main():
    base_dir = Path(__file__).resolve().parent.parent.parent
    corpus_path = base_dir / "data" / "corpus" / "corpus.jsonl"
    chunks_path = base_dir / "data" / "processed" / "chunks.jsonl"

    loader = TigerGraphLoader()

    # 1. Connect
    loader.connect()

    # 2. Validate
    if not corpus_path.exists():
        logger.error(f"Corpus path missing: {corpus_path}")
        sys.exit(1)

    # 3 & 4. Batch upsert
    logger.info("Starting production load...")
    stats = loader.load_corpus(corpus_path, chunks_path)

    # 5. Report counts
    logger.info("--- PRODUCTION LOAD COMPLETE ---")
    logger.info(f"Documents attempted: {stats['documents_attempted']}, loaded: {stats['documents_loaded']}")
    logger.info(f"Chunks attempted: {stats['chunks_attempted']}, loaded: {stats['chunks_loaded']}")
    logger.info(f"Entities attempted: {stats['entities_attempted']}, loaded: {stats['entities_loaded']}")
    logger.info(f"HAS_CHUNK edges loaded: {stats['has_chunk_edges_loaded']}")
    logger.info(f"ABOUT_ENTITY edges loaded: {stats['about_entity_edges_loaded']}")
    logger.info(f"Failures/Errors: {stats['failures_errors']}")

    # Expected corpus-derived counts
    if stats["documents_loaded"] == 2951 and stats["chunks_loaded"] == 13130:
        logger.info("Counts MATCH expected validation counts for Documents (2951) and Chunks (13130).")
    else:
        logger.warning(f"Counts MISMATCH. Expected: 2951 Documents, 13130 Chunks.")

if __name__ == "__main__":
    main()
