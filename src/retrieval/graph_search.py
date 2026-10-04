import logging
import os
import time
from typing import List, Dict, Any

import pyTigerGraph as tg


logger = logging.getLogger(__name__)


class GraphRetriever:
    """Retrieves context using TigerGraph multi-hop traversal."""

    def __init__(self, conn: tg.TigerGraphConnection = None):
        """
        Initialize the TigerGraph connection.

        If a connection is supplied, reuse it.
        Otherwise, create a live TigerGraph connection from:
            TIGERGRAPH_HOST
            TIGERGRAPH_GRAPH
            TIGERGRAPH_SECRET

        No mock or fallback connection is used here.
        """

        if conn is None:
            try:
                from dotenv import load_dotenv
                load_dotenv()
            except ImportError:
                pass
            host = os.environ.get("TIGERGRAPH_HOST")
            graphname = os.environ.get("TIGERGRAPH_GRAPH")
            secret = os.environ.get("TIGERGRAPH_SECRET")

            if not all([host, graphname, secret]):
                raise RuntimeError(
                    "TigerGraph configuration missing: "
                    "TIGERGRAPH_HOST, TIGERGRAPH_GRAPH, TIGERGRAPH_SECRET"
                )

            self.conn = tg.TigerGraphConnection(
                host=host,
                graphname=graphname,
                gsqlSecret=secret,
            )

            # Obtain a live authentication token.
            self.conn.getToken(secret)

        else:
            self.conn = conn

        # Enforce a hard 120-second timeout on all TigerGraph network requests
        # to prevent indefinite blocking in pyTigerGraph's requests.Session.
        if hasattr(self.conn, "_session") and hasattr(self.conn._session, "request"):
            orig_req = self.conn._session.request
            def timeout_req(method, url, **kwargs):
                if kwargs.get("timeout") is None:
                    kwargs["timeout"] = 120.0
                return orig_req(method, url, **kwargs)
            self.conn._session.request = timeout_req

    def get_graph_context(
        self,
        seed_doc_ids: List[str],
        max_entities: int = 10,
        max_documents_per_entity: int = 10,
        max_chunks_per_document: int = 20,
        max_total_chunks: int = 50,
    ) -> Dict[str, Any]:
        """
        Execute an interpreted TigerGraph query for:

            SeedDocs -> Entities -> ExpandedDocs -> Chunks

        The graph traversal uses only relationships present in the
        GRAPHRAG graph.
        """

        # Ensure unique seed IDs while preserving deterministic ordering.
        seed_doc_ids = list(dict.fromkeys(seed_doc_ids))

        if not seed_doc_ids:
            return {
                "chunks": [],
                "provenance": [],
                "trace": {
                    "graph_retrieval_duration": 0.0,
                    "entity_count": 0,
                    "related_document_count": 0,
                    "retrieved_chunk_count": 0,
                },
            }

        start_time = time.time()

        query = """
        INTERPRET QUERY (SET<STRING> seed_doc_ids) FOR GRAPH __GRAPH_NAME__ {
            SetAccum<STRING> @seed_docs;
            SetAccum<STRING> @connected_entities;
            MaxAccum<STRING> @parent_doc;

            AllDocs = {Document.*};

            SeedDocs =
                SELECT s
                FROM AllDocs:s
                WHERE s.doc_id IN seed_doc_ids;

            E1 = SELECT t FROM SeedDocs:s -(ABOUT_ENTITY:e)-> Entity:t ACCUM t.@seed_docs += s.doc_id;
            E2 = SELECT t FROM SeedDocs:s -(IN_GAMES:e)-> Games:t ACCUM t.@seed_docs += s.doc_id;
            E3 = SELECT t FROM SeedDocs:s -(AT_VENUE:e)-> Venue:t ACCUM t.@seed_docs += s.doc_id;
            E4 = SELECT t FROM SeedDocs:s -(HAS_EVENT:e)-> Event:t ACCUM t.@seed_docs += s.doc_id;
            E5 = SELECT t FROM SeedDocs:s -(GOLD_NOC:e)-> NOC:t ACCUM t.@seed_docs += s.doc_id;
            E6 = SELECT t FROM SeedDocs:s -(SILVER_NOC:e)-> NOC:t ACCUM t.@seed_docs += s.doc_id;
            E7 = SELECT t FROM SeedDocs:s -(BRONZE_NOC:e)-> NOC:t ACCUM t.@seed_docs += s.doc_id;
            Entities = E1 UNION E2 UNION E3 UNION E4 UNION E5 UNION E6 UNION E7;

            D1 = SELECT s FROM AllDocs:s -(ABOUT_ENTITY:e)-> Entity:t WHERE t.@seed_docs.size() > 0 ACCUM s.@connected_entities += ("ABOUT_ENTITY:" + t.wikidata_qid);
            D2 = SELECT s FROM AllDocs:s -(IN_GAMES:e)-> Games:t WHERE t.@seed_docs.size() > 0 ACCUM s.@connected_entities += ("IN_GAMES:" + t.id);
            D3 = SELECT s FROM AllDocs:s -(AT_VENUE:e)-> Venue:t WHERE t.@seed_docs.size() > 0 ACCUM s.@connected_entities += ("AT_VENUE:" + t.id);
            D4 = SELECT s FROM AllDocs:s -(HAS_EVENT:e)-> Event:t WHERE t.@seed_docs.size() > 0 ACCUM s.@connected_entities += ("HAS_EVENT:" + t.id);
            D5 = SELECT s FROM AllDocs:s -(GOLD_NOC:e)-> NOC:t WHERE t.@seed_docs.size() > 0 ACCUM s.@connected_entities += ("GOLD_NOC:" + t.id);
            D6 = SELECT s FROM AllDocs:s -(SILVER_NOC:e)-> NOC:t WHERE t.@seed_docs.size() > 0 ACCUM s.@connected_entities += ("SILVER_NOC:" + t.id);
            D7 = SELECT s FROM AllDocs:s -(BRONZE_NOC:e)-> NOC:t WHERE t.@seed_docs.size() > 0 ACCUM s.@connected_entities += ("BRONZE_NOC:" + t.id);
            ExpandedDocs = D1 UNION D2 UNION D3 UNION D4 UNION D5 UNION D6 UNION D7;

            Chunks =
                SELECT t
                FROM ExpandedDocs:s -(HAS_CHUNK:e)-> Chunk:t
                ACCUM t.@parent_doc = s.doc_id;

            PRINT Entities[
                Entities.@seed_docs
            ];

            PRINT ExpandedDocs[
                ExpandedDocs.title,
                ExpandedDocs.url,
                ExpandedDocs.doc_id,
                ExpandedDocs.@connected_entities
            ];

            PRINT Chunks[
                Chunks.text,
                Chunks.chunk_index,
                Chunks.chunk_id,
                Chunks.@parent_doc
            ];
        }
        """

        query = query.replace("__GRAPH_NAME__", self.conn.graphname)

        try:
            res = self.conn.runInterpretedQuery(
                query,
                params={"seed_doc_ids": seed_doc_ids},
            )
        except Exception as e:
            logger.error(
                "Error running TigerGraph interpreted query: %s",
                e,
            )
            raise

        # ------------------------------------------------------------
        # Parse TigerGraph response
        # ------------------------------------------------------------

        entities_data = (
            res[0].get("Entities", [])
            if len(res) > 0
            else []
        )

        expanded_docs_data = (
            res[1].get("ExpandedDocs", [])
            if len(res) > 1
            else []
        )

        chunks_data = (
            res[2].get("Chunks", [])
            if len(res) > 2
            else []
        )

        # ------------------------------------------------------------
        # Build entity lookup
        # ------------------------------------------------------------

        entities: Dict[str, List[str]] = {}

        for ent in entities_data:
            attr = ent.get("attributes", {})

            # For ANY type, the ID might be in v_id, or wikidata_qid
            qid = (
                attr.get("Entities.wikidata_qid")
                or attr.get("Entities.id")
                or attr.get("wikidata_qid")
                or attr.get("id")
                or ent.get("v_id")
            )

            seed_docs = attr.get(
                "Entities.@seed_docs",
                [],
            )

            if qid:
                entities[qid] = seed_docs

        # ------------------------------------------------------------
        # Build document lookup
        # ------------------------------------------------------------

        docs: Dict[str, Dict[str, Any]] = {}

        for doc in expanded_docs_data:
            attr = doc.get("attributes", {})

            doc_id = (
                attr.get("ExpandedDocs.doc_id")
                or attr.get("doc_id")
                or doc.get("v_id")
            )

            if not doc_id:
                continue

            title = attr.get(
                "ExpandedDocs.title",
                "",
            )

            url = attr.get(
                "ExpandedDocs.url",
                "",
            )

            connected_entities = attr.get(
                "ExpandedDocs.@connected_entities",
                [],
            )

            docs[doc_id] = {
                "title": title,
                "url": url,
                "entities": connected_entities,
                "chunks": [],
            }

        # ------------------------------------------------------------
        # Attach chunks to their parent documents
        # ------------------------------------------------------------

        for chunk in chunks_data:
            attr = chunk.get("attributes", {})

            chunk_id = (
                attr.get("Chunks.chunk_id")
                or attr.get("chunk_id")
                or chunk.get("v_id")
            )

            text = attr.get(
                "Chunks.text",
                "",
            )

            chunk_index = attr.get(
                "Chunks.chunk_index",
                0,
            )

            parent_doc = attr.get(
                "Chunks.@parent_doc",
                "",
            )

            if parent_doc in docs:
                docs[parent_doc]["chunks"].append(
                    {
                        "chunk_id": chunk_id,
                        "text": text,
                        "chunk_index": chunk_index,
                    }
                )

        # ------------------------------------------------------------
        # Apply limits and construct provenance
        # ------------------------------------------------------------

        final_chunks: List[Dict[str, Any]] = []
        provenance: List[Dict[str, Any]] = []
        seen_chunks = set()

        # Process seed documents first.
        doc_ids = list(docs.keys())

        doc_ids.sort(
            key=lambda x: (
                0 if x in seed_doc_ids else 1,
                x,
            )
        )

        for doc_id in doc_ids:
            doc_info = docs[doc_id]

            doc_chunks = sorted(
                doc_info["chunks"],
                key=lambda x: x["chunk_index"],
            )

            # Deduplicate chunks before applying per-document limit
            unique_doc_chunks = []
            for chunk in doc_chunks:
                chunk_id = chunk["chunk_id"]
                if chunk_id not in seen_chunks:
                    unique_doc_chunks.append(chunk)
                    seen_chunks.add(chunk_id)

            # Limit chunks per document
            doc_chunks = unique_doc_chunks[:max_chunks_per_document]

            for chunk in doc_chunks:
                chunk_id = chunk["chunk_id"]

                if len(final_chunks) >= max_total_chunks:
                    break

                # ----------------------------------------------------
                # Construct graph provenance path
                # ----------------------------------------------------

                path = None
                seed_docs_used = []

                if doc_id in seed_doc_ids:
                    path = (
                        f"{doc_id} -> "
                        f"HAS_CHUNK -> "
                        f"{chunk_id}"
                    )
                    seed_docs_used = [doc_id]
                else:
                    connected_entities = doc_info.get("entities", [])
                    conn_ent = connected_entities[0] if connected_entities else None

                    if conn_ent:
                        # conn_ent is like "IN_GAMES:2016 Summer"
                        parts = conn_ent.split(":", 1)
                        if len(parts) == 2:
                            rel_type = parts[0]
                            ent_id = parts[1]
                        else:
                            rel_type = "ABOUT_ENTITY"
                            ent_id = conn_ent

                        conn_seeds = entities.get(ent_id, [])
                        if conn_seeds:
                            seed_docs_used = conn_seeds
                            first_seed = conn_seeds[0]
                            path = (
                                f"{first_seed} -> {rel_type} -> "
                                f"{ent_id} -> {rel_type} -> "
                                f"{doc_id} -> HAS_CHUNK -> "
                                f"{chunk_id}"
                            )

                # ----------------------------------------------------
                # Store retrieved chunk
                # ----------------------------------------------------

                final_chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "doc_id": doc_id,
                        "title": doc_info["title"],
                        "url": doc_info["url"],
                        "chunk_index": chunk["chunk_index"],
                        "text": chunk["text"],
                    }
                )

                # ----------------------------------------------------
                # Store provenance
                # ----------------------------------------------------

                if path is not None:
                    provenance.append(
                        {
                            "chunk_id": chunk_id,
                            "doc_id": doc_id,
                            "title": doc_info["title"],
                            "url": doc_info["url"],
                            "chunk_index": chunk["chunk_index"],
                            "text": chunk["text"],
                            "seed_documents": seed_docs_used,
                            "entities_used": doc_info["entities"],
                            "graph_path": path,
                        }
                    )

            if len(final_chunks) >= max_total_chunks:
                break

        duration = time.time() - start_time

        return {
            "chunks": final_chunks,
            "provenance": provenance,
            "trace": {
                "graph_retrieval_duration": duration,
                "entity_count": len(entities),
                "related_document_count": len(docs),
                "retrieved_chunk_count": len(final_chunks),
            },
        }
