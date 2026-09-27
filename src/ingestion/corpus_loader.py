import json
import logging
from pathlib import Path
from typing import Iterator

from src.ingestion.models import Document

logger = logging.getLogger(__name__)

def _check_path_safety(file_path: Path) -> None:
    path_str = str(file_path.resolve()).replace('\\', '/').lower()
    if 'data/hidden' in path_str or 'eval_hidden.jsonl' in path_str:
        raise ValueError(f"SECURITY ALERT: Attempted to access forbidden hidden dataset path: {file_path}")

def load_documents(path: Path | str) -> Iterator[Document]:
    """
    Incrementally load and yield Document objects from a JSONL file.
    Validates required fields, skips malformed records, and detects duplicates.
    """
    file_path = Path(path)
    _check_path_safety(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {file_path}")

    seen_ids = set()

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)

                # Extract and validate required fields
                doc_id = record.get("doc_id")
                if not doc_id:
                    logger.warning(f"Line {line_number}: Missing 'doc_id', skipping.")
                    continue

                if doc_id in seen_ids:
                    logger.warning(f"Line {line_number}: Duplicate doc_id found: {doc_id}, skipping.")
                    continue

                seen_ids.add(doc_id)

                # Check for missing required fields
                required_keys = ["title", "url", "wikidata_qid", "wikipedia_pageid", "approx_tokens", "text"]
                missing_keys = [k for k in required_keys if k not in record or record[k] is None]
                if missing_keys:
                    logger.warning(f"Line {line_number} (doc_id={doc_id}): Missing fields {missing_keys}, skipping.")
                    continue

                yield Document(
                    doc_id=doc_id,
                    title=str(record["title"]),
                    url=str(record["url"]),
                    wikidata_qid=str(record["wikidata_qid"]),
                    wikipedia_pageid=int(record["wikipedia_pageid"]),
                    approx_tokens=int(record["approx_tokens"]),
                    text=str(record["text"])
                )

            except json.JSONDecodeError as e:
                logger.error(f"Line {line_number}: Malformed JSON record: {e}")
            except (ValueError, TypeError) as e:
                logger.error(f"Line {line_number}: Type conversion error: {e}")
