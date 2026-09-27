import json
import logging
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def check_path_safety(file_path: Path) -> None:
    """Ensure the path does not point to the hidden dataset."""
    path_str = str(file_path.resolve()).replace('\\', '/').lower()
    if 'data/hidden' in path_str or 'eval_hidden.jsonl' in path_str:
        raise ValueError(f"SECURITY ALERT: Attempted to access forbidden hidden dataset path: {file_path}")

def get_type_str(value: Any) -> str:
    """Return a string representation of a value's type."""
    if value is None:
        return "null"
    return type(value).__name__

def analyze_lengths(texts: List[str]) -> Dict[str, Any]:
    """Calculate length statistics for a list of strings."""
    if not texts:
        return {"count": 0}

    char_lengths = [len(t) for t in texts]
    word_lengths = [len(t.split()) for t in texts]

    return {
        "count": len(texts),
        "chars": {
            "min": min(char_lengths),
            "max": max(char_lengths),
            "mean": round(statistics.mean(char_lengths), 2),
            "median": statistics.median(char_lengths)
        },
        "words": {
            "min": min(word_lengths),
            "max": max(word_lengths),
            "mean": round(statistics.mean(word_lengths), 2),
            "median": statistics.median(word_lengths)
        }
    }

def profile_corpus(file_path: Path) -> Dict[str, Any]:
    """Profile the corpus dataset."""
    check_path_safety(file_path)
    logger.info(f"Profiling corpus from {file_path}")

    stats: Dict[str, Any] = {
        "total_records": 0,
        "valid_records": 0,
        "malformed_records": 0,
        "empty_records": 0,
        "fields_presence": Counter(),
        "fields_types": defaultdict(set),
        "null_counts": Counter(),
        "duplicate_ids_count": 0,
    }

    seen_ids = set()
    texts_to_analyze = []

    if not file_path.exists():
        logger.warning(f"File not found: {file_path}")
        return stats

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                stats["empty_records"] += 1
                continue

            stats["total_records"] += 1
            try:
                record = json.loads(line)
                stats["valid_records"] += 1

                for key, value in record.items():
                    stats["fields_presence"][key] += 1
                    stats["fields_types"][key].add(get_type_str(value))
                    if value is None or value == "" or value == []:
                        stats["null_counts"][key] += 1

                doc_id = record.get("doc_id") or record.get("id")
                if doc_id:
                    if doc_id in seen_ids:
                        stats["duplicate_ids_count"] += 1
                    seen_ids.add(doc_id)

                text_val = record.get("text") or record.get("content")
                if text_val and isinstance(text_val, str):
                    texts_to_analyze.append(text_val)

            except json.JSONDecodeError:
                stats["malformed_records"] += 1

    # Convert sets to lists for JSON serialization
    stats["fields_types"] = {k: list(v) for k, v in stats["fields_types"].items()}
    stats["fields_presence"] = dict(stats["fields_presence"])
    stats["null_counts"] = dict(stats["null_counts"])

    stats["text_length_statistics"] = analyze_lengths(texts_to_analyze)
    return stats

def profile_public_questions(file_path: Path) -> Dict[str, Any]:
    """Profile the public questions dataset."""
    check_path_safety(file_path)
    logger.info(f"Profiling public questions from {file_path}")

    stats: Dict[str, Any] = {
        "total_records": 0,
        "valid_records": 0,
        "malformed_records": 0,
        "empty_records": 0,
        "fields_presence": Counter(),
        "fields_types": defaultdict(set),
        "null_counts": Counter(),
        "question_types": Counter(),
        "answer_formats": Counter(),
        "unique_source_docs_referenced": set(),
    }

    questions_to_analyze = []
    answers_to_analyze = []

    if not file_path.exists():
        logger.warning(f"File not found: {file_path}")
        return stats

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                stats["empty_records"] += 1
                continue

            stats["total_records"] += 1
            try:
                record = json.loads(line)
                stats["valid_records"] += 1

                for key, value in record.items():
                    stats["fields_presence"][key] += 1
                    stats["fields_types"][key].add(get_type_str(value))
                    if value is None or value == "" or value == []:
                        stats["null_counts"][key] += 1

                if "qtype" in record:
                    stats["question_types"][record["qtype"]] += 1

                if "answer" in record:
                    stats["answer_formats"][get_type_str(record["answer"])] += 1
                    if isinstance(record["answer"], list) and len(record["answer"]) > 0:
                        answers_to_analyze.append(str(record["answer"][0]))
                    elif isinstance(record["answer"], str):
                        answers_to_analyze.append(record["answer"])

                if "gold_doc_ids" in record and isinstance(record["gold_doc_ids"], list):
                    stats["unique_source_docs_referenced"].update(record["gold_doc_ids"])

                q_val = record.get("question")
                if q_val and isinstance(q_val, str):
                    questions_to_analyze.append(q_val)

            except json.JSONDecodeError:
                stats["malformed_records"] += 1

    stats["fields_types"] = {k: list(v) for k, v in stats["fields_types"].items()}
    stats["fields_presence"] = dict(stats["fields_presence"])
    stats["null_counts"] = dict(stats["null_counts"])
    stats["question_types"] = dict(stats["question_types"])
    stats["answer_formats"] = dict(stats["answer_formats"])
    stats["unique_source_docs_referenced"] = len(stats["unique_source_docs_referenced"])

    stats["question_length_statistics"] = analyze_lengths(questions_to_analyze)
    stats["answer_length_statistics"] = analyze_lengths(answers_to_analyze)
    return stats

def main():
    base_dir = Path(__file__).resolve().parent.parent.parent
    corpus_path = base_dir / "data" / "corpus" / "corpus" / "corpus.jsonl"
    if not corpus_path.exists():
        # Fallback to the provided standard path if double 'corpus' is fixed
        corpus_path = base_dir / "data" / "corpus" / "corpus.jsonl"

    public_path = base_dir / "data" / "public" / "eval_public.jsonl"

    report_dir = base_dir / "reports"
    report_dir.mkdir(exist_ok=True)

    corpus_stats = profile_corpus(corpus_path)
    public_stats = profile_public_questions(public_path)

    # Heuristic analysis based on data observed
    heuristic_analysis = {
        "observation": "Question types are explicitly provided. Answers are generally lists containing string elements.",
    }

    architecture_observations = [
        f"Found {public_stats.get('unique_source_docs_referenced', 0)} unique source documents referenced across all public questions.",
        "BENCHMARK INTEGRITY: 'gold_doc_ids' are for EVALUATION ONLY and MUST NEVER be used in the retrieval path or by the agent.",
        "BENCHMARK INTEGRITY: 'qtype' is for EVALUATION METADATA ONLY and MUST NOT be used for application input, question routing, or retrieval strategy.",
        f"Answer formats primarily observed as: {public_stats.get('answer_formats', {})}",
        "The corpus contains Wikipedia-like structure with titles, text, tokens, and URLs, which is rich for Graph properties."
    ]

    final_report = {
        "corpus": corpus_stats,
        "public_questions": public_stats,
        "schema": {
            "corpus_fields": corpus_stats.get("fields_types", {}),
            "public_questions_fields": public_stats.get("fields_types", {})
        },
        "quality": {
            "corpus_malformed": corpus_stats.get("malformed_records", 0),
            "corpus_nulls": corpus_stats.get("null_counts", {}),
            "public_malformed": public_stats.get("malformed_records", 0),
            "public_nulls": public_stats.get("null_counts", {})
        },
        "heuristic_analysis": heuristic_analysis,
        "architecture_observations": architecture_observations
    }

    json_out = report_dir / "dataset_profile.json"
    with open(json_out, 'w', encoding='utf-8') as f:
        json.dump(final_report, f, indent=2)

    txt_out = report_dir / "dataset_profile.txt"
    with open(txt_out, 'w', encoding='utf-8') as f:
        f.write("DATASET PROFILE REPORT\n")
        f.write("======================\n\n")
        f.write("CORPUS STATS\n")
        for k, v in corpus_stats.items():
            f.write(f"{k}: {v}\n")
        f.write("\nPUBLIC QUESTIONS STATS\n")
        for k, v in public_stats.items():
            f.write(f"{k}: {v}\n")
        f.write("\nARCHITECTURE OBSERVATIONS\n")
        for obs in architecture_observations:
            f.write(f"- {obs}\n")

    logger.info(f"Reports generated successfully at {report_dir}")

if __name__ == "__main__":
    main()
