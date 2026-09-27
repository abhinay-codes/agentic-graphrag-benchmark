import re
from typing import Dict, Any, List

def normalize_text(text: str) -> str:
    """Normalize text for deterministic matching."""
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return " ".join(text.split())

def extract_numbers(text: str) -> List[str]:
    """Extract all numbers from text."""
    return re.findall(r'\b\d+\b', str(text))

def exact_match(prediction: str, gold: str) -> bool:
    return normalize_text(prediction) == normalize_text(gold)

def numeric_match(prediction: str, gold: str) -> bool:
    pred_nums = set(extract_numbers(prediction))
    gold_nums = set(extract_numbers(gold))
    if not gold_nums:
        return False
    return gold_nums.issubset(pred_nums)

def answer_contains_gold(prediction: str, gold: str) -> bool:
    if not gold:
        return False
    return normalize_text(gold) in normalize_text(prediction)

def calculate_accuracy(prediction: str, gold: str) -> Dict[str, Any]:
    if not gold:
        return {
            "exact_match": False,
            "numeric_match": False,
            "answer_contains_gold": False
        }

    return {
        "exact_match": exact_match(prediction, gold),
        "numeric_match": numeric_match(prediction, gold),
        "answer_contains_gold": answer_contains_gold(prediction, gold)
    }

def calculate_completeness(prediction: str, expected_entities: List[str] = None) -> Dict[str, Any]:
    """Calculate completeness based on expected entities."""
    if not expected_entities:
        return {"entities_found": 0, "entities_expected": 0, "completeness_score": 0.0}

    norm_pred = normalize_text(prediction)
    found = 0
    for ent in expected_entities:
        if normalize_text(ent) in norm_pred:
            found += 1

    return {
        "entities_found": found,
        "entities_expected": len(expected_entities),
        "completeness_score": found / len(expected_entities)
    }

def calculate_grounding(citations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate grounding metrics from citations."""
    doc_ids = list(set([c.get("doc_id") for c in citations if c.get("doc_id")]))
    chunk_ids = list(set([c.get("chunk_id") for c in citations if c.get("chunk_id")]))
    urls = list(set([c.get("url") for c in citations if c.get("url")]))

    return {
        "has_citations": len(citations) > 0,
        "citation_count": len(citations),
        "citation_doc_ids": doc_ids,
        "citation_chunk_ids": chunk_ids,
        "citation_urls": urls
    }
