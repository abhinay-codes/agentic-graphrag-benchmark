from dataclasses import dataclass
from typing import Optional

@dataclass
class Document:
    doc_id: str
    title: str
    url: str
    wikidata_qid: str
    wikipedia_pageid: int
    approx_tokens: int
    text: str
