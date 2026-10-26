"""Offline library analytics and keyword search (no API calls, no cost)."""
from __future__ import annotations

import re
from collections import Counter

from .index import _STOPWORDS, DocumentIndex, tokenize

_WORD = re.compile(r"[^\W\d_][\w'-]*", re.UNICODE)
from .pipeline import Hit, top_indices


def word_count(text: str) -> int:
    return len(text.split())


def doc_stats(index: DocumentIndex) -> list[dict]:
    """One row per document: passages, words and average passage length."""
    words: Counter = Counter()
    for c in index.chunks:
        words[c["doc_id"]] += word_count(c["text"])
    rows = []
    for d in index.docs.values():
        n = d.get("passages", 0) or 0
        rows.append({"id": d["id"], "name": d["name"], "kind": d["kind"], "passages": n,
                     "words": words[d["id"]], "avg_words": round(words[d["id"]] / n) if n else 0})
    return rows


def top_terms(index: DocumentIndex, doc_id: str | None = None, n: int = 12) -> list[tuple[str, int]]:
    """Most frequent meaningful terms in one document (or the whole library)."""
    counts: Counter = Counter()
    for c in index.chunks:
        if doc_id is None or c["doc_id"] == doc_id:
            counts.update(w for w in (m.lower().strip("'-") for m in _WORD.findall(c["text"]))
                          if len(w) > 3 and w not in _STOPWORDS)
    return counts.most_common(n)


def keyword_search(index: DocumentIndex, query: str, n: int = 10, doc_ids=()) -> list[Hit]:
    """BM25 keyword search across the library; returns hits sorted by score."""
    tokens = tokenize(query)
    if not tokens or not len(index):
        return []
    scores = index.bm25.scores(tokens)
    order = top_indices(scores, index.mask(doc_ids), n, positive_only=True)
    return [Hit(i, index.chunks[i], -1.0, float(scores[i]), float(scores[i])) for i in order]
