"""Retrieval pipeline.

question ─► (rewrite with chat history) ─► (multi-query expansion)
         ─► semantic search + BM25 keyword search per query
         ─► Reciprocal Rank Fusion ─► (LLM re-rank | MMR diversification)
         ─► (neighbour expansion) ─► numbered context for the answer model
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from . import config as C
from .config import RetrievalSettings
from .index import DocumentIndex, tokenize


@dataclass
class Hit:
    idx: int
    chunk: dict
    dense: float = 0.0
    keyword: float = 0.0
    fused: float = 0.0
    rerank: Optional[float] = None

    def as_source(self) -> dict:
        c = self.chunk
        return {"file": c["file"], "doc_id": c["doc_id"], "location": c["location"], "page": c.get("page"),
                "text": c["text"], "dense": round(self.dense, 4), "keyword": round(self.keyword, 3),
                "fused": round(self.fused, 5), "rerank": self.rerank}


@dataclass
class RetrievalResult:
    question: str
    standalone: str
    queries: list
    hits: list
    context: str
    notes: list = field(default_factory=list)


# ── ranking primitives ──────────────────────────────────────────────────
def top_indices(scores: np.ndarray, mask: np.ndarray, n: int, positive_only: bool = False) -> list[int]:
    s = np.where(mask, scores, -np.inf)
    if positive_only:
        s = np.where(s > 0, s, -np.inf)
    n = min(n, int(np.isfinite(s).sum()))
    if n <= 0:
        return []
    idx = np.argpartition(-s, n - 1)[:n]
    return idx[np.argsort(-s[idx], kind="stable")].tolist()


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    fused: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, i in enumerate(ranking):
            fused[i] += 1.0 / (k + rank + 1)
    return dict(fused)


def mmr(candidates: list[int], relevance: dict[int, float], vectors: np.ndarray,
        k: int, lam: float) -> list[int]:
    """Maximal Marginal Relevance: balance relevance against redundancy."""
    if not candidates:
        return []
    values = np.array([relevance[i] for i in candidates], dtype=np.float32)
    span = float(values.max() - values.min())
    rel = (values - values.min()) / span if span > 1e-12 else np.ones_like(values)
    vecs = vectors[candidates]
    max_sim = np.zeros(len(candidates), dtype=np.float32)
    chosen: list[int] = []
    available = np.ones(len(candidates), dtype=bool)
    for _ in range(min(k, len(candidates))):
        score = lam * rel - (1 - lam) * max_sim
        score[~available] = -np.inf
        best = int(np.argmax(score))
        chosen.append(candidates[best])
        available[best] = False
        max_sim = np.maximum(max_sim, vecs @ vecs[best])
    return chosen


# ── search ──────────────────────────────────────────────────────────────
def search(index: DocumentIndex, queries: list[str], query_vectors: Optional[np.ndarray],
           settings: RetrievalSettings) -> list[Hit]:
    n = len(index)
    if n == 0:
        return []
    mask = index.mask(settings.files)
    pool = max(settings.candidates, settings.top_k)
    rankings: list[list[int]] = []
    dense = np.full(n, -1.0, dtype=np.float32)
    keyword = np.zeros(n, dtype=np.float32)

    if settings.mode != "Keyword" and query_vectors is not None:
        for q in query_vectors:
            scores = index.vectors @ q
            dense = np.maximum(dense, scores)
            rankings.append(top_indices(scores, mask, pool))
    if settings.mode != "Semantic":
        for q in queries:
            scores = index.bm25.scores(tokenize(q))
            keyword = np.maximum(keyword, scores)
            rankings.append(top_indices(scores, mask, pool, positive_only=True))

    fused = reciprocal_rank_fusion(rankings)
    order = sorted(fused, key=lambda i: (-fused[i], i))[:pool]
    if settings.min_similarity > 0 and settings.mode != "Keyword" and query_vectors is not None:
        order = [i for i in order if dense[i] >= settings.min_similarity]
    return [Hit(i, index.chunks[i], float(dense[i]), float(keyword[i]), fused[i]) for i in order]


def build_context(hits: list[Hit], index: DocumentIndex, neighbors: bool) -> str:
    parts, used = [], 0
    for number, hit in enumerate(hits, 1):
        body = hit.chunk["text"]
        if neighbors:
            prev, nxt = index.neighbors(hit.idx)
            if prev and prev["location"] == hit.chunk["location"]:
                body = f"(preceding) …{prev['text'][-500:]}\n\n{body}"
            if nxt and nxt["location"] == hit.chunk["location"]:
                body = f"{body}\n\n(following) {nxt['text'][:500]}…"
        block = f"[S{number}] {hit.chunk['file']} · {hit.chunk['location']}\n{body}"
        if used + len(block) > C.MAX_CONTEXT_CHARS and parts:
            break
        parts.append(block)
        used += len(block)
    return "\n\n---\n\n".join(parts)


def retrieve(llm, index: DocumentIndex, question: str, history: list, settings: RetrievalSettings,
             log: Callable[[str], None] = lambda _: None) -> RetrievalResult:
    notes = []
    standalone = question
    if settings.rewrite and history:
        try:
            standalone = llm.rewrite_query(question, history)
            if standalone.strip().lower() != question.strip().lower():
                log(f"Interpreted as: *{standalone}*")
        except Exception:
            notes.append("Query rewriting failed; used the question as typed.")

    queries = [standalone]
    if settings.multi_query:
        try:
            extra = llm.expand_queries(standalone, n=3)
            queries += [q for q in extra if q.lower() != standalone.lower()]
            if extra:
                log(f"Also searching: {'; '.join(extra)}")
        except Exception:
            notes.append("Multi-query expansion failed; searched the main query only.")

    query_vectors = None
    if settings.mode != "Keyword" or settings.use_mmr:
        query_vectors = llm.embed(queries)
    log(f"Searching {len(index.docs) if not settings.files else len(settings.files)} document(s) "
        f"in {settings.mode.lower()} mode…")
    hits = search(index, queries, query_vectors if settings.mode != "Keyword" else None, settings)

    if hits and settings.rerank:
        candidates = hits[:20]
        try:
            log(f"Re-ranking {len(candidates)} candidates…")
            scores = llm.rerank(standalone, [(str(i), h.chunk["text"]) for i, h in enumerate(candidates)])
            for i, h in enumerate(candidates):
                h.rerank = scores.get(str(i))
            hits = sorted(candidates, key=lambda h: (-(h.rerank if h.rerank is not None else -1), -h.fused))
            hits = [h for h in hits if h.rerank is None or h.rerank >= 2] or hits[:2]
            hits = hits[:settings.top_k]
        except Exception:
            notes.append("Re-ranking failed; used fused ranking.")
            hits = hits[:settings.top_k]
    elif hits and settings.use_mmr and query_vectors is not None:
        relevance = {h.idx: h.fused for h in hits}
        chosen = mmr([h.idx for h in hits], relevance, index.vectors, settings.top_k, settings.mmr_lambda)
        by_idx = {h.idx: h for h in hits}
        hits = [by_idx[i] for i in chosen]
    else:
        hits = hits[:settings.top_k]

    context = build_context(hits, index, settings.neighbors)
    return RetrievalResult(question, standalone, queries, hits, context, notes)


# ── citations ───────────────────────────────────────────────────────────
_CITE = re.compile(r"\[(S\d+(?:\s*[,;]\s*S?\d+)*)\]")


def citations(answer: str, n_sources: int) -> tuple[list[int], list[int]]:
    """Return (valid cited source numbers, invalid numbers) found in the answer."""
    found = set()
    for group in _CITE.findall(answer or ""):
        found.update(int(x) for x in re.findall(r"\d+", group))
    valid = sorted(i for i in found if 1 <= i <= n_sources)
    invalid = sorted(i for i in found if not 1 <= i <= n_sources)
    return valid, invalid
