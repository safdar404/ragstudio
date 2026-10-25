"""In-memory document index: dense vectors + BM25 keyword index.

Documents can be added and removed incrementally, and the whole library can be
saved to / restored from a .zip so nothing has to be re-embedded.
"""
from __future__ import annotations

import io
import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np

FORMAT_VERSION = 2

_STOPWORDS = frozenset("""
a an and are as at be been being but by can could did do does for from had has have he her
here hers him his how i if in into is it its me my no nor not of on or our ours she should so
than that the their theirs them then there these they this those to too us was we were what
when where which while who whom why will with would you your yours also about any all each
some such only other more most own same just very per via
""".split())
_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)


def _stem(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN.findall(text.lower())
            if t not in _STOPWORDS and (len(t) > 1 or t.isdigit())]


class BM25:
    def __init__(self, corpus: list[list[str]], k1: float = 1.4, b: float = 0.75):
        self.k1, self.b, self.n = k1, b, len(corpus)
        self.lengths = np.array([len(doc) for doc in corpus], dtype=np.float32)
        self.avgdl = float(self.lengths.mean()) if self.n else 1.0
        postings: dict[str, list] = defaultdict(list)
        for i, doc in enumerate(corpus):
            for term, tf in Counter(doc).items():
                postings[term].append((i, tf))
        self.postings = {
            t: (np.array([d for d, _ in p], dtype=np.int64), np.array([f for _, f in p], dtype=np.float32))
            for t, p in postings.items()
        }
        self.idf = {t: math.log(1 + (self.n - len(p[0]) + 0.5) / (len(p[0]) + 0.5))
                    for t, p in self.postings.items()}
        self._norm = self.k1 * (1 - self.b + self.b * self.lengths / max(self.avgdl, 1e-9))

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        scores = np.zeros(self.n, dtype=np.float32)
        for term in set(query_tokens):
            if term not in self.postings:
                continue
            docs, tf = self.postings[term]
            scores[docs] += self.idf[term] * tf * (self.k1 + 1) / (tf + self._norm[docs])
        return scores


class DocumentIndex:
    def __init__(self, embed_model: str):
        self.embed_model = embed_model
        self.docs: dict[str, dict] = {}
        self.chunks: list[dict] = []
        self.vectors = np.zeros((0, 0), dtype=np.float32)
        self._bm25 = None
        self._positions = None

    # ── basic info ──
    def __len__(self):
        return len(self.chunks)

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1]) if self.vectors.size else 0

    def _invalidate(self):
        self._bm25 = None
        self._positions = None

    # ── mutation ──
    def add(self, meta: dict, chunks: list[dict], vectors: np.ndarray):
        if meta["id"] in self.docs:
            return
        if len(chunks) != len(vectors):
            raise ValueError("Passage and vector counts differ.")
        vectors = np.asarray(vectors, dtype=np.float32)
        if self.vectors.size == 0:
            self.vectors = vectors
        else:
            if vectors.shape[1] != self.dim:
                raise ValueError("Embedding size differs from the library. Use the same embedding model.")
            self.vectors = np.vstack([self.vectors, vectors])
        meta = {**meta, "passages": len(chunks)}
        self.docs[meta["id"]] = meta
        self.chunks.extend(chunks)
        self._invalidate()

    def remove(self, doc_id: str):
        keep = [i for i, c in enumerate(self.chunks) if c["doc_id"] != doc_id]
        dim = self.dim
        self.chunks = [self.chunks[i] for i in keep]
        self.vectors = self.vectors[keep] if keep else np.zeros((0, dim), dtype=np.float32)
        self.docs.pop(doc_id, None)
        self._invalidate()

    # ── lookup ──
    @property
    def bm25(self) -> BM25:
        if self._bm25 is None:
            self._bm25 = BM25([tokenize(f"{c['location']} {c['text']}") for c in self.chunks])
        return self._bm25

    def mask(self, doc_ids=()) -> np.ndarray:
        if not doc_ids:
            return np.ones(len(self.chunks), dtype=bool)
        wanted = set(doc_ids)
        return np.array([c["doc_id"] in wanted for c in self.chunks], dtype=bool)

    def neighbors(self, i: int):
        if self._positions is None:
            self._positions = {(c["doc_id"], c["seq"]): j for j, c in enumerate(self.chunks)}
        c = self.chunks[i]
        prev = self._positions.get((c["doc_id"], c["seq"] - 1))
        nxt = self._positions.get((c["doc_id"], c["seq"] + 1))
        return (self.chunks[prev] if prev is not None else None,
                self.chunks[nxt] if nxt is not None else None)

    def doc_chunks(self, doc_id: str) -> list[dict]:
        return [c for c in self.chunks if c["doc_id"] == doc_id]

    # ── persistence ──
    def to_bytes(self) -> bytes:
        buffer = io.BytesIO()
        vec = io.BytesIO()
        np.save(vec, self.vectors, allow_pickle=False)
        manifest = {
            "format": FORMAT_VERSION,
            "app": "Document Studio",
            "embed_model": self.embed_model,
            "saved": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "docs": self.docs,
        }
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=1))
            z.writestr("chunks.json", json.dumps(self.chunks, ensure_ascii=False))
            z.writestr("vectors.npy", vec.getvalue())
        return buffer.getvalue()

    @classmethod
    def from_bytes(cls, data: bytes) -> "DocumentIndex":
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                if sum(i.file_size for i in z.infolist()) > 1024 * 1024 * 1024:
                    raise ValueError("Saved library is too large.")
                manifest = json.loads(z.read("manifest.json"))
                chunks = json.loads(z.read("chunks.json"))
                vectors = np.load(io.BytesIO(z.read("vectors.npy")), allow_pickle=False)
        except (KeyError, zipfile.BadZipFile, json.JSONDecodeError):
            raise ValueError("This isn't a Document Studio library file.")
        if manifest.get("format") != FORMAT_VERSION:
            raise ValueError("This library was saved by an incompatible version.")
        if len(chunks) != len(vectors):
            raise ValueError("Library file is damaged (passage/vector mismatch).")
        index = cls(manifest["embed_model"])
        index.docs = manifest["docs"]
        index.chunks = chunks
        index.vectors = vectors.astype(np.float32)
        return index
