"""Offline tests — no API key needed (a fake model stands in for OpenAI)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from studio import ui  # noqa: E402
from studio.chunking import build_chunks, chunk_text  # noqa: E402
from studio.config import RetrievalSettings  # noqa: E402
from studio.extractors import extract_file  # noqa: E402
from studio.index import DocumentIndex  # noqa: E402
from studio.insights import doc_stats, keyword_search, top_terms  # noqa: E402
from studio.pipeline import citations, retrieve  # noqa: E402


class FakeLLM:
    """Bag-of-words embeddings; deterministic and free."""
    DIM = 64

    def embed(self, texts):
        out = np.zeros((len(texts), self.DIM), dtype=np.float32)
        for r, t in enumerate(texts):
            for w in t.lower().split():
                out[r, hash(w.strip(".,")) % self.DIM] += 1
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)

    def rewrite_query(self, q, history):
        return q


def make_index():
    llm, idx = FakeLLM(), DocumentIndex("fake")
    docs = {"a.md": b"# Budget\n\nThe budget is 50,000 USD.\n\n# Timeline\n\nThe deadline is March 2027.",
            "b.txt": b"The contract requires 30 days notice for termination. Payment within 15 days."}
    for n, (name, data) in enumerate(docs.items()):
        ex = extract_file(name, data)
        chunks = build_chunks(f"d{n}", name, ex.sections)
        idx.add({"id": f"d{n}", "name": name, "kind": ex.kind, "size": len(data)}, chunks,
                llm.embed([c["text"] for c in chunks]))
    return llm, idx


def test_chunking_respects_size():
    parts = chunk_text("Sentence number one. " * 200, size=300, overlap=50)
    assert len(parts) > 3 and all(len(p) <= 400 for p in parts)


@pytest.mark.parametrize("mode", ["Hybrid", "Semantic", "Keyword"])
def test_retrieve_finds_budget(mode):
    llm, idx = make_index()
    res = retrieve(llm, idx, "budget 50,000", [], RetrievalSettings(mode=mode, top_k=3))
    assert res.hits and "50,000" in res.context


def test_scope_filter():
    llm, idx = make_index()
    res = retrieve(llm, idx, "budget", [], RetrievalSettings(files=("d1",)))
    assert all(h.chunk["doc_id"] == "d1" for h in res.hits)


def test_citations():
    assert citations("A [S1] b [S2, S9]", 3) == ([1, 2], [9])


def test_save_restore_roundtrip():
    _, idx = make_index()
    back = DocumentIndex.from_bytes(idx.to_bytes())
    assert len(back) == len(idx) and back.docs.keys() == idx.docs.keys()
    with pytest.raises(ValueError):
        DocumentIndex.from_bytes(b"not a zip")


def test_insights():
    _, idx = make_index()
    stats = doc_stats(idx)
    assert {s["name"] for s in stats} == {"a.md", "b.txt"} and all(s["words"] > 0 for s in stats)
    assert any(w == "budget" for w, _ in top_terms(idx, "d0"))
    hits = keyword_search(idx, "termination notice")
    assert hits and hits[0].chunk["doc_id"] == "d1"
    assert keyword_search(idx, "") == []


def test_ui_escapes_html():
    card = ui.source_card(1, {"file": "<b>x</b>", "location": "l", "text": "<script>", "fused": 1}, False, 1)
    assert "<script>" not in card and "&lt;script&gt;" in card
    assert "violet-badge[S1]" in ui.style_citations("x [S1]")
    assert "hero-logo" in ui.hero(big=True)
