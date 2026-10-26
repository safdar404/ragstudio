"""Visual layer: CSS theme and small HTML components.

Every piece of document or model text placed in HTML here is escaped first.
"""
from __future__ import annotations

import html
import re

from .theme import CSS, logo_data_uri  # noqa: F401  (rainbow AI theme)

KIND_ICON = {"PDF": "📄", "Word": "📝", "Image": "🖼️", "Text": "📃", "Markdown": "📑", "CSV": "📊"}


def esc(text) -> str:
    return html.escape(str(text if text is not None else ""))


def _logo_img(cls: str) -> str:
    uri = logo_data_uri()
    return f'<img class="{cls}" src="{uri}" alt="AI agent logo">' if uri else f'<span class="{cls}">🤖</span>'


def sidebar_brand(version: str) -> str:
    return (f'<div class="side-brand">{_logo_img("")}<div><b>RAG Studio</b>'
            f'<span>AI document agent · v{esc(version)}</span></div></div>')


HERO_CHIPS = ["📄 PDF + scans", "📝 Word", "🖼️ Images", "📊 CSV · TXT · MD",
              "⚡ Hybrid search", "📌 Cited answers", "🌍 Multilingual"]


def hero(stats: list[tuple[str, str]] | None = None, big: bool = False) -> str:
    stats_html = "".join(f'<div class="stat"><b>{esc(v)}</b><small>{esc(k)}</small></div>' for k, v in stats or [])
    chips = ('<div class="chips">' + "".join(f"<span>{esc(c)}</span>" for c in HERO_CHIPS) + "</div>") if big else ""
    stats_block = f'<div class="stats">{stats_html}</div>' if stats_html else ""
    return (f'<div class="hero{" big" if big else ""}"><div class="hero-main">{_logo_img("hero-logo")}<div>'
            f'<div class="eyebrow"><i></i>AI AGENT · DOCUMENT INTELLIGENCE</div>'
            f'<h1><span>RAG Document Studio</span> ✨</h1>'
            f'<p>Chat with your documents 💬 get grounded, cited answers 📌 and inspect every source 🔍</p>'
            f'{chips}</div></div>{stats_block}</div>')


def section_title(text: str) -> str:
    return f'<div class="section-title">{esc(text)}</div>'


def keyword_chips(pairs: list[tuple[str, int]]) -> str:
    return '<div class="kw">' + "".join(f"<span>{esc(w)}<b>{n}</b></span>" for w, n in pairs) + "</div>"


def empty_state(icon: str, title: str, body: str) -> str:
    return f'<div class="empty"><div class="icon">{icon}</div><h3>{esc(title)}</h3><p>{esc(body)}</p></div>'


def pills(items: list[tuple[str, str]]) -> str:
    """items: [(text, variant)] where variant in '', 'ok', 'warn', 'kind'"""
    return '<div class="pills">' + "".join(f'<span class="pill {v}">{esc(t)}</span>' for t, v in items) + "</div>"


def _highlight(text: str, terms: list[str]) -> str:
    safe = esc(text)
    terms = sorted({t for t in terms if len(t) > 2}, key=len, reverse=True)[:12]
    if not terms:
        return safe
    pattern = re.compile(r"\b((?:" + "|".join(re.escape(esc(t)) for t in terms) + r")\w*)", re.I)
    return pattern.sub(r"<mark>\1</mark>", safe)


def source_card(number: int, source: dict, cited: bool, max_fused: float, terms: list[str] | None = None) -> str:
    text = source["text"]
    if len(text) > 1800:
        text = text[:1800] + " …"
    width = 0 if not max_fused else max(4, min(100, int(100 * source.get("fused", 0) / max_fused)))
    meta = []
    if source.get("dense", -1) > -1:
        meta.append(f"semantic {source['dense']:.2f}")
    if source.get("keyword"):
        meta.append(f"keyword {source['keyword']:.1f}")
    if source.get("rerank") is not None:
        meta.append(f"re-rank {source['rerank']:.0f}/10")
    tag = '<span class="src-tag">cited</span>' if cited else ""
    return (f'<div class="src{" cited" if cited else ""}"><div class="src-head">'
            f'<span class="src-id">S{number}</span><span class="src-file">{esc(source["file"])}</span>'
            f'<span class="src-loc">· {esc(source["location"])}</span>{tag}</div>'
            f'<div class="src-body">{_highlight(text, terms or [])}</div>'
            f'<div class="src-meta"><div class="bar"><i style="width:{width}%"></i></div>'
            f'{"".join(f"<span>{esc(m)}</span>" for m in meta)}</div></div>')


_CITE = re.compile(r"\[(S\d+(?:\s*[,;]\s*S?\d+)*)\]")


def style_citations(answer: str) -> str:
    """Turn [S1] / [S1, S3] into Streamlit-native coloured badges (no raw HTML)."""
    def repl(match):
        ids = re.findall(r"\d+", match.group(1))
        return " ".join(f":violet-badge[S{i}]" for i in ids)
    return _CITE.sub(repl, answer or "")
