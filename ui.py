"""Visual layer: CSS theme and small HTML components.

Every piece of document or model text placed in HTML here is escaped first.
"""
from __future__ import annotations

import html
import re

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
:root{
  --ink:#0f172a; --muted:#64748b; --line:#e5e9f0; --surface:#ffffff; --bg:#f6f7fb;
  --accent:#0f766e; --accent-soft:#e6f4f2; --accent-ink:#115e59;
  --amber:#b45309; --amber-soft:#fef3c7; --rose:#be123c;
  --radius:14px; --shadow:0 1px 2px rgba(15,23,42,.04),0 4px 16px rgba(15,23,42,.04);
}
html,body,.stApp,[class*="css"]{font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
.stApp{background:var(--bg);color:var(--ink)}
[data-testid="stHeader"]{background:transparent}
.block-container{max-width:1240px;padding-top:1.4rem;padding-bottom:4rem}

/* Sidebar */
[data-testid="stSidebar"]{background:var(--surface);border-right:1px solid var(--line)}
[data-testid="stSidebar"] [data-testid="stExpander"]{border:1px solid var(--line);border-radius:12px;box-shadow:none}
.side-brand{display:flex;align-items:center;gap:10px;margin:.2rem 0 1rem}
.side-brand .mark{width:34px;height:34px;border-radius:10px;display:grid;place-items:center;
  background:linear-gradient(135deg,#0f766e,#0e7490);color:#fff;font-weight:700}
.side-brand b{display:block;font-size:15px;color:var(--ink)}
.side-brand span{font-size:11px;letter-spacing:.08em;color:var(--muted);text-transform:uppercase}
.usage{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin:.3rem 0 .6rem}
.usage div{background:var(--bg);border:1px solid var(--line);border-radius:10px;padding:7px 9px}
.usage small{display:block;color:var(--muted);font-size:10.5px;text-transform:uppercase;letter-spacing:.05em}
.usage b{font-family:"JetBrains Mono",monospace;font-size:13px;color:var(--ink)}

/* Header */
.hero{display:flex;justify-content:space-between;align-items:flex-end;gap:24px;flex-wrap:wrap;
  padding:26px 30px;border-radius:20px;margin-bottom:18px;color:#fff;
  background:radial-gradient(1200px 300px at 0% 0%,#14b8a6 0%,transparent 55%),
             linear-gradient(120deg,#0b2a3c 0%,#0f4c5c 55%,#0f766e 100%)}
.hero .eyebrow{font-size:11px;letter-spacing:.18em;font-weight:600;color:#99f6e4}
.hero h1{margin:6px 0 4px;font-size:30px;font-weight:700;letter-spacing:-.02em;color:#fff;padding:0}
.hero p{margin:0;color:#cfe7ea;font-size:14.5px}
.hero.big{padding:44px 40px;flex-direction:column;align-items:flex-start}
.hero.big h1{font-size:40px}
.stats{display:flex;gap:10px;flex-wrap:wrap}
.stat{background:rgba(255,255,255,.1);border:1px solid rgba(255,255,255,.18);backdrop-filter:blur(4px);
  border-radius:12px;padding:8px 14px;min-width:92px}
.stat b{display:block;font-size:20px;font-weight:700;line-height:1.1}
.stat small{font-size:11px;color:#cfe7ea;letter-spacing:.04em}
.chips{display:flex;gap:8px;flex-wrap:wrap;margin-top:16px}
.chips span{padding:5px 12px;border:1px solid rgba(255,255,255,.28);border-radius:999px;font-size:12px;color:#fff}

/* Tabs */
[data-baseweb="tab-list"]{gap:4px;border-bottom:1px solid var(--line)}
[data-baseweb="tab"]{padding:10px 14px;border-radius:10px 10px 0 0}
[data-baseweb="tab"][aria-selected="true"]{color:var(--accent)}
[data-baseweb="tab-highlight"]{background:var(--accent)}

/* Cards & containers */
[data-testid="stExpander"]{background:var(--surface);border:1px solid var(--line);border-radius:var(--radius);box-shadow:var(--shadow)}
[data-testid="stVerticalBlockBorderWrapper"]{border-radius:var(--radius)}
[data-testid="stChatMessage"]{background:var(--surface);border:1px solid var(--line);border-radius:16px;
  padding:14px 18px;margin-bottom:10px;box-shadow:var(--shadow)}
[data-testid="stChatInput"]{border-radius:14px}
[data-testid="stFileUploaderDropzone"]{background:var(--bg);border:1.5px dashed #b6c2d2;border-radius:14px}
.stButton>button,.stDownloadButton>button,.stFormSubmitButton>button{border-radius:10px;font-weight:500}
button[kind="primary"]{background:var(--accent)!important;border:0!important;color:#fff!important}
button[kind="primary"]:hover{background:var(--accent-ink)!important}

/* Empty state */
.empty{text-align:center;padding:42px 20px;border:1.5px dashed #cbd5e1;border-radius:18px;background:var(--surface)}
.empty .icon{font-size:34px}
.empty h3{margin:.4rem 0 .2rem;font-size:19px}
.empty p{color:var(--muted);margin:0}

/* Pills */
.pills{display:flex;gap:6px;flex-wrap:wrap;margin:.35rem 0 .2rem}
.pill{font-size:11.5px;padding:3px 9px;border-radius:999px;background:var(--bg);border:1px solid var(--line);color:var(--muted)}
.pill.ok{background:var(--accent-soft);border-color:#b7e3dc;color:var(--accent-ink)}
.pill.warn{background:var(--amber-soft);border-color:#fcd88a;color:var(--amber)}
.pill.kind{font-weight:600;letter-spacing:.03em}

/* Source cards */
.src{border:1px solid var(--line);border-radius:12px;padding:12px 14px;margin:8px 0;background:var(--surface)}
.src.cited{border-left:3px solid var(--accent);background:linear-gradient(90deg,#f0faf8,transparent 40%)}
.src-head{display:flex;gap:8px;align-items:center;flex-wrap:wrap;font-size:13px}
.src-id{font-family:"JetBrains Mono",monospace;font-weight:600;font-size:12px;padding:1px 7px;border-radius:6px;background:var(--ink);color:#fff}
.src.cited .src-id{background:var(--accent)}
.src-file{font-weight:600}
.src-loc{color:var(--muted)}
.src-tag{margin-left:auto;font-size:11px;color:var(--accent-ink);background:var(--accent-soft);padding:2px 8px;border-radius:999px}
.src-body{margin-top:8px;font-size:13.5px;line-height:1.55;color:#334155;white-space:pre-wrap;max-height:220px;overflow:auto}
.src-meta{display:flex;gap:12px;align-items:center;margin-top:9px;font-family:"JetBrains Mono",monospace;font-size:11px;color:var(--muted);flex-wrap:wrap}
.bar{flex:1;min-width:80px;max-width:180px;height:5px;background:var(--line);border-radius:9px;overflow:hidden}
.bar i{display:block;height:100%;background:linear-gradient(90deg,#14b8a6,#0f766e)}
mark{background:#fde68a;padding:0 2px;border-radius:3px}

@media(max-width:700px){.hero{padding:20px}.hero h1{font-size:24px}.hero.big h1{font-size:30px}}
</style>
"""

KIND_ICON = {"PDF": "📄", "Word": "📝", "Image": "🖼️", "Text": "📃", "Markdown": "📑", "CSV": "📊"}


def esc(text) -> str:
    return html.escape(str(text if text is not None else ""))


def sidebar_brand(version: str) -> str:
    return (f'<div class="side-brand"><div class="mark">✦</div><div><b>Document Studio</b>'
            f'<span>RAG workspace · v{esc(version)}</span></div></div>')


def hero(stats: list[tuple[str, str]] | None = None, big: bool = False) -> str:
    stats_html = "".join(f'<div class="stat"><b>{esc(v)}</b><small>{esc(k)}</small></div>' for k, v in stats or [])
    chips = ('<div class="chips"><span>📄 PDF + scanned pages</span><span>📝 Word</span><span>🖼️ Images</span>'
             '<span>📊 CSV · TXT · MD</span><span>🔎 Hybrid search</span><span>📌 Cited answers</span></div>') if big else ""
    stats_block = f'<div class="stats">{stats_html}</div>' if stats_html else ""
    return (f'<div class="hero{" big" if big else ""}"><div><div class="eyebrow">DOCUMENT INTELLIGENCE WORKSPACE</div>'
            f'<h1>✦ Document Studio</h1><p>Ask questions across your documents and inspect every source behind the answer.</p>'
            f'{chips}</div>{stats_block}</div>')


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
        return " ".join(f":green-badge[S{i}]" for i in ids)
    return _CITE.sub(repl, answer or "")
