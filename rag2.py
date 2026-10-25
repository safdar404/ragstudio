"""RAG Document Studio — retrieval-augmented Q&A over PDFs, Word, images and text files.

Run:  streamlit run rag2.py
"""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime

import streamlit as st
from openai import AuthenticationError

from studio import config as C
from studio import ui
from studio.chunking import build_chunks, embed_text
from studio.config import RetrievalSettings
from studio.extractors import extract_file
from studio.index import DocumentIndex, tokenize
from studio.llm import LLM, error_message
from studio.pipeline import build_context, citations, retrieve

APP_TITLE = "RAG Document Studio"

# ═════════════════════════════════════════════════════════════════════════
# Vibrant AI theme (layered on top of studio.ui.CSS)
# ═════════════════════════════════════════════════════════════════════════
THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
:root{
  --ink:#140f2e; --muted:#6c6890; --line:#e8e4f8; --surface:rgba(255,255,255,.82); --bg:#f8f6ff;
  --violet:#7c3aed; --fuchsia:#d946ef; --pink:#ec4899; --cyan:#06b6d4; --indigo:#4f46e5;
  --accent:var(--violet); --accent-soft:#f1eaff; --accent-ink:#5b21b6;
  --grad:linear-gradient(120deg,#7c3aed 0%,#c026d3 50%,#06b6d4 100%);
  --grad-soft:linear-gradient(120deg,rgba(124,58,237,.10),rgba(217,70,239,.08),rgba(6,182,212,.10));
  --glow:0 10px 30px -10px rgba(124,58,237,.45);
  --shadow:0 1px 2px rgba(20,15,46,.04),0 8px 28px -12px rgba(76,29,149,.18);
}
html,body,.stApp,[class*="css"],button,input,textarea{font-family:"Plus Jakarta Sans",Inter,ui-sans-serif,system-ui,"Segoe UI",sans-serif}
.stApp{color:var(--ink);
  background:
    radial-gradient(900px 500px at 105% -10%,rgba(6,182,212,.13),transparent 60%),
    radial-gradient(800px 500px at -10% 10%,rgba(124,58,237,.13),transparent 60%),
    radial-gradient(700px 500px at 50% 110%,rgba(217,70,239,.10),transparent 60%),
    #f8f6ff;background-attachment:fixed}
.block-container{max-width:1260px}
h1,h2,h3,h4,h5{letter-spacing:-.015em}

/* ── Sidebar ── */
[data-testid="stSidebar"]{background:rgba(255,255,255,.72);backdrop-filter:blur(14px);border-right:1px solid var(--line)}
[data-testid="stSidebar"] [data-testid="stExpander"]{background:#fff;border:1px solid var(--line);border-radius:14px}
[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover{color:var(--violet)}
.side-brand{display:flex;align-items:center;gap:11px;margin:.1rem 0 1.1rem}
.side-brand .mark{width:40px;height:40px;border-radius:12px;display:grid;place-items:center;font-size:18px;color:#fff;
  background:var(--grad);box-shadow:var(--glow)}
.side-brand b{display:block;font-size:15.5px;font-weight:800;color:var(--ink);line-height:1.15}
.side-brand b em{font-style:normal;background:var(--grad);-webkit-background-clip:text;background-clip:text;color:transparent}
.side-brand span{font-size:10.5px;letter-spacing:.12em;color:var(--muted);text-transform:uppercase}
.usage div{background:#fff;border:1px solid var(--line);border-radius:12px;position:relative;overflow:hidden}
.usage div:before{content:"";position:absolute;left:0;top:0;bottom:0;width:3px;background:var(--grad)}
.usage b{color:var(--accent-ink)}

/* ── Hero ── */
.rx-hero{position:relative;overflow:hidden;border-radius:24px;padding:30px 34px;margin-bottom:20px;color:#fff;
  background:#0d0824;box-shadow:0 24px 60px -28px rgba(76,29,149,.7);isolation:isolate;
  display:flex;justify-content:space-between;align-items:flex-end;gap:24px;flex-wrap:wrap}
.rx-hero:before{content:"";position:absolute;inset:-40%;z-index:-1;
  background:
    radial-gradient(28% 40% at 32% 38%,rgba(124,58,237,.95),transparent 70%),
    radial-gradient(26% 36% at 68% 36%,rgba(6,182,212,.7),transparent 70%),
    radial-gradient(30% 34% at 52% 66%,rgba(217,70,239,.75),transparent 70%);
  filter:blur(36px);animation:rxDrift 18s ease-in-out infinite alternate}
.rx-hero:after{content:"";position:absolute;inset:0;z-index:-1;opacity:.18;
  background-image:linear-gradient(rgba(255,255,255,.12) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.12) 1px,transparent 1px);
  background-size:28px 28px;mask-image:linear-gradient(180deg,#000,transparent 85%)}
@keyframes rxDrift{0%{transform:translate(0,0) rotate(0)}100%{transform:translate(4%,-3%) rotate(8deg)}}
.rx-eyebrow{display:inline-flex;align-items:center;gap:8px;font-size:11px;letter-spacing:.2em;font-weight:700;
  padding:5px 12px;border-radius:999px;background:rgba(255,255,255,.1);border:1px solid rgba(255,255,255,.2)}
.rx-eyebrow i{width:7px;height:7px;border-radius:50%;background:#22d3ee;box-shadow:0 0 10px #22d3ee;animation:rxPulse 2s infinite}
@keyframes rxPulse{50%{opacity:.35}}
.rx-hero h1{margin:12px 0 6px;padding:0;font-size:34px;font-weight:800;color:#fff;letter-spacing:-.03em}
.rx-hero h1 span{background:linear-gradient(90deg,#c4b5fd,#f0abfc 45%,#67e8f9);-webkit-background-clip:text;background-clip:text;color:transparent}
.rx-hero p{margin:0;color:#d6d0f5;font-size:14.5px;max-width:560px}
.rx-hero.big{flex-direction:column;align-items:flex-start;padding:48px 44px}
.rx-hero.big h1{font-size:46px}
.rx-chips{display:flex;gap:8px;flex-wrap:wrap;margin-top:18px}
.rx-chips span{padding:6px 13px;border-radius:999px;font-size:12px;color:#fff;background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.2)}
.rx-stats{display:flex;gap:10px;flex-wrap:wrap}
.rx-stat{min-width:104px;padding:11px 16px;border-radius:16px;background:rgba(255,255,255,.08);
  border:1px solid rgba(255,255,255,.18);backdrop-filter:blur(8px)}
.rx-stat b{display:block;font-size:24px;font-weight:800;line-height:1.1;
  background:linear-gradient(90deg,#fff,#e9d5ff);-webkit-background-clip:text;background-clip:text;color:transparent}
.rx-stat small{font-size:11px;color:#c9c2ef;letter-spacing:.06em;text-transform:uppercase}

/* ── Tabs: gradient pills (Streamlit ≥1.60 React-Aria markup, plus legacy BaseWeb) ── */
[data-testid="stTabs"] [role="tablist"],[data-baseweb="tab-list"]{gap:6px;padding:6px;border:1px solid var(--line)!important;
  border-radius:16px;background:rgba(255,255,255,.75);backdrop-filter:blur(10px);width:fit-content;max-width:100%;
  box-shadow:var(--shadow);margin-bottom:.6rem}
[data-testid="stTab"],[data-baseweb="tab"]{padding:8px 16px!important;border-radius:11px!important;font-weight:600;
  color:var(--muted);transition:all .2s;height:auto!important}
[data-testid="stTab"]:hover,[data-baseweb="tab"]:hover{color:var(--violet);background:var(--accent-soft)}
[data-testid="stTab"][aria-selected="true"],[data-baseweb="tab"][aria-selected="true"]{background:var(--grad)!important;box-shadow:var(--glow)}
[data-testid="stTab"][aria-selected="true"] p,[data-baseweb="tab"][aria-selected="true"] p{color:#fff!important}
[data-testid="stTab"] > div:not([data-testid]),[data-baseweb="tab-highlight"],[data-baseweb="tab-border"]{display:none!important}

/* ── Cards, chat, inputs ── */
[data-testid="stExpander"]{background:var(--surface);backdrop-filter:blur(10px);border:1px solid var(--line);border-radius:18px;box-shadow:var(--shadow)}
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]){border-color:var(--line)!important;border-radius:18px}
[data-testid="stChatMessage"]{background:var(--surface);backdrop-filter:blur(10px);border:1px solid var(--line);
  border-radius:20px;padding:16px 20px;box-shadow:var(--shadow);position:relative;overflow:hidden}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarCustom"]):before{content:"";position:absolute;left:0;top:0;bottom:0;width:4px;background:var(--grad)}
[data-testid="stChatMessageAvatarCustom"],[data-testid="stChatMessage"] > div:first-child img{border-radius:12px}
[data-testid="stChatInput"]{border-radius:18px!important;border:1.5px solid transparent!important;
  background:linear-gradient(#fff,#fff) padding-box,var(--grad) border-box!important;box-shadow:var(--glow)}
[data-testid="stFileUploaderDropzone"]{background:var(--grad-soft);border:2px dashed #c4b5fd;border-radius:18px}
[data-testid="stFileUploaderDropzone"]:hover{border-color:var(--fuchsia)}
.stTextInput input:focus,.stTextArea textarea:focus{border-color:var(--violet)!important;box-shadow:0 0 0 3px rgba(124,58,237,.15)!important}

/* ── Buttons ── */
.stButton>button,.stDownloadButton>button,.stFormSubmitButton>button{border-radius:12px;font-weight:600;border:1px solid var(--line);transition:all .2s}
.stButton>button:hover,.stDownloadButton>button:hover{border-color:#c4b5fd;color:var(--violet);transform:translateY(-1px);box-shadow:var(--shadow)}
button[kind="primary"],button[kind="primaryFormSubmit"]{background:var(--grad)!important;background-size:160% 100%!important;
  border:0!important;color:#fff!important;box-shadow:var(--glow)}
button[kind="primary"]:hover,button[kind="primaryFormSubmit"]:hover{background-position:100% 0!important;color:#fff!important;transform:translateY(-1px)}
button[kind="tertiary"]{color:var(--violet)!important}

/* ── Accent recolouring of built-in widgets (config.toml primary colour is teal) ── */
[data-testid="stSlider"] [role="group"] > div > div:first-child{filter:hue-rotate(88deg) saturate(1.35)}
[data-testid="stSlider"] [role="group"] > div > div[data-rac]{background-color:var(--violet)!important;
  box-shadow:0 0 0 4px rgba(124,58,237,.18),0 0 12px rgba(217,70,239,.5)!important}
[data-testid="stSliderThumbValue"] p{color:var(--violet)!important;font-weight:700}
[data-testid="stCheckbox"] label[data-selected="true"] > div:first-of-type{background:var(--grad)!important}
[data-testid="stRadioOption"][data-selected="true"] > div > div:first-child{background-color:var(--violet)!important;
  border-color:var(--violet)!important;box-shadow:0 0 0 3px rgba(124,58,237,.18)}
[data-baseweb="slider"] [role="slider"]{background:var(--violet)!important}
[data-baseweb="tag"]{background:var(--accent-soft)!important;color:var(--accent-ink)!important}
[data-baseweb="tag"] span{color:var(--accent-ink)!important}

/* ── Pills, empty states, source cards ── */
.pill{background:#fff;border-color:var(--line)}
.pill.ok{background:var(--accent-soft);border-color:#ddd0ff;color:var(--accent-ink)}
.pill.kind{background:var(--grad);color:#fff;border:0}
.empty{background:var(--surface);border:2px dashed #d8ccfb;border-radius:22px}
.empty .icon{display:inline-grid;place-items:center;width:64px;height:64px;border-radius:20px;background:var(--grad);box-shadow:var(--glow);font-size:30px}
.empty h3{margin-top:.8rem}
.src{border-radius:14px;background:#fff;transition:box-shadow .2s}
.src:hover{box-shadow:var(--shadow)}
.src.cited{border-left:0;background:linear-gradient(#fff,#fff) padding-box,var(--grad) border-box;border:1.5px solid transparent}
.src-id{background:var(--ink)}
.src.cited .src-id{background:var(--grad)}
.src-tag{background:var(--accent-soft);color:var(--accent-ink)}
.bar i{background:var(--grad)}
mark{background:linear-gradient(90deg,#f5d0fe,#ddd6fe);color:#3b0764}
[data-testid="stCode"],[data-testid="stDataFrame"]{border-radius:14px;overflow:hidden}
::selection{background:#e9d5ff}
[data-testid="stMarkdownContainer"] code{color:var(--accent-ink);background:var(--accent-soft);border-radius:6px;padding:1px 6px}
@media(max-width:700px){.rx-hero{padding:22px}.rx-hero h1{font-size:26px}.rx-hero.big{padding:28px 22px}.rx-hero.big h1{font-size:32px}}
</style>
"""


def hero(stats: list[tuple[str, str]] | None = None, big: bool = False) -> str:
    stats_html = "".join(f'<div class="rx-stat"><b>{ui.esc(v)}</b><small>{ui.esc(k)}</small></div>'
                         for k, v in stats or [])
    chips = ('<div class="rx-chips"><span>📄 PDF + scans</span><span>📝 Word</span><span>🖼️ Images</span>'
             '<span>📊 CSV · TXT · MD</span><span>⚡ Hybrid search</span><span>📌 Cited answers</span></div>'
             if big else "")
    stats_block = f'<div class="rx-stats">{stats_html}</div>' if stats_html else ""
    return (f'<div class="rx-hero{" big" if big else ""}"><div>'
            f'<div class="rx-eyebrow"><i></i>AI DOCUMENT INTELLIGENCE</div>'
            f'<h1>✦ <span>RAG Document Studio</span></h1>'
            f'<p>Chat with your documents, get grounded answers, and inspect every source behind them.</p>'
            f'{chips}</div>{stats_block}</div>')


def sidebar_brand(version: str) -> str:
    return (f'<div class="side-brand"><div class="mark">✦</div><div><b><em>RAG</em> Document Studio</b>'
            f'<span>AI workspace · v{ui.esc(version)}</span></div></div>')


def cite_badges(answer: str) -> str:
    """[S1] / [S1, S3] → violet Streamlit badges (no raw HTML from model output)."""
    return ui.style_citations(answer).replace(":green-badge[", ":violet-badge[")

st.set_page_config(page_title="RAG Document Studio", page_icon="✦", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown(ui.CSS + THEME_CSS, unsafe_allow_html=True)

ss = st.session_state
for key, default in {"history": [], "usage": {}, "index": None, "queued": None, "failed": None,
                     "lab": None, "nonce": 0, "report": None}.items():
    ss.setdefault(key, default)


# ═════════════════════════════════════════════════════════════════════════
# Connect
# ═════════════════════════════════════════════════════════════════════════
def connect_screen():
    st.markdown(hero(big=True), unsafe_allow_html=True)
    _, col, _ = st.columns([1, 1.5, 1])
    with col, st.container(border=True):
        st.markdown("#### 🔑 Connect your workspace")
        env_key = os.environ.get("OPENAI_API_KEY", "")
        with st.form("connect"):
            key = st.text_input("OpenAI API key", type="password", placeholder="sk-…")
            base = st.text_input("API base URL (optional)", placeholder="https://api.openai.com/v1",
                                 help="For OpenAI-compatible servers such as Azure gateways, LiteLLM, vLLM or Ollama.")
            go = st.form_submit_button("Open workspace →", type="primary", width="stretch")
        if go:
            chosen = key.strip() or env_key
            if chosen:
                ss.api_key, ss.base_url = chosen, base.strip()
                st.rerun()
            st.warning("Please enter an API key.")
        if env_key:
            st.caption("✓ `OPENAI_API_KEY` found in the environment — leave the key field empty to use it.")
        st.caption("Your key stays in this browser session only. Document text, images and questions are sent "
                   "to the API when processed; usage may incur charges.")
    st.stop()


if not ss.get("api_key"):
    connect_screen()


def logout():
    for k in ("api_key", "base_url"):
        ss.pop(k, None)


# ═════════════════════════════════════════════════════════════════════════
# Sidebar settings
# ═════════════════════════════════════════════════════════════════════════
index: DocumentIndex | None = ss.index
has_docs = bool(index and index.docs)

with st.sidebar:
    st.markdown(sidebar_brand(C.VERSION), unsafe_allow_html=True)

    with st.expander("🧠 Models"):
        chat_model = st.text_input("Answer model", C.DEFAULT_CHAT_MODEL)
        vision_model = st.text_input("Vision model (images & scans)", C.DEFAULT_VISION_MODEL)
        embed_model = st.text_input(
            "Embedding model", index.embed_model if has_docs else C.DEFAULT_EMBED_MODEL,
            disabled=has_docs, help="Locked while the library has documents: all vectors must come from one model.")

    with st.expander("🔎 Retrieval", expanded=True):
        mode = st.radio("Search mode", C.RETRIEVAL_MODES, horizontal=True,
                        help="Hybrid combines meaning-based (semantic) and exact-word (BM25 keyword) search.")
        top_k = st.slider("Passages per answer", 2, 15, 6)
        candidates = st.slider("Candidate pool", 10, 80, 30, step=5,
                               help="How many passages are considered before diversification or re-ranking.")
        use_mmr = st.toggle("Diversify sources (MMR)", True,
                            help="Avoids sending several near-duplicate passages.")
        mmr_lambda = st.slider("Relevance ↔ diversity", 0.3, 1.0, 0.7, 0.05, disabled=not use_mmr)
        neighbors = st.toggle("Add surrounding context", False,
                              help="Includes the text just before and after each passage.")
        min_sim = st.slider("Minimum semantic similarity", 0.0, 0.6, 0.0, 0.05,
                            help="Drop weak matches. 0 keeps everything.")

    with st.expander("✨ Answering", expanded=True):
        style = st.selectbox("Answer style", list(C.ANSWER_STYLES))
        rewrite = st.toggle("Understand follow-up questions", True,
                            help="Rewrites questions like “what about the second one?” using the chat history.")
        multi_query = st.toggle("Multi-query search", False,
                                help="Searches with extra AI-generated phrasings to improve recall.")
        rerank = st.toggle("AI re-ranking", False, help="An extra model call grades candidates before answering.")
        want_followups = st.toggle("Suggest follow-up questions", True)

    u = ss.usage
    st.markdown(
        '<div class="usage">'
        f'<div><small>Prompt tokens</small><b>{u.get("prompt", 0):,}</b></div>'
        f'<div><small>Output tokens</small><b>{u.get("completion", 0):,}</b></div>'
        f'<div><small>Embedding tokens</small><b>{u.get("embedding", 0):,}</b></div>'
        f'<div><small>API calls</small><b>{u.get("calls", 0):,}</b></div></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    if c1.button("🧹 Clear chat", width="stretch", disabled=not ss.history):
        ss.history, ss.failed = [], None
        st.rerun()
    c2.button("🔑 Change key", width="stretch", on_click=logout)
    st.caption("Session-only workspace. Save your library from the Export tab to reuse it later.")


def get_llm() -> LLM:
    return LLM(ss.api_key, base_url=ss.get("base_url"), chat_model=chat_model or C.DEFAULT_CHAT_MODEL,
               vision_model=vision_model or C.DEFAULT_VISION_MODEL,
               embed_model=(index.embed_model if has_docs else embed_model) or C.DEFAULT_EMBED_MODEL,
               usage=ss.usage)


def settings(files=()) -> RetrievalSettings:
    return RetrievalSettings(mode=mode, top_k=top_k, candidates=candidates, use_mmr=use_mmr,
                             mmr_lambda=mmr_lambda, neighbors=neighbors, min_similarity=min_sim,
                             rewrite=rewrite, multi_query=multi_query, rerank=rerank, files=tuple(files))


# ═════════════════════════════════════════════════════════════════════════
# Ingestion
# ═════════════════════════════════════════════════════════════════════════
def ingest(files, ocr: bool):
    llm = get_llm()
    if ss.index is None or not ss.index.docs:
        ss.index = DocumentIndex(llm.embed_model)
    idx: DocumentIndex = ss.index
    added, skipped, failed = [], [], []
    with st.status("Processing documents…", expanded=True) as status:
        bar = st.progress(0.0)
        for n, f in enumerate(files):
            data = f.getvalue()
            doc_id = hashlib.sha256(data).hexdigest()[:16]
            if doc_id in idx.docs:
                skipped.append(f.name)
                continue
            if len(data) > C.MAX_FILE_MB * 1024 * 1024:
                failed.append((f.name, f"Larger than {C.MAX_FILE_MB} MB."))
                continue
            try:
                st.write(f"📖 Reading **{f.name}**…")
                ex = extract_file(f.name, data, transcribe=llm.transcribe_image, ocr_scanned=ocr)
                chunks = build_chunks(doc_id, f.name, ex.sections)
                if len(idx) + len(chunks) > C.MAX_LIBRARY_PASSAGES:
                    raise ValueError(f"Library limit of {C.MAX_LIBRARY_PASSAGES:,} passages would be exceeded.")
                st.write(f"🧮 Embedding {len(chunks)} passages…")
                vectors = llm.embed([embed_text(c) for c in chunks])
                idx.add({"id": doc_id, "name": f.name, "kind": ex.kind, "size": len(data), "pages": ex.pages,
                         "sections": len(ex.sections), "ocr_pages": ex.ocr_pages, "title": ex.title,
                         "warnings": ex.warnings, "added": datetime.now().strftime("%Y-%m-%d %H:%M"),
                         "brief": None}, chunks, vectors)
                added.append(f.name)
            except AuthenticationError as e:
                failed.extend([(f.name, error_message(e))] + [(g.name, "Skipped.") for g in files[n + 1:]])
                break
            except Exception as e:  # noqa: BLE001 — report per file, keep going
                failed.append((f.name, error_message(e)))
            bar.progress((n + 1) / len(files))
        state = "error" if failed and not added else "complete"
        status.update(label=f"Added {len(added)} · skipped {len(skipped)} · failed {len(failed)}",
                      state=state, expanded=bool(failed))
    ss.report = {"added": added, "skipped": skipped, "failed": failed}
    ss.nonce += 1          # clears the uploader
    st.rerun()


def restore_library(data: bytes):
    loaded = DocumentIndex.from_bytes(data)
    current: DocumentIndex | None = ss.index
    if not current or not current.docs:
        ss.index = loaded
        return len(loaded.docs)
    if loaded.embed_model != current.embed_model:
        raise ValueError(f"Saved library uses {loaded.embed_model}; current library uses {current.embed_model}.")
    count = 0
    for doc_id, meta in loaded.docs.items():
        if doc_id in current.docs:
            continue
        rows = [i for i, c in enumerate(loaded.chunks) if c["doc_id"] == doc_id]
        current.add(meta, [loaded.chunks[i] for i in rows], loaded.vectors[rows])
        count += 1
    return count


# ═════════════════════════════════════════════════════════════════════════
# Header + upload panel
# ═════════════════════════════════════════════════════════════════════════
answers = sum(m["role"] == "assistant" for m in ss.history)
st.markdown(hero([("documents", len(index.docs) if index else 0),
                     ("passages", f"{len(index) if index else 0:,}"),
                     ("answers", answers)]), unsafe_allow_html=True)

with st.expander("📂 Add documents", expanded=not has_docs):
    uploads = st.file_uploader("Drop files here", type=C.SUPPORTED_TYPES, accept_multiple_files=True,
                               key=f"upload_{ss.nonce}", label_visibility="collapsed")
    col_a, col_b = st.columns([3, 1], vertical_alignment="center")
    ocr = col_a.toggle("Read scanned PDF pages with the vision model", True,
                       help=f"Pages without a text layer are sent to the vision model (max "
                            f"{C.MAX_OCR_PAGES_PER_PDF} per PDF).")
    if col_b.button("➕ Add to library", type="primary", width="stretch", disabled=not uploads):
        if len(uploads) > C.MAX_FILES_PER_BATCH:
            st.error(f"Add at most {C.MAX_FILES_PER_BATCH} files at a time.")
        else:
            ingest(uploads, ocr)
    st.caption(f"PDF · Word (.docx) · PNG/JPG/WebP · TXT · Markdown · CSV — up to {C.MAX_FILE_MB} MB each. "
               "Files already in the library are skipped automatically.")

if ss.report:
    r = ss.report
    if r["added"]:
        st.success(f"Added {len(r['added'])} document(s): {', '.join(r['added'])}", icon="✅")
    if r["skipped"]:
        st.info(f"Already in library: {', '.join(r['skipped'])}", icon="ℹ️")
    for name, reason in r["failed"]:
        st.error(f"**{name}** — {reason}", icon="⚠️")
    ss.report = None

index = ss.index
has_docs = bool(index and index.docs)
tab_chat, tab_lib, tab_lab, tab_export = st.tabs(["💬 Chat", "📚 Library", "🔬 Retrieval lab", "⇩ Export & save"])


# ═════════════════════════════════════════════════════════════════════════
# Chat
# ═════════════════════════════════════════════════════════════════════════
def queue(question: str):
    ss.queued = question


def regenerate():
    if len(ss.history) >= 2 and ss.history[-1]["role"] == "assistant":
        ss.history.pop()
        ss.queued = ss.history.pop()["content"]


def render_assistant(m: dict, is_last: bool):
    st.markdown(cite_badges(m["content"]))
    tags = []
    if m.get("standalone") and m["standalone"].strip().lower() != m.get("question", "").strip().lower():
        tags.append((f"🔁 searched: {m['standalone'][:90]}", ""))
    if len(m.get("queries", [])) > 1:
        tags.append((f"+{len(m['queries']) - 1} query variants", ""))
    n_sources, n_cited = len(m.get("sources", [])), len(m.get("cited", []))
    if n_sources:
        tags.append((f"📌 {n_cited}/{n_sources} sources cited", "ok" if n_cited else "warn"))
    if m.get("invalid"):
        tags.append((f"⚠ unknown citation(s): {', '.join('S' + str(i) for i in m['invalid'])}", "warn"))
    if n_sources and not n_cited:
        tags.append(("no citations — verify against sources", "warn"))
    for note in m.get("notes", []):
        tags.append((note, "warn"))
    if tags:
        st.markdown(ui.pills(tags), unsafe_allow_html=True)
    if m.get("sources"):
        cited = set(m.get("cited", []))
        max_fused = max(s.get("fused", 0) for s in m["sources"]) or 1
        terms = tokenize(m.get("standalone") or m.get("question", ""))
        with st.expander(f"🔎 Evidence · {n_sources} passages"):
            only_cited = st.toggle("Show cited only", value=False, key=f"oc_{m['id']}", disabled=not cited)
            for i, s in enumerate(m["sources"], 1):
                if only_cited and i not in cited:
                    continue
                st.markdown(ui.source_card(i, s, i in cited, max_fused, terms), unsafe_allow_html=True)
    cols = st.columns([1, 1, 6], vertical_alignment="center")
    with cols[0]:
        rating = st.feedback("thumbs", key=f"fb_{m['id']}")
        m["rating"] = {0: "down", 1: "up"}.get(rating)
    if is_last:
        cols[1].button("🔁 Regenerate", key=f"rg_{m['id']}", on_click=regenerate, type="tertiary")


def answer(question: str, files, live):
    llm = get_llm()
    history = list(ss.history)
    cfg = settings(files)
    with live:
        with st.chat_message("user", avatar=":material/person:"):
            st.markdown(question)
        with st.chat_message("assistant", avatar=":material/auto_awesome:"):
            try:
                with st.status("Searching your documents…", expanded=False) as status:
                    result = retrieve(llm, index, question, history, cfg, log=status.write)
                    status.update(label=f"Found {len(result.hits)} relevant passages", state="complete")
                if not result.hits:
                    text = ("I couldn't find any passages related to this question in the selected documents. "
                            "Try rephrasing, widening the search scope, or lowering the minimum similarity.")
                    st.markdown(text)
                else:
                    text = st.write_stream(llm.stream_answer(question, result.context, history, style))
                    if not isinstance(text, str):
                        text = "".join(str(t) for t in text)
                    if not text.strip():
                        raise ValueError("The model returned an empty answer.")
                followups = []
                if want_followups and result.hits:
                    try:
                        followups = llm.followups(result.standalone, text)
                    except Exception:  # noqa: BLE001 — optional extra
                        pass
            except Exception as e:  # noqa: BLE001
                st.error(error_message(e))
                ss.failed = question
                return
    sources = [h.as_source() for h in result.hits]
    valid, invalid = citations(text, len(sources))
    ss.history.append({"id": uuid.uuid4().hex, "role": "user", "content": question,
                       "time": datetime.now().isoformat(timespec="seconds")})
    ss.history.append({"id": uuid.uuid4().hex, "role": "assistant", "content": text, "question": question,
                       "standalone": result.standalone, "queries": result.queries, "sources": sources,
                       "cited": valid, "invalid": invalid, "followups": followups, "notes": result.notes,
                       "settings": {"mode": cfg.mode, "top_k": cfg.top_k, "mmr": cfg.use_mmr,
                                    "rerank": cfg.rerank, "multi_query": cfg.multi_query, "style": style,
                                    "model": llm.chat_model},
                       "time": datetime.now().isoformat(timespec="seconds"), "rating": None})
    ss.failed = None
    st.rerun()


with tab_chat:
    if not has_docs:
        st.markdown(ui.empty_state("📂", "Your library is empty",
                                   "Add documents above, then ask questions. Answers cite the exact passages used."),
                    unsafe_allow_html=True)
    else:
        names = {d["id"]: d["name"] for d in index.docs.values()}
        scope = st.multiselect("Search in", list(names), format_func=lambda i: names.get(i, i),
                               placeholder="All documents", key="scope")
        scope = [s for s in scope if s in names]

        for i, m in enumerate(ss.history):
            if m["role"] == "user":
                with st.chat_message("user", avatar=":material/person:"):
                    st.markdown(m["content"])
            else:
                with st.chat_message("assistant", avatar=":material/auto_awesome:"):
                    render_assistant(m, is_last=i == len(ss.history) - 1)

        live = st.container()

        if ss.failed:
            c1, c2 = st.columns([4, 1], vertical_alignment="center")
            c1.warning(f"Last question failed: “{ss.failed[:120]}”")
            c2.button("🔄 Retry", on_click=queue, args=(ss.failed,), width="stretch")

        suggestions = (ss.history[-1].get("followups") if ss.history else C.STARTER_QUESTIONS) or []
        if suggestions:
            st.caption("Suggested next questions" if ss.history else "Try asking")
            cols = st.columns(min(len(suggestions), 4))
            for col, q in zip(cols, suggestions):
                col.button(q, key=f"sg_{hash(q)}_{len(ss.history)}", on_click=queue, args=(q,), width="stretch")

        prompt = st.chat_input("Ask about your documents…")
        question = (prompt or "").strip() or ss.queued
        ss.queued = None
        if question:
            answer(question, scope, live)


# ═════════════════════════════════════════════════════════════════════════
# Library
# ═════════════════════════════════════════════════════════════════════════
def remove_doc(doc_id: str):
    ss.index.remove(doc_id)
    if not ss.index.docs:
        ss.index = None
    ss.lab = None
    if "scope" in ss:
        ss.scope = [d for d in ss.scope if d != doc_id]


def clear_library():
    ss.index, ss.lab = None, None
    if "scope" in ss:
        ss.scope = []


def make_brief(doc_id: str):
    chunks = index.doc_chunks(doc_id)
    step = max(1, len(chunks) // 16)
    sample = chunks[::step][:16]
    context = "\n\n".join(f"[S{i}] {c['location']}\n{c['text']}" for i, c in enumerate(sample, 1))
    meta = index.docs[doc_id]
    meta["brief"] = get_llm().brief(meta["name"], context)
    meta["brief_sources"] = [c["location"] for c in sample]


def fmt_size(n: int) -> str:
    return f"{n / 1024 / 1024:.1f} MB" if n > 1024 * 1024 else f"{max(1, n // 1024)} KB"


with tab_lib:
    if not has_docs:
        st.markdown(ui.empty_state("📚", "No documents yet", "Documents you add will appear here."),
                    unsafe_allow_html=True)
    else:
        head_l, head_r = st.columns([4, 1], vertical_alignment="center")
        head_l.markdown(f"**{len(index.docs)} documents · {len(index):,} passages** · "
                        f"embeddings: `{index.embed_model}`")
        head_r.button("🗑 Clear library", on_click=clear_library, width="stretch")

        for doc in list(index.docs.values()):
            with st.container(border=True):
                left, right = st.columns([5, 2], vertical_alignment="center")
                with left:
                    st.markdown(f"**{ui.KIND_ICON.get(doc['kind'], '📄')} {doc['name']}**")
                    info = [(doc["kind"], "kind"), (fmt_size(doc["size"]), ""),
                            (f"{doc['passages']} passages", "")]
                    if doc.get("pages"):
                        info.append((f"{doc['pages']} pages", ""))
                    if doc.get("ocr_pages"):
                        info.append((f"{doc['ocr_pages']} OCR pages", "warn"))
                    if doc.get("title"):
                        info.append((f"“{doc['title'][:50]}”", ""))
                    info.append((f"added {doc['added']}", ""))
                    st.markdown(ui.pills(info), unsafe_allow_html=True)
                with right:
                    b1, b2 = st.columns(2)
                    if b1.button("✨ Brief", key=f"br_{doc['id']}", width="stretch",
                                 help="Generate an AI overview of this document"):
                        with st.spinner("Writing brief…"):
                            try:
                                make_brief(doc["id"])
                            except Exception as e:  # noqa: BLE001
                                st.error(error_message(e))
                    b2.button("Remove", key=f"rm_{doc['id']}", on_click=remove_doc, args=(doc["id"],),
                              width="stretch")
                for w in doc.get("warnings", []):
                    st.caption(f"⚠️ {w}")
                if doc.get("brief"):
                    with st.expander("✨ Brief", expanded=True):
                        st.markdown(cite_badges(doc["brief"]))
                        st.caption("Sampled from: " + ", ".join(doc.get("brief_sources", [])[:16]))

        st.markdown("##### Browse passages")
        names = {d["id"]: d["name"] for d in index.docs.values()}
        c1, c2 = st.columns([2, 3])
        pick = c1.selectbox("Document", list(names), format_func=names.get)
        needle = c2.text_input("Filter text", placeholder="Find a word or phrase…")
        rows = [{"#": c["seq"] + 1, "Location": c["location"], "Passage": c["text"]}
                for c in index.doc_chunks(pick) if not needle or needle.lower() in c["text"].lower()]
        st.dataframe(rows, hide_index=True, width="stretch", height=360,
                     column_config={"#": st.column_config.NumberColumn(width="small"),
                                    "Location": st.column_config.TextColumn(width="medium"),
                                    "Passage": st.column_config.TextColumn(width="large")})
        st.caption(f"{len(rows)} passage(s). Passages overlap slightly so sentences at boundaries aren't lost.")


# ═════════════════════════════════════════════════════════════════════════
# Retrieval lab
# ═════════════════════════════════════════════════════════════════════════
with tab_lab:
    st.markdown("Test what the retriever finds for a query **without** generating an answer. "
                "Useful for tuning the sidebar settings.")
    if not has_docs:
        st.info("Add documents first.")
    else:
        with st.form("lab_form"):
            lab_q = st.text_input("Test query", placeholder="e.g. termination clause notice period")
            run = st.form_submit_button("Run retrieval", type="primary")
        if run and lab_q.strip():
            try:
                with st.spinner("Retrieving…"):
                    cfg = settings([d for d in ss.get("scope", []) if d in index.docs])
                    res = retrieve(get_llm(), index, lab_q.strip(), [], cfg)
                ss.lab = {"query": lab_q.strip(), "queries": res.queries,
                          "sources": [h.as_source() for h in res.hits], "context": res.context,
                          "notes": res.notes, "mode": cfg.mode}
            except Exception as e:  # noqa: BLE001
                st.error(error_message(e))
        if ss.lab:
            lab = ss.lab
            st.markdown(ui.pills([(f"mode: {lab['mode']}", "kind"), (f"{len(lab['sources'])} results", "ok")]
                                 + [(f"variant: {q}", "") for q in lab["queries"][1:]]
                                 + [(n, "warn") for n in lab["notes"]]), unsafe_allow_html=True)
            st.dataframe([{"Rank": i, "Document": s["file"], "Location": s["location"],
                           "Semantic": s["dense"] if s["dense"] > -1 else None, "Keyword": s["keyword"],
                           "Fused (RRF)": s["fused"], "Re-rank": s["rerank"], "Preview": s["text"][:160]}
                          for i, s in enumerate(lab["sources"], 1)],
                         hide_index=True, width="stretch",
                         column_config={"Semantic": st.column_config.ProgressColumn(min_value=0, max_value=1,
                                                                                    format="%.3f"),
                                        "Fused (RRF)": st.column_config.NumberColumn(format="%.4f")})
            with st.expander("Context exactly as sent to the answer model"):
                st.code(lab["context"] or "(empty)", language=None, wrap_lines=True)


# ═════════════════════════════════════════════════════════════════════════
# Export & save
# ═════════════════════════════════════════════════════════════════════════
def transcript_markdown() -> str:
    lines = [f"# {APP_TITLE} conversation", f"_Exported {datetime.now():%Y-%m-%d %H:%M}_", ""]
    for m in ss.history:
        if m["role"] == "user":
            lines += [f"## Q: {m['content']}", ""]
        else:
            lines += [m["content"], ""]
            if m.get("sources"):
                lines.append("**Sources**")
                for i, s in enumerate(m["sources"], 1):
                    mark = " ✓" if i in m.get("cited", []) else ""
                    lines.append(f"- [S{i}]{mark} {s['file']} · {s['location']}")
                lines.append("")
    return "\n".join(lines)


with tab_export:
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    c1, c2 = st.columns(2)
    with c1, st.container(border=True):
        st.markdown("**💬 Conversation**")
        st.caption("Questions, answers, cited sources and your ratings.")
        st.download_button("⇩ Markdown transcript", transcript_markdown(), f"conversation-{stamp}.md",
                           "text/markdown", width="stretch", disabled=not ss.history)
        st.download_button("⇩ Full JSON (with evidence)", json.dumps(ss.history, ensure_ascii=False, indent=2),
                           f"conversation-{stamp}.json", "application/json", width="stretch",
                           disabled=not ss.history)
    with c2, st.container(border=True):
        st.markdown("**📚 Library**")
        st.caption("Save the processed library (text + embeddings) and restore it later — no re-processing or "
                   "embedding costs.")
        if has_docs:
            st.download_button("⇩ Save library (.zip)", index.to_bytes(), f"rag-document-studio-library-{stamp}.zip",
                               "application/zip", width="stretch", type="primary")
            st.download_button("⇩ Passages (JSON)", json.dumps(index.chunks, ensure_ascii=False, indent=1),
                               f"passages-{stamp}.json", "application/json", width="stretch")
        saved = st.file_uploader("Restore a saved library", type=["zip"], key=f"restore_{ss.nonce}")
        if saved and st.button("↺ Restore library", width="stretch"):
            try:
                count = restore_library(saved.getvalue())
                ss.report = {"added": [f"{count} document(s) from saved library"], "skipped": [], "failed": []}
                ss.nonce += 1
                st.rerun()
            except ValueError as e:
                st.error(str(e))
    st.caption("Saved libraries contain your document text. Store them as carefully as the originals.")
