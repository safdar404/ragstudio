"""Document Studio — retrieval-augmented Q&A over PDFs, Word, images and text files.

Run:  streamlit run rag.py
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

st.set_page_config(page_title="Document Studio", page_icon="✦", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown(ui.CSS, unsafe_allow_html=True)

ss = st.session_state
for key, default in {"history": [], "usage": {}, "index": None, "queued": None, "failed": None,
                     "lab": None, "nonce": 0, "report": None}.items():
    ss.setdefault(key, default)


# ═════════════════════════════════════════════════════════════════════════
# Connect
# ═════════════════════════════════════════════════════════════════════════
def connect_screen():
    st.markdown(ui.hero(big=True), unsafe_allow_html=True)
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
    st.markdown(ui.sidebar_brand(C.VERSION), unsafe_allow_html=True)

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
st.markdown(ui.hero([("documents", len(index.docs) if index else 0),
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
    st.markdown(ui.style_citations(m["content"]))
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
                        st.markdown(ui.style_citations(doc["brief"]))
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
    lines = [f"# Document Studio conversation", f"_Exported {datetime.now():%Y-%m-%d %H:%M}_", ""]
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
            st.download_button("⇩ Save library (.zip)", index.to_bytes(), f"document-studio-library-{stamp}.zip",
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
