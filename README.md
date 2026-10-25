# ✦ Document Studio 2.0

Ask questions across PDFs, Word files, images, text, Markdown and CSV, and inspect every passage behind each answer.

## Run it (Python 3.12)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run rag.py --server.port 8503
```

macOS/Linux: `python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/streamlit run rag.py`

Enter your OpenAI API key (or set `OPENAI_API_KEY` and leave the field empty), add documents, and ask.
An optional **API base URL** lets you use OpenAI-compatible servers (Azure gateways, LiteLLM, vLLM, Ollama).

## What's new in 2.0

**Retrieval (RAG)**
- **Hybrid search** — semantic embeddings + BM25 keyword search, merged with Reciprocal Rank Fusion. Exact terms (codes, names, numbers) are no longer missed. Semantic-only and keyword-only modes are also available.
- **Follow-up understanding** — questions like “what about the second one?” are rewritten into standalone queries using the chat history.
- **Multi-query search** (optional) — extra AI-generated phrasings improve recall.
- **AI re-ranking** (optional) — candidates are graded by the model before answering.
- **MMR diversification** — avoids sending several near-duplicate passages.
- **Surrounding context** (optional) — adds text just before/after each passage.
- **Minimum similarity** filter and **document scope** (“Search in…”) per question.
- **Smarter chunking** — sentence-aligned passages with sentence overlap; each passage is embedded with its document name and section heading.
- **Citation checking** — cited sources are highlighted; invalid or missing citations are flagged.

**Documents**
- **Scanned PDF pages** are read by the vision model (up to 25 pages per PDF) instead of being dropped.
- **Word**: reading order preserved, sections labelled by heading (e.g. `Budget › Costs · ¶4–7`), merged table cells de-duplicated, **text boxes** and **headers/footers** extracted.
- New formats: **TXT, Markdown** (split by heading), **CSV** (rows indexed as `column: value`).
- Password-protected PDF detection and clearer per-page warnings.

**Workflow**
- **Incremental library** — add files any time; duplicates are skipped; one failing file no longer blocks the others; remove individual documents.
- **Save / restore library** (.zip with text + embeddings) — no re-processing or embedding cost next session.
- **Streaming answers**, answer styles, suggested follow-up questions, regenerate, 👍/👎 ratings.
- **Document briefs** — one-click AI overview of any document.
- **Retrieval lab** — test queries and see semantic / keyword / fused / re-rank scores and the exact context sent to the model.
- **Token usage** counter; Markdown and JSON exports with evidence.

**Design** — redesigned interface: compact header with live stats, source cards with relevance bars and term highlighting, citation badges, grouped sidebar settings, empty states.

## Project layout

```
rag.py                 Streamlit app (UI and flow)
studio/config.py       limits, defaults, RetrievalSettings
studio/extractors.py   PDF / Word / image / TXT / MD / CSV extraction
studio/chunking.py     sentence-aware chunking + contextual headers
studio/index.py        vector store + BM25, add/remove, save/restore
studio/pipeline.py     rewrite → multi-query → hybrid search → RRF → re-rank/MMR → context
studio/llm.py          OpenAI wrapper: embeddings, streaming, helpers, usage tracking
studio/prompts.py      prompt templates
studio/ui.py           CSS theme and HTML components (all text escaped)
tests/                 offline tests (fake model, no API key needed)
```

Run tests: `pip install pytest reportlab` then `python -m pytest -q`.

## Tuning tips
- Precise lookups (IDs, clause numbers): **Keyword** or **Hybrid** mode.
- Vague or conceptual questions: turn on **Multi-query search**.
- Answers missing detail: raise **Passages per answer** or enable **Surrounding context**.
- Answers drawing on irrelevant text: enable **AI re-ranking** or raise **Minimum similarity**.
- Use the **Retrieval lab** to see the effect of any change before asking.

## Limits & privacy
Up to 20 files per batch, 25 MB each, 500 PDF pages, 6,000 passages per library. Images are resized to 2,000 px per side; small text may lose detail.
Document text, images and questions are sent to the configured API; usage may incur charges. The key, library and chat live in the browser session only — save the library and export chats to keep them. Saved libraries contain your document text. No authentication or multi-user storage is included. Transcriptions and AI citations should be reviewed for important decisions.
