"""OpenAI (or OpenAI-compatible) client wrapper with usage tracking."""
from __future__ import annotations

import base64
import io
import json
import re
from typing import Iterator, Optional

import numpy as np
from openai import (APIConnectionError, APITimeoutError, AuthenticationError, BadRequestError,
                    NotFoundError, OpenAI, PermissionDeniedError, RateLimitError)
from PIL import Image, ImageOps

from . import config as C
from . import prompts as P


def normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        return vectors / max(float(np.linalg.norm(vectors)), 1e-12)
    return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)


def _parse_json(text: str) -> dict:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return {}


def _image_to_data_url(image) -> str:
    if isinstance(image, (bytes, bytearray)):
        image = Image.open(io.BytesIO(image))
    with image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((C.IMAGE_MAX_SIDE, C.IMAGE_MAX_SIDE))
        out = io.BytesIO()
        image.save(out, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(out.getvalue()).decode()


def _history_messages(history: list, limit: int = 6, chars: int = 700) -> list:
    messages = []
    for m in history[-limit:]:
        if m.get("role") in ("user", "assistant") and m.get("content"):
            messages.append({"role": m["role"], "content": m["content"][:chars]})
    return messages


class LLM:
    def __init__(self, api_key: str, *, chat_model: str, vision_model: str, embed_model: str,
                 base_url: Optional[str] = None, usage: Optional[dict] = None):
        self.client = OpenAI(api_key=api_key, base_url=base_url or None, timeout=120, max_retries=2)
        self.chat_model = chat_model.strip()
        self.vision_model = vision_model.strip()
        self.embed_model = embed_model.strip()
        self.usage = usage if usage is not None else {}
        for key in ("prompt", "completion", "embedding", "calls"):
            self.usage.setdefault(key, 0)

    # ── accounting ──
    def _track(self, usage, embedding: bool = False):
        self.usage["calls"] += 1
        if not usage:
            return
        if embedding:
            self.usage["embedding"] += getattr(usage, "total_tokens", 0) or 0
        else:
            self.usage["prompt"] += getattr(usage, "prompt_tokens", 0) or 0
            self.usage["completion"] += getattr(usage, "completion_tokens", 0) or 0

    # ── primitives ──
    def embed(self, texts: list[str], progress=None) -> np.ndarray:
        vectors = []
        for start in range(0, len(texts), C.EMBED_BATCH):
            batch = [t[:24000] or " " for t in texts[start:start + C.EMBED_BATCH]]
            response = self.client.embeddings.create(model=self.embed_model, input=batch)
            self._track(response.usage, embedding=True)
            vectors.extend(d.embedding for d in sorted(response.data, key=lambda d: d.index))
            if progress:
                progress(min(1.0, (start + len(batch)) / len(texts)))
        return normalize(np.asarray(vectors, dtype=np.float32))

    def _chat(self, messages: list, json_mode: bool = False, model: Optional[str] = None) -> str:
        kwargs = {"model": model or self.chat_model, "messages": messages}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            response = self.client.chat.completions.create(**kwargs)
        except BadRequestError:
            if not json_mode:
                raise
            kwargs.pop("response_format")            # model/server without JSON mode
            response = self.client.chat.completions.create(**kwargs)
        self._track(response.usage)
        return response.choices[0].message.content or ""

    # ── document reading ──
    def transcribe_image(self, image) -> str:
        text = self._chat([{"role": "user", "content": [
            {"type": "text", "text": P.OCR_PROMPT},
            {"type": "image_url", "image_url": {"url": _image_to_data_url(image)}},
        ]}], model=self.vision_model).strip()
        if not text or text == "[NO LEGIBLE TEXT]":
            raise ValueError("No legible text was found in the image.")
        return text

    # ── retrieval helpers ──
    def rewrite_query(self, question: str, history: list) -> str:
        context = _history_messages(history, limit=6, chars=500)
        if not context:
            return question
        transcript = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in context)
        text = self._chat([
            {"role": "system", "content": P.REWRITE_SYSTEM},
            {"role": "user", "content": f"CONVERSATION:\n{transcript}\n\nLATEST QUESTION:\n{question}"},
        ]).strip().strip('"')
        return text[:500] if text else question

    def expand_queries(self, question: str, n: int = 3) -> list[str]:
        data = _parse_json(self._chat([
            {"role": "system", "content": P.MULTI_QUERY_SYSTEM.format(n=n)},
            {"role": "user", "content": question},
        ], json_mode=True))
        queries = [q.strip() for q in data.get("queries", []) if isinstance(q, str) and q.strip()]
        return queries[:n]

    def rerank(self, question: str, passages: list[tuple[str, str]]) -> dict[str, float]:
        """passages: [(id, text)] -> {id: score 0..10}"""
        listing = "\n\n".join(f"<passage id=\"{pid}\">\n{text[:900]}\n</passage>" for pid, text in passages)
        data = _parse_json(self._chat([
            {"role": "system", "content": P.RERANK_SYSTEM},
            {"role": "user", "content": f"QUESTION: {question}\n\nPASSAGES:\n{listing}"},
        ], json_mode=True))
        scores = {}
        for pid, value in (data.get("scores") or {}).items():
            try:
                scores[str(pid)] = float(value)
            except (TypeError, ValueError):
                pass
        return scores

    # ── answering ──
    def _answer_messages(self, question: str, context: str, history: list, style: str) -> list:
        return [
            {"role": "system", "content": P.ANSWER_SYSTEM.format(style=C.ANSWER_STYLES.get(style, style))},
            *_history_messages(history, limit=4),
            {"role": "user", "content": f"PASSAGES:\n{context}\n\nQUESTION:\n{question}"},
        ]

    def stream_answer(self, question: str, context: str, history: list, style: str) -> Iterator[str]:
        kwargs = {"model": self.chat_model, "messages": self._answer_messages(question, context, history, style),
                  "stream": True, "stream_options": {"include_usage": True}}
        try:
            stream = self.client.chat.completions.create(**kwargs)
        except BadRequestError:
            kwargs.pop("stream_options")             # some compatible servers reject it
            stream = self.client.chat.completions.create(**kwargs)
        tracked = False
        for chunk in stream:
            if getattr(chunk, "usage", None):
                self._track(chunk.usage)
                tracked = True
            if chunk.choices:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        if not tracked:
            self._track(None)

    def followups(self, question: str, answer: str) -> list[str]:
        data = _parse_json(self._chat([
            {"role": "system", "content": P.FOLLOWUP_SYSTEM},
            {"role": "user", "content": f"QUESTION: {question}\n\nANSWER: {answer[:2500]}"},
        ], json_mode=True))
        return [q.strip() for q in data.get("questions", []) if isinstance(q, str) and q.strip()][:3]

    def brief(self, file_name: str, context: str) -> str:
        return self._chat([
            {"role": "system", "content": P.BRIEF_SYSTEM},
            {"role": "user", "content": f"DOCUMENT: {file_name}\n\nPASSAGES:\n{context}"},
        ])


def error_message(error: Exception) -> str:
    if isinstance(error, AuthenticationError):
        return "API key rejected. Use “Change API key” in the sidebar."
    if isinstance(error, PermissionDeniedError):
        return "Your API key doesn't have access to this model or feature."
    if isinstance(error, RateLimitError):
        return "Rate limit or quota reached. Check your account's billing and limits, then retry."
    if isinstance(error, NotFoundError):
        return "Model not found. Check the model names in the sidebar (Models)."
    if isinstance(error, BadRequestError):
        msg = getattr(error, "message", "") or str(error)
        return f"The API rejected the request: {msg[:300]}"
    if isinstance(error, (APIConnectionError, APITimeoutError)):
        return "Couldn't reach the API. Check your connection or base URL and retry."
    if isinstance(error, ValueError):
        return str(error)
    return f"Something went wrong: {type(error).__name__}."
