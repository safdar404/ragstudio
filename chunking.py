"""Sentence-aware chunking with overlap and contextual headers.

Passages never cut a word or (where possible) a sentence in half, overlap is
made of whole sentences, and each passage is embedded together with its
document name and section so that short passages keep their context.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from . import config as C

_SENTENCE_END = re.compile(r"(?<=[.!?…。！？])[\"'”’)\]]*\s+")


@dataclass
class _Unit:
    text: str
    new_paragraph: bool


def _hard_split(text: str, size: int):
    while len(text) > size:
        cut = text.rfind(" ", int(size * 0.6), size)
        if cut <= 0:
            cut = size
        yield text[:cut].strip()
        text = text[cut:].strip()
    if text:
        yield text


def _units(text: str, size: int):
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        pieces = [paragraph] if len(paragraph) <= size else _SENTENCE_END.split(paragraph)
        first = True
        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            for part in _hard_split(piece, size):
                yield _Unit(part, first)
                first = False


def _join(units) -> str:
    out = ""
    for u in units:
        if not out:
            out = u.text
        else:
            out += ("\n\n" if u.new_paragraph else " ") + u.text
    return out


def chunk_text(text: str, size: int = C.CHUNK_SIZE, overlap: int = C.CHUNK_OVERLAP) -> list[str]:
    chunks, current, length = [], [], 0
    for unit in _units(text, size):
        if current and length + len(unit.text) + 1 > size:
            chunks.append(_join(current))
            tail, tail_len = [], 0
            for prev in reversed(current):
                if tail_len + len(prev.text) > overlap:
                    break
                tail.insert(0, prev)
                tail_len += len(prev.text) + 1
            current, length = tail, tail_len
        current.append(unit)
        length += len(unit.text) + 1
    if current:
        chunks.append(_join(current))
    return [c for c in chunks if c.strip()]


def build_chunks(doc_id: str, file_name: str, sections, size: int = C.CHUNK_SIZE,
                 overlap: int = C.CHUNK_OVERLAP) -> list[dict]:
    records, seq = [], 0
    for section in sections:
        for text in chunk_text(section.text, size, overlap):
            digest = hashlib.sha1(f"{doc_id}|{seq}|{text}".encode()).hexdigest()[:12]
            records.append({
                "id": digest,
                "doc_id": doc_id,
                "file": file_name,
                "location": section.location,
                "page": section.page,
                "heading": section.heading,
                "seq": seq,
                "text": text,
            })
            seq += 1
    return records


def embed_text(chunk: dict) -> str:
    """Text actually sent to the embedding model: passage plus its context."""
    return f"Document: {chunk['file']}\nSection: {chunk['location']}\n\n{chunk['text']}"
