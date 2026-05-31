"""Simple normalization and code-switch tagging utilities.

This module implements a lightweight, dependency-free heuristic to detect
Indonesian vs English segments, perform basic text normalization, and
wrap segments with language tags so the LLM can preserve code-switching.

It's intentionally conservative: it preserves negation words and avoids
aggressive tokenization so downstream TTS/LLM behavior remains predictable.
"""
from __future__ import annotations

import re
from typing import List, Tuple

# Small lists of keywords to help heuristic language detection.
ID_KEYWORDS = {
    "saya",
    "kamu",
    "kita",
    "tidak",
    "belum",
    "tapi",
    "ada",
    "tanpa",
    "dan",
    "di",
    "ke",
    "dari",
}

EN_KEYWORDS = {
    "the",
    "is",
    "you",
    "i",
    "we",
    "and",
    "to",
    "of",
    "in",
}


def _split_sentences(text: str) -> List[str]:
    # Very small splitter based on punctuation, keeps delimiter
    parts = re.split(r"([\.!?]+)\s+", text)
    sentences: List[str] = []
    for i in range(0, len(parts), 2):
        main = parts[i].strip()
        delim = parts[i + 1] if i + 1 < len(parts) else ""
        if main:
            sentences.append((main + delim).strip())
    if not sentences and text.strip():
        return [text.strip()]
    return sentences


def detect_language_heuristic(text: str) -> str:
    """Return 'id' or 'en' based on keyword counts; defaults to 'id' for short texts."""
    if not text or not text.strip():
        return "id"
    tokens = re.findall(r"\w+", text.lower())
    id_count = sum(1 for t in tokens if t in ID_KEYWORDS)
    en_count = sum(1 for t in tokens if t in EN_KEYWORDS)
    # Favor Indonesian when counts equal or text is short
    if id_count >= en_count:
        return "id"
    return "en"


def normalize_segment(text: str) -> str:
    """Basic normalization: trim spaces, normalize whitespace, keep case minimal.

    Preserves negation words (tidak, belum, etc.) and does not expand abbreviations.
    """
    s = text.strip()
    s = re.sub(r"\s+", " ", s)
    # Normalize fancy quotes and dashes
    s = s.replace("“", '"').replace("”", '"').replace("–", "-")
    return s


def tag_code_switching(text: str) -> Tuple[str, List[Tuple[str, str]]]:
    """Split text into language-tagged segments and return tagged text

    Returns (tagged_text, segments) where segments is list of (lang, segment).
    Tag format: [ID]...[/ID] and [EN]...[/EN]
    """
    sentences = _split_sentences(text)
    segments: List[Tuple[str, str]] = []
    for sent in sentences:
        lang = detect_language_heuristic(sent)
        norm = normalize_segment(sent)
        segments.append((lang, norm))

    # Merge consecutive segments with same lang
    merged: List[Tuple[str, str]] = []
    for lang, seg in segments:
        if merged and merged[-1][0] == lang:
            merged[-1] = (lang, merged[-1][1] + " " + seg)
        else:
            merged.append((lang, seg))

    tagged_parts: List[str] = []
    for lang, seg in merged:
        if lang == "id":
            tagged_parts.append(f"[ID]{seg}[/ID]")
        else:
            tagged_parts.append(f"[EN]{seg}[/EN]")

    return " ".join(tagged_parts), merged


def normalize_and_tag(text: str) -> dict:
    """Convenience function returning normalized text and metadata."""
    tagged, segments = tag_code_switching(text)
    return {"original": text, "tagged": tagged, "segments": segments}
