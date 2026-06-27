#!/usr/bin/env python
"""Ingest source/reference excerpts, align sentences, translate via Google Cloud."""

from __future__ import annotations

import argparse
import html
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, List, Literal

from google.cloud import translate_v2 as translate

logger = logging.getLogger(__name__)

AlignmentStatus = Literal["aligned", "adjusted", "forced_truncate"]

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_PUNKT_RESOURCES = ("punkt", "punkt_tab")


def resolve_credentials_path() -> str | None:
    """Use GOOGLE_APPLICATION_CREDENTIALS, or the first *.json key in secrets/."""
    env_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if env_path and Path(env_path).is_file():
        return env_path

    secrets_dir = Path(__file__).resolve().parent.parent / "secrets"
    keys = sorted(secrets_dir.glob("*.json"))
    if keys:
        return str(keys[0])

    return env_path


def _ensure_nltk_data() -> None:
    import nltk

    for resource in _PUNKT_RESOURCES:
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


def normalize_text(text: str) -> str:
    """Collapse whitespace and strip; punctuation is preserved."""
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def sentence_split(text: str) -> List[str]:
    """Split text into sentences using NLTK, with regex fallback."""
    normalized = normalize_text(text)
    if not normalized:
        return []

    try:
        from nltk.tokenize import sent_tokenize

        sentences = sent_tokenize(normalized)
    except Exception:
        logger.debug("NLTK sentence tokenization failed; using regex fallback.")
        sentences = _SENTENCE_SPLIT_RE.split(normalized)

    return [s.strip() for s in sentences if s.strip()]


def validate_alignment(source: List[str], reference: List[str]) -> bool:
    return len(source) == len(reference)


def _merge_last_into_previous(sentences: List[str]) -> List[str]:
    if len(sentences) < 2:
        return sentences
    return sentences[:-2] + [f"{sentences[-2]} {sentences[-1]}".strip()]


def align_source_reference(
    source: List[str],
    reference: List[str],
    entry_id: Any,
) -> tuple[List[str], List[str], AlignmentStatus]:
    """Align source and reference sentence lists deterministically."""
    source = list(source)
    reference = list(reference)

    if validate_alignment(source, reference):
        return source, reference, "aligned"

    logger.warning("Sentence mismatch for id=%s", entry_id)

    shortest = min(len(source), len(reference))
    source = source[:shortest]
    reference = reference[:shortest]

    if validate_alignment(source, reference):
        return source, reference, "forced_truncate"

    max_steps = len(source) + len(reference)
    for _ in range(max_steps):
        if validate_alignment(source, reference):
            return source, reference, "adjusted"

        if len(source) > len(reference) and len(source) > 1:
            source = _merge_last_into_previous(source)
        elif len(reference) > len(source) and len(reference) > 1:
            reference = _merge_last_into_previous(reference)
        else:
            break

    shortest = min(len(source), len(reference))
    return source[:shortest], reference[:shortest], "adjusted"


def translate_sentences(
    client: translate.Client,
    source_sentences: List[str],
    target_language: str = "fr",
) -> List[str]:
    prediction_sentences: List[str] = []
    for sentence in source_sentences:
        result = client.translate(sentence, target_language=target_language)
        prediction_sentences.append(html.unescape(result["translatedText"]))
    return prediction_sentences


def _load_entries(path: Path) -> List[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError("Input JSON must be a list of records")

    entries: List[dict[str, Any]] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Record {i} must be a JSON object")

        entry_id = item.get("id", i)
        source = item.get("source")
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"Record {i} must have a non-empty string 'source'")

        reference = item.get("reference", item.get("references"))
        if isinstance(reference, list):
            if len(reference) != 1 or not isinstance(reference[0], str):
                raise ValueError(
                    f"Record {i} must have a string 'reference' (or single-item list)"
                )
            reference = reference[0]
        if not isinstance(reference, str) or not reference.strip():
            raise ValueError(f"Record {i} must have a non-empty string 'reference'")

        entries.append({"id": entry_id, "source": source, "reference": reference})

    return entries


def process_entry(
    entry: dict[str, Any],
    client: translate.Client,
) -> tuple[dict[str, Any], dict[str, Any]]:
    entry_id = entry["id"]
    source_text = normalize_text(entry["source"])
    reference_text = normalize_text(entry["reference"])

    source_sentences = sentence_split(source_text)
    reference_sentences = sentence_split(reference_text)

    source_sentences, reference_sentences, alignment_status = align_source_reference(
        source_sentences,
        reference_sentences,
        entry_id,
    )

    prediction_sentences = translate_sentences(client, source_sentences)
    prediction_text = " ".join(prediction_sentences)
    num_sentences = len(source_sentences)

    excerpt = {
        "id": entry_id,
        "source": source_text,
        "prediction": prediction_text,
        "references": [reference_text],
        "metadata": {
            "num_sentences": num_sentences,
            "alignment_status": alignment_status,
        },
    }

    sentences = {
        "id": entry_id,
        "source_sentences": source_sentences,
        "prediction_sentences": prediction_sentences,
        "reference_sentences": reference_sentences,
        "num_sentences": num_sentences,
    }

    return excerpt, sentences


def write_entry_outputs(
    entry_id: Any,
    excerpt: dict[str, Any],
    sentences: dict[str, Any],
    excerpt_dir: Path,
    sentences_dir: Path,
) -> None:
    excerpt_dir.mkdir(parents=True, exist_ok=True)
    sentences_dir.mkdir(parents=True, exist_ok=True)

    excerpt_path = excerpt_dir / f"{entry_id}_excerpt.json"
    sentences_path = sentences_dir / f"{entry_id}_sentences.json"

    with excerpt_path.open("w", encoding="utf-8") as f:
        json.dump(excerpt, f, ensure_ascii=False, indent=2)
        f.write("\n")

    with sentences_path.open("w", encoding="utf-8") as f:
        json.dump(sentences, f, ensure_ascii=False, indent=2)
        f.write("\n")

    logger.info(
        "Wrote %s and %s (%d sentences, %s)",
        excerpt_path,
        sentences_path,
        sentences["num_sentences"],
        excerpt["metadata"]["alignment_status"],
    )


def process_dataset(
    input_path: Path,
    excerpt_dir: Path,
    sentences_dir: Path,
    client: translate.Client,
) -> int:
    entries = _load_entries(input_path)
    for entry in entries:
        excerpt, sentences = process_entry(entry, client)
        write_entry_outputs(entry["id"], excerpt, sentences, excerpt_dir, sentences_dir)
    return len(entries)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description=(
            "Convert manual source/reference excerpts into aligned excerpt- and "
            "sentence-level JSON with Google Translate predictions."
        )
    )
    parser.add_argument("--input", required=True, help="Path to input JSON (list of records).")
    parser.add_argument(
        "--excerpt-dir",
        default="data/excerpts",
        help="Output directory for excerpt files (default: data/excerpts).",
    )
    parser.add_argument(
        "--sentences-dir",
        default="data/sentences",
        help="Output directory for sentence files (default: data/sentences).",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        parser.error(f"Input file not found: {input_path}")

    creds_path = resolve_credentials_path()
    if not creds_path or not Path(creds_path).is_file():
        parser.error(
            "Google credentials not found. Set GOOGLE_APPLICATION_CREDENTIALS "
            "or place a service-account JSON key in secrets/."
        )
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path

    _ensure_nltk_data()
    client = translate.Client()
    count = process_dataset(
        input_path,
        Path(args.excerpt_dir),
        Path(args.sentences_dir),
        client,
    )
    logger.info("Processed %d entries.", count)


if __name__ == "__main__":
    main()
