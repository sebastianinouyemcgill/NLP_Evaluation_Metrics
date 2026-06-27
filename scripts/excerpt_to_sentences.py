#!/usr/bin/env python
"""Convert excerpt JSON to aligned sentence JSON using spaCy."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, List


def normalize_text(text: str) -> str:
    """Replace newlines with spaces, collapse whitespace, and strip."""
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _load_spacy_models():
    import spacy

    try:
        nlp_en = spacy.load("en_core_web_sm")
    except OSError as e:
        raise OSError(
            "spaCy English model not found. Run: python -m spacy download en_core_web_sm"
        ) from e

    try:
        nlp_fr = spacy.load("fr_core_news_sm")
    except OSError as e:
        raise OSError(
            "spaCy French model not found. Run: python -m spacy download fr_core_news_sm"
        ) from e

    return nlp_en, nlp_fr


def split_sentences(text: str, nlp) -> List[str]:
    doc = nlp(normalize_text(text))
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def _reference_text(record: dict, index: int) -> str:
    if "reference" in record and isinstance(record["reference"], str):
        return record["reference"]

    refs = record.get("references")
    if isinstance(refs, list) and refs and isinstance(refs[0], str):
        return refs[0]

    raise ValueError(f"Record {index} must have a string 'reference' or non-empty 'references' list")


def convert_excerpt(
    record: dict,
    index: int,
    nlp_en,
    nlp_fr,
    on_mismatch: str = "skip",
) -> List[dict[str, Any]] | None:
    page_id = str(record.get("id", index))

    for field in ("source", "prediction"):
        if not isinstance(record.get(field), str):
            raise ValueError(f"Record {page_id} must have a string '{field}'")

    reference = _reference_text(record, index)

    source_sents = split_sentences(record["source"], nlp_en)
    pred_sents = split_sentences(record["prediction"], nlp_fr)
    ref_sents = split_sentences(reference, nlp_fr)

    counts = (len(source_sents), len(pred_sents), len(ref_sents))
    if len(set(counts)) != 1:
        msg = (
            f"page {page_id!r} — sentence count mismatch "
            f"(source={counts[0]}, prediction={counts[1]}, reference={counts[2]})"
        )
        if on_mismatch == "skip":
            print(f"Warning: skipping {msg}", file=sys.stderr)
            return None

        n = min(counts)
        print(
            f"Warning: truncating {msg} to first {n} sentence(s)",
            file=sys.stderr,
        )
        source_sents = source_sents[:n]
        pred_sents = pred_sents[:n]
        ref_sents = ref_sents[:n]

    if not source_sents:
        print(f"Warning: skipping page {page_id!r} — no sentences after splitting", file=sys.stderr)
        return None

    return [
        {
            "page_id": page_id,
            "sentence_id": sent_idx,
            "source": src,
            "prediction": pred,
            "reference": ref,
        }
        for sent_idx, (src, pred, ref) in enumerate(
            zip(source_sents, pred_sents, ref_sents), start=1
        )
    ]


def convert_file(input_path: Path, output_path: Path, on_mismatch: str = "skip") -> int:
    with input_path.open(encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of records")

    nlp_en, nlp_fr = _load_spacy_models()

    sentences: List[dict[str, Any]] = []
    for i, record in enumerate(data):
        if not isinstance(record, dict):
            raise ValueError(f"Record {i} must be a JSON object")

        page_sentences = convert_excerpt(record, i, nlp_en, nlp_fr, on_mismatch)
        if page_sentences is not None:
            sentences.extend(page_sentences)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(sentences, f, ensure_ascii=False, indent=2)
        f.write("\n")

    return len(sentences)


def resolve_output_path(input_path: Path, output_arg: str | None) -> Path:
    """Map ``data/excerpts/{id}_excerpt.json`` → ``data/sentences/{id}_sentences.json``."""
    if output_arg:
        return Path(output_arg)

    stem = input_path.stem
    if not stem.endswith("_excerpt"):
        raise ValueError(
            f"Input filename must end with '_excerpt' (got {input_path.name!r}). "
            "Pass --output explicitly or rename the file."
        )

    base = stem[: -len("_excerpt")]
    if input_path.parent.name == "excerpts":
        return input_path.parent.parent / "sentences" / f"{base}_sentences.json"

    return input_path.parent / "sentences" / f"{base}_sentences.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split excerpt-level JSON into sentence-level JSON using spaCy."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to excerpt JSON (e.g. data/excerpts/7_excerpt.json).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path (default: data/sentences/{id}_sentences.json from input name).",
    )
    parser.add_argument(
        "--on-mismatch",
        choices=["skip", "truncate"],
        default="skip",
        help="When sentence counts differ: skip the page (default) or pair the first N sentences.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    try:
        output_path = resolve_output_path(input_path, args.output)
    except ValueError as e:
        parser.error(str(e))

    if not input_path.is_file():
        parser.error(f"Input file not found: {input_path}")

    count = convert_file(input_path, output_path, on_mismatch=args.on_mismatch)
    print(f"Wrote {count} sentences to {output_path}")


if __name__ == "__main__":
    main()
