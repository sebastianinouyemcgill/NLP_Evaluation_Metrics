#!/usr/bin/env python
"""CLI entry point for MT evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.evaluate import evaluate  # noqa: E402

METRIC_CHOICES = ["bleu", "meteor", "bleurt", "cometkiwi"]
METEOR_LANGUAGES = ["en", "cz", "de", "es", "fr"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate machine translation outputs.")
    parser.add_argument("--data", required=True, help="Path to input JSON file.")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["bleu", "meteor"],
        choices=METRIC_CHOICES,
        help="Metrics to compute.",
    )
    parser.add_argument(
        "--bleurt-checkpoint",
        default=None,
        help="Path to a BLEURT checkpoint directory (required for bleurt).",
    )
    parser.add_argument(
        "--meteor-language",
        default="en",
        choices=METEOR_LANGUAGES,
        help="Target language for METEOR (official JVM implementation).",
    )
    parser.add_argument(
        "--cometkiwi-model",
        default="Unbabel/wmt22-cometkiwi-da",
        help="COMET model name or checkpoint path (for cometkiwi).",
    )
    args = parser.parse_args()

    if "bleurt" in args.metrics and not args.bleurt_checkpoint:
        parser.error("--bleurt-checkpoint is required when --metrics includes bleurt")

    results = evaluate(
        data_path=args.data,
        metrics=args.metrics,
        bleurt_checkpoint=args.bleurt_checkpoint,
        meteor_language=args.meteor_language,
        cometkiwi_model=args.cometkiwi_model,
    )

    num_examples = results.pop("num_examples")
    scores = {name: round(value, 4) for name, value in results.items()}

    print(f"Evaluated {num_examples} examples")
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
