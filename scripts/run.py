import argparse
import json

from evaluation.evaluate import evaluate


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    predictions = []
    references = []

    for item in data:
        predictions.append(item["prediction"])
        references.append(item["references"])

    return predictions, references


def main():
    parser = argparse.ArgumentParser(description="Run BLEU, METEOR, BLEURT evaluation")

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to JSON file with predictions and references"
    )

    parser.add_argument(
        "--bleurt_checkpoint",
        type=str,
        default="BLEURT-20",
        help="Path or name of BLEURT checkpoint (e.g., BLEURT-20)"
    )

    args = parser.parse_args()

    predictions, references = load_json(args.input)

    results = evaluate(
        predictions=predictions,
        references=references,
        bleurt_checkpoint=args.bleurt_checkpoint
    )

    print("\n=== Evaluation Results ===")
    print(f"BLEU:   {results['bleu']:.4f}")
    print(f"METEOR: {results['meteor']:.4f}")
    print(f"BLEURT: {results['bleurt']:.4f}")


if __name__ == "__main__":
    main()