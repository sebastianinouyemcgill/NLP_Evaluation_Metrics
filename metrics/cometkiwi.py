"""Corpus-level COMETKiwi (reference-free, requires source per example)."""

from __future__ import annotations

from typing import List


def corpus_cometkiwi(
    sources: List[str],
    predictions: List[str],
    model_name: str = "Unbabel/wmt22-cometkiwi-da",
) -> float:
    """Return mean sentence-level COMETKiwi score (typically between 0 and 1)."""
    if len(sources) != len(predictions):
        raise ValueError("sources and predictions must have the same length")

    try:
        import torch
        from comet import download_model, load_from_checkpoint
    except ImportError as e:
        raise ImportError(
            "COMET is not installed. Install dependencies with "
            "pip install -r requirements.txt"
        ) from e

    model_path = download_model(model_name)
    model = load_from_checkpoint(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    scores: List[float] = []
    with torch.no_grad():
        for src, pred in zip(sources, predictions):
            prepared = model.prepare_sample([{"src": src, "mt": pred}], stage="predict")
            score = _score_prepared(model, prepared, device)
            scores.append(float(score.cpu()))

    return sum(scores) / len(scores)


def _score_prepared(model, prepared, device):
    """Score one example; handles UnifiedMetric (tuple) and legacy Regression models (dict)."""
    import torch

    if isinstance(prepared, tuple):
        if len(prepared) == 3:
            seq_scores = []
            for seq in prepared:
                inputs = {key: value.to(device) for key, value in seq.items()}
                seq_scores.append(model(**inputs).score)
            return torch.stack(seq_scores, dim=0).mean(dim=0)

        prepared = prepared[0]

    inputs = {key: value.to(device) for key, value in prepared.items()}
    return model(**inputs).score
