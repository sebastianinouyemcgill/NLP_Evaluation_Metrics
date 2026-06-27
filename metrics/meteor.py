"""Corpus-level METEOR (JVM via aac-metrics). Requires Java 8–13."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path
from typing import List

_AAC_ROOT: Path | None = None
_JAVA_CONFIGURED = False

_METEOR_JAR = Path.home() / ".cache/aac-metrics/meteor/meteor-1.5.jar"
_STANFORD_JAR = (
    Path.home() / ".cache/aac-metrics/stanford_nlp/stanford-corenlp-3.4.1.jar"
)
_PARAPHRASE_FILES = {
    "en": Path.home() / ".cache/aac-metrics/meteor/data/paraphrase-en.gz",
    "fr": Path.home() / ".cache/aac-metrics/meteor/data/paraphrase-fr.gz",
    "de": Path.home() / ".cache/aac-metrics/meteor/data/paraphrase-de.gz",
    "es": Path.home() / ".cache/aac-metrics/meteor/data/paraphrase-es.gz",
    "cz": Path.home() / ".cache/aac-metrics/meteor/data/paraphrase-cz.gz",
}


def _aac_root() -> Path:
    global _AAC_ROOT
    if _AAC_ROOT is None:
        import site

        for base in site.getsitepackages():
            candidate = Path(base) / "aac_metrics"
            if candidate.is_dir():
                _AAC_ROOT = candidate
                break
        if _AAC_ROOT is None:
            raise ImportError(
                "aac-metrics is not installed. See requirements-meteor.txt"
            )
    return _AAC_ROOT


def _ensure_package(name: str, path: Path | None = None) -> types.ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    module = types.ModuleType(name)
    if path is not None:
        module.__path__ = [str(path)]
    sys.modules[name] = module
    return module


def _load_aac_module(qualified_name: str, relative_path: str):
    if qualified_name in sys.modules:
        return sys.modules[qualified_name]

    root = _aac_root()
    parts = qualified_name.split(".")
    for i in range(1, len(parts)):
        pkg_name = ".".join(parts[:i])
        pkg_path = root.joinpath(*parts[1:i]) if i > 1 else root
        _ensure_package(pkg_name, pkg_path if pkg_path.is_dir() else None)

    filepath = root / relative_path
    spec = importlib.util.spec_from_file_location(qualified_name, filepath)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load aac-metrics module at {filepath}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = module
    spec.loader.exec_module(module)
    return module


def _meteor_api():
    _load_aac_module("aac_metrics.utils.collections", "utils/collections.py")
    _load_aac_module("aac_metrics.utils.checks", "utils/checks.py")
    _load_aac_module("aac_metrics.utils.globals", "utils/globals.py")
    tokenization = _load_aac_module(
        "aac_metrics.utils.tokenization", "utils/tokenization.py"
    )
    meteor_mod = _load_aac_module("aac_metrics.functional.meteor", "functional/meteor.py")
    return meteor_mod.meteor, tokenization.preprocess_mono_sents, tokenization.preprocess_mult_sents


def _java_candidates() -> List[str]:
    candidates: List[str] = []

    env_java = os.environ.get("AAC_METRICS_JAVA_PATH")
    if env_java:
        candidates.append(env_java)

    for path in (
        "/opt/homebrew/opt/openjdk@11/bin/java",
        "/opt/homebrew/opt/openjdk@13/bin/java",
        "/usr/local/opt/openjdk@11/bin/java",
        "/usr/local/opt/openjdk@13/bin/java",
    ):
        candidates.append(path)

    for version in ("11", "13", "1.8"):
        try:
            result = subprocess.run(
                ["/usr/libexec/java_home", "-v", version],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                candidates.append(os.path.join(result.stdout.strip(), "bin", "java"))
        except (FileNotFoundError, OSError):
            pass

    default_java = shutil.which("java")
    if default_java:
        candidates.append(default_java)

    seen = set()
    unique: List[str] = []
    for path in candidates:
        if path and path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def _configure_java() -> None:
    global _JAVA_CONFIGURED
    if _JAVA_CONFIGURED:
        return

    checks = _load_aac_module("aac_metrics.utils.checks", "utils/checks.py")
    globals_mod = _load_aac_module("aac_metrics.utils.globals", "utils/globals.py")

    if os.environ.get("AAC_METRICS_JAVA_PATH"):
        _JAVA_CONFIGURED = True
        return

    for java_path in _java_candidates():
        if not os.path.isfile(java_path):
            continue
        if checks.check_java_path(java_path):
            globals_mod.set_default_java_path(java_path)
            _JAVA_CONFIGURED = True
            return

    _JAVA_CONFIGURED = True
    raise RuntimeError(
        "METEOR requires Java 8–13 for PTB tokenization. "
        "Install with `brew install openjdk@11` or set AAC_METRICS_JAVA_PATH "
        "to a supported Java binary. The default macOS `java` is often too new."
    )


def _ensure_meteor_assets(language: str) -> None:
    missing = []
    if not _METEOR_JAR.is_file():
        missing.append(str(_METEOR_JAR))
    if not _STANFORD_JAR.is_file():
        missing.append(str(_STANFORD_JAR))

    paraphrase = _PARAPHRASE_FILES.get(language)
    if paraphrase is not None and not paraphrase.is_file():
        missing.append(str(paraphrase))

    if missing:
        raise FileNotFoundError(
            "METEOR setup is incomplete. Missing files:\n"
            + "\n".join(f"  - {path}" for path in missing)
            + "\nSee the Installation section in README.md for download commands."
        )


def corpus_meteor(
    predictions: List[str],
    references: List[List[str]],
    language: str = "en",
) -> float:
    """Return corpus-level METEOR (scores are typically between 0 and 1)."""
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")

    _ensure_meteor_assets(language)
    _configure_java()

    meteor, preprocess_mono_sents, preprocess_mult_sents = _meteor_api()
    candidates = preprocess_mono_sents(predictions)
    mult_refs = preprocess_mult_sents(references)
    corpus_scores, _ = meteor(candidates, mult_refs, language=language)
    return float(corpus_scores["meteor"])
