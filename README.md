# NLP Evaluation Metrics

Automatic evaluation for machine translation using BLEU, METEOR, and BLEURT.

## Installation

```bash
pip install -r requirements.txt
```

NLTK data (WordNet, tokenizers) is downloaded automatically on first METEOR run.

## Input Format

Provide a JSON list of records:

```json
[
  {
    "id": 1,
    "prediction": "generated sentence",
    "references": ["reference 1", "reference 2"]
  }
]
```

- `id` is optional (defaults to the list index).
- `references` may contain any number of reference strings per example.

## Usage

BLEU and METEOR:

```bash
python scripts/run.py --data data/sample_en.json
```

Select metrics explicitly:

```bash
python scripts/run.py --data data/sample_en.json --metrics bleu meteor
```

BLEURT (requires a downloaded checkpoint):

```bash
python scripts/run.py --data data/sample_en.json --metrics bleurt \
  --bleurt-checkpoint BLEURT-20-D12
```

All three metrics:

```bash
python scripts/run.py --data data/sample_en.json --metrics bleu meteor bleurt \
  --bleurt-checkpoint BLEURT-20-D12
```

## Example Output

```
Evaluated 3 examples
{
  "bleu": 66.7074,
  "meteor": 0.8081,
  "bleurt": 0.7891
}
```

- **BLEU** is reported on SacreBLEU's 0–100 scale (WMT convention).
- **METEOR** and **BLEURT** are typically between 0 and 1.

## Metrics

### BLEU

Computed with [SacreBLEU](https://github.com/mjpost/sacrebleu) (`tokenize=13a`).
Returns a single **corpus-level** score. When examples have different numbers
of references, shorter lists are padded by repeating the first reference so
SacreBLEU can aggregate n-grams across the corpus.

### METEOR

Computed with [NLTK's `meteor_score`](https://www.nltk.org/api/nltk.translate.meteor_score.html),
which implements the METEOR 1.5 algorithm using **English WordNet** synonyms
and an English stemmer. Use this when candidate and reference text are in
**English**. It is not suitable for non-English target text.

Corpus score = mean of sentence-level METEOR scores. With multiple references,
NLTK scores against each reference and uses the best match per sentence.

### BLEURT

Uses the [official Google Research BLEURT](https://github.com/google-research/bleurt)
(TensorFlow). Corpus score = mean of sentence-level scores; with multiple
references, the highest score per example is kept.

#### BLEURT Setup

1. Install dependencies (included in `requirements.txt`):

   ```bash
   pip install -r requirements.txt
   ```

   **macOS (Apple Silicon):** TensorFlow 2.20+ can crash when loading a BLEURT
   checkpoint (`mutex lock failed: Invalid argument`). This project pins
   `tensorflow<2.20`. If you already upgraded TensorFlow, run:

   ```bash
   pip install "tensorflow>=2.12.0,<2.20.0"
   ```

2. Download a checkpoint. The distilled 12-layer model is a good default:

   ```bash
   wget https://storage.googleapis.com/bleurt-oss-21/BLEURT-20-D12.zip
   unzip BLEURT-20-D12.zip
   ```

   This creates a `BLEURT-20-D12/` directory containing the checkpoint files.

3. Run evaluation:

   ```bash
   python scripts/run.py --data data/sample.json --metrics bleurt \
     --bleurt-checkpoint BLEURT-20-D12
   ```

Other checkpoints (full BLEURT-20, BLEURT-20-D3, etc.) are listed in the
[official checkpoints page](https://github.com/google-research/bleurt/blob/master/checkpoints.md).
