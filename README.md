# NLP Evaluation Metrics

Automatic evaluation for English→French machine translation using BLEU, METEOR, BLEURT, and COMETKiwi.

## Project layout

```
NLP_Codebase/
├── data/
│   ├── sample_en.json          # example data (committed)
│   ├── sample_fr.json
│   ├── data.json               # raw ingestion input (gitignored)
│   ├── excerpts/               # {id}_excerpt.json (gitignored, generated)
│   └── sentences/              # {id}_sentences.json (gitignored, generated)
├── evaluation/evaluate.py      # data loading + metric orchestration
├── metrics/                    # bleu, meteor, bleurt, cometkiwi
├── scripts/
│   ├── run.py                  # evaluation CLI
│   ├── data_to_excerpt.py      # ingest + Google Translate pipeline
│   └── excerpt_to_sentences.py # spaCy excerpt → sentence split (optional)
├── secrets/                    # Google service-account JSON (gitignored)
└── test_translation.py         # Translate API smoke test (gitignored)
```

## Installation

Create the virtualenv on your **internal drive** (exFAT SSDs waste space with 1 MB clusters):

```bash
python -m venv ~/venvs/NLP_Codebase
source ~/venvs/NLP_Codebase/bin/activate
pip install -r requirements.txt
pip install -r requirements-meteor.txt --no-deps
```

spaCy models (only if using `excerpt_to_sentences.py`):

```bash
python -m spacy download en_core_web_sm fr_core_news_sm
```

### METEOR (Java + JARs)

METEOR uses the JVM implementation via [aac-metrics](https://github.com/Labbeti/aac-metrics). Requires **Java 8–13** (11 recommended on macOS):

```bash
brew install openjdk@11
export PATH="/opt/homebrew/opt/openjdk@11/bin:$PATH"
export AAC_METRICS_JAVA_PATH="/opt/homebrew/opt/openjdk@11/bin/java"
```

Download JARs (once per machine):

```bash
mkdir -p ~/.cache/aac-metrics/meteor/data ~/.cache/aac-metrics/stanford_nlp
curl -L -o ~/.cache/aac-metrics/meteor/meteor-1.5.jar \
  https://github.com/tylin/coco-caption/raw/master/pycocoevalcap/meteor/meteor-1.5.jar
curl -L -o ~/.cache/aac-metrics/stanford_nlp/stanford-corenlp-3.4.1.jar \
  https://github.com/tylin/coco-caption/raw/master/pycocoevalcap/tokenizer/stanford-corenlp-3.4.1.jar
curl -L -o ~/.cache/aac-metrics/meteor/data/paraphrase-en.gz \
  https://github.com/tylin/coco-caption/raw/master/pycocoevalcap/meteor/data/paraphrase-en.gz
```

For French METEOR (`--meteor-language fr`):

```bash
curl -L -o ~/.cache/aac-metrics/meteor/data/paraphrase-fr.gz \
  https://github.com/cmu-mtlab/meteor/raw/master/data/paraphrase-fr.gz
```

### COMETKiwi (Hugging Face)

1. Accept the license at [Unbabel/wmt22-cometkiwi-da](https://huggingface.co/Unbabel/wmt22-cometkiwi-da).
2. Authenticate: `huggingface-cli login`
3. Verify: `huggingface-cli whoami`

### Google Cloud Translate (data ingestion)

1. Enable the [Cloud Translation API](https://console.cloud.google.com/apis/library/translate.googleapis.com) for your project.
2. Place a service-account JSON key in `secrets/`.
3. Add to `~/.zshrc`:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/NLP_Codebase/secrets/your-key.json"
```

4. Smoke test:

```bash
python test_translation.py
```

### BLEURT checkpoint

```bash
curl -L -o BLEURT-20-D12.zip https://storage.googleapis.com/bleurt-oss-21/BLEURT-20-D12.zip
unzip BLEURT-20-D12.zip
```

On Apple Silicon, keep `tensorflow<2.20` (pinned in `requirements.txt`) to avoid checkpoint load crashes.

## Data pipeline

### 1. Ingest raw excerpts (`data_to_excerpt.py`)

Input `data/data.json` — a list of manually copied EN source + FR reference pairs:

```json
[
  {
    "id": 8,
    "source": "English excerpt from website...",
    "reference": "French excerpt from website..."
  }
]
```

Run:

```bash
python scripts/data_to_excerpt.py --input data/data.json
```

For each entry this:
- Cleans and sentence-splits source + reference (NLTK)
- Aligns sentence counts (truncate, then merge if needed)
- Translates each source sentence to French via Google Cloud Translate
- Writes `data/excerpts/{id}_excerpt.json` and `data/sentences/{id}_sentences.json`

### 2. Optional: spaCy sentence split (`excerpt_to_sentences.py`)

Re-splits an existing excerpt file with spaCy (when prediction is already present):

```bash
python scripts/excerpt_to_sentences.py --input data/excerpts/7_excerpt.json
```

Output defaults to `data/sentences/7_sentences.json`.

## Input formats for evaluation

`scripts/run.py` accepts three JSON shapes via `evaluation/evaluate.py`:

| File | Format | Examples evaluated |
|------|--------|-------------------|
| Sample / excerpt | `{id, source, prediction, references}` | 1 per file |
| Sentence bundle | `{id, source_sentences, prediction_sentences, reference_sentences}` | 1 per sentence |
| Sentence rows | `[{page_id, sentence_id, source, prediction, reference}, ...]` | 1 per row |

Sample format (also used in `data/sample_en.json`):

```json
[
  {
    "id": 1,
    "source": "original sentence",
    "prediction": "generated translation",
    "references": ["reference translation"]
  }
]
```

- `source` is required for COMETKiwi.
- `references` may contain multiple strings per example.

## Usage

Activate the venv from the project root:

```bash
source ~/venvs/NLP_Codebase/bin/activate
```

BLEU + METEOR on sample data:

```bash
python scripts/run.py --data data/sample_en.json
```

All four metrics on an excerpt:

```bash
python scripts/run.py --data data/excerpts/7_excerpt.json \
  --metrics bleu meteor bleurt cometkiwi \
  --bleurt-checkpoint BLEURT-20-D12
```

Same metrics at sentence level:

```bash
python scripts/run.py --data data/sentences/7_sentences.json \
  --metrics bleu meteor bleurt cometkiwi \
  --bleurt-checkpoint BLEURT-20-D12
```

French METEOR target:

```bash
python scripts/run.py --data data/sample_fr.json --metrics meteor --meteor-language fr
```

## Example output

```
Evaluated 3 examples
{
  "bleu": 74.0716,
  "meteor": 0.5561,
  "bleurt": 0.8662,
  "cometkiwi": 0.8723
}
```

- **BLEU** — SacreBLEU 0–100 scale.
- **METEOR**, **BLEURT**, **COMETKiwi** — typically 0–1.

## Metrics summary

| Metric | Library | Notes |
|--------|---------|-------|
| BLEU | SacreBLEU | Corpus-level; pads unequal reference counts |
| METEOR | aac-metrics (JVM) | Sentence mean; best reference per sentence |
| BLEURT | google-research/bleurt | Sentence mean; max over references |
| COMETKiwi | unbabel-comet | Reference-free; needs `source` |

## SSD / exFAT notes

macOS creates `._*` sidecar files on exFAT volumes. They are gitignored; clean with:

```bash
find . -name '._*' -not -path './.git/*' -delete
```

To reduce new sidecars when copying: `export COPYFILE_DISABLE=1`. Keeping the repo on APFS avoids them entirely.

Keep `.venv` off the SSD; use `~/venvs/NLP_Codebase` instead.
