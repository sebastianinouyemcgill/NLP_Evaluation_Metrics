# NLP_Evaluation_Metrics

Implements automatic evaluation for machine translation using:
- BLEU
- METEOR
- BLEURT

## Installation
pip install -r requirements.txt

## Input Format
All inputs are provided as a JSON file (`sample.json`) with the following structure:

```json
[
  {
    "id": 1,
    "prediction": "generated sentence",
    "references": ["reference sentence 1", "reference sentence 2"]
  }
]
```

## Usage
python scripts/run.py \
  --input data/sample.json \
  --bleurt_checkpoint BLEURT-20 #default model

## Output
=== Evaluation Results === #example data
BLEU:   0.3251
METEOR: 0.4187
BLEURT: -0.1234

