# C1: Russian vs. English Surname Classifier

A project to build and evaluate a classifier that discriminates between Russian and English surnames using n-gram (bigram/trigram) frequency analysis.

## Project Overview

This assignment explores character-level language models for name classification:

1. **Bigram Analysis** -- Compute n-gram frequencies, find most/least frequent
2. **Feature Selection** -- Identify least informative bigrams for classification
3. **Model Building** -- Build classifier and evaluate precision/recall
4. **LM Improvements** -- Add smoothing/backoff if using LM approach
5. **Data Extension** -- Find more English data and improve performance

## Directory Structure

```
C1-Surname-Classifier/
├── README.md
├── data/
│   └── Russian-and-English-dev.txt
├── src/
│   ├── __init__.py
│   ├── utils.py
│   ├── classifier.py
│   ├── task1_analysis.py
│   ├── task2_informativeness.py
│   ├── task3_model.py
│   ├── task4_smoothing.py
│   └── task5_extension.py
├── tasks/
│   ├── task1_compute_bigrams.md
│   ├── task2_least_informative_bigram.md
│   ├── task3_build_classifier.md
│   ├── task4_smoothing_backoff.md
│   └── task5_extended_data.md
├── results/
│   └── findings.md
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Tasks

| # | Task | Status |
|---|------|--------|
| 1 | Compute bigram frequencies (most/least frequent) | Done |
| 2 | Find least informative bigram for classification | Done |
| 3 | Build classifier, report precision and recall | Done |
| 4 | Add smoothing/backoff (LM approach) | Done |
| 5 | Find more English data and retrain | Done |

See `tasks/` for detailed descriptions. See `results/findings.md` for a summary of results.

## Running

```bash
cd src
python task1_analysis.py
python task2_informativeness.py
python task3_model.py
python task4_smoothing.py
python task5_extension.py
```

## References

- PEP 8 Style Guide
- scikit-learn documentation
- NLTK language model resources
