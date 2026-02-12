# C1: Russian vs. English Surname Classifier

A machine learning project to build and evaluate a classifier that discriminates between Russian and English surnames using n-gram (bigram/trigram) frequency analysis.

## 📋 Project Overview

This assignment explores character-level language models for name classification:

1. **Bigram Analysis** — Compute n-gram frequencies, find most/least frequent
2. **Feature Selection** — Identify least informative bigrams for classification
3. **Model Building** — Build classifier and evaluate precision/recall
4. **LM Improvements** — Add smoothing/backoff if using LM approach
5. **Data Extension** — Find more English data and improve performance

## 📁 Directory Structure

```
C1-Surname-Classifier/
├── README.md                    # This file
├── data/                        # Training/evaluation data
│   └── Russian-and-English-dev.txt
├── src/                         # Source code
│   ├── __init__.py
│   └── (implementation files)
├── tasks/                       # Task descriptions
│   ├── task1_compute_bigrams.md
│   ├── task2_least_informative_bigram.md
│   ├── task3_build_classifier.md
│   ├── task4_smoothing_backoff.md
│   └── task5_extended_data.md
├── results/                     # Output and analysis results
└── requirements.txt             # Python dependencies
```

## 🚀 Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📝 Tasks

| # | Task | Status |
|---|------|--------|
| 1 | Compute bigram frequencies (most/least frequent) | ✅ Done |
| 2 | Find least informative bigram for classification | ✅ Done |
| 3 | Build classifier, report precision and recall | ✅ Done |
| 4 | Add smoothing/backoff (if using LM approach) | ✅ Done |
| 5 | Find more English data and retrain | ✅ Done |

See the `tasks/` folder for detailed descriptions of each task.

## 📊 Expected Outputs

- Bigram frequency tables
- Most/least frequent n-grams analysis
- Least informative bigram identification
- Model performance metrics (precision, recall)
- Comparative analysis with extended data

## 📚 References

- PEP 8 Style Guide
- scikit-learn documentation
- NLTK language model resources
