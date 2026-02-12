import os
from collections import Counter


DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data"
)
RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "results"
)


def load_data(filepath=None):
    if filepath is None:
        filepath = os.path.join(DATA_DIR, "Russian-and-English-dev.txt")
    names, labels = [], []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().replace("\r", "")
            if not line:
                continue
            parts = [p for p in line.split(",") if p.strip()]
            if len(parts) < 2:
                continue
            name, lang = parts[0].strip(), parts[-1].strip()
            if lang not in ("Russian", "English"):
                continue
            # skip noise entries
            if " " in name:
                continue
            names.append(name)
            labels.append(lang)
    return names, labels


def extract_ngrams(name, n=2):
    name = name.lower()
    return [name[i:i + n] for i in range(len(name) - n + 1)]


def compute_frequencies(names, n=2):
    counts = Counter()
    for name in names:
        grams = extract_ngrams(name, n)
        counts.update(grams)
    return counts


def split_by_language(names, labels):
    eng, rus = [], []
    for name, lang in zip(names, labels):
        if lang == "English":
            eng.append(name)
        else:
            rus.append(name)
    return eng, rus


def ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)
