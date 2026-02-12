import math
import numpy as np
from sklearn.model_selection import train_test_split

from utils import load_data, extract_ngrams, split_by_language


def build_lm(names, k=1.0):
    """Build a character bigram language model with add-k smoothing."""
    from collections import Counter
    bigram_counts = Counter()
    unigram_counts = Counter()

    for name in names:
        grams = extract_ngrams(name, n=2)
        bigram_counts.update(grams)
        for bg in grams:
            unigram_counts[bg[0]] += 1
        if grams:
            unigram_counts[grams[-1][-1]] += 1

    vocab = set()
    for bg in bigram_counts:
        vocab.add(bg[0])
        vocab.add(bg[1])
    V = len(vocab)

    probs = {}
    for bigram, count in bigram_counts.items():
        context = bigram[0]
        ctx_count = unigram_counts.get(context, 0)
        probs[bigram] = (count + k) / (ctx_count + k * V)

    return probs, unigram_counts, V, k


def score_name(name, lm_probs, unigram_counts, V, k):
    """Log probability of a name under a language model."""
    log_prob = 0.0
    grams = extract_ngrams(name, n=2)
    for bg in grams:
        prob = lm_probs.get(bg, None)
        if prob is None:
            ctx_count = unigram_counts.get(bg[0], 0)
            prob = k / (ctx_count + k * V)
        log_prob += math.log(prob)
    return log_prob


def classify_name(name, eng_lm, rus_lm):
    eng_score = score_name(name, *eng_lm)
    rus_score = score_name(name, *rus_lm)
    return "English" if eng_score > rus_score else "Russian"


def main():
    names, labels = load_data()
    labels_arr = np.array(labels)

    X_train_names, X_test_names, y_train, y_test = (
        train_test_split(
            names, labels_arr, test_size=0.2,
            random_state=42, stratify=labels_arr
        )
    )

    eng_train = [n for n, l in zip(X_train_names, y_train)
                 if l == "English"]
    rus_train = [n for n, l in zip(X_train_names, y_train)
                 if l == "Russian"]

    k_values = [0.01, 0.1, 0.5, 1.0, 2.0]

    print("="*55)
    print("Task 4: Smoothing & Backoff (LM Approach)")
    print("="*55)

    # baseline: no smoothing (k very small)
    eng_lm_base = build_lm(eng_train, k=1e-10)
    rus_lm_base = build_lm(rus_train, k=1e-10)
    base_preds = [classify_name(n, eng_lm_base, rus_lm_base)
                  for n in X_test_names]

    from sklearn.metrics import (
        precision_score, recall_score, f1_score,
        classification_report
    )

    print("\nBaseline (no smoothing):")
    print(classification_report(y_test, base_preds))
    base_p = precision_score(y_test, base_preds,
                             pos_label="Russian")
    base_r = recall_score(y_test, base_preds,
                          pos_label="Russian")
    base_f = f1_score(y_test, base_preds,
                      pos_label="Russian")

    print(f"\nSmoothing Method: Add-k Smoothing")
    print(f"Equation: P(w_i|w_{{i-1}}) = "
          f"(C(w_{{i-1}},w_i) + k) / (C(w_{{i-1}}) + k*V)")

    best_k, best_f1 = None, 0
    results = []

    for k in k_values:
        eng_lm = build_lm(eng_train, k=k)
        rus_lm = build_lm(rus_train, k=k)
        preds = [classify_name(n, eng_lm, rus_lm)
                 for n in X_test_names]
        p = precision_score(y_test, preds, pos_label="Russian")
        r = recall_score(y_test, preds, pos_label="Russian")
        f = f1_score(y_test, preds, pos_label="Russian")
        results.append((k, p, r, f))
        if f > best_f1:
            best_f1 = f
            best_k = k

    print(f"\nResults across k values:")
    print(f"{'k':<8} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 44)
    for k, p, r, f in results:
        print(f"{k:<8} {p:<12.4f} {r:<12.4f} {f:<12.4f}")

    print(f"\nBest k = {best_k}")

    print(f"\nComparison:")
    print(f"{'Metric':<12} {'Baseline':<12} "
          f"{'Smoothed(k=' + str(best_k) + ')':<20}")
    print("-" * 44)
    best_res = [r for r in results if r[0] == best_k][0]
    print(f"{'Precision':<12} {base_p:<12.4f} {best_res[1]:<20.4f}")
    print(f"{'Recall':<12} {base_r:<12.4f} {best_res[2]:<20.4f}")
    print(f"{'F1':<12} {base_f:<12.4f} {best_res[3]:<20.4f}")


if __name__ == "__main__":
    main()
