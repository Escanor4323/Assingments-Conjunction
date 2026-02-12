import os
import string
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import (
    load_data, extract_ngrams, compute_frequencies,
    split_by_language, ensure_results_dir, RESULTS_DIR
)


def run_bigram_analysis(names, label="all"):
    freqs = compute_frequencies(names, n=2)
    total = sum(freqs.values())
    top20 = freqs.most_common(20)
    hapax = [bg for bg, c in freqs.items() if c == 1]

    print(f"\n{'='*50}")
    print(f"Bigram Frequency Analysis — {label}")
    print(f"{'='*50}")
    print(f"Total bigrams counted: {total}")
    print(f"Unique bigrams: {len(freqs)}")
    print(f"\nTop 20 Most Frequent Bigrams:")
    for rank, (bg, count) in enumerate(top20, 1):
        pct = count / total * 100
        print(f"  {rank:>2}. '{bg}' — {count} ({pct:.1f}%)")

    print(f"\nLeast Frequent Bigrams:")
    print(f"  Bigrams appearing exactly once (hapax): {len(hapax)}")
    if hapax[:10]:
        print(f"  Examples: {hapax[:10]}")

    # all possible bigrams from lowercase letters
    alpha = string.ascii_lowercase
    possible = {a + b for a in alpha for b in alpha}
    unobserved = possible - set(freqs.keys())
    print(f"  Unobserved letter-only bigrams: {len(unobserved)}")
    if list(unobserved)[:5]:
        print(f"  Examples: {sorted(unobserved)[:5]}")

    print(f"\n  NOTE: 'Least frequent' is ambiguous —")
    print(f"  {len(hapax)} bigrams appear once, but "
          f"{len(unobserved)} possible bigrams never appear.")
    print(f"  The truly 'least frequent' are the unobserved ones "
          f"(frequency 0).")

    return freqs, top20, hapax, unobserved


def run_trigram_analysis(names, label="all"):
    freqs = compute_frequencies(names, n=3)
    total = sum(freqs.values())
    top20 = freqs.most_common(20)
    hapax = [tg for tg, c in freqs.items() if c == 1]

    print(f"\n{'='*50}")
    print(f"Trigram Frequency Analysis — {label}")
    print(f"{'='*50}")
    print(f"Total trigrams counted: {total}")
    print(f"Unique trigrams: {len(freqs)}")
    print(f"\nTop 20 Most Frequent Trigrams:")
    for rank, (tg, count) in enumerate(top20, 1):
        pct = count / total * 100
        print(f"  {rank:>2}. '{tg}' — {count} ({pct:.1f}%)")

    print(f"\nTrigrams appearing exactly once: {len(hapax)}")
    print(f"  Trigrams are sparser but more distinctive than bigrams.")

    return freqs, top20, hapax


def plot_top_ngrams(top_items, title, filename):
    labels = [bg for bg, _ in top_items]
    counts = [c for _, c in top_items]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(labels[::-1], counts[::-1], color="#4a90d9")
    ax.set_xlabel("Frequency")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=150)
    plt.close()
    print(f"  Plot saved: results/{filename}")


def save_frequency_table(freqs, filepath):
    sorted_freqs = sorted(freqs.items(), key=lambda x: -x[1])
    with open(filepath, "w") as f:
        f.write("ngram,count\n")
        for gram, count in sorted_freqs:
            f.write(f"{gram},{count}\n")


def main():
    names, labels = load_data()
    eng_names, rus_names = split_by_language(names, labels)
    ensure_results_dir()

    print(f"Loaded {len(names)} names "
          f"({len(eng_names)} English, {len(rus_names)} Russian)")

    # English bigrams
    eng_bi, eng_top20, _, _ = run_bigram_analysis(
        eng_names, "English"
    )
    # Russian bigrams
    rus_bi, rus_top20, _, _ = run_bigram_analysis(
        rus_names, "Russian"
    )

    # Trigram analysis
    eng_tri, eng_tri_top, _ = run_trigram_analysis(
        eng_names, "English"
    )
    rus_tri, rus_tri_top, _ = run_trigram_analysis(
        rus_names, "Russian"
    )

    # Save tables
    save_frequency_table(
        eng_bi, os.path.join(RESULTS_DIR, "english_bigrams.csv")
    )
    save_frequency_table(
        rus_bi, os.path.join(RESULTS_DIR, "russian_bigrams.csv")
    )
    save_frequency_table(
        eng_tri, os.path.join(RESULTS_DIR, "english_trigrams.csv")
    )
    save_frequency_table(
        rus_tri, os.path.join(RESULTS_DIR, "russian_trigrams.csv")
    )
    print("\nFrequency tables saved to results/")

    # Plots
    plot_top_ngrams(
        eng_top20, "Top 20 English Bigrams",
        "english_bigrams_top20.png"
    )
    plot_top_ngrams(
        rus_top20, "Top 20 Russian Bigrams",
        "russian_bigrams_top20.png"
    )
    plot_top_ngrams(
        eng_tri_top, "Top 20 English Trigrams",
        "english_trigrams_top20.png"
    )
    plot_top_ngrams(
        rus_tri_top, "Top 20 Russian Trigrams",
        "russian_trigrams_top20.png"
    )


if __name__ == "__main__":
    main()
