import os

from utils import (
    load_data, compute_frequencies,
    split_by_language, ensure_results_dir, RESULTS_DIR
)


def informativeness_score(bigram, eng_freq, rus_freq,
                          eng_total, rus_total):
    eng_prob = eng_freq.get(bigram, 0) / eng_total
    rus_prob = rus_freq.get(bigram, 0) / rus_total
    return abs(eng_prob - rus_prob)


def main():
    names, labels = load_data()
    eng_names, rus_names = split_by_language(names, labels)
    ensure_results_dir()

    eng_freq = compute_frequencies(eng_names, n=2)
    rus_freq = compute_frequencies(rus_names, n=2)
    eng_total = sum(eng_freq.values())
    rus_total = sum(rus_freq.values())

    all_bigrams = set(eng_freq.keys()) | set(rus_freq.keys())

    scores = {}
    for bg in all_bigrams:
        scores[bg] = informativeness_score(
            bg, eng_freq, rus_freq, eng_total, rus_total
        )

    ranked = sorted(scores.items(), key=lambda x: x[1])

    print("="*50)
    print("Task 2: Least Informative Bigrams")
    print("="*50)

    print("\n20 Least Informative Bigrams (lowest score):")
    print(f"{'Rank':<5} {'Bigram':<8} {'Score':<10} "
          f"{'Eng Freq':<10} {'Rus Freq':<10}")
    print("-" * 43)
    for i, (bg, score) in enumerate(ranked[:20], 1):
        ep = eng_freq.get(bg, 0) / eng_total * 100
        rp = rus_freq.get(bg, 0) / rus_total * 100
        print(f"{i:<5} '{bg}'    {score:.6f}  "
              f"{ep:.2f}%      {rp:.2f}%")

    print(f"\nLeast informative bigram: '{ranked[0][0]}' "
          f"(score: {ranked[0][1]:.6f})")
    print("This bigram appears at nearly the same frequency "
          "in both languages,")
    print("making it a poor discriminator for classification.")

    print("\n20 Most Informative Bigrams (highest score):")
    print(f"{'Rank':<5} {'Bigram':<8} {'Score':<10} "
          f"{'Eng Freq':<10} {'Rus Freq':<10}")
    print("-" * 43)
    for i, (bg, score) in enumerate(reversed(ranked[-20:]), 1):
        ep = eng_freq.get(bg, 0) / eng_total * 100
        rp = rus_freq.get(bg, 0) / rus_total * 100
        print(f"{i:<5} '{bg}'    {score:.6f}  "
              f"{ep:.2f}%      {rp:.2f}%")

    # save results
    outpath = os.path.join(RESULTS_DIR, "task2_informativeness.csv")
    with open(outpath, "w") as f:
        f.write("bigram,score,eng_pct,rus_pct\n")
        for bg, score in ranked:
            ep = eng_freq.get(bg, 0) / eng_total * 100
            rp = rus_freq.get(bg, 0) / rus_total * 100
            f.write(f"{bg},{score:.6f},{ep:.2f},{rp:.2f}\n")
    print(f"\nFull results saved to results/task2_informativeness.csv")


if __name__ == "__main__":
    main()
