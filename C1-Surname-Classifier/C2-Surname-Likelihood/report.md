# C2: English Surname Analysis Report

## Part a) Name Likelihoods

The following table shows the estimated log-likelihoods and probabilities for the specified names using a character-level bigram model trained on 366 English names.

| Name | Log-Likelihood | Probability |
| :--- | :--- | :--- |
| Fergus | -18.5543 | 8.75e-09 |
| Angus | -14.2186 | 6.68e-07 |
| Boston | -15.5021 | 1.85e-07 |
| Austin | -16.8246 | 4.93e-08 |
| Dankworth | -62.9943 | 4.38e-28 |
| Denkworth | -62.4709 | 7.40e-28 |
| Birtwistle | -27.1566 | 1.61e-12 |
| Birdwhistle | -30.0526 | 8.88e-14 |

**Observation**: "Dankworth" and "Denkworth" receive extremely low likelihoods. This is because they contain bigrams (like 'kw', 'nk', or 'wo') that are likely unseen or very rare in the small training set of 366 names. The model assigns a small epsilon probability ($10^{-10}$) to unseen bigrams, which accumulates to a very small total probability.

## Part b) Most Likely Completions

Using greedy decoding (always picking the next most probable character), the model generated the following completions for the given prefixes:

-   **Lou** $\rightarrow$ **Lour**
-   **Ber** $\rightarrow$ **Ber**
-   **Cul** $\rightarrow$ **Culer**
-   **Ede** $\rightarrow$ **Eder**
-   **Zjo** $\rightarrow$ **Zjon**

**Observation**:
-   "Ber" completes immediately to "Ber" because 'r' is often the last letter in English names (e.g., "Miller", "Baker"), so $P(\text{End} | \text{'r'})$ is likely higher than any other character transition.
-   "Zjo" is an unusual prefix for English. The model likely followed 'o' with 'n' (common in "son", "on", "ton" names) to get "Zjon".

## Part c) Critique and Improvement

**Least Plausible Result**: The likelihood estimate for **"Dankworth"** ($4.38 \times 10^{-28}$) is implausibly low for a valid English surname.

**Why it happened**:
The training data (366 names) is too sparse to cover all valid character transitions in English surnames. The specific transitions in "Dankworth" (possibly `dk`, `kw`, `nk`) were likely not observed in the training set. A simple bigram model without sophisticated smoothing assumes these transitions are nearly impossible.

**How to Improve**:
1.  **Smoothing**: Implement **Add-k smoothing** (Laplace smoothing) or **Kneser-Ney smoothing**. Instead of assigning an arbitrary small epsilon to unseen bigrams, we should redistribute probability mass from seen to unseen events. This would give "Dankworth" a low but reasonable probability, rather than an effectively impossible one.
2.  **More Data**: Training on a larger corpus of English names would naturally increase the coverage of valid bigrams.
3.  **Trigram Model**: A bigram model misses context. For example, 'w' might be rare after 'k', but common in the context of 'ank' (though 'ankw' is still rare). However, with such small data, a trigram model would suffer even more from sparsity.
4.  **Subword Units**: Instead of characters, using subword units (like BPE) closer to morphemes (e.g., "worth", "ton", "field") could capture common name suffixes effectively.
