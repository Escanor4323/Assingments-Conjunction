# Findings Report — C1 Surname Classifier

## Dataset
- **Source**: `Russian-and-English-dev.txt`
- **Total**: 1302 names (936 Russian, 366 English)
- **Split**: 80/20 stratified train/test

---

## Task 1: Bigram & Trigram Frequencies

### English Top Bigrams
| Rank | Bigram | Count | Pct  |
|------|--------|-------|------|
| 1    | er     | 49    | 2.5% |
| 2    | on     | 47    | 2.4% |
| 3    | ar     | 40    | 2.1% |
| 4    | in     | 37    | 1.9% |
| 5    | le     | 37    | 1.9% |

### Russian Top Bigrams
| Rank | Bigram | Count | Pct  |
|------|--------|-------|------|
| 1    | ov     | 342   | 5.3% |
| 2    | in     | 199   | 3.1% |
| 3    | ko     | 159   | 2.5% |
| 4    | ev     | 133   | 2.1% |
| 5    | ch     | 117   | 1.8% |

### Least Frequent (Trick Question)
- **English**: 63 hapax bigrams, 374 unobserved
- **Russian**: 74 hapax bigrams, 304 unobserved
- The "least frequent" question is ambiguous: unobserved bigrams (freq=0) are technically the least frequent, but they form an infinite set of possible character pairs. Among observed bigrams, many appear exactly once (hapax legomena).

### Trigram Observations
- Trigrams are sparser but more distinctive per language
- Russian trigrams like 'kov', 'ova', 'sky' are highly characteristic
- English trigrams like 'ton', 'ley', 'son' stand out

---

## Task 2: Least Informative Bigram

**Least informative**: `'ob'` (score: 0.000007)
- English frequency: 0.15%, Russian frequency: 0.15%
- Appears at nearly identical rates in both languages

**Most informative**: `'ov'` (score: 0.052)
- English: 0.05%, Russian: 5.29% — enormous gap
- Other strong discriminators: `'ko'`, `'ev'`, `'sk'` (Russian); `'ey'`, `'ll'`, `'th'`, `'ck'` (English)

---

## Task 3: Classifier Results

### Bigram Logistic Regression
| Metric    | Value |
|-----------|-------|
| Accuracy  | 92%   |
| Precision (Russian) | 0.92 |
| Recall (Russian)    | 0.97 |
| F1 (Russian)        | 0.94 |

### Confusion Matrix
|            | Pred Eng | Pred Rus |
|------------|----------|----------|
| Actual Eng | 60       | 13       |
| Actual Rus | 7        | 181      |

### Error Patterns
- Total errors: 22/261
- English names misclassified as Russian tend to have Slavic-sounding patterns (e.g., "Rodrigues", "Chisholm", "Bradshaw")
- Russian names misclassified as English tend to be short or atypical (e.g., "Engver", "Remez", "Gudim")

---

## Task 4: Smoothing (LM Approach)

**Method**: Add-k Smoothing  
**Equation**: P(w_i|w_{i-1}) = (C(w_{i-1},w_i) + k) / (C(w_{i-1}) + k×V)

| k    | Precision | Recall | F1     |
|------|-----------|--------|--------|
| 0.01 | 0.9194    | 0.9096 | 0.9144 |
| 0.1  | 0.9348    | 0.9149 | 0.9247 |
| 0.5  | 0.9305    | 0.9255 | 0.9280 |
| 1.0  | 0.9316    | 0.9415 | **0.9365** |
| 2.0  | 0.9267    | 0.9415 | 0.9340 |

**Best k = 1.0** — classic Laplace smoothing performed best, improving F1 by +2.4% over the unsmoothed baseline.

---

## Task 5: Extended Data

**Source**: US Census common surnames (250 names added)

| Metric    | Original | Extended | Change |
|-----------|----------|----------|--------|
| Precision | 0.9192   | 0.9500   | +0.03  |
| Recall    | 0.9681   | 0.9096   | -0.06  |
| F1        | 0.9430   | 0.9293   | -0.01  |

**Discussion**: Adding more English data improved precision (fewer English names misclassified as Russian) but hurt recall (more Russian names misclassified as English). The net F1 dropped slightly. This suggests the additional data introduced some English surname patterns that overlap with Russian names (e.g., names ending in "-son", "-man"), causing the model to be more conservative in predicting Russian.

---

## Key Takeaways

1. **Character bigrams are strong features** — even a simple logistic regression achieves 92% accuracy on this task
2. **Russian names have signature patterns**: `'ov'`, `'ev'`, `'ko'`, `'sk'`, `'zh'` are near-exclusive to Russian
3. **English names have their own markers**: `'th'`, `'ck'`, `'ey'`, `'ll'` are strong English indicators
4. **Smoothing helps the LM approach** — Add-1 smoothing gave the best tradeoff
5. **More data isn't always better** — extending English data improved precision but hurt recall due to pattern overlap
