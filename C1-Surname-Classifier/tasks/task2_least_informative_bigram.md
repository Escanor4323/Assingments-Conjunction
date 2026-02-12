# Task 2: Least Informative Bigram
## For Joel Martinez

## Objective
Determine which bigram is **least informative** for distinguishing between English and Russian names.

## Description
Not all bigrams are equally informative for classification. Some bigrams appear with similar frequency in both English and Russian names, making them poor discriminators.

## Requirements

### 2.1 Compute Bigram Frequencies for Both Languages
- Compute bigram frequencies for English names
- Compute bigram frequencies for Russian names
- Normalize frequencies for fair comparison

### 2.2 Calculate Informativeness Score
Possible approaches:

**Approach A: Frequency Ratio**
```
informativeness = |freq_english(bigram) - freq_russian(bigram)|
```
Low difference = least informative.

**Approach B: Information Gain / Mutual Information**
```
MI(bigram, language) = measure of statistical dependence
```

**Approach C: Chi-squared statistic**
- Test independence between bigram occurrence and language

### 2.3 Identify Least Informative Bigram
- The bigram with the smallest informativeness score
- Likely candidates: common letter pairs in both languages (e.g., vowel pairs, 'in', 'an')

## Implementation Hints

```python
def informativeness_score(bigram, eng_freq, rus_freq):
    """
    Compute how informative a bigram is for classification.
    Lower score = less informative.
    """
    eng_prob = eng_freq.get(bigram, 0) / sum(eng_freq.values())
    rus_prob = rus_freq.get(bigram, 0) / sum(rus_freq.values())
    return abs(eng_prob - rus_prob)

# Find least informative
all_bigrams = set(eng_freq.keys()) | set(rus_freq.keys())
scores = {bg: informativeness_score(bg, eng_freq, rus_freq) for bg in all_bigrams}
least_informative = min(scores, key=scores.get)
```

## Deliverables
- [ ] Comparative frequency table (English vs. Russian)
- [ ] Informativeness scores for common bigrams
- [ ] Identification of least informative bigram(s)
- [ ] Brief explanation of why this bigram is uninformative

## Expected Insights
- Bigrams common to both languages (e.g., 'in', 'an', 'er') may be least informative
- Language-specific bigrams (e.g., 'ov', 'sk' for Russian; 'th', 'ck' for English) are most informative
