# Task 1: Compute Bigram Frequencies
## For Joel Martinez

## Objective
Compute bigram frequencies for English names. Identify the most frequent and least frequent bigrams.

## Description
A **bigram** is a sequence of two consecutive characters. For surname classification, character-level bigrams capture common patterns in name structure.

## Requirements

### 1.1 Compute Bigram Frequencies
- Load English surname data
- Extract all character bigrams from each name
- Count frequency of each unique bigram
- Normalize frequencies (optional: by total count or per-name)

### 1.2 Identify Most Frequent Bigrams
- Sort bigrams by frequency (descending)
- Report top 10-20 most frequent bigrams
- Discuss patterns observed (e.g., common endings like "-on", "-er")

### 1.3 Identify Least Frequent Bigrams (Trick Question!)
- **Key insight**: Many bigrams may have frequency = 1 (hapax legomena)
- Or certain bigrams may never appear at all
- Discuss why "least frequent" is ambiguous:
  - Unobserved bigrams have frequency 0
  - Single-occurrence bigrams are equally "least frequent"
  - Which is "least"? The ones that appear once, or the infinite set that never appear?

### 1.4 Trigram Analysis (Optional)
- Extend the same methodology to 3-character sequences
- Compare: Are trigrams more distinctive but sparser?

## Implementation Hints

```python
from collections import Counter

def extract_bigrams(name):
    """Extract character bigrams from a name."""
    name = name.lower()
    return [name[i:i+2] for i in range(len(name) - 1)]

def compute_frequencies(names):
    """Compute bigram frequencies across all names."""
    bigram_counts = Counter()
    for name in names:
        bigrams = extract_bigrams(name)
        bigram_counts.update(bigrams)
    return bigram_counts
```

## Deliverables
- [ ] Frequency table of bigrams
- [ ] Top 10 most frequent bigrams with counts
- [ ] Discussion of "least frequent" ambiguity (the trick question)
- [ ] Visualization (bar chart of top bigrams)
- [ ] Optional: Trigram analysis

## Expected Output Format
```
Most Frequent Bigrams:
1. 'er' - 245 occurrences
2. 'on' - 198 occurrences
...

Least Frequent Analysis:
- 42 bigrams appear exactly once
- Unobserved bigrams: 'xz', 'qx', etc.
- Discussion: The "least frequent" question is a trick because...
```
