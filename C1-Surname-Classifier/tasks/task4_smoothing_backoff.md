# Task 4: Smoothing and Backoff (If Using LM Approach)

## Objective
If you used a Language Model approach in Task 3, add smoothing and/or backoff to improve performance.

## Description
Raw probability estimates suffer from the **zero-frequency problem**: unseen bigrams get P = 0, making entire names have P = 0. Smoothing addresses this.

## Requirements

### 4.1 Choose a Smoothing Method

**Add-k Smoothing (Laplace)**
```
P(w_i | w_{i-1}) = (count(w_{i-1}, w_i) + k) / (count(w_{i-1}) + k × V)
```
- k = smoothing parameter (commonly k = 1 or k = 0.1)
- V = vocabulary size

**Good-Turing Smoothing**
- Redistribute probability mass from seen to unseen events

**Kneser-Ney Smoothing**
- State-of-the-art, uses absolute discounting

### 4.2 Implement Backoff (Optional)
When bigram is unseen, back off to unigram probability:
```
P(w_i | w_{i-1}) = λ × P_bigram(w_i | w_{i-1}) + (1-λ) × P_unigram(w_i)
```

### 4.3 Document Your Implementation

**You MUST document:**
1. **Method chosen**: (e.g., Add-k smoothing)
2. **Equation**: Full mathematical formulation
3. **Metaparameters**: Values used (e.g., k = 0.5)
4. **Results**: Updated precision and recall

## Implementation Hints

```python
def add_k_smoothing(bigram_counts, unigram_counts, k=1.0):
    """
    Apply add-k smoothing to bigram probabilities.
    
    Method: Add-k (Laplace) Smoothing
    Equation: P(w_i|w_{i-1}) = (C(w_{i-1},w_i) + k) / (C(w_{i-1}) + k*V)
    Metaparameters: k = 1.0
    """
    vocab_size = len(set(w for bg in bigram_counts for w in bg))
    
    smoothed_probs = {}
    for bigram, count in bigram_counts.items():
        context = bigram[0]
        context_count = unigram_counts.get(context, 0)
        smoothed_probs[bigram] = (count + k) / (context_count + k * vocab_size)
    
    return smoothed_probs
```

## Deliverables
- [ ] Smoothing method implemented
- [ ] Documentation: method name, equation, metaparameter values
- [ ] Updated precision and recall results
- [ ] Comparison table: baseline vs. smoothed

## Expected Output Format
```
Smoothing Method: Add-k Smoothing
Equation: P(w_i|w_{i-1}) = (C(w_{i-1},w_i) + k) / (C(w_{i-1}) + k×V)
Metaparameters: k = 0.5, V = 702 (observed vocabulary)

Results Comparison:
                Baseline    Smoothed    Change
Precision       0.85        0.87        +2%
Recall          0.78        0.82        +4%
F1              0.81        0.84        +3%
```
