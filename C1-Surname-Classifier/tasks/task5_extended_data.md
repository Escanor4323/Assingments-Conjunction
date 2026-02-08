# Task 5: Extend with More English Data

## Objective
Find more English data and retrain to improve your performance.

## Description
More training data can improve model robustness, but may also introduce noise or domain mismatch. This task explores the effect of additional training data.

## Requirements

### 5.1 Find Additional English Data

**Potential Sources:**
- US Census surname lists
- Baby name databases
- English dictionaries (for word patterns)
- Public name corpora (Kaggle, GitHub)

### 5.2 Document Your Data

**You MUST specify:**
1. **Data source**: Where did you get it?
2. **Size**: How many names/entries?
3. **Why you chose it**: Rationale for selection
4. **Preprocessing**: Any cleaning/filtering applied?

### 5.3 Retrain and Evaluate
- Combine original + new data
- Retrain classifier with same approach
- Evaluate on same dev set

### 5.4 Report and Discuss Results

**Performance comparison table:**
| Metric | Original | Extended | Change |
|--------|----------|----------|--------|
| Precision | ? | ? | ↑/↓ |
| Recall | ? | ? | ↑/↓ |
| F1-Score | ? | ? | ↑/↓ |

**Discussion points:**
- Did performance increase or get worse?
- **Why?** Provide evidence for your reasoning
- What patterns in the new data affected results?

## Potential Issues to Discuss

1. **Domain mismatch**: First names vs. last names behave differently
2. **Class imbalance**: More English data skews the model
3. **Noise**: Lower quality sources may introduce errors
4. **Ethnic overlap**: Some "English" names may have Slavic origins

## Deliverables
- [ ] Data source documented (URL, description)
- [ ] Data size specified
- [ ] Rationale for why you chose this data
- [ ] Updated model trained on extended data
- [ ] Performance comparison table (original vs. extended)
- [ ] Discussion of results with evidence

## Expected Output Format
```
Extended Data Source: US Census 2000 Surnames
- URL: https://www.census.gov/topics/population/genealogy.html
- Size: 151,671 unique surnames
- Rationale: Official source, diverse English surnames
- Preprocessing: Filtered to 5+ character names, lowercase

Results Comparison:
                Original    Extended    Change
Precision       0.85        0.83        -2%
Recall          0.78        0.84        +6%
F1              0.81        0.83        +2%

Discussion:
The extended data improved recall but slightly hurt precision.
This suggests the model now captures more Russian patterns correctly
but also incorrectly classifies some English names as Russian.

Evidence: Examining misclassified names reveals that Census data 
includes Slavic-origin surnames (e.g., "Kowalski", "Petrosky") that
are now common in English-speaking countries. These names share
bigram patterns with Russian names, causing the precision drop.
```
