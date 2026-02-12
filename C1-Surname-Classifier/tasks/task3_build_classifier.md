# Task 3: Build Classifier Using Bigram Frequencies
## For Nairuti

## Objective
Using your bigram frequencies, build a simple model for identifying Russian names. Report precision and recall on `Russian-and-English-dev.txt`.

## Description
Using bigram frequencies as features, construct a classifier to predict the language origin of surnames.

## Approach Options

### Option A: Bag of Bigrams + Linear Regression
1. Create feature vector: count of each bigram in the name
2. Train logistic regression classifier
3. Predict language based on model output

### Option B: Language Model Probability Comparison
1. Build separate language models for English and Russian
2. For each name, compute P(name | English) and P(name | Russian)
3. Classify based on higher probability

## Requirements

### 3.1 Feature Extraction
- Convert each name to a feature vector of bigram counts
- Handle vocabulary: all bigrams seen in training, or top-K

### 3.2 Model Training

**Option A: Bag of Bigrams**
```python
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

def train_classifier(names, labels):
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 2))
    X = vectorizer.fit_transform(names)
    model = LogisticRegression()
    model.fit(X, labels)
    return model, vectorizer
```

**Option B: LM Probability**
```python
import math

def compute_lm_probability(name, bigram_probs):
    """
    Compute log probability of name under language model.
    P(name) = product of P(c_i | c_{i-1})
    """
    log_prob = 0
    name = name.lower()
    for i in range(1, len(name)):
        bigram = name[i-1:i+1]
        prob = bigram_probs.get(bigram, 1e-10)
        log_prob += math.log(prob)
    return log_prob
```

### 3.3 Evaluation on Dev Set

```python
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

def evaluate(y_true, y_pred):
    print(classification_report(y_true, y_pred))
    return {
        'precision': precision_score(y_true, y_pred, pos_label='Russian'),
        'recall': recall_score(y_true, y_pred, pos_label='Russian'),
        'f1': f1_score(y_true, y_pred, pos_label='Russian')
    }
```

## Deliverables
- [ ] Trained classifier model
- [ ] Feature extraction pipeline
- [ ] Classification function
- [ ] **Precision** on Russian-and-English-dev.txt
- [ ] **Recall** on Russian-and-English-dev.txt
- [ ] Confusion matrix and error analysis

## Expected Output Format
```
Results on Russian-and-English-dev.txt
=======================================
Precision (Russian): 0.85
Recall (Russian):    0.78
F1-Score:            0.81

Confusion Matrix:
              Predicted
            Eng    Rus
Actual Eng  142     18
       Rus   28     92
```
