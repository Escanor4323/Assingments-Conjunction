import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score
)


def build_vectorizer(names, ngram_range=(2, 2)):
    vectorizer = CountVectorizer(
        analyzer="char", ngram_range=ngram_range
    )
    X = vectorizer.fit_transform(names)
    return vectorizer, X


def train_logistic(X, y):
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y)
    return model


def evaluate(y_true, y_pred, label="Results"):
    print(f"\n{label}")
    print("=" * 40)
    print(classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred,
                          labels=["English", "Russian"])
    print("Confusion Matrix:")
    print(f"              Predicted")
    print(f"            Eng    Rus")
    print(f"Actual Eng  {cm[0][0]:<6} {cm[0][1]}")
    print(f"       Rus  {cm[1][0]:<6} {cm[1][1]}")

    p = precision_score(y_true, y_pred, pos_label="Russian")
    r = recall_score(y_true, y_pred, pos_label="Russian")
    f = f1_score(y_true, y_pred, pos_label="Russian")
    return {"precision": p, "recall": r, "f1": f}
