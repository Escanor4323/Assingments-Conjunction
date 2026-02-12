import numpy as np
from sklearn.model_selection import train_test_split

from utils import load_data
from classifier import (
    build_vectorizer, train_logistic, evaluate
)


def main():
    names, labels = load_data()
    labels_arr = np.array(labels)

    # 80/20 stratified split
    X_train_names, X_test_names, y_train, y_test = (
        train_test_split(
            names, labels_arr, test_size=0.2,
            random_state=42, stratify=labels_arr
        )
    )

    print(f"Training set: {len(X_train_names)} names")
    print(f"Test set:     {len(X_test_names)} names")

    # bigram features
    vectorizer, X_train = build_vectorizer(
        X_train_names, ngram_range=(2, 2)
    )
    X_test = vectorizer.transform(X_test_names)

    model = train_logistic(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = evaluate(
        y_test, y_pred,
        "Task 3: Bigram Classifier on dev set"
    )

    # bigram + trigram features
    vec_23, X_train_23 = build_vectorizer(
        X_train_names, ngram_range=(2, 3)
    )
    X_test_23 = vec_23.transform(X_test_names)
    model_23 = train_logistic(X_train_23, y_train)
    y_pred_23 = model_23.predict(X_test_23)

    metrics_23 = evaluate(
        y_test, y_pred_23,
        "Bigram+Trigram Classifier"
    )

    # error analysis
    print("\nError Analysis (bigram model):")
    misclassified = []
    for name, true, pred in zip(X_test_names, y_test, y_pred):
        if true != pred:
            misclassified.append((name, true, pred))

    print(f"Total errors: {len(misclassified)} / {len(y_test)}")
    if misclassified:
        print("\nMisclassified names:")
        for name, true, pred in misclassified[:15]:
            print(f"  {name:<25} true={true}, pred={pred}")


if __name__ == "__main__":
    main()
