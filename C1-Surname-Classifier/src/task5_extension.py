import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    classification_report
)

from utils import load_data, ensure_results_dir, RESULTS_DIR
from classifier import build_vectorizer, train_logistic


# additional English names for data extension
EXTRA_ENGLISH = [
    "Smith", "Johnson", "Williams", "Brown", "Jones",
    "Garcia", "Miller", "Davis", "Rodriguez", "Wilson",
    "Anderson", "Taylor", "Thomas", "Hernandez", "Moore",
    "Martin", "Jackson", "Thompson", "White", "Lopez",
    "Lee", "Gonzalez", "Harris", "Clark", "Lewis",
    "Robinson", "Walker", "Perez", "Hall", "Young",
    "Allen", "Sanchez", "Wright", "King", "Scott",
    "Green", "Baker", "Adams", "Nelson", "Hill",
    "Ramirez", "Campbell", "Mitchell", "Roberts", "Carter",
    "Phillips", "Evans", "Turner", "Torres", "Parker",
    "Collins", "Edwards", "Stewart", "Flores", "Morris",
    "Nguyen", "Murphy", "Rivera", "Cook", "Rogers",
    "Morgan", "Peterson", "Cooper", "Reed", "Bailey",
    "Bell", "Gomez", "Kelly", "Howard", "Ward",
    "Cox", "Diaz", "Richardson", "Wood", "Watson",
    "Brooks", "Bennett", "Gray", "James", "Reyes",
    "Cruz", "Hughes", "Price", "Myers", "Long",
    "Foster", "Sanders", "Ross", "Morales", "Powell",
    "Sullivan", "Russell", "Ortiz", "Jenkins", "Gutierrez",
    "Perry", "Butler", "Barnes", "Fisher", "Henderson",
    "Coleman", "Simmons", "Patterson", "Jordan", "Reynolds",
    "Hamilton", "Graham", "Kim", "Gonzales", "Alexander",
    "Ramos", "Wallace", "Griffin", "West", "Cole",
    "Hayes", "Chavez", "Gibson", "Bryant", "Ellis",
    "Stevens", "Murray", "Ford", "Marshall", "Owens",
    "Mcdonald", "Harrison", "Ruiz", "Kennedy", "Wells",
    "Alvarez", "Woods", "Mendoza", "Castillo", "Olson",
    "Webb", "Washington", "Tucker", "Freeman", "Burns",
    "Henry", "Vasquez", "Snyder", "Simpson", "Crawford",
    "Jimenez", "Porter", "Mason", "Shaw", "Gordon",
    "Wagner", "Hunter", "Romero", "Hicks", "Dixon",
    "Hunt", "Palmer", "Robertson", "Black", "Holmes",
    "Stone", "Meyer", "Boyd", "Mills", "Warren",
    "Fox", "Rose", "Rice", "Moreno", "Schmidt",
    "Patel", "Ferguson", "Nichols", "Herrera", "Medina",
    "Ryan", "Fernandez", "Weaver", "Daniels", "Stephens",
    "Gardner", "Payne", "Kelley", "Dunn", "Pierce",
    "Arnold", "Tran", "Spencer", "Peters", "Hawkins",
    "Grant", "Hansen", "Castro", "Hoffman", "Hart",
    "Elliott", "Cunningham", "Knight", "Bradley", "Carroll",
    "Hudson", "Duncan", "Armstrong", "Berry", "Andrews",
    "Johnston", "Ray", "Lane", "Riley", "Carpenter",
    "Perkins", "Aguilar", "Silva", "Richards", "Willis",
    "Matthews", "Chapman", "Lawrence", "Garza", "Vargas",
    "Watkins", "Wheeler", "Larson", "Carlson", "Harper",
]

EXTRA_ENGLISH_LABELS = ["English"] * len(EXTRA_ENGLISH)


def main():
    names, labels = load_data()
    labels_arr = np.array(labels)
    ensure_results_dir()

    X_train_orig, X_test, y_train_orig, y_test = (
        train_test_split(
            names, labels_arr, test_size=0.2,
            random_state=42, stratify=labels_arr
        )
    )

    # baseline on original
    vec_orig, X_tr_orig = build_vectorizer(
        X_train_orig, ngram_range=(2, 2)
    )
    X_te_orig = vec_orig.transform(X_test)
    model_orig = train_logistic(X_tr_orig, y_train_orig)
    y_pred_orig = model_orig.predict(X_te_orig)

    p_orig = precision_score(y_test, y_pred_orig,
                             pos_label="Russian")
    r_orig = recall_score(y_test, y_pred_orig,
                          pos_label="Russian")
    f_orig = f1_score(y_test, y_pred_orig,
                      pos_label="Russian")

    # extended training set
    X_train_ext = list(X_train_orig) + EXTRA_ENGLISH
    y_train_ext = list(y_train_orig) + EXTRA_ENGLISH_LABELS

    vec_ext, X_tr_ext = build_vectorizer(
        X_train_ext, ngram_range=(2, 2)
    )
    X_te_ext = vec_ext.transform(X_test)
    model_ext = train_logistic(X_tr_ext, y_train_ext)
    y_pred_ext = model_ext.predict(X_te_ext)

    p_ext = precision_score(y_test, y_pred_ext,
                            pos_label="Russian")
    r_ext = recall_score(y_test, y_pred_ext,
                         pos_label="Russian")
    f_ext = f1_score(y_test, y_pred_ext,
                     pos_label="Russian")

    print("="*55)
    print("Task 5: Extended English Data")
    print("="*55)

    print(f"\nExtended Data Source: US Census Common Surnames")
    print(f"  Size: {len(EXTRA_ENGLISH)} additional English names")
    print(f"  Rationale: Official census data, diverse English "
          f"surname patterns")
    print(f"  Preprocessing: No filtering, already clean")

    print(f"\nOriginal training: {len(X_train_orig)} names")
    print(f"Extended training: {len(X_train_ext)} names")
    print(f"Test set: {len(X_test)} names (unchanged)")

    print(f"\nOriginal model:")
    print(classification_report(y_test, y_pred_orig))

    print(f"Extended model:")
    print(classification_report(y_test, y_pred_ext))

    print(f"\nComparison Table:")
    print(f"{'Metric':<12} {'Original':<12} {'Extended':<12} "
          f"{'Change':<10}")
    print("-" * 46)
    dp = p_ext - p_orig
    dr = r_ext - r_orig
    df = f_ext - f_orig
    print(f"{'Precision':<12} {p_orig:<12.4f} {p_ext:<12.4f} "
          f"{dp:+.4f}")
    print(f"{'Recall':<12} {r_orig:<12.4f} {r_ext:<12.4f} "
          f"{dr:+.4f}")
    print(f"{'F1':<12} {f_orig:<12.4f} {f_ext:<12.4f} "
          f"{df:+.4f}")

    # save
    outpath = os.path.join(RESULTS_DIR, "task5_comparison.txt")
    with open(outpath, "w") as f:
        f.write("Task 5: Extended Data Results\n")
        f.write(f"Extra names: {len(EXTRA_ENGLISH)}\n")
        f.write(f"Original P={p_orig:.4f} R={r_orig:.4f} "
                f"F1={f_orig:.4f}\n")
        f.write(f"Extended P={p_ext:.4f} R={r_ext:.4f} "
                f"F1={f_ext:.4f}\n")
    print(f"\nResults saved to results/task5_comparison.txt")


if __name__ == "__main__":
    main()
