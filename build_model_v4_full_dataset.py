"""Evaluate landslide risk classification models on nationwide dataset."""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

INPUT_CSV = "all_districts_dataset.csv"
FEATURE_COLS = ["Avg_Annual_Rainfall_mm", "Avg_Slope_Degrees", "Elevation_m"]

def add_risk_labels(df):
    df = df.copy()
    df["risk_binary"] = df["Rank"].apply(
        lambda r: "At-Risk" if r <= 60 else "Lower-Risk"
    )
    return df

def random_oversample(X_train, y_train, majority_label, minority_label, rng):
    maj_mask = y_train == majority_label
    min_mask = y_train == minority_label
    X_maj, y_maj = X_train[maj_mask], y_train[maj_mask]
    X_min, y_min = X_train[min_mask], y_train[min_mask]

    n_needed = len(X_maj) - len(X_min)
    if n_needed <= 0:
        return X_train, y_train
    idx = rng.choice(len(X_min), size=n_needed, replace=True)
    X_over = np.vstack([X_maj, X_min, X_min[idx]])
    y_over = np.concatenate([y_maj, y_min, y_min[idx]])
    return X_over, y_over

def make_models():
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
        ),
        "Gradient Boosting (HistGB)": HistGradientBoostingClassifier(
            max_iter=300,
            max_depth=4,
            learning_rate=0.05,
            class_weight="balanced",
            random_state=42,
        ),
        "SVM (RBF kernel)": SVC(
            kernel="rbf",
            class_weight="balanced",
            C=1.0,
            gamma="scale",
            probability=True,
            random_state=42,
        ),
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            C=0.1,
            solver="lbfgs",
            max_iter=1000,
            random_state=42,
        ),
    }

def evaluate_model(model, X, y_arr, cv, majority, minority):
    rng = np.random.default_rng(42)
    y_pred_all = np.empty(len(y_arr), dtype=object)
    y_prob_all = np.zeros(len(y_arr), dtype=float)

    for train_idx, test_idx in cv.split(X, y_arr):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y_arr[train_idx]

        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc  = scaler.transform(X_test)

        X_res, y_res = random_oversample(X_train_sc, y_train, majority, minority, rng)

        model.fit(X_res, y_res)
        y_pred_all[test_idx] = model.predict(X_test_sc)
        
        # Get probability for At-Risk class
        classes = list(model.classes_)
        at_risk_idx = classes.index("At-Risk")
        probs = model.predict_proba(X_test_sc)
        y_prob_all[test_idx] = probs[:, at_risk_idx]

    return y_pred_all, y_prob_all

def main():
    print("=" * 70)
    print("LANDSLIDE RISK MODEL EVALUATION - FULL DATASET (147 DISTRICTS)")
    print("=" * 70)
    
    print("\nLoading data from:", INPUT_CSV)
    df = pd.read_csv(INPUT_CSV)
    total_raw_rows = len(df)
    print(f"Total raw rows in dataset: {total_raw_rows}")

    df = add_risk_labels(df)

    # Convert feature columns to numeric, coercion if needed
    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df_clean = df.dropna(subset=FEATURE_COLS + ["risk_binary"]).copy()
    dropped_rows = total_raw_rows - len(df_clean)
    print(f"Dropped {dropped_rows} rows due to missing feature data/coordinates.")
    print(f"Clean usable rows for training/CV: {len(df_clean)}")

    print("\nClass distribution in clean dataset:")
    counts = df_clean["risk_binary"].value_counts()
    for cls_name, cnt in counts.items():
        pct = (cnt / len(df_clean)) * 100
        print(f"  {cls_name:<12s}: {cnt:>3d} rows ({pct:.1f}%)")

    X = df_clean[FEATURE_COLS].values
    y = df_clean["risk_binary"].values
    majority, minority = "Lower-Risk", "At-Risk"

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = make_models()

    print("\n" + "-" * 70)
    print("Running 5-fold Stratified Cross-Validation...")
    print("-" * 70)

    results = []

    for name, model in models.items():
        y_pred, y_prob = evaluate_model(model, X, y, cv, majority, minority)
        
        prec, rec, f1, _ = precision_recall_fscore_support(
            y, y_pred, pos_label="At-Risk", average="binary", zero_division=0
        )
        
        max_prob = np.max(y_prob) * 100
        overall_acc = (y == y_pred).mean() * 100

        results.append({
            "Model": name,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "MaxProb": max_prob,
            "Accuracy": overall_acc,
            "y_pred": y_pred,
            "y_prob": y_prob
        })

        labels = ["At-Risk", "Lower-Risk"]
        print(f"\n>>> {name} <<<")
        print(classification_report(y, y_pred, labels=labels, zero_division=0))
        cm = confusion_matrix(y, y_pred, labels=labels)
        print(f"  Confusion matrix (rows=actual, cols=predicted):")
        print(f"               {labels}")
        for label, row in zip(labels, cm):
            print(f"  {label:14s} {row}")
        print(f"  Max Predicted At-Risk Probability: {max_prob:.1f}%")

    # Sort results by At-Risk F1 score descending
    results.sort(key=lambda x: x["F1"], reverse=True)
    best_f1 = results[0]["F1"]

    print("\n" + "=" * 75)
    print("SUMMARY COMPARISON TABLE (Ranked by At-Risk F1 Score)")
    print("=" * 75)
    header = f"{'Model':<28} | {'At-Risk F1':>10} | {'Precision':>9} | {'Recall':>8} | {'Max Prob %':>10} | {'Acc %':>6}"
    print(header)
    print("-" * 75)
    for r in results:
        marker = " <-- BEST" if r["F1"] == best_f1 else ""
        row_str = (f"{r['Model']:<28} | {r['F1']:>10.3f} | {r['Precision']:>9.3f} | "
                   f"{r['Recall']:>8.3f} | {r['MaxProb']:>9.1f}% | {r['Accuracy']:>5.1f}%{marker}")
        print(row_str)
    print("=" * 75)

if __name__ == "__main__":
    main()
