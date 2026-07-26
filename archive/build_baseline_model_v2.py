"""Binary classification baseline with cross-validation."""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

INPUT_CSV = "districts_with_rainfall_and_slope.csv"
OUTPUT_CSV = "districts_final_with_risk_labels_v2.csv"


def add_risk_columns(df):
    df["risk_score"] = 148 - df["Rank"]

    # Binary this time: top 60 ranked (nationally) = At-Risk, rest = Lower-Risk
    df["risk_binary"] = df["Rank"].apply(lambda r: "At-Risk" if r <= 60 else "Lower-Risk")
    return df


def main():
    print("Loading data...")
    df = pd.read_csv(INPUT_CSV)
    df = add_risk_columns(df)

    df_clean = df.dropna(subset=["Avg_Annual_Rainfall_mm", "Avg_Slope_Degrees", "risk_binary"])
    dropped = len(df) - len(df_clean)
    if dropped > 0:
        print(f"Dropped {dropped} rows with missing data.")

    print("\nClass counts:")
    print(df_clean["risk_binary"].value_counts())

    df_clean.to_csv(OUTPUT_CSV, index=False)

    X = df_clean[["Avg_Annual_Rainfall_mm", "Avg_Slope_Degrees"]]
    y = df_clean["risk_binary"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 5-fold cross-validation: with 57 rows, ~11-12 per fold — reasonable
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # Random forests can capture non-linear feature combinations (e.g. "high
    # rainfall AND high slope together" being riskier than either alone) that
    # logistic regression can't. class_weight="balanced" still applies here too.
    model = RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42
    )

    # cross_val_predict gives us a prediction for EVERY row, each made by a model
    # that never saw that row during training — so it's a fair evaluation
    # across your whole dataset, not just one 25% slice.
    y_pred = cross_val_predict(model, X_scaled, y, cv=cv)

    print("\n--- Cross-validated results (across all 5 folds) ---")
    print(classification_report(y, y_pred, zero_division=0))

    print("Confusion matrix (rows=actual, columns=predicted):")
    labels = sorted(y.unique())
    cm = confusion_matrix(y, y_pred, labels=labels)
    print(f"           {labels}")
    for label, row in zip(labels, cm):
        print(f"  {label:10s} {row}")

    # Fit on ALL data now (for feature importance — cross_val_predict doesn't give us one model)
    model.fit(X_scaled, y)
    print("\nFeature importance (higher = more influence on the model's decisions):")
    for feature, importance in zip(X.columns, model.feature_importances_):
        print(f"  {feature}: {round(importance, 3)}")


if __name__ == "__main__":
    main()