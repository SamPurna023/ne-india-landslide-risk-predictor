"""Baseline multiclass logistic regression model for landslide risk tiers."""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

INPUT_CSV = "districts_with_rainfall_and_slope.csv"
OUTPUT_CSV = "districts_final_with_risk_labels.csv"


def add_risk_columns(df):
    # Higher score = more at risk (rank 1 = most exposed nationally, so we flip it)
    df["risk_score"] = 148 - df["Rank"]

    # Since our 57 districts are ALL from the national top-147 most-at-risk list,
    # we bucket relative to each other, not the full country
    def tier(rank):
        if rank <= 60:
            return "High"
        elif rank <= 110:
            return "Medium"
        else:
            return "Low"

    df["risk_tier"] = df["Rank"].apply(tier)
    return df


def main():
    print("Loading data...")
    df = pd.read_csv(INPUT_CSV)

    df = add_risk_columns(df)
    print("\nRisk tier counts:")
    print(df["risk_tier"].value_counts())

    # Drop any rows with missing data (e.g. districts the geocoder couldn't find earlier)
    df_clean = df.dropna(subset=["Avg_Annual_Rainfall_mm", "Avg_Slope_Degrees", "risk_tier"])
    dropped = len(df) - len(df_clean)
    if dropped > 0:
        print(f"\nDropped {dropped} rows with missing data before training.")

    df_clean.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved labeled dataset to {OUTPUT_CSV}")

    # ---- Baseline model ----
    X = df_clean[["Avg_Annual_Rainfall_mm", "Avg_Slope_Degrees"]]
    y = df_clean["risk_tier"]

    # NOTE: With only ~57 rows total, a train/test split leaves very few test examples.
    # This is expected and fine for a first baseline — the real point right now is to
    # confirm your pipeline works end-to-end, not to get a publishable accuracy number.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Scale features since rainfall (~1000s) and slope (~0-40) are on very different scales
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    print("\n--- Baseline model results ---")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("Feature importance (coefficients, larger magnitude = more influence):")
    for feature, coef in zip(X.columns, model.coef_[0]):
        print(f"  {feature}: {round(coef, 3)}")


if __name__ == "__main__":
    main()
