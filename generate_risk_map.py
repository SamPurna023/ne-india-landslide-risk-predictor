"""Train Gradient Boosting model on nationwide dataset and output NE India predictions."""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

INPUT_CSV  = "all_districts_dataset.csv"
OUTPUT_CSV = "final_risk_predictions.csv"
MODEL_FILE = "final_gb_model.joblib"

FEATURE_COLS = ["Avg_Annual_Rainfall_mm", "Avg_Slope_Degrees", "Elevation_m"]

NE_STATES = [
    "Assam", "Meghalaya", "Manipur", "Mizoram",
    "Nagaland", "Tripura", "Arunachal Pradesh"
]

def add_labels(df):
    df = df.copy()
    df["Actual_Label"] = df["Rank"].apply(lambda r: "At-Risk" if r <= 60 else "Lower-Risk")
    return df

def main():
    print("Loading nationwide dataset:", INPUT_CSV)
    df = pd.read_csv(INPUT_CSV)
    df = add_labels(df)

    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df_clean = df.dropna(subset=FEATURE_COLS + ["Actual_Label"]).copy()
    total_clean = len(df_clean)
    print(f"Loaded {total_clean} clean nationwide districts for training.")

    X = df_clean[FEATURE_COLS].values
    y = df_clean["Actual_Label"].values

    print("\nTraining HistGradientBoostingClassifier on all 131 nationwide districts...")
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("gb", HistGradientBoostingClassifier(
            max_iter=300,
            max_depth=4,
            learning_rate=0.05,
            class_weight="balanced",
            random_state=42,
        ))
    ])
    pipeline.fit(X, y)
    print("Training complete.")

    joblib.dump(pipeline, MODEL_FILE)
    print(f"Saved model -> {MODEL_FILE}")

    # Generate predictions across dataset
    df_clean["Predicted_Label"] = pipeline.predict(X)
    probs = pipeline.predict_proba(X)
    
    classes = list(pipeline.named_steps["gb"].classes_)
    at_risk_idx = classes.index("At-Risk")
    df_clean["AtRisk_Probability"] = (probs[:, at_risk_idx] * 100).round(1)
    df_clean["Correct"] = (df_clean["Actual_Label"] == df_clean["Predicted_Label"])

    # Filter to NE India districts
    ne_df = df_clean[df_clean["State"].isin(NE_STATES)].copy()
    
    # Sort NE districts by model confidence (highest At-Risk probability first)
    ne_df = ne_df.sort_values("AtRisk_Probability", ascending=False).reset_index(drop=True)
    ne_df.index = ne_df.index + 1
    ne_df.index.name = "Model_Risk_Rank"
    ne_df = ne_df.reset_index()

    # Save filtered NE dataset to CSV
    out_cols = [
        "Model_Risk_Rank", "Rank", "District", "State", "Latitude", "Longitude",
        "Avg_Annual_Rainfall_mm", "Avg_Slope_Degrees", "Elevation_m",
        "Actual_Label", "Predicted_Label", "AtRisk_Probability", "Correct"
    ]
    # Rename Elevation_m to Avg_Elevation_m if desired for consistency
    ne_df = ne_df.rename(columns={"Elevation_m": "Avg_Elevation_m"})
    ne_df["Rainfall_x_Slope"] = ne_df["Avg_Annual_Rainfall_mm"] * ne_df["Avg_Slope_Degrees"]
    
    final_cols = [
        "Model_Risk_Rank", "Rank", "District", "State", "Latitude", "Longitude",
        "Avg_Annual_Rainfall_mm", "Avg_Slope_Degrees", "Avg_Elevation_m", "Rainfall_x_Slope",
        "Actual_Label", "Predicted_Label", "AtRisk_Probability", "Correct"
    ]
    
    ne_df[final_cols].to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved NE India predictions ({len(ne_df)} districts) -> {OUTPUT_CSV}")

    # Summary Console Table
    print("\n" + "=" * 76)
    print(f"  {'Rank':<5} {'District':<22} {'State':<18} {'Prob%':>6}  {'Predicted':<12} {'Actual':<12} {'OK?'}")
    print(f"  {'-'*5} {'-'*22} {'-'*18} {'-'*6}  {'-'*12} {'-'*12} {'-'*4}")
    for _, row in ne_df.iterrows():
        ok = "YES" if row["Correct"] else "MISS"
        highlight = " <--" if row["Predicted_Label"] == "At-Risk" else ""
        print(f"  {row['Model_Risk_Rank']:<5} {row['District']:<22} {row['State']:<18} {row['AtRisk_Probability']:>5.1f}%"
              f"  {row['Predicted_Label']:<12} {row['Actual_Label']:<12} {ok}{highlight}")
    print("=" * 76)

    correct_cnt = ne_df["Correct"].sum()
    total_ne = len(ne_df)
    caught_cnt = ((ne_df["Actual_Label"] == "At-Risk") & (ne_df["Predicted_Label"] == "At-Risk")).sum()
    total_at_risk = (ne_df["Actual_Label"] == "At-Risk").sum()
    max_prob = ne_df["AtRisk_Probability"].max()

    print(f"\n  NE India Summary Stats:")
    print(f"    - Flagged At-Risk : {(ne_df['Predicted_Label'] == 'At-Risk').sum()}")
    print(f"    - Correctly Caught: {caught_cnt}/{total_at_risk} districts")
    print(f"    - NE Accuracy     : {correct_cnt}/{total_ne} ({100*correct_cnt/total_ne:.1f}%)")
    print(f"    - Max Risk Prob   : {max_prob:.1f}%\n")

if __name__ == "__main__":
    main()
