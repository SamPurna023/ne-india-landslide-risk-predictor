"""Model comparison pipeline with feature engineering and oversampling."""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score

DEM_FILE   = "ne_india_dem.tif"
INPUT_CSV  = "districts_with_rainfall_and_slope.csv"
OUTPUT_CSV = "districts_v3_features.csv"


# -- 1. Elevation extraction ---------------------------------------------------

def get_elevation_at_point(elev_array, transform, lat, lon, window=1):
    """Mean elevation (m) in a small pixel window around a lat/lon."""
    row, col = rowcol(transform, lon, lat)
    r0, r1 = max(0, row - window), row + window + 1
    c0, c1 = max(0, col - window), col + window + 1
    patch = elev_array[r0:r1, c0:c1]
    patch = patch[~np.isnan(patch)]
    return round(float(np.mean(patch)), 1) if patch.size > 0 else None


def extract_elevation(df):
    print(f"\n[1/4] Extracting elevation from {DEM_FILE}...")
    with rasterio.open(DEM_FILE) as dem:
        elev = dem.read(1).astype(float)
        elev[elev < -1000] = np.nan          # mask no-data sentinel
        transform = dem.transform

    elevations = []
    for _, row in df.iterrows():
        lat, lon = row["Latitude"], row["Longitude"]
        if pd.isna(lat) or pd.isna(lon):
            elevations.append(None)
        else:
            e = get_elevation_at_point(elev, transform, lat, lon)
            elevations.append(e)
            print(f"   {row['District']:30s} -> {e} m")

    df = df.copy()
    df["Avg_Elevation_m"] = elevations
    return df


# -- 2. Feature engineering ----------------------------------------------------

def engineer_features(df):
    """Add Rainfall x Slope synergy term."""
    print("\n[2/4] Engineering features...")
    df = df.copy()
    df["Rainfall_x_Slope"] = df["Avg_Annual_Rainfall_mm"] * df["Avg_Slope_Degrees"]
    print("   Features: Rainfall, Slope, Elevation, Rainfall x Slope")
    return df


def add_risk_labels(df):
    df = df.copy()
    df["risk_binary"] = df["Rank"].apply(
        lambda r: "At-Risk" if r <= 60 else "Lower-Risk"
    )
    return df


# -- 3. Random oversampling (no scipy needed) ----------------------------------

def random_oversample(X_train, y_train, majority_label, minority_label, rng):
    """
    Duplicate minority-class rows until both classes are equal in size.
    Applied only inside each training fold => no data leakage.
    """
    maj_mask = y_train == majority_label
    min_mask = y_train == minority_label
    X_maj, y_maj = X_train[maj_mask], y_train[maj_mask]
    X_min, y_min = X_train[min_mask], y_train[min_mask]

    n_needed = len(X_maj) - len(X_min)          # how many duplicates to add
    idx = rng.choice(len(X_min), size=n_needed, replace=True)
    X_over = np.vstack([X_maj, X_min, X_min[idx]])
    y_over = np.concatenate([y_maj, y_min, y_min[idx]])
    return X_over, y_over


# -- 4. Model zoo (pure sklearn) -----------------------------------------------

def make_models():
    return {
        "Random Forest (baseline)": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
        ),
        # HistGradientBoostingClassifier = sklearn's histogram-based gradient
        # boosting. Same idea as XGBoost: sequential trees that fix residual
        # errors. Supports class_weight natively; no C++ compiler needed.
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


# -- 5. Cross-validation with per-fold oversampling ---------------------------

def evaluate_model(model, X, y_arr, cv, majority, minority):
    """
    Manual CV loop:
      - scale features with StandardScaler fit on training fold only
      - oversample minority in training fold
      - predict on unmodified test fold
    """
    rng = np.random.default_rng(42)
    y_pred_all = np.empty(len(y_arr), dtype=object)

    for train_idx, test_idx in cv.split(X, y_arr):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y_arr[train_idx]

        # Scale (fit only on training data)
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc  = scaler.transform(X_test)

        # Oversample minority class in training fold only
        X_res, y_res = random_oversample(X_train_sc, y_train, majority, minority, rng)

        model.fit(X_res, y_res)
        y_pred_all[test_idx] = model.predict(X_test_sc)

    return y_pred_all


# -- 6. Report -----------------------------------------------------------------

def print_report(name, y_true, y_pred):
    labels = ["At-Risk", "Lower-Risk"]
    print(f"\n{'='*62}")
    print(f"  {name}")
    print(f"{'='*62}")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("  Confusion matrix (rows=actual, cols=predicted):")
    print(f"               {labels}")
    for label, row in zip(labels, cm):
        print(f"  {label:14s} {row}")

    return f1_score(y_true, y_pred, pos_label="At-Risk",
                    average="binary", zero_division=0)


# -- main ----------------------------------------------------------------------

def main():
    print("Loading data...")
    df = pd.read_csv(INPUT_CSV)
    df = add_risk_labels(df)

    df = extract_elevation(df)
    df = engineer_features(df)

    feature_cols = [
        "Avg_Annual_Rainfall_mm",
        "Avg_Slope_Degrees",
        "Avg_Elevation_m",
        "Rainfall_x_Slope",
    ]

    df_clean = df.dropna(subset=feature_cols + ["risk_binary"])
    dropped = len(df) - len(df_clean)
    if dropped:
        print(f"\nDropped {dropped} rows with missing data.")

    df_clean.to_csv(OUTPUT_CSV, index=False)
    print(f"\nEnriched dataset saved -> {OUTPUT_CSV}")

    print("\n[3/4] Class distribution:")
    print(df_clean["risk_binary"].value_counts().to_string())

    X    = df_clean[feature_cols].values
    y    = df_clean["risk_binary"].values
    majority, minority = "Lower-Risk", "At-Risk"

    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = make_models()

    print("\n[4/4] Running 5-fold CV (oversampling applied inside each fold)...")
    results = {}

    for name, model in models.items():
        print(f"\n  > {name}...")
        y_pred = evaluate_model(model, X, y, cv, majority, minority)
        results[name] = print_report(name, y, y_pred)

    # -- Summary table --
    best = max(results.values())
    print("\n" + "="*62)
    print("  SUMMARY  --  ranked by At-Risk F1")
    print("  (F1 balances precision & recall for the minority class)")
    print("="*62)
    print(f"  {'Model':<35} {'At-Risk F1':>10}")
    print(f"  {'-'*35} {'-'*10}")
    for name, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
        marker = "  <-- BEST" if score == best else ""
        print(f"  {name:<35} {score:>10.3f}{marker}")
    print()


if __name__ == "__main__":
    main()
