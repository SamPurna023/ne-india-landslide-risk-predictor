"""Update dashboard/data.js with latest landslide risk predictions."""

import json
import os
import pandas as pd

INPUT_CSV     = "final_risk_predictions.csv"
DATA_JS_PATH  = os.path.join("dashboard", "data.js")


def main():
    df = pd.read_csv(INPUT_CSV)
    data_list = df.to_dict(orient="records")

    os.makedirs("dashboard", exist_ok=True)

    with open(DATA_JS_PATH, "w", encoding="utf-8") as f:
        f.write(f"window.DASHBOARD_DATA = {json.dumps(data_list, indent=2)};\n")

    print(f"Updated dashboard data -> {DATA_JS_PATH}")


if __name__ == "__main__":
    main()
