"""Compute terrain slope from elevation DEM raster for NE India districts."""

import os
import requests
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol

API_KEY = os.environ.get("OPENTOPOGRAPHY_API_KEY", "")

# Bounding box roughly covering NE India
SOUTH, NORTH = 22.0, 29.6
WEST, EAST = 88.0, 97.6

DEM_FILE = "ne_india_dem.tif"
INPUT_CSV = "districts_with_rainfall.csv"
OUTPUT_CSV = "districts_with_rainfall_and_slope.csv"


def download_dem():
    """Download one elevation raster covering all of NE India (only runs if not already downloaded)."""
    import os
    if os.path.exists(DEM_FILE):
        print(f"{DEM_FILE} already exists, skipping download.")
        return

    print("Downloading elevation data for NE India (this may take a minute)...")
    url = "https://portal.opentopography.org/API/globaldem"
    params = {
        "demtype": "SRTMGL3",  # 90m resolution — good enough for district-level slope
        "south": SOUTH, "north": NORTH,
        "west": WEST, "east": EAST,
        "outputFormat": "GTiff",
        "API_Key": API_KEY,
    }
    r = requests.get(url, params=params, timeout=120)
    r.raise_for_status()  # will error clearly if the API key is wrong or request failed
    with open(DEM_FILE, "wb") as f:
        f.write(r.content)
    print(f"Saved elevation data to {DEM_FILE}")


def compute_slope_degrees(elevation, pixel_size_m):
    """Turn a grid of elevation values into a grid of slope (in degrees)."""
    dy, dx = np.gradient(elevation, pixel_size_m)
    slope_radians = np.arctan(np.sqrt(dx**2 + dy**2))
    slope_degrees = np.degrees(slope_radians)
    return slope_degrees


def get_slope_at_point(slope_array, transform, lat, lon, window=1):
    """Get the average slope near a given lat/long point."""
    row, col = rowcol(transform, lon, lat)
    row_start, row_end = max(0, row - window), row + window + 1
    col_start, col_end = max(0, col - window), col + window + 1
    patch = slope_array[row_start:row_end, col_start:col_end]
    patch = patch[~np.isnan(patch)]
    if patch.size == 0:
        return None
    return round(float(np.mean(patch)), 2)


def main():
    download_dem()

    print("Computing slope from elevation data...")
    with rasterio.open(DEM_FILE) as dem:
        elevation = dem.read(1).astype(float)
        elevation[elevation < -1000] = np.nan  # clean up bad/no-data values

        # Approximate pixel size in meters (SRTMGL3 is ~90m, but we compute it properly)
        pixel_size_deg = dem.transform[0]
        pixel_size_m = pixel_size_deg * 111_000  # rough degrees-to-meters conversion

        slope = compute_slope_degrees(elevation, pixel_size_m)
        transform = dem.transform

    print("Reading your districts CSV...")
    df = pd.read_csv(INPUT_CSV)

    slopes = []
    for _, row in df.iterrows():
        lat, lon = row["Latitude"], row["Longitude"]
        if pd.isna(lat) or pd.isna(lon):
            slopes.append(None)
            continue
        s = get_slope_at_point(slope, transform, lat, lon)
        slopes.append(s)
        print(f"  {row['District']}: slope = {s} degrees")

    df["Avg_Slope_Degrees"] = slopes
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDone! Saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
