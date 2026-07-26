"""Geocode districts and fetch historical annual rainfall data."""

import requests
import time
import csv

# Your district list (Rank, District, State) — copied from your sheet
districts = [
    (11, "Imphal West", "Manipur"), (22, "Cachar", "Assam"), (31, "West Garo Hills", "Meghalaya"),
    (34, "East Khasi Hills", "Meghalaya"), (40, "Kamrup", "Assam"), (45, "Lawngtlai", "Mizoram"),
    (47, "Hailakandi", "Assam"), (49, "Goalpara", "Assam"), (55, "Karbi Anglong", "Assam"),
    (56, "Lohit", "Arunachal Pradesh"), (60, "Kolasib", "Mizoram"), (63, "Churachandpur", "Manipur"),
    (64, "East Garo Hills", "Meghalaya"), (66, "West Khasi Hills", "Meghalaya"), (67, "Ribhoi", "Meghalaya"),
    (69, "Jaintia Hills", "Meghalaya"), (71, "Mamit", "Mizoram"), (73, "Papum Pare", "Arunachal Pradesh"),
    (74, "Tawang", "Arunachal Pradesh"), (75, "West Tripura", "Tripura"), (76, "North Tripura", "Tripura"),
    (78, "East Siang", "Arunachal Pradesh"), (82, "Serchhip", "Mizoram"), (83, "North Cachar Hills", "Assam"),
    (84, "Lower Subansiri", "Arunachal Pradesh"), (87, "West Kameng", "Arunachal Pradesh"), (89, "Phek", "Nagaland"),
    (90, "South Garo Hills", "Meghalaya"), (91, "Changlang", "Arunachal Pradesh"), (93, "Mon", "Nagaland"),
    (94, "Tuensang", "Nagaland"), (95, "Bongaigaon", "Assam"), (96, "Lower Dibang Valley", "Arunachal Pradesh"),
    (97, "Senapati", "Manipur"), (100, "Tamenglong", "Manipur"), (101, "East Kameng", "Arunachal Pradesh"),
    (104, "Upper Siang", "Arunachal Pradesh"), (105, "Mokokchung", "Nagaland"), (106, "Kohima", "Nagaland"),
    (108, "West Siang", "Arunachal Pradesh"), (110, "Dhalai", "Tripura"), (112, "Ukhrul", "Manipur"),
    (113, "Kurung Kumey", "Arunachal Pradesh"), (115, "Tirap", "Arunachal Pradesh"), (118, "Champhai", "Mizoram"),
    (120, "Upper Subansiri", "Arunachal Pradesh"), (122, "Chandel", "Manipur"), (125, "Anjaw", "Arunachal Pradesh"),
    (127, "Dhubri", "Assam"), (130, "South Tripura", "Tripura"), (135, "Dibang Valley", "Arunachal Pradesh"),
    (137, "Karimganj", "Assam"), (142, "Imphal East", "Manipur"), (143, "Bishnupur", "Manipur"),
    (144, "Zunheboto", "Nagaland"), (145, "Wokha", "Nagaland"),
]

HEADERS = {"User-Agent": "student-landslide-project"}  # required by OpenStreetMap's policy

def get_lat_long(district, state):
    """Look up coordinates for a district using OpenStreetMap's free geocoder."""
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": f"{district} district, {state}, India", "format": "json", "limit": 1}
    r = requests.get(url, params=params, headers=HEADERS, timeout=10)
    data = r.json()
    if data:
        return float(data[0]["lat"]), float(data[0]["lon"])
    return None, None

def get_avg_rainfall(lat, lon):
    """Get average annual rainfall (mm) for a location, 2015-2024, from Open-Meteo."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": "2015-01-01", "end_date": "2024-12-31",
        "daily": "precipitation_sum",
        "timezone": "Asia/Kolkata",
    }
    r = requests.get(url, params=params, timeout=30)
    data = r.json()
    daily_values = data.get("daily", {}).get("precipitation_sum", [])
    daily_values = [v for v in daily_values if v is not None]
    if not daily_values:
        return None
    total_over_10_years = sum(daily_values)
    avg_annual = total_over_10_years / 10
    return round(avg_annual, 1)

def main():
    results = []
    total = len(districts)
    for i, (rank, district, state) in enumerate(districts, 1):
        print(f"[{i}/{total}] {district}, {state} ...")
        lat, lon = get_lat_long(district, state)
        if lat is None:
            print(f"   -> Could not find coordinates, skipping rainfall for this one.")
            results.append((rank, district, state, "", "", ""))
            time.sleep(1)
            continue

        rainfall = get_avg_rainfall(lat, lon)
        results.append((rank, district, state, lat, lon, rainfall))
        print(f"   -> lat={lat}, lon={lon}, avg_annual_rainfall_mm={rainfall}")

        time.sleep(1.2)  # be polite to the free geocoding API so it doesn't block us

    with open("districts_with_rainfall.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Rank", "District", "State", "Latitude", "Longitude", "Avg_Annual_Rainfall_mm"])
        writer.writerows(results)

    print("\nDone! Saved to districts_with_rainfall.csv")

if __name__ == "__main__":
    main()
