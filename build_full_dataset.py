"""Build full national district dataset with rainfall, elevation, and slope."""

import requests
import time
import csv
import math

HEADERS = {"User-Agent": "student-landslide-project"}

# Full national list: (Rank, District, State) — all 147 from the ISRO Landslide Atlas
districts = [
    (1, "Rudraprayag", "Uttaranchal"), (2, "Tehri Garhwal", "Uttaranchal"), (3, "Thrissur", "Kerala"),
    (4, "Rajouri", "J&K"), (5, "Palakkad", "Kerala"), (6, "Poonch", "J&K"),
    (7, "Malappuram", "Kerala"), (8, "South District", "Sikkim"), (9, "East District", "Sikkim"),
    (10, "Kozhikode", "Kerala"), (11, "Imphal West", "Manipur"), (12, "Kodagu", "Karnataka"),
    (13, "Wayanad", "Kerala"), (14, "Jammu", "J&K"), (15, "Ernakulam", "Kerala"),
    (16, "Mandi", "Himachal Pradesh"), (17, "Udhampur", "J&K"), (18, "Idukki", "Kerala"),
    (19, "Chamoli", "Uttaranchal"), (20, "West District", "Sikkim"), (21, "Uttarkashi", "Uttaranchal"),
    (22, "Cachar", "Assam"), (23, "Garhwal", "Uttaranchal"), (24, "Kottayam", "Kerala"),
    (25, "Hamirpur", "Himachal Pradesh"), (26, "Kannur", "Kerala"), (27, "Pulwama", "J&K"),
    (28, "Thiruvananthapuram", "Kerala"), (29, "Dehra Dun", "Uttaranchal"), (30, "Bilaspur", "Himachal Pradesh"),
    (31, "West Garo Hills", "Meghalaya"), (32, "Chamba", "Himachal Pradesh"), (33, "Pathanamthitta", "Kerala"),
    (34, "East Khasi Hills", "Meghalaya"), (35, "Darjiling", "West Bengal"), (36, "Coimbatore", "Tamil Nadu"),
    (37, "Solan", "Himachal Pradesh"), (38, "Aizawl", "Mizoram"), (39, "Lunglei", "Mizoram"),
    (40, "Kamrup", "Assam"), (41, "Dindigul", "Tamil Nadu"), (42, "Kathua", "J&K"),
    (43, "Kanniyakumari", "Tamil Nadu"), (44, "Kasaragod", "Kerala"), (45, "Lawngtlai", "Mizoram"),
    (46, "Kinnaur", "Himachal Pradesh"), (47, "Hailakandi", "Assam"), (48, "Kollam", "Kerala"),
    (49, "Goalpara", "Assam"), (50, "Bageshwar", "Uttaranchal"), (51, "North District", "Sikkim"),
    (52, "Anantnag", "J&K"), (53, "Hassan", "Karnataka"), (54, "Dakshina Kannada", "Karnataka"),
    (55, "Karbi Anglong", "Assam"), (56, "Lohit", "Arunachal Pradesh"), (57, "Kullu", "Himachal Pradesh"),
    (58, "Baramulla", "J&K"), (59, "Theni", "Tamil Nadu"), (60, "Kolasib", "Mizoram"),
    (61, "Shimla", "Himachal Pradesh"), (62, "Kangra", "Himachal Pradesh"), (63, "Churachandpur", "Manipur"),
    (64, "East Garo Hills", "Meghalaya"), (65, "Champawat", "Uttaranchal"), (66, "West Khasi Hills", "Meghalaya"),
    (67, "Ribhoi", "Meghalaya"), (68, "Naini Tal", "Uttaranchal"), (69, "Jaintia Hills", "Meghalaya"),
    (70, "Una", "Himachal Pradesh"), (71, "Mamit", "Mizoram"), (72, "Tirunelveli", "Tamil Nadu"),
    (73, "Papum Pare", "Arunachal Pradesh"), (74, "Tawang", "Arunachal Pradesh"), (75, "West Tripura", "Tripura"),
    (76, "North Tripura", "Tripura"), (77, "Udupi", "Karnataka"), (78, "East Siang", "Arunachal Pradesh"),
    (79, "Doda", "J&K"), (80, "Thane", "Maharashtra"), (81, "Almora", "Uttaranchal"),
    (82, "Serchhip", "Mizoram"), (83, "North Cachar Hills", "Assam"), (84, "Lower Subansiri", "Arunachal Pradesh"),
    (85, "The Nilgiris", "Tamil Nadu"), (86, "Pithoragarh", "Uttaranchal"), (87, "West Kameng", "Arunachal Pradesh"),
    (88, "Sirmaur", "Himachal Pradesh"), (89, "Phek", "Nagaland"), (90, "South Garo Hills", "Meghalaya"),
    (91, "Changlang", "Arunachal Pradesh"), (92, "Chikmagalur", "Karnataka"), (93, "Mon", "Nagaland"),
    (94, "Tuensang", "Nagaland"), (95, "Bongaigaon", "Assam"), (96, "Lower Dibang Valley", "Arunachal Pradesh"),
    (97, "Senapati", "Manipur"), (98, "Srinagar", "J&K"), (99, "Morigaon", "Assam"),
    (100, "Tamenglong", "Manipur"), (101, "East Kameng", "Arunachal Pradesh"), (102, "Saiha", "Mizoram"),
    (103, "Shimoga", "Karnataka"), (104, "Upper Siang", "Arunachal Pradesh"), (105, "Mokokchung", "Nagaland"),
    (106, "Kohima", "Nagaland"), (107, "Pune", "Maharashtra"), (108, "West Siang", "Arunachal Pradesh"),
    (109, "Raygarh", "Maharashtra"), (110, "Dhalai", "Tripura"), (111, "North Goa", "Goa"),
    (112, "Ukhrul", "Manipur"), (113, "Kurung Kumey", "Arunachal Pradesh"), (114, "Sindhudurg", "Maharashtra"),
    (115, "Tirap", "Arunachal Pradesh"), (116, "Uttara Kannada", "Karnataka"), (117, "Nagaon", "Assam"),
    (118, "Champhai", "Mizoram"), (119, "Budgam", "J&K"), (120, "Upper Subansiri", "Arunachal Pradesh"),
    (121, "South Goa", "Goa"), (122, "Chandel", "Manipur"), (123, "Kargil", "J&K"),
    (124, "Haveri", "Karnataka"), (125, "Anjaw", "Arunachal Pradesh"), (126, "Lahul & Spiti", "Himachal Pradesh"),
    (127, "Dhubri", "Assam"), (128, "Nashik", "Maharashtra"), (129, "Ratnagiri", "Maharashtra"),
    (130, "South Tripura", "Tripura"), (131, "Ahmadnagar", "Maharashtra"), (132, "Kupwara", "J&K"),
    (133, "Kolhapur", "Maharashtra"), (134, "Satara", "Maharashtra"), (135, "Dibang Valley", "Arunachal Pradesh"),
    (136, "Leh-Ladakh", "J&K"), (137, "Karimganj", "Assam"), (138, "Alappuzha", "Kerala"),
    (139, "Mumbai Suburban", "Maharashtra"), (140, "Mumbai", "Maharashtra"), (141, "Thoubal", "Manipur"),
    (142, "Imphal East", "Manipur"), (143, "Bishnupur", "Manipur"), (144, "Zunheboto", "Nagaland"),
    (145, "Wokha", "Nagaland"), (146, "Haridwar", "Uttaranchal"), (147, "Udham Singh Nagar", "Uttaranchal"),
]


def get_lat_long(district, state):
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": f"{district} district, {state}, India", "format": "json", "limit": 1}
    r = requests.get(url, params=params, headers=HEADERS, timeout=10)
    data = r.json()
    if data:
        return float(data[0]["lat"]), float(data[0]["lon"])
    return None, None


def get_avg_rainfall(lat, lon):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": "2015-01-01", "end_date": "2024-12-31",
        "daily": "precipitation_sum", "timezone": "Asia/Kolkata",
    }
    r = requests.get(url, params=params, timeout=30)
    data = r.json()
    daily_values = [v for v in data.get("daily", {}).get("precipitation_sum", []) if v is not None]
    if not daily_values:
        return None
    return round(sum(daily_values) / 10, 1)


def get_elevation_batch(points):
    """Query elevation for up to 100 (lat, lon) points at once via Open-Topo-Data (free, no key)."""
    locations = "|".join(f"{lat},{lon}" for lat, lon in points)
    url = "https://api.opentopodata.org/v1/srtm90m"
    r = requests.get(url, params={"locations": locations}, timeout=30)
    r.raise_for_status()
    data = r.json()
    return [res["elevation"] for res in data["results"]]


def compute_slope_degrees(center_elev, n, s, e, w, spacing_m=1100):
    """Approximate slope (degrees) from elevation at center + 4 surrounding points."""
    if any(v is None for v in [center_elev, n, s, e, w]):
        return None
    dz_dy = (n - s) / (2 * spacing_m)
    dz_dx = (e - w) / (2 * spacing_m)
    slope_rad = math.atan(math.sqrt(dz_dx**2 + dz_dy**2))
    return round(math.degrees(slope_rad), 2)


def main():
    results = []
    total = len(districts)

    for i, (rank, district, state) in enumerate(districts, 1):
        print(f"[{i}/{total}] {district}, {state} ...")
        lat, lon = get_lat_long(district, state)
        if lat is None:
            print("   -> Could not geocode, skipping.")
            results.append((rank, district, state, "", "", "", "", ""))
            time.sleep(1)
            continue

        rainfall = get_avg_rainfall(lat, lon)

        # ~0.01 degrees is roughly 1.1km — small enough to represent local terrain
        offset = 0.01
        points = [
            (lat, lon),                # center
            (lat + offset, lon),       # north
            (lat - offset, lon),       # south
            (lat, lon + offset),       # east
            (lat, lon - offset),       # west
        ]
        try:
            elevations = get_elevation_batch(points)
            center, n, s, e, w = elevations
            slope = compute_slope_degrees(center, n, s, e, w)
        except Exception as ex:
            print(f"   -> Elevation lookup failed: {ex}")
            center, slope = None, None

        results.append((rank, district, state, lat, lon, rainfall, center, slope))
        print(f"   -> rainfall={rainfall}mm, elevation={center}m, slope={slope}deg")

        time.sleep(1.2)

    with open("full_147_districts_dataset.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Rank", "District", "State", "Latitude", "Longitude",
                          "Avg_Annual_Rainfall_mm", "Elevation_m", "Avg_Slope_Degrees"])
        writer.writerows(results)

    print("\nDone! Saved to full_147_districts_dataset.csv")


if __name__ == "__main__":
    main()
