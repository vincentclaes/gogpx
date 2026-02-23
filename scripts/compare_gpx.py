#!/usr/bin/env python3
import argparse
import math
import xml.etree.ElementTree as ET


def parse_gpx(path):
    tree = ET.parse(path)
    root = tree.getroot()
    ns = {"gpx": "http://www.topografix.com/GPX/1/1"}
    points = []
    for trkpt in root.findall(".//gpx:trkpt", ns):
        lat = float(trkpt.get("lat"))
        lon = float(trkpt.get("lon"))
        points.append((lat, lon))
    return points


def haversine_km(a, b):
    lat1, lon1 = a
    lat2, lon2 = b
    r = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    sa = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return 2 * r * math.asin(math.sqrt(sa))


def path_length_km(points):
    total = 0.0
    for i in range(1, len(points)):
        total += haversine_km(points[i - 1], points[i])
    return total


def avg_nearest_km(a, b, step=5):
    if not a or not b:
        return float("inf")
    b_sample = b[::step] if len(b) > step else b
    total = 0.0
    count = 0
    for i in range(0, len(a), step):
        p = a[i]
        best = min(haversine_km(p, q) for q in b_sample)
        total += best
        count += 1
    return total / max(count, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", required=True)
    parser.add_argument("--cand", required=True)
    args = parser.parse_args()

    ref = parse_gpx(args.ref)
    cand = parse_gpx(args.cand)

    ref_len = path_length_km(ref)
    cand_len = path_length_km(cand)
    length_ratio = cand_len / ref_len if ref_len else 0.0
    avg_near = avg_nearest_km(cand, ref)

    print(f"ref_len_km={ref_len:.3f}")
    print(f"cand_len_km={cand_len:.3f}")
    print(f"length_ratio={length_ratio:.3f}")
    print(f"avg_nearest_km={avg_near:.3f}")


if __name__ == "__main__":
    main()
