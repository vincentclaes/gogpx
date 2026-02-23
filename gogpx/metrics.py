from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from typing import List, Tuple


def parse_gpx(path: str) -> List[Tuple[float, float]]:
    tree = ET.parse(path)
    root = tree.getroot()
    ns = {"gpx": "http://www.topografix.com/GPX/1/1"}
    points: List[Tuple[float, float]] = []
    for trkpt in root.findall(".//gpx:trkpt", ns):
        lat = float(trkpt.get("lat"))
        lon = float(trkpt.get("lon"))
        points.append((lat, lon))
    return points


def haversine_km(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    lat1, lon1 = a
    lat2, lon2 = b
    r = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    sa = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return 2 * r * math.asin(math.sqrt(sa))


def path_length_km(points: List[Tuple[float, float]]) -> float:
    total = 0.0
    for i in range(1, len(points)):
        total += haversine_km(points[i - 1], points[i])
    return total


def avg_nearest_km(a: List[Tuple[float, float]], b: List[Tuple[float, float]], step: int = 5) -> float:
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
