from __future__ import annotations

import math
from typing import List, Tuple


def haversine_km(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    lat1, lon1 = a
    lat2, lon2 = b
    r = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    sa = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return 2 * r * math.asin(math.sqrt(sa))


def invert_3x3(m: List[List[float]]) -> List[List[float]]:
    a, b, c = m[0]
    d, e, f = m[1]
    g, h, i = m[2]
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if abs(det) < 1e-12:
        raise ValueError("Singular matrix")
    return [
        [(e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det],
        [(f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det],
        [(d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det],
    ]


def mat_vec_mul(m: List[List[float]], v: List[float]) -> List[float]:
    return [sum(m[i][j] * v[j] for j in range(len(v))) for i in range(len(m))]


def solve_affine(points: List[Tuple[float, float, float]]) -> Tuple[float, float, float]:
    # least squares for t = a*x + b*y + c
    ata = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    atb = [0.0, 0.0, 0.0]
    for x, y, t in points:
        row = [x, y, 1.0]
        for i in range(3):
            atb[i] += row[i] * t
            for j in range(3):
                ata[i][j] += row[i] * row[j]
    inv = invert_3x3(ata)
    coeffs = mat_vec_mul(inv, atb)
    return coeffs[0], coeffs[1], coeffs[2]
