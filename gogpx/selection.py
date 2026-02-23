from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from statistics import median
from typing import Dict, Iterable, List, Tuple

from .geo import haversine_km, solve_affine
from .geocode import get_lat_lon


@dataclass
class AnchorCandidate:
    selected: List[Tuple[Dict, Dict]]
    transform: Tuple[float, float, float, float, float, float]
    score: float
    meta: Dict[str, float]


def pixel_path_length(polyline: List[Dict]) -> float:
    total = 0.0
    for i in range(1, len(polyline)):
        ax = float(polyline[i - 1]["x"])
        ay = float(polyline[i - 1]["y"])
        bx = float(polyline[i]["x"])
        by = float(polyline[i]["y"])
        total += math.hypot(bx - ax, by - ay)
    return total


def route_length_from_transform(
    polyline: List[Dict], transform: Tuple[float, float, float, float, float, float]
) -> float:
    if len(polyline) < 2:
        return 0.0
    a_lat, b_lat, c_lat, a_lon, b_lon, c_lon = transform
    total = 0.0
    prev = None
    for p in polyline:
        x = float(p["x"])
        y = float(p["y"])
        lat = a_lat * x + b_lat * y + c_lat
        lon = a_lon * x + b_lon * y + c_lon
        if prev is not None:
            total += haversine_km(prev, (lat, lon))
        prev = (lat, lon)
    return total


def _extract_distance_candidates_km(text: str) -> List[float]:
    out: List[float] = []
    if not text:
        return out
    text = text.lower()
    pattern = r"(\d+(?:[.,]\d+)?)\s*(km|kilometer|kilometre|mi|mile|m)\b"
    for raw, unit in re.findall(pattern, text):
        val = float(raw.replace(",", "."))
        if unit in ("km", "kilometer", "kilometre"):
            out.append(val)
        elif unit in ("mi", "mile"):
            out.append(val * 1.60934)
        elif unit == "m":
            out.append(val / 1000.0)
    return out


def parse_distance_hint_km(distance_text: str, ocr_text: Iterable[str]) -> float | None:
    candidates: List[float] = []
    candidates.extend(_extract_distance_candidates_km(distance_text))
    for term in ocr_text:
        candidates.extend(_extract_distance_candidates_km(term))
    if not candidates:
        return None
    # Prefer km-scale values; drop tiny values that are likely scale bars.
    filtered = [c for c in candidates if c >= 0.2]
    use = filtered or candidates
    return median(use)


def _median_scale_km_per_px(selected: List[Tuple[Dict, Dict]]) -> float | None:
    ratios: List[float] = []
    for i in range(len(selected)):
        a1, h1 = selected[i]
        x1 = float(a1["x"])
        y1 = float(a1["y"])
        lat1, lon1 = get_lat_lon(h1)
        for j in range(i + 1, len(selected)):
            a2, h2 = selected[j]
            x2 = float(a2["x"])
            y2 = float(a2["y"])
            pix = math.hypot(x2 - x1, y2 - y1)
            if pix <= 0:
                continue
            lat2, lon2 = get_lat_lon(h2)
            geo = haversine_km((lat1, lon1), (lat2, lon2))
            if geo <= 0:
                continue
            ratios.append(geo / pix)
    if not ratios:
        return None
    return median(ratios)


def _anchor_fit_rmse_km(
    selected: List[Tuple[Dict, Dict]], transform: Tuple[float, float, float, float, float, float]
) -> float:
    if not selected:
        return float("inf")
    a_lat, b_lat, c_lat, a_lon, b_lon, c_lon = transform
    errs = []
    for anchor, hit in selected:
        x = float(anchor["x"])
        y = float(anchor["y"])
        pred_lat = a_lat * x + b_lat * y + c_lat
        pred_lon = a_lon * x + b_lon * y + c_lon
        lat, lon = get_lat_lon(hit)
        errs.append(haversine_km((pred_lat, pred_lon), (lat, lon)))
    if not errs:
        return float("inf")
    return (sum(e * e for e in errs) / len(errs)) ** 0.5


def _cluster_spread_km(selected: List[Tuple[Dict, Dict]]) -> float:
    if not selected:
        return 0.0
    coords = [get_lat_lon(h) for _, h in selected]
    lat = sum(c[0] for c in coords) / len(coords)
    lon = sum(c[1] for c in coords) / len(coords)
    center = (lat, lon)
    return max(haversine_km(center, c) for c in coords)


def score_candidate(
    polyline: List[Dict],
    selected: List[Tuple[Dict, Dict]],
    transform: Tuple[float, float, float, float, float, float],
    center: Tuple[float, float] | None,
    distance_hint_km: float | None,
) -> Tuple[float, Dict[str, float]]:
    route_len_km = route_length_from_transform(polyline, transform)
    fit_km = _anchor_fit_rmse_km(selected, transform)
    score = 0.0
    meta: Dict[str, float] = {
        "route_len_km": route_len_km,
        "fit_km": fit_km,
    }

    w_fit = float(os.environ.get("GOGPX_SCORE_W_FIT", "1.0"))
    w_scale = float(os.environ.get("GOGPX_SCORE_W_SCALE", "2.0"))
    w_hint = float(os.environ.get("GOGPX_SCORE_W_HINT", "3.0"))
    w_center = float(os.environ.get("GOGPX_SCORE_W_CENTER", "0.3"))
    w_spread = float(os.environ.get("GOGPX_SCORE_W_SPREAD", "0.2"))

    score += w_fit * fit_km

    scale = _median_scale_km_per_px(selected)
    if scale:
        pix_len = pixel_path_length(polyline)
        expected = pix_len * scale
        if expected > 0:
            ratio = route_len_km / expected
            meta["scale_ratio"] = ratio
            score += w_scale * abs(math.log(ratio))

    if distance_hint_km:
        ratio = route_len_km / distance_hint_km if distance_hint_km > 0 else 0.0
        meta["hint_ratio"] = ratio
        if ratio > 0:
            score += w_hint * abs(math.log(ratio))

    if center:
        center_d = sum(haversine_km(center, get_lat_lon(h)) for _, h in selected) / max(len(selected), 1)
        meta["center_km"] = center_d
        score += w_center * center_d

    spread = _cluster_spread_km(selected)
    if route_len_km > 0:
        spread_ratio = spread / route_len_km
        meta["spread_ratio"] = spread_ratio
        if spread_ratio > 2.0:
            score += w_spread * (spread_ratio - 2.0)

    return score, meta


def rank_anchor_combos(
    anchors: List[Dict],
    hits_by_label: Dict[str, List[Dict]],
    polyline: List[Dict],
    center: Tuple[float, float] | None,
    distance_hint_km: float | None,
    center_radius_km: float = 20.0,
    max_anchors: int = 6,
    max_hits: int = 5,
    max_combos: int = 1500,
    max_candidates: int | None = None,
) -> List[AnchorCandidate]:
    max_anchors = int(os.environ.get("GOGPX_MAX_ANCHORS", str(max_anchors)))
    max_hits = int(os.environ.get("GOGPX_MAX_HITS", str(max_hits)))
    scored = []
    for anchor in anchors:
        label = anchor.get("label", "")
        hits = hits_by_label.get(label, [])
        if not hits:
            continue
        score = len(label) + 2 * len(hits)
        scored.append((score, anchor))
    scored.sort(key=lambda x: x[0], reverse=True)
    anchors = [a for _, a in scored][:max_anchors]

    labels = [a["label"] for a in anchors]
    hit_lists = []
    for label in labels:
        hits = hits_by_label.get(label, [])[:max_hits]
        if center:
            hits = [h for h in hits if haversine_km(center, get_lat_lon(h)) <= center_radius_km]
        hit_lists.append(hits)
    if not labels or any(len(h) == 0 for h in hit_lists):
        return []

    from itertools import product

    candidates: List[AnchorCandidate] = []
    for idx, combo_hits in enumerate(product(*hit_lists)):
        if idx >= max_combos:
            break
        lat_points = []
        lon_points = []
        for anchor, hit in zip(anchors, combo_hits):
            x = float(anchor["x"])
            y = float(anchor["y"])
            lat, lon = get_lat_lon(hit)
            lat_points.append((x, y, lat))
            lon_points.append((x, y, lon))
        try:
            a_lat, b_lat, c_lat = solve_affine(lat_points)
            a_lon, b_lon, c_lon = solve_affine(lon_points)
        except ValueError:
            continue
        transform = (a_lat, b_lat, c_lat, a_lon, b_lon, c_lon)
        selected = list(zip(anchors, combo_hits))
        score, meta = score_candidate(polyline, selected, transform, center, distance_hint_km)
        if "hint_ratio" in meta:
            max_hint = float(os.environ.get("GOGPX_HINT_RATIO_MAX", "4.0"))
            ratio = meta["hint_ratio"]
            if ratio < (1.0 / max_hint) or ratio > max_hint:
                continue
        if "scale_ratio" in meta:
            max_scale = float(os.environ.get("GOGPX_SCALE_RATIO_MAX", "6.0"))
            ratio = meta["scale_ratio"]
            if ratio < (1.0 / max_scale) or ratio > max_scale:
                continue
        candidates.append(AnchorCandidate(selected=selected, transform=transform, score=score, meta=meta))

    candidates.sort(key=lambda c: c.score)
    if max_candidates is None:
        max_candidates = int(os.environ.get("GOGPX_MAX_CANDIDATES", "6"))
    return candidates[:max_candidates]
