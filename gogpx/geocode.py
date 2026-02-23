from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

from .geo import haversine_km
from .graphhopper import geocode as gh_geocode


def _cache_dir() -> Path:
    root = os.environ.get("GOGPX_CACHE_DIR")
    if root:
        path = Path(root)
    else:
        path = Path.home() / ".cache" / "gogpx"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cache_path(prefix: str, key: str) -> Path:
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return _cache_dir() / f"{prefix}_{h}.json"


def geocode_label(key: str, label: str, limit: int = 5, hint: str = "") -> List[Dict]:
    q = f"{label}, {hint}" if hint else label
    cache_path = _cache_path("geocode", q)
    if cache_path.exists():
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        return data.get("hits", [])
    try:
        hits = gh_geocode(key, q, limit=limit)
    except RuntimeError as exc:
        if "rate limited" in str(exc).lower() or "429" in str(exc):
            return []
        raise
    if not hits and hint:
        # fallback without hint if the combined query yields nothing
        try:
            hits = gh_geocode(key, label, limit=limit)
        except RuntimeError as exc:
            if "rate limited" in str(exc).lower() or "429" in str(exc):
                return []
            raise
    data = {"hits": hits}
    cache_path.write_text(json.dumps(data), encoding="utf-8")
    return data.get("hits", [])


def extract_country_filters(region_hint: str) -> Tuple[List[str], List[str]]:
    hint = region_hint.lower()
    countries = []
    codes = []
    if "belgium" in hint or "belgique" in hint or "belgie" in hint:
        countries.append("belgium")
        codes.append("be")
    return countries, codes


def filter_hits_by_country(hits: List[Dict], countries: List[str], codes: List[str]) -> List[Dict]:
    if not countries and not codes:
        return hits
    out = []
    for h in hits:
        country = (h.get("country") or "").lower()
        code = (h.get("countrycode") or h.get("country_code") or "").lower()
        if (country and country in countries) or (code and code in codes):
            out.append(h)
    return out if out else hits


def get_lat_lon(hit: Dict) -> Tuple[float, float]:
    if "point" in hit and isinstance(hit["point"], dict):
        lat = hit["point"].get("lat")
        lon = hit["point"].get("lng") or hit["point"].get("lon")
        if lat is not None and lon is not None:
            return float(lat), float(lon)
    if "lat" in hit and ("lng" in hit or "lon" in hit):
        lat = hit.get("lat")
        lon = hit.get("lng") or hit.get("lon")
        return float(lat), float(lon)
    raise ValueError("Unexpected geocode hit format")


def choose_cluster(
    all_hits: List[Tuple[str, Dict]], radius_km: float = 20.0
) -> Tuple[Tuple[float, float] | None, List[Tuple[str, Dict]]]:
    if not all_hits:
        return None, []
    coords = [(label, hit, get_lat_lon(hit)) for label, hit in all_hits]
    best = None
    best_count = -1
    for _, _, coord in coords:
        count = 0
        for _, _, coord2 in coords:
            if haversine_km(coord, coord2) <= radius_km:
                count += 1
        if count > best_count:
            best_count = count
            best = coord
    cluster = [(label, hit) for label, hit, coord in coords if haversine_km(best, coord) <= radius_km]
    return best, cluster


def find_region_center(
    key: str, ocr_terms: List[str], region_hint: str = "", max_terms: int = 8
) -> Tuple[Tuple[float, float] | None, List[Tuple[str, Dict]]]:
    all_hits: List[Tuple[str, Dict]] = []
    unique_terms = []
    seen = set()
    for term in ocr_terms:
        t = term.strip()
        if t.lower() in seen:
            continue
        seen.add(t.lower())
        unique_terms.append(t)
    # Prefer longer terms to reduce ambiguity
    unique_terms = sorted(unique_terms, key=len, reverse=True)[:max_terms]

    for term in unique_terms:
        term = term.strip()
        if len(term) < 4:
            continue
        hits = geocode_label(key, term, limit=5, hint=region_hint)
        if not hits:
            continue
        countries, codes = extract_country_filters(region_hint)
        hits = filter_hits_by_country(hits, countries, codes)
        for h in hits:
            all_hits.append((term, h))
    center, cluster = choose_cluster(all_hits, radius_km=20.0)
    return center, cluster


def order_hits_by_center(hits: List[Dict], center: Tuple[float, float] | None) -> List[Dict]:
    if not center:
        return hits
    return sorted(hits, key=lambda h: haversine_km(center, get_lat_lon(h)))


def pick_anchor_hits(
    anchors: List[Dict],
    key: str,
    max_hits: int = 5,
    region_hint: str = "",
    radius_km: float = 20.0,
    center: Tuple[float, float] | None = None,
) -> List[Tuple[Dict, Dict]]:
    all_hits: List[Tuple[str, Dict]] = []
    hits_by_label: Dict[str, List[Dict]] = {}
    for anchor in anchors:
        label = anchor.get("label", "").strip()
        if not label:
            continue
        hits = geocode_label(key, label, limit=max_hits, hint=region_hint)
        countries, codes = extract_country_filters(region_hint)
        hits = filter_hits_by_country(hits, countries, codes)
        hits_by_label[label] = hits
        for h in hits:
            all_hits.append((label, h))

    cluster_center, cluster = choose_cluster(all_hits, radius_km=radius_km)
    cluster_by_label: Dict[str, List[Dict]] = {}
    for label, hit in cluster:
        cluster_by_label.setdefault(label, []).append(hit)

    selected: List[Tuple[Dict, Dict]] = []
    for anchor in anchors:
        label = anchor.get("label", "").strip()
        if not label:
            continue
        hits = cluster_by_label.get(label) or hits_by_label.get(label) or []
        if not hits:
            continue
        if cluster_center:
            hits = order_hits_by_center(hits, cluster_center)
        elif center:
            hits = order_hits_by_center(hits, center)
        selected.append((anchor, hits[0]))
    return selected


def collect_anchor_hits(
    anchors: List[Dict],
    key: str,
    region_hint: str = "",
    max_hits: int = 5,
    center: Tuple[float, float] | None = None,
    radius_km: float = 60.0,
) -> Dict[str, List[Dict]]:
    hits_by_label: Dict[str, List[Dict]] = {}
    for anchor in anchors:
        label = anchor.get("label", "").strip()
        if not label:
            continue
        hits = geocode_label(key, label, limit=max_hits, hint=region_hint)
        countries, codes = extract_country_filters(region_hint)
        hits = filter_hits_by_country(hits, countries, codes)
        if center:
            hits = [h for h in hits if haversine_km(center, get_lat_lon(h)) <= radius_km]
        if center:
            hits = order_hits_by_center(hits, center)
        hits_by_label[label] = hits
    return hits_by_label


def center_from_hits(hits_by_label: Dict[str, List[Dict]], radius_km: float = 30.0) -> Tuple[Tuple[float, float] | None, List[Tuple[str, Dict]]]:
    all_hits: List[Tuple[str, Dict]] = []
    for label, hits in hits_by_label.items():
        for hit in hits:
            all_hits.append((label, hit))
    center, cluster = choose_cluster(all_hits, radius_km=radius_km)
    return center, cluster
