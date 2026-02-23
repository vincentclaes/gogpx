from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from .config import get_graphhopper_key, get_openai_key, load_env
from .geo import solve_affine
import math

from .geocode import (
    center_from_hits,
    collect_anchor_hits,
    find_region_center,
    get_lat_lon,
    pick_anchor_hits,
)
from .gpx import build_gpx
from .metrics import path_length_km
import os

from .graphhopper import map_match, route_segment, route_via_points_chunked
from .selection import parse_distance_hint_km, rank_anchor_combos
from .vision import VisionConfig, VisionOutput, extract_route_data


@dataclass
class PipelineConfig:
    model_name: str = "gpt-5.2"
    profile: str = "foot"
    debug_json: str | None = None
    debug_raw_gpx: str | None = None
    debug_selected: str | None = None
    use_agent: bool = True


def _route_via_anchors(
    key: str, profile: str, anchors_latlon: List[Tuple[float, float]]
) -> List[Tuple[float, float]]:
    if len(anchors_latlon) < 2:
        raise ValueError("Need at least 2 anchors to route")
    stitched: List[Tuple[float, float]] = []
    for i in range(1, len(anchors_latlon)):
        seg = route_segment(key, profile, anchors_latlon[i - 1], anchors_latlon[i])
        if stitched and seg:
            seg = seg[1:]
        stitched.extend(seg)
    return stitched


def _sample_points(points: List[Tuple[float, float]], max_points: int) -> List[Tuple[float, float]]:
    if len(points) <= max_points:
        return points
    step = (len(points) - 1) / float(max_points - 1)
    sampled: List[Tuple[float, float]] = []
    for i in range(max_points):
        idx = int(round(i * step))
        sampled.append(points[idx])
    return sampled


def _point_to_segment_distance(px, py, ax, ay, bx, by) -> float:
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay
    ab_len_sq = abx * abx + aby * aby
    if ab_len_sq == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, (apx * abx + apy * aby) / ab_len_sq))
    projx = ax + t * abx
    projy = ay + t * aby
    return math.hypot(px - projx, py - projy)


def _min_distance_to_polyline(px: float, py: float, polyline: List[dict]) -> float:
    if len(polyline) < 2:
        return float("inf")
    best = float("inf")
    for i in range(1, len(polyline)):
        ax = float(polyline[i - 1]["x"])
        ay = float(polyline[i - 1]["y"])
        bx = float(polyline[i]["x"])
        by = float(polyline[i]["y"])
        best = min(best, _point_to_segment_distance(px, py, ax, ay, bx, by))
    return best


def _anchor_order_by_polyline(anchors: List[dict], polyline: List[dict]) -> List[dict]:
    if len(polyline) < 2:
        return anchors
    indexed = []
    for anchor in anchors:
        ax = float(anchor["x"])
        ay = float(anchor["y"])
        best_idx = 0
        best_dist = float("inf")
        for i in range(len(polyline)):
            dx = ax - float(polyline[i]["x"])
            dy = ay - float(polyline[i]["y"])
            d = dx * dx + dy * dy
            if d < best_dist:
                best_dist = d
                best_idx = i
        indexed.append((best_idx, anchor))
    indexed.sort(key=lambda x: x[0])
    return [a for _, a in indexed]


def _ensure_keys() -> Tuple[str, str]:
    openai_key = get_openai_key()
    gh_key = get_graphhopper_key()
    if not openai_key:
        raise RuntimeError("OPENAI_API_KEY is not set in environment")
    if not gh_key:
        raise RuntimeError("GRAPHOPPER_API_KEY (or GRAPHHOPPER_API_KEY) is not set in environment")
    return openai_key, gh_key


def run_pipeline(image_path: str, out_path: str, config: PipelineConfig) -> None:
    load_env()
    _openai_key, gh_key = _ensure_keys()
    if config.use_agent:
        from .agent import run_agent_pipeline

        run_agent_pipeline(
            image_path=image_path,
            out_path=out_path,
            model_name=config.model_name,
            profile=config.profile,
            debug_json=config.debug_json,
            debug_raw_gpx=config.debug_raw_gpx,
            debug_selected=config.debug_selected,
        )
        return

    vision = extract_route_data(image_path, VisionConfig(model_name=config.model_name))
    _write_debug(config.debug_json, vision)

    ocr_terms = set(t.strip().lower() for t in (vision.ocr_text or []))
    polyline = [p.model_dump() for p in vision.polyline]

    # Prefer label_points (OCR text positions) that are close to the route line
    label_points = [
        lp.model_dump()
        for lp in vision.label_points
        if lp.label.strip().lower() in ocr_terms
    ]
    ranked = []
    max_dist = float(os.environ.get("GOGPX_LABEL_MAX_DIST_PX", "120"))
    for lp in label_points:
        label = lp.get("label", "").strip()
        if not label or len(label) < 4:
            continue
        dist = _min_distance_to_polyline(lp["x"], lp["y"], polyline)
        if dist > max_dist:
            continue
        score = len(label) - (dist * 0.05)
        ranked.append((score, lp, dist))
    ranked.sort(key=lambda x: x[0], reverse=True)
    anchors = [lp for _, lp, _ in ranked[:8]]

    if len(anchors) < 3:
        raw_anchors = [a.model_dump() for a in vision.anchors]
        anchors = [a for a in raw_anchors if a.get("label", "").strip().lower() in ocr_terms]
    if len(anchors) < 3:
        raise RuntimeError("Need at least 3 anchors from vision output to georeference")
    if len(polyline) < 10:
        raise RuntimeError("Need more polyline points from vision output")

    region_hint = (vision.region_hint or "").strip()
    # Use labels near the route for region detection to avoid unrelated OCR noise.
    center_terms = [lp["label"] for _, lp, _ in ranked]
    max_center_terms = int(os.environ.get("GOGPX_CENTER_LABEL_MAX", "8"))
    center_terms = center_terms[:max_center_terms]
    center, _ = find_region_center(gh_key, center_terms or (vision.ocr_text or []), region_hint=region_hint)

    def _count_hit_labels(hits_map: dict[str, list[dict]]) -> int:
        return sum(1 for v in hits_map.values() if v)

    radius_primary = float(os.environ.get("GOGPX_GH_ANCHOR_RADIUS_KM", "15"))
    hits_by_label = collect_anchor_hits(
        anchors,
        gh_key,
        region_hint=region_hint,
        center=center,
        radius_km=radius_primary,
        max_hits=5,
    )
    center2, cluster2 = center_from_hits(hits_by_label, radius_km=30.0)
    if center2 and (center is None or len(cluster2) >= 2):
        center = center2
        hits_by_label = collect_anchor_hits(
            anchors,
            gh_key,
            region_hint=region_hint,
            center=center,
            radius_km=radius_primary,
            max_hits=5,
        )
    if _count_hit_labels(hits_by_label) < 3:
        hits_by_label = collect_anchor_hits(
            anchors,
            gh_key,
            region_hint=region_hint,
            center=center,
            radius_km=40.0,
            max_hits=5,
        )
    distance_hint_km = parse_distance_hint_km(vision.distance_text, vision.ocr_text)
    candidates = rank_anchor_combos(
        anchors,
        hits_by_label,
        polyline,
        center=center,
        distance_hint_km=distance_hint_km,
        center_radius_km=float(os.environ.get("GOGPX_GH_CENTER_RADIUS_KM", "20")),
    )
    if not candidates:
        selected = pick_anchor_hits(anchors, gh_key, region_hint=region_hint, center=center)
        if len(selected) < 3:
            selected = pick_anchor_hits(anchors, gh_key, region_hint=region_hint, radius_km=40.0, center=center)
        if len(selected) < 3:
            selected = pick_anchor_hits(anchors, gh_key, region_hint="", radius_km=40.0, center=center)
        if len(selected) < 3:
            raise RuntimeError("Could not geocode enough anchors to build transform")

        lat_points = []
        lon_points = []
        for anchor, hit in selected:
            x = float(anchor["x"])
            y = float(anchor["y"])
            lat, lon = get_lat_lon(hit)
            lat_points.append((x, y, lat))
            lon_points.append((x, y, lon))

        a_lat, b_lat, c_lat = solve_affine(lat_points)
        a_lon, b_lon, c_lon = solve_affine(lon_points)
        candidates = [
            (selected, (a_lat, b_lat, c_lat, a_lon, b_lon, c_lon)),
        ]
    else:
        candidates = [(c.selected, c.transform) for c in candidates]

    matched = None
    last_error: Exception | None = None
    final_selected = None
    max_locations = int(os.environ.get("GOGPX_GH_MAX_LOCATIONS", "30"))
    max_points = int(os.environ.get("GOGPX_GH_ROUTE_MAX_POINTS", "5"))
    max_match_points = int(os.environ.get("GOGPX_GH_MATCH_MAX_POINTS", "0"))
    min_ratio = float(os.environ.get("GOGPX_MATCH_MIN_RATIO", "0.85"))
    max_ratio = float(os.environ.get("GOGPX_MATCH_MAX_RATIO", "1.2"))

    for selected, transform in candidates:
        a_lat, b_lat, c_lat, a_lon, b_lon, c_lon = transform
        route_latlon = []
        for p in polyline:
            x = float(p["x"])
            y = float(p["y"])
            lat = a_lat * x + b_lat * y + c_lat
            lon = a_lon * x + b_lon * y + c_lon
            route_latlon.append((lat, lon))
        final_selected = selected

        gpx_in = build_gpx(route_latlon, name="raw")
        if config.debug_raw_gpx:
            Path(config.debug_raw_gpx).write_text(gpx_in, encoding="utf-8")

        raw_len = path_length_km(route_latlon)
        mm_failed = None
        alt_failed = None

        matched = None
        if max_match_points > 0 and len(route_latlon) <= max_match_points:
            try:
                mm = map_match(gh_key, gpx_in, profile=config.profile)
                if not mm.get("paths"):
                    raise RuntimeError("Map matching returned no paths")
                coords = mm["paths"][0]["points"]["coordinates"]
                matched = [(lat, lon) for lon, lat in coords]
            except RuntimeError as exc:
                mm_failed = exc

        alt = None
        if len(route_latlon) > 2 and max_locations >= 2:
            sampled = _sample_points(route_latlon, max_locations)
            try:
                alt = route_via_points_chunked(gh_key, config.profile, sampled, max_points=max_points)
            except RuntimeError as exc:
                alt_failed = exc

        candidates_match = []
        if matched:
            ratio = path_length_km(matched) / raw_len if raw_len else 0.0
            candidates_match.append((abs(1.0 - ratio), matched))
        if alt:
            ratio = path_length_km(alt) / raw_len if raw_len else 0.0
            candidates_match.append((abs(1.0 - ratio), alt))

        if candidates_match:
            candidates_match.sort(key=lambda x: x[0])
            matched = candidates_match[0][1]
        else:
            if mm_failed and ("match" in str(mm_failed).lower() or "Map Matching" in str(mm_failed)):
                anchors_latlon = [(get_lat_lon(hit)[0], get_lat_lon(hit)[1]) for _, hit in selected]
                matched = _route_via_anchors(gh_key, config.profile, anchors_latlon)
            else:
                last_error = mm_failed or alt_failed
                matched = None

        if matched and raw_len > 0:
            ratio = path_length_km(matched) / raw_len
            if ratio < min_ratio or ratio > max_ratio:
                anchor_map = {a.get("label"): h for a, h in selected}
                ordered = _anchor_order_by_polyline([a for a, _ in selected], polyline)
                ordered_hits = [anchor_map[a.get("label")] for a in ordered if a.get("label") in anchor_map]
                if len(ordered_hits) >= 2:
                    anchors_latlon = [(get_lat_lon(h)[0], get_lat_lon(h)[1]) for h in ordered_hits]
                    matched = route_via_points_chunked(
                        gh_key, config.profile, anchors_latlon, max_points=max_points
                    )
            if matched:
                break

    if not matched:
        raise last_error or RuntimeError("Route matching failed for all candidates")

    if config.debug_selected and final_selected:
        payload = []
        for anchor, hit in final_selected:
            lat, lon = get_lat_lon(hit)
            payload.append(
                {
                    "label": anchor.get("label"),
                    "x": anchor.get("x"),
                    "y": anchor.get("y"),
                    "lat": lat,
                    "lon": lon,
                }
            )
        Path(config.debug_selected).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    gpx_out = build_gpx(matched, name="matched")
    Path(out_path).write_text(gpx_out, encoding="utf-8")


def _write_debug(path: str | None, vision: VisionOutput) -> None:
    if not path:
        return
    payload = vision.model_dump()
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
