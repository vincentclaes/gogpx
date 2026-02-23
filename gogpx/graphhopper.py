from __future__ import annotations

from typing import Dict, List, Tuple

import requests
import os

from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential


class GraphHopperRateLimitError(RuntimeError):
    pass


class GraphHopperServerError(RuntimeError):
    pass


def _retrying() -> Retrying:
    rps = float(os.environ.get("GOGPX_GH_RPS", "1"))
    backoff_max = float(os.environ.get("GOGPX_GH_BACKOFF_MAX", "90"))
    attempts = int(os.environ.get("GOGPX_GH_RETRY_ATTEMPTS", "10"))
    min_wait = max(1.0 / max(rps, 0.1), 0.2)
    return Retrying(
        retry=retry_if_exception_type((GraphHopperRateLimitError, GraphHopperServerError)),
        wait=wait_exponential(min=min_wait, max=backoff_max),
        stop=stop_after_attempt(attempts),
        reraise=True,
    )


def _request_json(url: str, params: Dict, timeout: int = 30) -> Dict:
    for attempt in _retrying():
        with attempt:
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code == 429:
                raise GraphHopperRateLimitError(f"Rate limited: {resp.text}")
            if 500 <= resp.status_code < 600:
                raise GraphHopperServerError(f"Server error {resp.status_code}: {resp.text}")
            resp.raise_for_status()
            return resp.json()
    raise RuntimeError("Unreachable")


def _request_raw(method: str, url: str, params, data, headers, timeout: int = 60) -> requests.Response:
    for attempt in _retrying():
        with attempt:
            resp = requests.request(method, url, params=params, data=data, headers=headers, timeout=timeout)
            if resp.status_code == 429:
                raise GraphHopperRateLimitError(f"Rate limited: {resp.text}")
            if 500 <= resp.status_code < 600:
                raise GraphHopperServerError(f"Server error {resp.status_code}: {resp.text}")
            return resp
    raise RuntimeError("Unreachable")


def geocode(key: str, query: str, limit: int = 5) -> List[Dict]:
    url = "https://graphhopper.com/api/1/geocode"
    params = {"q": query, "limit": limit, "key": key}
    data = _request_json(url, params=params, timeout=30)
    return data.get("hits", [])


def map_match(key: str, gpx_xml: str, profile: str) -> Dict:
    url = "https://graphhopper.com/api/1/match"
    params = {
        "key": key,
        "profile": profile,
        "points_encoded": "false",
        "instructions": "false",
        "calc_points": "true",
    }
    headers = {"Content-Type": "application/gpx+xml"}
    resp = _request_raw(
        "POST",
        url,
        params=params,
        data=gpx_xml.encode("utf-8"),
        headers=headers,
        timeout=60,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"GraphHopper map-match failed: {resp.status_code} {resp.text}")
    return resp.json()


def route_segment(
    key: str, profile: str, start: Tuple[float, float], end: Tuple[float, float]
) -> List[Tuple[float, float]]:
    url = "https://graphhopper.com/api/1/route"
    params = [
        ("key", key),
        ("profile", profile),
        ("points_encoded", "false"),
        ("instructions", "false"),
        ("calc_points", "true"),
        ("point", f"{start[0]},{start[1]}"),
        ("point", f"{end[0]},{end[1]}"),
    ]
    resp = _request_raw("GET", url, params=params, data=None, headers=None, timeout=60)
    if resp.status_code >= 400:
        raise RuntimeError(f"GraphHopper route failed: {resp.status_code} {resp.text}")
    data = resp.json()
    coords = data["paths"][0]["points"]["coordinates"]
    return [(lat, lon) for lon, lat in coords]


def route_via_points(
    key: str, profile: str, points: List[Tuple[float, float]]
) -> List[Tuple[float, float]]:
    if len(points) < 2:
        raise ValueError("Need at least 2 points to route")
    url = "https://graphhopper.com/api/1/route"
    params: List[Tuple[str, str]] = [
        ("key", key),
        ("profile", profile),
        ("points_encoded", "false"),
        ("instructions", "false"),
        ("calc_points", "true"),
    ]
    for lat, lon in points:
        params.append(("point", f"{lat},{lon}"))
    resp = _request_raw("GET", url, params=params, data=None, headers=None, timeout=60)
    if resp.status_code >= 400:
        raise RuntimeError(f"GraphHopper route failed: {resp.status_code} {resp.text}")
    data = resp.json()
    coords = data["paths"][0]["points"]["coordinates"]
    return [(lat, lon) for lon, lat in coords]


def route_via_points_chunked(
    key: str, profile: str, points: List[Tuple[float, float]], max_points: int
) -> List[Tuple[float, float]]:
    if max_points < 2:
        raise ValueError("max_points must be at least 2")
    if len(points) <= max_points:
        return route_via_points(key, profile, points)
    stitched: List[Tuple[float, float]] = []
    step = max_points - 1
    i = 0
    while i < len(points) - 1:
        chunk = points[i : i + max_points]
        if len(chunk) < 2:
            break
        seg = route_via_points(key, profile, chunk)
        if stitched and seg:
            seg = seg[1:]
        stitched.extend(seg)
        i += step
    return stitched
