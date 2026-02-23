from __future__ import annotations

import json
import multiprocessing as mp
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential

from .config import get_graphhopper_key, get_openai_key, load_env
from .console import ConsoleUX
from .geo import solve_affine
from .geocode import center_from_hits, collect_anchor_hits, find_region_center, get_lat_lon, pick_anchor_hits
from .gpx import build_gpx
from .graphhopper import map_match, route_via_points_chunked
from .metrics import path_length_km
from .selection import AnchorCandidate, parse_distance_hint_km, rank_anchor_combos
from .vision import VisionConfig, VisionOutput, extract_route_data


class AgentDecision(BaseModel):
    action: str  # continue | ask | finish
    question: Optional[str] = None
    summary: Optional[str] = None


@dataclass
class AgentState:
    vision: VisionOutput | None = None
    polyline: List[Dict] = field(default_factory=list)
    anchors: List[Dict] = field(default_factory=list)
    region_center: Tuple[float, float] | None = None
    hits_by_label: Dict[str, List[Dict]] = field(default_factory=dict)
    selected: List[Tuple[Dict, Dict]] = field(default_factory=list)
    candidates: List[AnchorCandidate] = field(default_factory=list)
    candidate_index: int = 0
    transform: Tuple[float, float, float, float, float, float] | None = None
    route_latlon: List[Tuple[float, float]] = field(default_factory=list)
    matched: List[Tuple[float, float]] = field(default_factory=list)
    match_method: str | None = None
    distance_hint_km: float | None = None


@dataclass
class AgentDeps:
    image_path: str
    out_path: str
    model_name: str
    profile: str
    debug_json: str | None = None
    debug_raw_gpx: str | None = None
    debug_selected: str | None = None
    state: AgentState = field(default_factory=AgentState)
    console: ConsoleUX = field(default_factory=ConsoleUX)


def _vision_worker(image_path: str, model_name: str, queue: mp.Queue) -> None:
    try:
        result = extract_route_data(image_path, VisionConfig(model_name=model_name))
        queue.put({"ok": True, "data": result.model_dump()})
    except Exception as exc:
        queue.put({"ok": False, "error": repr(exc)})


def _point_to_segment_distance(px, py, ax, ay, bx, by) -> float:
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay
    ab_len_sq = abx * abx + aby * aby
    if ab_len_sq == 0:
        return ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5
    t = max(0.0, min(1.0, (apx * abx + apy * aby) / ab_len_sq))
    projx = ax + t * abx
    projy = ay + t * aby
    return ((px - projx) ** 2 + (py - projy) ** 2) ** 0.5


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


def _build_route_latlon(
    polyline: List[Dict], transform: Tuple[float, float, float, float, float, float]
) -> List[Tuple[float, float]]:
    a_lat, b_lat, c_lat, a_lon, b_lon, c_lon = transform
    route_latlon = []
    for p in polyline:
        x = float(p["x"])
        y = float(p["y"])
        lat = a_lat * x + b_lat * y + c_lat
        lon = a_lon * x + b_lon * y + c_lon
        route_latlon.append((lat, lon))
    return route_latlon


def build_agent(model_name: str) -> Agent[AgentDeps, AgentDecision]:
    system_prompt = (
        "You are an orchestration agent for an image->GPX pipeline. "
        "Use the provided tools to progress through steps: "
        "extract vision, select anchors, find region, geocode anchors, "
        "select best anchor combo, build route, map match, write GPX. "
        "Do not hallucinate coordinates. Ask the user for hints only if required."
    )
    agent = Agent(model=f"openai:{model_name}", deps_type=AgentDeps, output_type=AgentDecision, system_prompt=system_prompt)

    @agent.tool
    def extract_vision(ctx: RunContext[AgentDeps]) -> str:
        ctx.deps.console.tick("vision", "extract")
        ctx.deps.console.log("vision: calling OpenAI")
        if ctx.deps.state.vision is None:
            timeout_s = float(os.environ.get("GOGPX_VISION_TIMEOUT", "60"))
            queue: mp.Queue = mp.Queue()
            proc = mp.Process(
                target=_vision_worker,
                args=(ctx.deps.image_path, ctx.deps.model_name, queue),
            )
            proc.start()
            proc.join(timeout=timeout_s)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)
                raise TimeoutError(f"vision timeout after {timeout_s}s")
            if queue.empty():
                raise RuntimeError("vision worker returned no output")
            payload = queue.get()
            if not payload.get("ok"):
                raise RuntimeError(payload.get("error", "vision worker failed"))
            ctx.deps.state.vision = VisionOutput.model_validate(payload["data"])
            if ctx.deps.debug_json:
                Path(ctx.deps.debug_json).write_text(
                    json.dumps(ctx.deps.state.vision.model_dump(), indent=2), encoding="utf-8"
                )
        ctx.deps.console.log("vision: done")
        return "vision_extracted"

    @agent.tool
    def prepare_anchors(ctx: RunContext[AgentDeps]) -> str:
        ctx.deps.console.tick("anchors", "select")
        ctx.deps.console.log("anchors: preparing")
        vision = ctx.deps.state.vision
        if not vision:
            return "missing_vision"
        polyline = [p.model_dump() for p in vision.polyline]
        ctx.deps.state.polyline = polyline
        ocr_terms = set(t.strip().lower() for t in (vision.ocr_text or []))
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
            anchors = [a.model_dump() for a in vision.anchors if a.label.strip().lower() in ocr_terms]
        ctx.deps.state.anchors = anchors
        ctx.deps.console.log(f"anchors: {len(anchors)}")
        return f"anchors={len(anchors)}"

    @agent.tool
    def find_region(ctx: RunContext[AgentDeps]) -> str:
        ctx.deps.console.tick("region", "infer")
        ctx.deps.console.log("region: searching")
        vision = ctx.deps.state.vision
        if not vision:
            return "missing_vision"
        region_hint = (vision.region_hint or "").strip()
        labels = [a.get("label", "") for a in ctx.deps.state.anchors]
        center_terms = [t for t in labels if t]
        center, _ = find_region_center(get_graphhopper_key(), center_terms or (vision.ocr_text or []), region_hint=region_hint)
        ctx.deps.state.region_center = center
        ctx.deps.console.log(f"region: {'yes' if center else 'no'}")
        return "region_ready"

    @agent.tool
    def geocode_anchors(ctx: RunContext[AgentDeps]) -> str:
        ctx.deps.console.tick("geocode", "anchors")
        ctx.deps.console.log("geocode: requesting hits")
        vision = ctx.deps.state.vision
        if not vision:
            return "missing_vision"
        region_hint = (vision.region_hint or "").strip()
        hits = collect_anchor_hits(
            ctx.deps.state.anchors,
            get_graphhopper_key(),
            region_hint=region_hint,
            center=ctx.deps.state.region_center,
            radius_km=float(os.environ.get("GOGPX_GH_ANCHOR_RADIUS_KM", "15")),
            max_hits=5,
        )
        center2, cluster2 = center_from_hits(hits, radius_km=30.0)
        if center2 and (ctx.deps.state.region_center is None or len(cluster2) >= 2):
            ctx.deps.state.region_center = center2
            hits = collect_anchor_hits(
                ctx.deps.state.anchors,
                get_graphhopper_key(),
                region_hint=region_hint,
                center=ctx.deps.state.region_center,
                radius_km=float(os.environ.get("GOGPX_GH_ANCHOR_RADIUS_KM", "15")),
                max_hits=5,
            )
        ctx.deps.state.hits_by_label = hits
        ctx.deps.console.log("geocode: done")
        return "hits_ready"

    @agent.tool
    def select_combo(ctx: RunContext[AgentDeps]) -> str:
        ctx.deps.console.tick("anchors", "fit")
        ctx.deps.console.log("anchors: selecting combo")
        vision = ctx.deps.state.vision
        distance_hint_km = None
        if vision:
            distance_hint_km = parse_distance_hint_km(vision.distance_text, vision.ocr_text)
        ctx.deps.state.distance_hint_km = distance_hint_km

        candidates = rank_anchor_combos(
            ctx.deps.state.anchors,
            ctx.deps.state.hits_by_label,
            ctx.deps.state.polyline,
            center=ctx.deps.state.region_center,
            distance_hint_km=distance_hint_km,
            center_radius_km=float(os.environ.get("GOGPX_GH_CENTER_RADIUS_KM", "20")),
        )
        if not candidates:
            vision = ctx.deps.state.vision
            region_hint = (vision.region_hint or "").strip() if vision else ""
            selected = pick_anchor_hits(
                ctx.deps.state.anchors,
                get_graphhopper_key(),
                region_hint=region_hint,
                center=ctx.deps.state.region_center,
                radius_km=float(os.environ.get("GOGPX_GH_ANCHOR_RADIUS_KM", "15")),
            )
            if len(selected) < 3:
                return "combo_failed"
            lat_points = []
            lon_points = []
            for anchor, hit in selected:
                x = float(anchor["x"])
                y = float(anchor["y"])
                lat, lon = get_lat_lon(hit)
                lat_points.append((x, y, lat))
                lon_points.append((x, y, lon))
            try:
                a_lat, b_lat, c_lat = solve_affine(lat_points)
                a_lon, b_lon, c_lon = solve_affine(lon_points)
                transform = (a_lat, b_lat, c_lat, a_lon, b_lon, c_lon)
            except ValueError:
                return "combo_failed"
            candidates = [AnchorCandidate(selected=selected, transform=transform, score=0.0, meta={})]

        ctx.deps.state.candidates = candidates
        ctx.deps.state.candidate_index = 0
        ctx.deps.state.selected = candidates[0].selected
        ctx.deps.state.transform = candidates[0].transform
        ctx.deps.console.log(f"anchors: candidates {len(candidates)}")
        if ctx.deps.debug_selected:
            payload = []
            for anchor, hit in ctx.deps.state.selected:
                lat, lon = get_lat_lon(hit)
                payload.append({"label": anchor.get("label"), "x": anchor.get("x"), "y": anchor.get("y"), "lat": lat, "lon": lon})
            Path(ctx.deps.debug_selected).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        ctx.deps.console.log("anchors: combo selected")
        return "combo_selected"

    @agent.tool
    def build_route(ctx: RunContext[AgentDeps]) -> str:
        ctx.deps.console.tick("route", "build")
        ctx.deps.console.log("route: building")
        if not ctx.deps.state.transform:
            return "missing_transform"
        route_latlon = _build_route_latlon(ctx.deps.state.polyline, ctx.deps.state.transform)
        ctx.deps.state.route_latlon = route_latlon
        if ctx.deps.debug_raw_gpx:
            Path(ctx.deps.debug_raw_gpx).write_text(build_gpx(route_latlon, name="raw"), encoding="utf-8")
        ctx.deps.console.log("route: built")
        return "route_ready"

    @agent.tool
    def match_route(ctx: RunContext[AgentDeps]) -> str:
        ctx.deps.console.tick("route", "match")
        ctx.deps.console.log("route: matching")
        if not ctx.deps.state.transform:
            return "missing_transform"

        gh_key = get_graphhopper_key()
        max_match_points = int(os.environ.get("GOGPX_GH_MATCH_MAX_POINTS", "0"))
        max_locations = int(os.environ.get("GOGPX_GH_MAX_LOCATIONS", "30"))
        max_points = int(os.environ.get("GOGPX_GH_ROUTE_MAX_POINTS", "5"))

        candidates = ctx.deps.state.candidates or [
            AnchorCandidate(selected=ctx.deps.state.selected, transform=ctx.deps.state.transform, score=0.0, meta={})
        ]
        start_idx = ctx.deps.state.candidate_index
        for idx in range(start_idx, len(candidates)):
            cand = candidates[idx]
            ctx.deps.state.candidate_index = idx
            ctx.deps.state.selected = cand.selected
            ctx.deps.state.transform = cand.transform
            route = _build_route_latlon(ctx.deps.state.polyline, cand.transform)
            ctx.deps.state.route_latlon = route
            if len(candidates) > 1:
                ctx.deps.console.log(f"route: candidate {idx + 1}/{len(candidates)}")

            gpx_in = build_gpx(route, name="raw")
            matched = None
            method = None
            if max_match_points > 0 and len(route) <= max_match_points:
                try:
                    mm = map_match(gh_key, gpx_in, profile=ctx.deps.profile)
                    if mm.get("paths"):
                        coords = mm["paths"][0]["points"]["coordinates"]
                        matched = [(lat, lon) for lon, lat in coords]
                        method = "map_match"
                    else:
                        ctx.deps.console.log("route: match empty")
                except RuntimeError as exc:
                    ctx.deps.console.log(f"route: match error {str(exc)[:200]}")
                    matched = None
            else:
                ctx.deps.console.log("route: match skipped")

            alt = None
            if len(route) > 2 and max_locations >= 2:
                step = (len(route) - 1) / float(max_locations - 1)
                sampled = [route[int(round(i * step))] for i in range(max_locations)]
                try:
                    alt = route_via_points_chunked(gh_key, ctx.deps.profile, sampled, max_points=max_points)
                except RuntimeError as exc:
                    ctx.deps.console.log(f"route: route_points error {str(exc)[:200]}")
                    alt = None

            raw_len = path_length_km(route)
            match_candidates = []
            if matched:
                match_candidates.append((abs(1.0 - path_length_km(matched) / raw_len), matched, method or "map_match"))
            if alt:
                match_candidates.append((abs(1.0 - path_length_km(alt) / raw_len), alt, "route_points"))

            if match_candidates:
                match_candidates.sort(key=lambda x: x[0])
                ctx.deps.state.matched = match_candidates[0][1]
                ctx.deps.state.match_method = match_candidates[0][2]
                ctx.deps.console.log(f"route: matched via {ctx.deps.state.match_method}")
                return "matched"

        ctx.deps.console.log("route: match_failed")
        return "match_failed"

    @agent.tool
    def write_output(ctx: RunContext[AgentDeps]) -> str:
        if not ctx.deps.state.matched:
            return "missing_match"
        gpx_out = build_gpx(ctx.deps.state.matched, name="matched")
        Path(ctx.deps.out_path).write_text(gpx_out, encoding="utf-8")
        where = "gpx_written"
        learned = f"anchors={len(ctx.deps.state.anchors)}, region={'yes' if ctx.deps.state.region_center else 'no'}, match={ctx.deps.state.match_method or 'unknown'}"
        next_step = "review GPX or provide a hint"
        ctx.deps.console.done(where, learned, next_step)
        return "gpx_written"

    return agent


def run_agent_pipeline(
    image_path: str,
    out_path: str,
    model_name: str,
    profile: str,
    debug_json: str | None,
    debug_raw_gpx: str | None,
    debug_selected: str | None,
) -> None:
    load_env()
    if not get_openai_key():
        raise RuntimeError("OPENAI_API_KEY is not set in environment")
    if not get_graphhopper_key():
        raise RuntimeError("GRAPHOPPER_API_KEY (or GRAPHHOPPER_API_KEY) is not set in environment")
    deps = AgentDeps(
        image_path=image_path,
        out_path=out_path,
        model_name=model_name,
        profile=profile,
        debug_json=debug_json,
        debug_raw_gpx=debug_raw_gpx,
        debug_selected=debug_selected,
    )
    agent = build_agent(model_name)
    prompt = (
        "Goal: produce a GPX from the image. "
        "If required data is missing, ask a single clarifying question. "
        "Otherwise call tools to finish."
    )

    def _should_retry(exc: Exception) -> bool:
        msg = str(exc).lower()
        return "rate limit" in msg or "request_limit" in msg or "429" in msg

    for _ in range(10):
        result = None
        for attempt in Retrying(
            retry=retry_if_exception(_should_retry),
            wait=wait_exponential(min=1, max=30),
            stop=stop_after_attempt(6),
            reraise=True,
        ):
            with attempt:
                result = agent.run_sync(prompt, deps=deps)
        decision = result.output
        if decision.action == "ask" and decision.question:
            raise RuntimeError(f"Agent requires input: {decision.question}")
        if decision.action == "finish":
            return
        prompt = "Continue until GPX is written."
    raise RuntimeError("Agent did not finish within max steps")
