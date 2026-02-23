from __future__ import annotations

from typing import List, Tuple


def build_gpx(points: List[Tuple[float, float]], name: str = "route") -> str:
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<gpx version="1.1" creator="gogpx" xmlns="http://www.topografix.com/GPX/1/1">',
        f"  <trk><name>{name}</name><trkseg>",
    ]
    for lat, lon in points:
        lines.append(f'    <trkpt lat="{lat:.7f}" lon="{lon:.7f}"></trkpt>')
    lines.append("  </trkseg></trk>")
    lines.append("</gpx>")
    return "\n".join(lines)
