from __future__ import annotations

from dataclasses import dataclass
import os
import time
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field
from pydantic_ai import Agent, BinaryContent


class PixelPoint(BaseModel):
    x: float
    y: float


class Anchor(BaseModel):
    label: str
    x: float
    y: float


class LabelPoint(BaseModel):
    label: str
    x: float
    y: float


class VisionOutput(BaseModel):
    ocr_text: List[str] = Field(default_factory=list)
    distance_text: str = ""
    region_hint: str = ""
    anchors: List[Anchor] = Field(default_factory=list)
    label_points: List[LabelPoint] = Field(default_factory=list)
    polyline: List[PixelPoint] = Field(default_factory=list)


@dataclass
class VisionConfig:
    model_name: str = "gpt-5.2"


def extract_route_data(image_path: str, config: VisionConfig) -> VisionOutput:
    image_bytes = Path(image_path).read_bytes()
    agent = Agent(
        model=f"openai:{config.model_name}",
        output_type=VisionOutput,
        system_prompt=(
            "You are extracting route data from a map screenshot. "
            "Return structured output exactly matching the schema."
        ),
    )
    prompt = (
        "Extract route data from this map screenshot.\n"
        "Rules:\n"
        "- anchors must use exact street/place labels that appear on the map (no 'waypoint' labels).\n"
        "- anchors must be on or very near the route line when possible.\n"
        "- include at least 4 anchors if possible.\n"
        "- polyline must be ordered along the route; include 80-200 points.\n"
        "- x,y are pixel coordinates with origin at top-left.\n"
        "- ocr_text should include street names, POIs, area names, and any distance labels.\n"
        "- region_hint should be a short guess of the area/country if visible.\n"
        "- label_points: provide coordinates for as many OCR labels as you can (center of the text).\n"
    )
    timeout_s = float(os.environ.get("GOGPX_VISION_TIMEOUT", "60"))
    start = time.time()
    result = agent.run_sync(
        [prompt, BinaryContent(data=image_bytes, media_type="image/png")],
    )
    duration = time.time() - start
    return result.output
