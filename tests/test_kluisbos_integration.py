import os
from pathlib import Path

import pytest

from gogpx.config import load_env
from gogpx.metrics import avg_nearest_km, parse_gpx, path_length_km
from gogpx.pipeline import PipelineConfig, run_pipeline


@pytest.mark.integration
def test_kluisbos_pipeline(tmp_path: Path) -> None:
    if os.environ.get("GOGPX_INTEGRATION") != "1":
        pytest.skip("integration test requires GOGPX_INTEGRATION=1")
    load_env()
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.fail("Missing OPENAI_API_KEY in environment")
    if not (os.environ.get("GRAPHOPPER_API_KEY") or os.environ.get("GRAPHHOPPER_API_KEY")):
        pytest.fail("Missing GRAPHOPPER_API_KEY (or GRAPHHOPPER_API_KEY) in environment")

    image_path = "data/kluisbos/kluisbos.png"
    ref_path = "data/kluisbos/kluisbos.gpx"
    out_path = tmp_path / "kluisbos_out.gpx"

    config = PipelineConfig(model_name="gpt-5.2", profile="foot")
    run_pipeline(image_path, str(out_path), config)

    ref = parse_gpx(ref_path)
    cand = parse_gpx(str(out_path))

    ref_len = path_length_km(ref)
    cand_len = path_length_km(cand)
    length_ratio = cand_len / ref_len if ref_len else 0.0
    avg_near = avg_nearest_km(cand, ref)

    assert 0.7 <= length_ratio <= 1.4
    assert avg_near < 2.0
