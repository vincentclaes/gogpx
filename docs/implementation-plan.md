**Implementation Plan**

**Goal**
Build a generic, managed‑API‑only pipeline that converts a route screenshot into a GPX. The chat UI should show route segments overlaid on the original image (no map tiles required). The pipeline must work on arbitrary images and only uses the included sample (`data/kluisbos/kluisbos.png`) for verification.

**Core Constraints**
- Managed APIs only (no self‑hosted routing, no GPU inference).
- Use OpenAI vision for image understanding.
- Use GraphHopper for geocoding and routing/map matching.
- Read `.env` if present, but do not require it (environment vars still supported).

**Stack**
- Python + `pydantic-ai` for structured vision extraction and agentic orchestration.
- `uv` for dependency management.
- Local CI scripts similar to `pogo` (lint, type, security, tests).

---

**Phase 1: Repo Scaffolding**
1. Add `pyproject.toml` with runtime deps:
   - `pydantic-ai-slim[openai]`, `requests`, `Pillow`, `python-dotenv`.
2. Add `ci/` scripts:
   - `ci/setup.sh`, `ci/run_tests.sh`, `ci/local_ci.sh`, `ci/bandit.yaml`.
3. Add package structure under `gogpx/`.

---

**Phase 2: Vision Extraction (OpenAI)**
1. Implement `gogpx/vision.py`:
   - Use `pydantic-ai` with `openai:gpt-5.2` (vision input).
   - Output schema:
     - `ocr_text`: list of detected labels
     - `distance_text`: optional distance label
     - `region_hint`: inferred region/country
     - `anchors`: street/POI labels with pixel coordinates on the route
     - `polyline`: ordered pixel coordinates (80–200 points)
2. Ensure structured output via `pydantic-ai` models.

---

**Phase 3: Candidate Region & Georeference**
1. Implement `gogpx/geocode.py`:
   - GraphHopper geocode lookup for OCR terms.
   - Cache responses to avoid rate limits.
   - Cluster hits to find a region center.
2. Use anchors + geocode hits to fit an affine transform from pixel space → lat/lon.
3. If fewer than 3 anchors resolve, return a status asking the user for more hints.

---

**Phase 4: Road‑Constrained Reconstruction**
1. Convert pixel polyline → rough lat/lon using the affine transform.
2. Attempt GraphHopper Map Matching (`/match`) to snap to real streets.
3. If Map Matching is unavailable or rate‑limited:
   - Fall back to GraphHopper routing (`/route`) between anchors.
4. Always output a route that follows known roads/trails.

---

**Phase 5: Validation**
1. If OCR extracted a distance label:
   - Compare matched length to expected distance.
2. If mismatch is large:
   - Try the next candidate region or ask the user for clarification.

---

**Phase 6: Chat‑Only Preview**
1. Render the matched route as an overlay on the original screenshot.
2. Label segments so the user can request edits in the chat.
3. Export GPX (`trk`).

---

**Phase 7: Tests**
**Unit tests**
- GPX parsing and length calculation.
- Geocode clustering behavior (mocked).

**Integration tests** (require API keys)
- Run pipeline on `data/kluisbos/kluisbos.png` and compare to `data/kluisbos/kluisbos.gpx`:
  - Length ratio within a tolerant range.
  - Average nearest distance under a loose threshold.
- Marked `integration` and gated by env var.

---

**Phase 8: CI**
1. `ci/setup.sh`: `uv sync --extra dev`
2. `ci/run_tests.sh`: unit or integration
3. `ci/local_ci.sh`: lint, type, security, tests

---

**Artifacts**
- `scripts/route_from_image.py` (CLI wrapper)
- `gogpx/` package with pipeline + modules
- `docs/implementation-plan.md` (this document)
- Tests in `tests/`

---

**Agent‑Driven Orchestration (Required)**
The pipeline is orchestrated by an AI agent with explicit tools, similar to the `pogo` agent loop.

**Agent tools (minimum)**
- `extract_vision`: call OpenAI vision for OCR + polyline + label points.
- `prepare_anchors`: select anchors near the route.
- `find_region`: infer a candidate region center from labels.
- `geocode_anchors`: fetch geocode hits for anchors.
- `select_combo`: choose the best anchor/geocode combination + affine transform.
- `build_route`: generate lat/lon polyline.
- `match_route`: map match or route via points.
- `write_output`: emit GPX.

**Agent loop**
- The agent runs tool calls until GPX is written.
- If it cannot proceed (missing anchors or missing region), it asks a single clarifying question.
- A short colored terminal UX shows stage + minimal debug lines.
