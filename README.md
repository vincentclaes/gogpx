# gogpx

Image → GPX pipeline using managed APIs (OpenAI vision + GraphHopper).

## Quick start
1. Ensure `OPENAI_API_KEY` and `GRAPHOPPER_API_KEY` are set (or `GRAPHHOPPER_API_KEY`).
2. Install deps: `uv sync --extra dev`
3. Run:
```bash
python scripts/route_from_image.py --image data/kluisbos/kluisbos.png --out /tmp/kluisbos_out.gpx
```
To disable the agent loop (deterministic pipeline), add `--no-agent`.

## Tests
- Unit tests: `ci/run_tests.sh unit`
- Integration tests (requires API keys): `ci/run_tests.sh integration`
