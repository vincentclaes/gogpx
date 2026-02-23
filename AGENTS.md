## Project Overview
This repo is for an AI-assisted "image -> GPX" pipeline. The app ingests a route screenshot, extracts the route, georeferences it, snaps it to real roads, and exports a GPX. The user interacts via a chat-style interface and sees route segments overlaid on the original screenshot (no map tiles required).

## Core Constraints
- Managed APIs only (no self-hosted routing, no local GPU).
- Do not read `.env` directly in code. Use runtime configuration (process env injection) and document required variables.
- Prefer chat-driven UX with annotated image overlays to show route segments.

## Primary Providers
- OpenAI vision + reasoning for image understanding (route polyline + OCR text).
- GraphHopper APIs for geocoding and map matching.

## Expected Environment Variables
- `GRAPHOPPER_API_KEY` or `GRAPHHOPPER_API_KEY`.
- `OPENAI_API_KEY`.
- Reading `.env` is allowed but optional; environment variables still take precedence.

## Agent Behavior
- Use an iterative loop until "exhausted": max steps, max queries, or no improvement.
- If exhausted, return best attempt + targeted questions for the user.
- Avoid hard confidence scoring; use qualitative status instead.

## Output Requirements
- Final route must follow real streets (map-matched).
- Export GPX (`trk` by default).
- Provide a preview overlay on the original image with segment labels.
