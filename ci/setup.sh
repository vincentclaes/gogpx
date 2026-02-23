#!/usr/bin/env bash
set -euo pipefail

echo "[setup] installing dependencies"
uv sync --extra dev
echo "[setup] done"
