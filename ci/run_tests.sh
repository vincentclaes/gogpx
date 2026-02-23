#!/usr/bin/env bash
set -euo pipefail

mode="${1:-unit}"

case "$mode" in
  unit)
    uv run pytest -m "not integration"
    ;;
  integration)
    GOGPX_INTEGRATION=1 uv run pytest -m integration
    ;;
  all)
    GOGPX_INTEGRATION=1 uv run pytest
    ;;
  *)
    echo "Usage: $0 [unit|integration|all]"
    exit 1
    ;;
esac
