#!/usr/bin/env bash
# Everything CI runs, locally, in CI's order. Run with the project venv active
# (or via `uv run scripts/preflight.sh`).
set -euo pipefail
cd "$(dirname "$0")/.."

bash scripts/check-hygiene.sh
python -m ruff check .
python -m ruff format --check .
python -m mypy
python -m pytest -q --cov --cov-report=term-missing:skip-covered "$@"
echo "preflight: ok"
