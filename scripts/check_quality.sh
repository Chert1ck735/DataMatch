#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
.venv/bin/ruff check src tests scripts
PYTHONPATH=src .venv/bin/python -m compileall -q src tests scripts
PYTHONPATH=src .venv/bin/python -m pytest -q tests
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git diff --check
fi
