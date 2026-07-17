#!/usr/bin/env sh
set -eu

if [ -x .venv/bin/ruff ]; then
    PATH="$(pwd)/.venv/bin:$PATH"
    export PATH
fi

ruff check .
ruff format --check .
pyright
pytest
