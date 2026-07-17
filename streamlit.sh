#!/usr/bin/env sh
set -eu
uv run streamlit run src/cancel_capture/streamlitapp/app.py "$@"
