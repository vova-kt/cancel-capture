#!/usr/bin/env sh
set -eu
uv run streamlit run src/cancel_capture/streamlit_app.py "$@"
