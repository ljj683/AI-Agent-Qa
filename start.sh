#!/usr/bin/env bash
set -e
if [ ! -d ".venv" ]; then
  python -m venv .venv
fi
source .venv/bin/activate
pip install -r requirements.txt
streamlit run src/streamlit_app.py
