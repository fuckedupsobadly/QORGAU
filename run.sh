#!/usr/bin/env bash
# QORGAU — one-command setup and launch.
set -euo pipefail
cd "$(dirname "$0")"

python3 -m pip install -q -r requirements.txt

echo "→ building corpus"
python3 -m training.prepare_dataset

echo "→ evaluating on the held-out test split"
python3 -m training.evaluate --split test

echo "→ starting dashboard on http://localhost:8501"
exec python3 -m streamlit run app/frontend/streamlit_app.py
