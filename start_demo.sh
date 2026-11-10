#!/bin/sh
# Starts the web dashboard for the demo and opens it in the browser.
# Usage (from the repo root):  sh start_demo.sh
cd "$(dirname "$0")/web" || exit 1

missing=""
for f in checkpoints/best_model_attn_sen2naip.pt checkpoints/best_model_attn_gan_sen2naip.pt; do
  [ -f "$f" ] || missing="$missing $f"
done
if [ -n "$missing" ]; then
  echo "Missing in web/:$missing"
  echo "Copy them from the root checkpoints/ folder (see web/README.md)."
  exit 1
fi
if ! ls data/pairs/*.npz >/dev/null 2>&1; then
  echo "No demo pairs in web/data/pairs/. From the repo root run:"
  echo "    .venv/bin/python -m src.data.sen2naip --export_web 8"
  exit 1
fi

PY=../.venv/bin/python
[ -x "$PY" ] || PY=python3
echo "Starting dashboard at http://localhost:8000  (Ctrl+C to stop)"
( sleep 4; open http://localhost:8000 2>/dev/null || xdg-open http://localhost:8000 2>/dev/null ) &
exec "$PY" -m uvicorn src.server:app --host 127.0.0.1 --port 8000
