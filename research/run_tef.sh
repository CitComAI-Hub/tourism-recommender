#!/usr/bin/env bash
# Build each TEF city and run the evaluation protocol on it, one after another
# (in parallel they would only fight over the Wikimedia rate limits).
# Usage: research/run_tef.sh [city ...]   (default: all, Paris last)
set -u
cd "$(dirname "$0")/.."
PY=venv/bin/python
CITIES=("$@")
[ ${#CITIES[@]} -eq 0 ] && CITIES=(brussels mechelen eindhoven luxembourg warsaw aarhus tampere copenhagen paris)
bbox() { $PY -c "import json,sys;b=json.load(open('data/raw/yfcc/tef_cities.json'))[sys.argv[1]];print(*b['lat'],*b['lon'])" "$1"; }
for city in "${CITIES[@]}"; do
  log="data/raw/tef_${city}.log"
  echo "== $city $(date +%T)" > "$log"
  ok=0
  for attempt in 1 2 3 4 5 6; do
    if $PY -m pmv_turismo.cli --config "configs/tef/$city.yaml" --step ingestion >> "$log" 2>&1; then ok=1; break; fi
    echo "== ingestion attempt $attempt failed $(date +%T)" >> "$log"; sleep 600
  done
  [ $ok = 1 ] || { echo "FAILED ingestion" >> "$log"; continue; }
  for step in embeddings geometry; do
    $PY -m pmv_turismo.cli --config "configs/tef/$city.yaml" --step $step >> "$log" 2>&1 || { echo "FAILED $step" >> "$log"; continue 2; }
  done
  PYTHONPATH=research $PY research/protocol.py "tef/$city" $(bbox "$city") 2012-01-01 >> "$log" 2>&1 || echo "FAILED protocol" >> "$log"
  echo "== done $city $(date +%T)" >> "$log"
done
