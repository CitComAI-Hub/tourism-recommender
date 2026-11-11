#!/usr/bin/env bash
# Evaluation protocol only, on the TEF cities already built (held-out
# confirmation: run once with frozen code, never used for tuning).
# Usage: research/run_tef_protocol.sh [city ...]
set -u
cd "$(dirname "$0")/.."
PY=venv/bin/python
CITIES=("$@")
[ ${#CITIES[@]} -eq 0 ] && CITIES=(aarhus copenhagen eindhoven luxembourg tampere mechelen)
bbox() { $PY -c "import json,sys;b=json.load(open('data/raw/yfcc/tef_cities.json'))[sys.argv[1]];print(*b['lat'],*b['lon'])" "$1"; }
commit=$(git rev-parse --short HEAD)
for city in "${CITIES[@]}"; do
  log="data/raw/tef_protocol_${city}.log"
  echo "== $city $(date +%T) code $commit" > "$log"
  PYTHONPATH=research $PY research/protocol.py "tef/$city" $(bbox "$city") 2012-01-01 >> "$log" 2>&1 || echo "FAILED protocol" >> "$log"
  echo "== done $city $(date +%T)" >> "$log"
done
