#!/usr/bin/env bash
# Full pipeline (Steps 1-6) for DEMAND noise.
#
# Usage:
#   ./run_demand.sh                                   # local faster-whisper "small"
#   BACKEND=groq MODEL=whisper-large-v3 ./run_demand.sh
#
# Env vars: HF_TOKEN, GROQ_API_KEY (or put them in ../.env), BACKEND, MODEL, MAX_HOURS
set -euo pipefail

NOISE=demand
NOISE_DIR=data/noise/demand
BACKEND="${BACKEND:-local}"
MODEL="${MODEL:-small}"
MAX_HOURS="${MAX_HOURS:-3}"
FRONT_ENDS="wiener_dd dfn"

# --- Pre-flight checks ---
ENCODERS="$(ffmpeg -hide_banner -encoders 2>/dev/null || true)"
grep -qi libopencore_amrnb <<<"$ENCODERS" || { echo "ERROR: ffmpeg lacks AMR-NB encoder (libopencore_amrnb)"; exit 1; }
if [[ ! -d "$NOISE_DIR" ]] || [[ -z "$(ls -A "$NOISE_DIR" 2>/dev/null)" ]]; then
  echo "ERROR: $NOISE_DIR is missing or empty."
  echo "Download DEMAND (e.g. the 16k zips) and extract one folder per environment into $NOISE_DIR."
  exit 1
fi

ASR_ARGS=(--data data --noise-type "$NOISE" --model "$MODEL")
[[ "$BACKEND" == "groq" ]] && ASR_ARGS+=(--backend groq)

# --- Step 1: extraction (skipped if already done) ---
if [[ ! -f data/manifest.csv ]]; then
  echo "== Step 1: extraction =="
  python scripts/extraction.py --out data --max-hours "$MAX_HOURS" ${HF_TOKEN:+--hf-token "$HF_TOKEN"}
else
  echo "== Step 1: skipped (data/manifest.csv exists) =="
fi

echo "== Step 2: degrade ($NOISE) =="
# --noise-group-by dir keeps each DEMAND environment as one group
python scripts/degrade.py --data data --noise-type "$NOISE" --noise-dir "$NOISE_DIR" --noise-group-by dir

echo "== Step 3: Gate 1 ASR ($BACKEND / $MODEL) =="
python scripts/eval_asr.py "${ASR_ARGS[@]}"

echo "== Step 4: denoise ($FRONT_ENDS) =="
python scripts/denoise.py --data data --noise-type "$NOISE" --methods $FRONT_ENDS

echo "== Step 5: enhancement metrics =="
python scripts/enh_metrics.py --data data --noise-type "$NOISE" --front-ends $FRONT_ENDS

echo "== Step 6: Gate 2 post-enhancement ASR =="
python scripts/eval_enh.py "${ASR_ARGS[@]}" --front-ends $FRONT_ENDS

echo "Done. Results in artifacts/Metrics/$NOISE/"
