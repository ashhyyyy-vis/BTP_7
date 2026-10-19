#!/usr/bin/env bash
# run_local_baseline.sh
# End-to-end execution of Steps 4-8 for local faster-whisper large-v3 evaluation
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

if [[ -f ~/project/baseline_env.sh ]]; then
  source ~/project/baseline_env.sh
elif [[ -f "$PROJECT_DIR/baseline_env.sh" ]]; then
  source "$PROJECT_DIR/baseline_env.sh"
fi

COMPUTE="${COMPUTE:-float16}"
MODEL="large-v3"
MAX_UTTS="167"

echo "=== [1/5] Smoke Test (5 utterances) ==="
python -m scripts.eval_asr \
  --data data --noise-type musan --backend local --model "$MODEL" \
  --device cuda --compute-type "$COMPUTE" --splits test --max-utts 5

echo "=== [2/5] Full Untreated Evaluation (167 utterances) ==="
python -m scripts.eval_asr \
  --data data --noise-type musan --backend local --model "$MODEL" \
  --device cuda --compute-type "$COMPUTE" --splits test --max-utts "$MAX_UTTS"

python -m scripts.eval_asr \
  --data data --noise-type demand --backend local --model "$MODEL" \
  --device cuda --compute-type "$COMPUTE" --splits test --max-utts "$MAX_UTTS"

echo "=== [3/5] Full Enhanced Evaluation (167 utterances) ==="
python -m scripts.eval_enh \
  --data data --noise-type musan --backend local --model "$MODEL" \
  --device cuda --compute-type "$COMPUTE" --splits test --max-utts "$MAX_UTTS" \
  --front-ends wiener_dd dfn --snrs 0 5 10

python -m scripts.eval_enh \
  --data data --noise-type demand --backend local --model "$MODEL" \
  --device cuda --compute-type "$COMPUTE" --splits test --max-utts "$MAX_UTTS" \
  --front-ends wiener_dd dfn --snrs 0 5 10

python -m scripts.eval_enh \
  --data data --noise-type musan --backend local --model "$MODEL" \
  --device cuda --compute-type "$COMPUTE" --splits test --max-utts "$MAX_UTTS" \
  --front-ends dfn_atten12 dfn_mix50 --snrs 0 5

echo "=== [4/5] Rescore Summaries ==="
python -m scripts.rescore \
  --data data --noise-type musan --model "$MODEL" --max-utts "$MAX_UTTS" --snrs 0 5 10

python -m scripts.rescore \
  --data data --noise-type demand --model "$MODEL" --max-utts "$MAX_UTTS" --snrs 0 5 10

echo "=== [5/5] Groq Checkpoint Sanity Check (100 utterances) ==="
python -m scripts.rescore \
  --data data --noise-type musan --model "$MODEL" --max-utts 100 --snrs 0 5 10

echo "=== Local Baseline Execution Complete ==="
