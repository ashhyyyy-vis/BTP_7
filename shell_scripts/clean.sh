#!/usr/bin/env bash
# Clean restart: wipes all generated data and results.
# Keeps data/noise/ (large MUSAN/DEMAND downloads) unless --all is passed.
set -euo pipefail

ALL=0
[[ "${1:-}" == "--all" ]] && ALL=1

echo "This will DELETE: data/clean data/clean8k data/degraded data/manifest.csv artifacts/"
[[ $ALL -eq 1 ]] && echo "...and data/noise/ (--all)"
read -r -p "Type 'yes' to continue: " ans
[[ "$ans" == "yes" ]] || { echo "Aborted."; exit 1; }

rm -rf data/clean data/clean8k data/degraded data/manifest.csv artifacts
[[ $ALL -eq 1 ]] && rm -rf data/noise

echo "Clean. Next: ./run_musan.sh and ./run_demand.sh"
