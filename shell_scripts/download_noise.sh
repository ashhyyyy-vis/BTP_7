#!/usr/bin/env bash
# Downloads MUSAN (noise only) and DEMAND (16 kHz) into data/noise/.
# Run from the project root. Safe to re-run: finished parts are skipped.
#
# Usage:
#   shell_scripts/download_noise.sh            # both
#   shell_scripts/download_noise.sh musan      # MUSAN only
#   shell_scripts/download_noise.sh demand     # DEMAND only
set -euo pipefail

WHAT="${1:-all}"
NOISE_ROOT=data/noise
mkdir -p "$NOISE_ROOT"

for cmd in tar unzip python3; do
  command -v "$cmd" >/dev/null || { echo "ERROR: '$cmd' not found. Install it first."; exit 1; }
done

# dl URL OUTFILE : resumable download, aria2c (16 connections) if present, else wget
dl() {
  local url="$1" out="$2"
  mkdir -p "$(dirname "$out")"
  if command -v aria2c >/dev/null; then
    aria2c -x16 -s16 -k1M -c --file-allocation=none \
      -d "$(dirname "$out")" -o "$(basename "$out")" "$url"
  elif command -v wget >/dev/null; then
    wget -c -O "$out" "$url"
  else
    echo "ERROR: need aria2c or wget"; exit 1
  fi
}
command -v aria2c >/dev/null || echo "Tip: install aria2 (e.g. sudo pacman -S aria2 / apt install aria2) for much faster downloads."

# ---------------- MUSAN (~11 GB download, noise subset kept) ----------------
get_musan() {
  local dest="$NOISE_ROOT/musan"
  if [[ -d "$dest" && -n "$(ls -A "$dest" 2>/dev/null)" ]]; then
    echo "== MUSAN: $dest already populated, skipping =="; return
  fi
  mkdir -p "$dest"
  echo "== MUSAN: downloading (~11 GB, resumable) =="
  dl https://www.openslr.org/resources/17/musan.tar.gz "$NOISE_ROOT/musan.tar.gz"
  echo "== MUSAN: extracting noise/ only =="
  tar -xzf "$NOISE_ROOT/musan.tar.gz" -C "$dest" --strip-components=2 musan/noise
  rm -f "$NOISE_ROOT/musan.tar.gz"
  echo "MUSAN wav files: $(find "$dest" -name '*.wav' | wc -l)"
}

# ---------------- DEMAND (Zenodo 1227121, 16 kHz zips) ----------------
get_demand() {
  local dest="$NOISE_ROOT/demand"
  if [[ -d "$dest" && -n "$(ls -A "$dest" 2>/dev/null)" ]]; then
    echo "== DEMAND: $dest already populated, skipping =="; return
  fi
  mkdir -p "$dest"
  echo "== DEMAND: fetching file list from Zenodo =="
  # Read the record's file list so URLs are not hardcoded
  python3 - "$NOISE_ROOT/demand_urls.txt" <<'PY'
import json, sys, urllib.request
out = sys.argv[1]
with urllib.request.urlopen("https://zenodo.org/api/records/1227121") as r:
    rec = json.load(r)
files = rec.get("files", [])
urls = []
for f in files:
    key = f.get("key") or f.get("filename") or ""
    if key.endswith("_16k.zip"):
        link = (f.get("links") or {}).get("self") or f"https://zenodo.org/records/1227121/files/{key}?download=1"
        urls.append(f"{link}\t{key}")
if not urls:
    sys.exit("No *_16k.zip files found in Zenodo record; check https://zenodo.org/records/1227121")
open(out, "w").write("\n".join(urls) + "\n")
print(f"{len(urls)} environments found")
PY
  echo "== DEMAND: downloading =="
  while IFS=$'\t' read -r url key; do
    [[ -n "$url" ]] || continue
    dl "$url" "$NOISE_ROOT/demand_zips/$key"
  done < "$NOISE_ROOT/demand_urls.txt"
  echo "== DEMAND: extracting (one folder per environment) =="
  for z in "$NOISE_ROOT"/demand_zips/*_16k.zip; do
    [[ -f "$z" ]] || continue
    unzip -oq "$z" -d "$dest"
  done
  rm -rf "$NOISE_ROOT/demand_zips" "$NOISE_ROOT/demand_urls.txt"
  echo "DEMAND environments: $(find "$dest" -mindepth 1 -maxdepth 1 -type d | wc -l)"
  echo "DEMAND wav files:    $(find "$dest" -name '*.wav' | wc -l)"
}

case "$WHAT" in
  musan)  get_musan ;;
  demand) get_demand ;;
  all)    get_musan; get_demand ;;
  *) echo "Usage: $0 [musan|demand|all]"; exit 1 ;;
esac

echo "Done."
