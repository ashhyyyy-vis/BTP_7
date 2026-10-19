# Local Baseline Runbook (RTX GPU Machine)

> **FOR AI AGENTS (e.g. Antigravity, Claude Code, Cursor, Codex):**
> You are running as an autonomous AI agent on the RTX GPU machine (Linux / WSL2 Ubuntu).
> Execute each step in this runbook sequentially. Run the verification commands after each step and assert that the output matches the **Expected Output / Assertions**.
> **Non-negotiable constraints:**
> - Do **NOT** modify the text normalizer (`scripts/hindi_norm.py`); it is frozen at v2.
> - Do **NOT** attempt to re-extract or re-degrade audio from scratch. All audio is pre-built in `baseline_bundle.tar`.
> - Do **NOT** change the model (`large-v3`) or the test set size (`167` test utterances).
> - If any assertion fails or an error occurs, halt and report the exact error.

---

## 0. Quick Reference & Configuration

| Variable | Value / Location | Notes |
|---|---|---|
| **Repository URL (HTTPS)** | `https://github.com/ashhyyyy-vis/BTP_7.git` | Primary clone URL |
| **Repository URL (SSH)** | `git@github.com:ashhyyyy-vis/BTP_7.git` | Alternative if SSH key is set |
| **Git Branch** | `WER-test` | Must checkout this branch |
| **Working Directory** | `~/project` | Inside Ubuntu/WSL2 home (never run under `/mnt/c/` for I/O performance) |
| **ASR Model** | `large-v3` (`Systran/faster-whisper-large-v3`) | Evaluated locally on CUDA |
| **Target Test Set** | `167` utterances | All utterances in the test split |
| **GDrive Bundle URL** | `https://drive.google.com/file/d/1Xl-oI3R9D1yF3r1jC2eL8z8HqgO5qgO5/view?usp=sharing` | Replace with active GDrive link for `baseline_bundle.tar` |
| **GDrive MD5 URL** | `https://drive.google.com/file/d/1Ym-pI3R9D1yF3r1jC2eL8z8HqgO5qgO5/view?usp=sharing` | Replace with active GDrive link for `baseline_bundle.md5` |

---

## Step 0. [Windows Host / WSL2] Pre-flight & GPU Verification

1. **[Windows Host]** Ensure the latest NVIDIA Game Ready / Studio drivers are installed on Windows (from GeForce Experience or [nvidia.com](https://www.nvidia.com/Download/index.aspx)).
2. **[Windows Host]** Disable system sleep: **Settings > System > Power > Screen and sleep > "When plugged in, put my device to sleep" -> Never**.
3. **[WSL2 / Ubuntu]** Open the WSL2 terminal and verify NVIDIA CUDA pass-through:
   ```bash
   nvidia-smi
   ```
   **Expected Output / Assertion:**
   - A GPU status table displays your NVIDIA GPU name (e.g. RTX 3060 / 4060 / 4070 / 3080 / etc.) and total VRAM (e.g., `8192MiB`, `12288MiB`, `16384MiB`).
   - If `nvidia-smi` outputs `command not found` or `no devices found`, halt: WSL2 NVIDIA driver pass-through is not working.

---

## Step 1. Clone the Code Repository

Install system prerequisites, clone the repository branch `WER-test`, and set up the project folder:

```bash
# 1. Install prerequisites
sudo apt update && sudo apt install -y git curl zip gh python3-pip

# 2. Clone repository to ~/project inside the Linux filesystem
rm -rf ~/project
git clone -b WER-test https://github.com/ashhyyyy-vis/BTP_7.git ~/project
cd ~/project

# 3. Verify directory structure
ls -d scripts docs Reports shell_scripts
```

**Expected Output / Assertion:**
- `scripts docs Reports shell_scripts` are all present.
- `scripts/` contains `eval_asr.py`, `eval_enh.py`, `rescore.py`, `hindi_norm.py`.
- `docs/` contains `local_baseline_runbook.md`.

---

## Step 2. Download, Verify & Unpack the Audio Bundle

The audio bundle `baseline_bundle.tar` (~1.7 GB) and its checksum `baseline_bundle.md5` contain:
- `data/manifest.csv`, `data/clean`, `data/clean8k`
- `data/degraded/musan` (3,716 files), `data/degraded/demand` (3,716 files)
- Pre-computed enhanced test audio for DEMAND and MUSAN (Wiener, DFN, DFN variants)

### Method A: Download via `gdown` (Direct from Google Drive)
```bash
cd ~/project
pip install -q gdown

# Paste your actual GDrive file IDs or share URLs below:
GDRIVE_BUNDLE_LINK="<PASTE_GDRIVE_LINK_FOR_baseline_bundle.tar>"
GDRIVE_MD5_LINK="<PASTE_GDRIVE_LINK_FOR_baseline_bundle.md5>"

gdown "$GDRIVE_BUNDLE_LINK" -O baseline_bundle.tar
gdown "$GDRIVE_MD5_LINK" -O baseline_bundle.md5
```

### Method B: Copy from Windows Downloads (If downloaded manually via browser)
```bash
cd ~/project
# Replace <YOUR_WINDOWS_USERNAME> with your Windows username:
WINDOWS_USER=$(ls /mnt/c/Users/ | grep -vE 'Public|Default|desktop.ini|All Users' | head -n1)
echo "Detected Windows user: $WINDOWS_USER"

cp "/mnt/c/Users/$WINDOWS_USER/Downloads/baseline_bundle.tar" ~/project/
cp "/mnt/c/Users/$WINDOWS_USER/Downloads/baseline_bundle.md5" ~/project/
```

### Verification & Extraction
Run the integrity check and unpack:
```bash
cd ~/project

# 1. Assert MD5 Checksum
md5sum -c baseline_bundle.md5
# Expected: baseline_bundle.tar: OK (exit code 0)

# 2. Extract bundle
tar -xf baseline_bundle.tar

# 3. Assert exact file counts
echo "--- File Count Verification ---"
echo "MUSAN degraded (expected 3716):  $(find data/degraded/musan -name '*.wav' | wc -l)"
echo "DEMAND degraded (expected 3716): $(find data/degraded/demand -name '*.wav' | wc -l)"
echo "DEMAND dfn (expected 501):       $(find artifacts/Metrics/demand/Enhancement/dfn -name '*.wav' | wc -l)"
echo "DEMAND wiener (expected 501):    $(find artifacts/Metrics/demand/Enhancement/wiener_dd -name '*.wav' | wc -l)"
echo "MUSAN dfn (expected 501):        $(find artifacts/Metrics/musan/Enhancement/dfn -name '*.wav' | wc -l)"
echo "MUSAN dfn_atten12 (expected 334): $(find artifacts/Metrics/musan/Enhancement/dfn_atten12 -name '*.wav' | wc -l)"
```

**Expected Output / Assertion:**
- `md5sum` prints `baseline_bundle.tar: OK` and returns exit code 0.
- Counts match exactly: `3716`, `3716`, `501`, `501`, `501`, `334`.

---

## Step 3. Set Up Python Environment & CUDA Runtime (`uv`)

We use `uv` for fast, reproducible dependency management and install `faster-whisper` with NVIDIA CUDA 12 libraries:

```bash
cd ~/project

# 1. Install Astral uv if not already installed
if ! command -v uv &> /dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# 2. Create isolated virtual environment
uv venv --python 3.12 .venv-asr
source .venv-asr/bin/activate

# 3. Install dependencies
uv pip install faster-whisper scipy soundfile numpy python-dotenv
uv pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"

# 4. Generate persistent environment loader with automatic VRAM detection
cat > ~/project/baseline_env.sh <<'EOF'
cd ~/project
source ~/project/.venv-asr/bin/activate

# Auto-detect VRAM to set compute type safely
TOTAL_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1 | tr -d ' ')
if [ "$TOTAL_VRAM" -ge 8000 ]; then
  export COMPUTE=float16
elif [ "$TOTAL_VRAM" -ge 6000 ]; then
  export COMPUTE=int8_float16
else
  echo "WARNING: VRAM is under 6000 MiB ($TOTAL_VRAM MiB). Using int8_float16, watch for OOM."
  export COMPUTE=int8_float16
fi

# Point LD_LIBRARY_PATH to WSL CUDA drivers and nvidia python packages
CUBLAS_PATH=$(python -c "import os, nvidia.cublas.lib; print(os.path.dirname(nvidia.cublas.lib.__file__))" 2>/dev/null || true)
CUDNN_PATH=$(python -c "import os, nvidia.cudnn.lib; print(os.path.dirname(nvidia.cudnn.lib.__file__))" 2>/dev/null || true)
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$CUBLAS_PATH:$CUDNN_PATH:${LD_LIBRARY_PATH:-}"

echo "[baseline_env.sh] Activated .venv-asr (COMPUTE=$COMPUTE, VRAM=${TOTAL_VRAM}MiB)"
EOF

# 5. Load environment and verify CUDA access in ctranslate2 & normalizer
source ~/project/baseline_env.sh
python -c "import ctranslate2; gpus = ctranslate2.get_cuda_device_count(); assert gpus > 0, 'No CUDA device visible'; print(f'CUDA check: {gpus} GPU(s) visible to ctranslate2')"
python -m scripts.hindi_norm
```

**Expected Output / Assertion:**
- `CUDA check: 1 GPU(s) visible to ctranslate2`
- `hindi_norm self-test ok, version v2`

---

## Step 4. Smoke Test (5 Utterances)

Before running the full suite, verify model download, weights caching, and inference on 5 utterances:

```bash
cd ~/project
source ~/project/baseline_env.sh

python -m scripts.eval_asr \
  --data data \
  --noise-type musan \
  --backend local \
  --model large-v3 \
  --device cuda \
  --compute-type $COMPUTE \
  --splits test \
  --max-utts 5
```

**Checks & Assertions:**
1. Whisper `large-v3` weights download automatically on first run (~3 GB cached in `~/.cache/huggingface/`).
2. Execution prints per-condition WER table.
3. Check the cached transcripts:
   ```bash
   head -n 3 artifacts/Metrics/musan/ASR/asr_large-v3.jsonl
   ```
   Confirm that the transcripts contain non-empty Hindi (Devanagari script) text, not English or blank strings.

---

## Step 5. Full Untreated Audio Evaluation (167 Test Utterances)

Evaluate Whisper `large-v3` on all 167 test utterances across all conditions (clean, clean8k, codec-only, and SNRs 20, 15, 10, 5, 0 dB):

```bash
cd ~/project
source ~/project/baseline_env.sh

# 1. MUSAN Untreated Evaluation (167 utts)
python -m scripts.eval_asr \
  --data data \
  --noise-type musan \
  --backend local \
  --model large-v3 \
  --device cuda \
  --compute-type $COMPUTE \
  --splits test \
  --max-utts 167

# 2. DEMAND Untreated Evaluation (167 utts)
python -m scripts.eval_asr \
  --data data \
  --noise-type demand \
  --backend local \
  --model large-v3 \
  --device cuda \
  --compute-type $COMPUTE \
  --splits test \
  --max-utts 167

# 3. Assert completion (167 utterances x 8 conditions = 1336 lines each)
wc -l artifacts/Metrics/musan/ASR/asr_large-v3.jsonl
wc -l artifacts/Metrics/demand/ASR/asr_large-v3.jsonl
```

**Expected Output / Assertion:**
- Line count for `artifacts/Metrics/musan/ASR/asr_large-v3.jsonl` is **1336**.
- Line count for `artifacts/Metrics/demand/ASR/asr_large-v3.jsonl` is **1336**.
- If interrupted, re-running resumes from the cache automatically.

---

## Step 6. Enhanced Audio Evaluation & Rescoring

Evaluate the front-ends on the enhanced audio and compute the paired bootstrap statistics:

```bash
cd ~/project
source ~/project/baseline_env.sh

# 1. MUSAN: wiener_dd and dfn (SNRs 0, 5, 10 dB)
python -m scripts.eval_enh \
  --data data \
  --noise-type musan \
  --backend local \
  --model large-v3 \
  --device cuda \
  --compute-type $COMPUTE \
  --splits test \
  --max-utts 167 \
  --front-ends wiener_dd dfn \
  --snrs 0 5 10

# 2. DEMAND: wiener_dd and dfn (SNRs 0, 5, 10 dB)
python -m scripts.eval_enh \
  --data data \
  --noise-type demand \
  --backend local \
  --model large-v3 \
  --device cuda \
  --compute-type $COMPUTE \
  --splits test \
  --max-utts 167 \
  --front-ends wiener_dd dfn \
  --snrs 0 5 10

# 3. MUSAN Variants: dfn_atten12 and dfn_mix50 (SNRs 0, 5 dB)
python -m scripts.eval_enh \
  --data data \
  --noise-type musan \
  --backend local \
  --model large-v3 \
  --device cuda \
  --compute-type $COMPUTE \
  --splits test \
  --max-utts 167 \
  --front-ends dfn_atten12 dfn_mix50 \
  --snrs 0 5

# 4. Generate Final Rescore Summaries
python -m scripts.rescore \
  --data data \
  --noise-type musan \
  --model large-v3 \
  --max-utts 167 \
  --snrs 0 5 10

python -m scripts.rescore \
  --data data \
  --noise-type demand \
  --model large-v3 \
  --max-utts 167 \
  --snrs 0 5 10
```

**Expected Output / Assertion:**
- Both `rescore` runs end with: `167 utterances scored`.
- Check that the output CSV files exist:
  - `artifacts/Metrics/musan/ASR/rescore_summary_large-v3.csv`
  - `artifacts/Metrics/demand/ASR/rescore_summary_large-v3.csv`

---

## Step 7. Sanity Check against Preliminary Groq Baseline

Run the 100-utterance rescore on MUSAN to verify alignment against the earlier Groq checkpoint:

```bash
cd ~/project
source ~/project/baseline_env.sh

python -m scripts.rescore \
  --data data \
  --noise-type musan \
  --model large-v3 \
  --max-utts 100 \
  --snrs 0 5 10
```

### Reference Benchmark Table (100 Utterances, Frozen Normalizer v2)
| Condition | Groq `whisper-large-v3` WER | Expected Local `large-v3` Range |
|---|---|---|
| **Clean** | 25.7% | 24.0% – 28.5% |
| **Clean 8k** | 26.7% | 25.0% – 29.5% |
| **Codec Only** | 30.3% | 28.5% – 33.0% |
| **SNR 10 dB** | 37.4% | 35.0% – 41.0% |
| **SNR 5 dB** | 42.1% | 39.5% – 45.5% |
| **SNR 0 dB** | 51.5% | 48.5% – 55.0% |

**Assertion:**
- Local WERs should be within **±3.0 percentage points** (maximum ±5.0%) of the Groq numbers.
- If any condition differs by > 5.0% absolute WER, inspect audio sampling rate and verify `hindi_norm.py`.

---

## Step 8. Export Results & Record Environment

Log exact library versions and export the results to the Windows host:

```bash
cd ~/project
source ~/project/baseline_env.sh

# 1. Snapshot python package versions
pip list | grep -E "faster-whisper|ctranslate2|torch|numpy|scipy|soundfile" > ~/project/versions_local_baseline.txt

# 2. Copy outputs to Windows Downloads folder
WINDOWS_USER=$(ls /mnt/c/Users/ | grep -vE 'Public|Default|desktop.ini|All Users' | head -n1)
DEST_DIR="/mnt/c/Users/$WINDOWS_USER/Downloads/BTP_Baseline_Results"
mkdir -p "$DEST_DIR"

cp artifacts/Metrics/musan/ASR/rescore_summary_large-v3.csv "$DEST_DIR/musan_rescore_summary_large-v3.csv"
cp artifacts/Metrics/demand/ASR/rescore_summary_large-v3.csv "$DEST_DIR/demand_rescore_summary_large-v3.csv"
cp ~/project/versions_local_baseline.txt "$DEST_DIR/"

echo "Results successfully exported to: $DEST_DIR"
```

---

## Autonomous Single-Script Runner (Optional)

If preferred, the entire Steps 4 through 8 workflow can be executed with a single command via [`shell_scripts/run_local_baseline.sh`](file:///home/viseash/Ashley/BTP/shell_scripts/run_local_baseline.sh):

```bash
cd ~/project
source ~/project/baseline_env.sh
bash shell_scripts/run_local_baseline.sh
```

---

## Completion Checklist for the Executing Agent

- [ ] **Step 0:** `nvidia-smi` confirms NVIDIA GPU with CUDA support.
- [ ] **Step 1:** Repository cloned on branch `WER-test` inside `~/project`.
- [ ] **Step 2:** `baseline_bundle.tar` MD5 checksum matches (`baseline_bundle.tar: OK`) and audio file counts verified (`3716` MUSAN, `3716` DEMAND, `501`/`501`/`501`/`334` enhancement).
- [ ] **Step 3:** Virtual environment active with `COMPUTE` set and `ctranslate2` reports visible GPU.
- [ ] **Step 4:** Smoke test passes with Devanagari Hindi text in `asr_large-v3.jsonl`.
- [ ] **Step 5:** Full untreated run produces `1336` entries in both MUSAN and DEMAND `.jsonl` files.
- [ ] **Step 6:** Enhanced evaluation and rescore complete for 167 utterances.
- [ ] **Step 7:** Sanity check on 100 utterances matches Groq baseline within tolerance.
- [ ] **Step 8:** `rescore_summary_large-v3.csv` and environment metadata exported.
