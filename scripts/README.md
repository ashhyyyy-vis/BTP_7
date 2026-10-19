# Hindi ASR Telephone Degradation Study

This repository contains scripts for studying the impact of telephone channel degradation on Hindi Automatic Speech Recognition (ASR) performance. The pipeline extracts Hindi speech from the IndicVoices dataset, simulates AMR-NB telephone channel degradation with noise injection, and evaluates Whisper ASR model performance across various degradation conditions.

## Prerequisites

### System Requirements
- Python 3.8+
- ffmpeg with AMR-NB encoder (libopencore_amrnb)
- Hugging Face account with access token

### Python Dependencies
```bash
pip install datasets numpy scipy soundfile faster-whisper python-dotenv
```

### Setup

1. **Install ffmpeg with AMR-NB encoder**
   ```bash
   # Check if ffmpeg has AMR-NB encoder
   ffmpeg -encoders | grep -i amr
   
   # If not present, install an ffmpeg build that includes libopencore_amrnb
   ```

2. **Authenticate with Hugging Face**
   ```bash
   huggingface-cli login
   # or set HF_TOKEN environment variable
   ```

3. **Create environment file** (optional)
   ```bash
   # Create .env file in project root
   echo "HF_TOKEN=your_token_here" > ../.env
   ```

---

# Hindi ASR Telephone Degradation & Speech Enhancement Study

This repository contains scripts for studying the impact of telephone channel degradation on Hindi Automatic Speech Recognition (ASR) performance and evaluating front-end speech enhancement methods.

## Prerequisites

### System Requirements
- Python 3.10+
- ffmpeg with AMR-NB encoder (`libopencore_amrnb`)
- Hugging Face account with access token (for IndicVoices dataset)
- Optional: Groq Cloud API key for cloud Whisper inference

### Python Dependencies
```bash
pip install datasets numpy scipy soundfile faster-whisper python-dotenv
# Optional:
pip install groq             # for Groq cloud Whisper backend
pip install pystoi pesq      # for STOI and PESQ metrics in enh_metrics.py
pip install deepfilternet    # for DeepFilterNet3 in denoise.py
```

---

## Directory Hierarchy

### Data Folder (`data/`)
```
data/
├── clean/                            # 16 kHz clean speech from IndicVoices
├── clean8k/                          # 8 kHz clean references (only regenerated if missing)
├── degraded/                         # Degraded audio per noise dataset
│   ├── synthetic/
│   │   ├── manifest_degraded.csv     # Synthetic degradation manifest
│   │   ├── babble/
│   │   ├── brown/
│   │   ├── pink/
│   │   ├── white/
│   │   └── none/
│   ├── musan/
│   │   ├── manifest_degraded.csv     # MUSAN degradation manifest
│   │   └── *.wav
│   └── demand/
│       ├── manifest_degraded.csv     # DEMAND degradation manifest
│       └── *.wav
├── noise/
│   ├── musan/                        # Extracted MUSAN noise files
│   └── demand/                       # Extracted DEMAND noise files
└── manifest.csv                      # Shared clean manifest
```

### Artifacts Folder (`artifacts/`)
```
artifacts/
├── shards/                                   # Shard jsonl files from eval_asr
│   └── asr_<model>_<noise_type>.shard*of*.jsonl
└── Metrics/
    ├── synthetic/                            # Or musan / demand
    │   ├── ASR/
    │   │   ├── asr_<model>.jsonl             # Transcription cache
    │   │   ├── asr_summary_<model>.csv       # Gate 1 ASR WER summary
    │   │   ├── asr_<model>_enh.jsonl         # Enhanced transcription cache
    │   │   └── enh_summary_<model>.csv       # Gate 2 post-enhancement ASR summary
    │   └── Enhancement/
    │       ├── <method>/                     # e.g., wiener_dd, dfn
    │       │   ├── <deg_id>.wav              # Enhanced audio files
    │       │   └── log.csv                   # Alignment delay logs
    │       ├── enh_metrics_per_file.csv      # SI-SDR, STOI, PESQ per utterance
    │       └── enh_metrics_summary.csv       # Mean metrics & improvements over none
    ├── musan/
    └── demand/
```

---

## 6-Step Execution Pipeline

### Step 1: Data Extraction (`extraction.py`)
Extract clean Hindi speech from IndicVoices.
```bash
python scripts/extraction.py --out data --max-hours 3 --hf-token <YOUR_HF_TOKEN>
```
- `--hf-token`: Hugging Face access token (overrides `HF_TOKEN` in env / `.env`).
- Writes to `data/clean/*.wav` and `data/manifest.csv`.

### Step 2: Channel & Noise Degradation (`degrade.py`)
Simulate 8 kHz AMR-NB telephone channel with noise injection.
```bash
# Synthetic noise (white, pink, brown, babble):
python scripts/degrade.py --data data --noise-type synthetic

# MUSAN noise:
python scripts/degrade.py --data data --noise-type musan --noise-dir data/noise/musan

# DEMAND noise:
python scripts/degrade.py --data data --noise-type demand --noise-dir data/noise/demand --noise-group-by dir
```
- `--noise-type`: `synthetic`, `musan`, or `demand`.
- `clean8k/` is checked first and only generated once.
- Each noise type writes its own manifest to `data/degraded/<noise_type>/manifest_degraded.csv`.

### Step 3: ASR Evaluation (`eval_asr.py`)
Evaluate Whisper ASR across clean, bandwidth-limited, codec-only, and noisy conditions.
```bash
# Local faster-whisper:
python scripts/eval_asr.py --data data --noise-type musan --model small

# Groq Cloud backend:
python scripts/eval_asr.py --data data --noise-type musan --backend groq --model whisper-large-v3 --groq-key <YOUR_GROQ_KEY>

# Sharded evaluation across multiple machines:
python scripts/eval_asr.py --data data --noise-type musan --num-shards 3 --shard-id 0
python scripts/eval_asr.py --data data --noise-type musan --num-shards 3 --shard-id 1
python scripts/eval_asr.py --data data --noise-type musan --num-shards 3 --shard-id 2
# Merge shards:
python scripts/eval_asr.py --data data --noise-type musan --merge
```
- `--groq-key`: Groq API key (overrides `GROQ_API_KEY` in env / `.env`).
- Shard files land in `artifacts/shards/`.
- Summary results land in `artifacts/Metrics/<noise_type>/ASR/`.

### Step 4: Speech Enhancement / Denoising (`denoise.py`)
Enhance degraded 8 kHz speech with classical or neural front-ends.
```bash
python scripts/denoise.py --data data --noise-type musan --methods wiener_dd dfn
```
- Enhanced audio lands in `artifacts/Metrics/<noise_type>/Enhancement/<method>/`.

### Step 5: Enhancement Metrics (`enh_metrics.py`)
Compute objective signal-quality metrics (SI-SDR, STOI, PESQ-NB) against `data/clean8k/`.
```bash
python scripts/enh_metrics.py --data data --noise-type musan --front-ends wiener_dd dfn
```
- Results saved to `artifacts/Metrics/<noise_type>/Enhancement/enh_metrics_summary.csv`.

### Step 6: Post-Enhancement ASR Evaluation (`eval_enh.py`)
Measure whether front-end enhancement actually lowers ASR Word Error Rate (WER).
```bash
python scripts/eval_enh.py --data data --noise-type musan --model small --front-ends wiener_dd dfn
```
- Summary saved to `artifacts/Metrics/<noise_type>/ASR/enh_summary_<model>.csv`.

---

## API Keys Configuration

You can either pass keys as CLI parameters or set them in `.env`:
- Hugging Face token: pass `--hf-token <token>` to `extraction.py` or set `HF_TOKEN` in `.env`.
- Groq API key: pass `--groq-key <key>` to `eval_asr.py` / `eval_enh.py` or set `GROQ_API_KEY` in `.env`.

