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

## Pipeline Overview

The scripts should be run in this order:

1. **extraction.py** - Extract clean Hindi speech from IndicVoices
2. **degrade.py** - Simulate telephone channel degradation
3. **eval_asr.py** - Evaluate ASR performance

---

## 1. Data Extraction (extraction.py)

Extracts Hindi speech from the IndicVoices Hugging Face dataset and creates speaker-level train/dev/test splits.

### Basic Usage

```bash
# Inspect dataset schema first (recommended)
python extraction.py --inspect

# Extract ~3 hours of 2-15 second clips
python extraction.py --out ../data --max-hours 3
```

### Advanced Options

```bash
# Extract with custom duration limits
python extraction.py --out ../data --max-hours 5 --min-dur 3.0 --max-dur 10.0

# Limit by number of utterances instead of hours
python extraction.py --out ../data --max-utts 1000

# Filter by specific scenario (e.g., conversational tasks)
python extraction.py --out ../data --scenario "conversational" --inspect  # first check available scenarios
python extraction.py --out ../data --scenario "conversational" --max-hours 3

# Custom column names (if dataset schema changes)
python extraction.py --out ../data --audio-col audio --text-col text --speaker-col speaker_id

# Change language or dataset split
python extraction.py --language hindi --hf-split train --out ../data
```

### Output

- `../data/clean/*.wav` - 16 kHz, mono, 16-bit WAV files
- `../data/manifest.csv` - Metadata with utterance IDs, splits, speaker IDs, transcripts, durations

### Speaker Splitting

Every speaker is hashed into train/dev/test (80/10/10) to ensure no speaker appears in more than one split.

---

## 2. Telephone Degradation (degrade.py)

Simulates 8 kHz AMR-NB telephone channel with noise injection at various SNR levels.

### Basic Usage

```bash
# Quick smoke test with 20 utterances (synthetic noise only)
python degrade.py --data ../data --limit 20

# Full run with synthetic noise (white, pink, brown, babble)
python degrade.py --data ../data
```

### Advanced Options

```bash
# Use real noise datasets (MUSAN, DEMAND, etc.)
python degrade.py --data ../data --noise-dir /path/to/musan/noise

# For DEMAND dataset, group noise by directory
python degrade.py --data ../data --noise-dir /path/to/demand --noise-group-by dir

# Custom SNR levels
python degrade.py --data ../data --snrs 0 5 10 15 20 25

# Custom AMR-NB bitrates for grid evaluation
python degrade.py --data ../data --grid-bitrates 12200 7950

# Adjust training data augmentation
python degrade.py --data ../data --train-copies 3 --bitrates 12200 7950 5900 4750

# Change codec-only probability for training
python degrade.py --data ../data --codec-only-prob 0.2

# Skip codec-only rows in grid
python degrade.py --data ../data --no-codec-only

# Adjust worker count for parallel processing
python degrade.py --data ../data --workers 4

# Custom random seed for reproducibility
python degrade.py --data ../data --seed 42
```

### Noise Types

- **Synthetic noise** (default when no --noise-dir): white, pink, brown, babble
- **File-based noise**: Use real noise recordings from MUSAN, DEMAND, or custom datasets
- **Babble noise**: Mixes multiple clean utterances to create speech-like noise

### Design Choices

- Dev/test utterances get the FULL GRID: every SNR (plus codec-only) with same noise segment
- Train utterances get random conditions (SNR drawn uniformly)
- Noise files are split train/dev/test by hashing (test noise never seen in training)
- SNR is measured in 8 kHz band against speech-active power
- Codec delay is measured by cross-correlation and removed for alignment
- All randomness is seeded for reproducibility

### Output

- `../data/clean8k/*.wav` - 8 kHz clean reference audio
- `../data/degraded/*.wav` - 8 kHz degraded audio (time-aligned to clean8k)
- `../data/manifest_degraded.csv` - Degradation recipes with full metadata (noise type, SNR, bitrate, delay, etc.)

---

## 3. ASR Evaluation (eval_asr.py)

Evaluates Whisper ASR model performance across degradation conditions and reports WER with bootstrap confidence intervals.

### Basic Usage

```bash
# Evaluate on test split, first 100 utterances, small model
python eval_asr.py --data ../data

# Use medium model (better accuracy, more RAM)
python eval_asr.py --data ../data --model medium
```

### Advanced Options

```bash
# Evaluate on specific splits
python eval_asr.py --data ../data --splits dev test

# Increase number of utterances
python eval_asr.py --data ../data --max-utts 500

# Use GPU for faster inference
python eval_asr.py --data ../data --device cuda --compute-type float16

# Change Whisper model size
python eval_asr.py --data ../data --model tiny  # fastest, lowest accuracy
python eval_asr.py --data ../data --model base
python eval_asr.py --data ../data --model small
python eval_asr.py --data ../data --model medium
python eval_asr.py --data ../data --model large-v2  # best accuracy, most RAM
```

### Sharded Evaluation (Multiple Machines)

For large-scale evaluation across multiple machines:

```bash
# Machine 0
python eval_asr.py --data ../data --num-shards 3 --shard-id 0

# Machine 1
python eval_asr.py --data ../data --num-shards 3 --shard-id 1

# Machine 2
python eval_asr.py --data ../data --num-shards 3 --shard-id 2
```

After all shards complete, copy all shard files to one `../data` folder and merge:

```bash
python eval_asr.py --data ../data --merge
```

### Evaluation Conditions

The script evaluates a ladder of conditions for each utterance:
- **clean** - Original 16 kHz audio
- **clean8k** - 8 kHz only (isolates bandwidth loss)
- **codec_only** - 8 kHz + AMR-NB, no noise (isolates codec)
- **snr_20, snr_15, snr_10, snr_5, snr_0** - Noise at specified SNR + 8 kHz + AMR-NB

### Output

- `../data/asr_<model>.jsonl` - Raw transcription hypotheses (cached for resumption)
- `../data/asr_summary_<model>.csv` - WER per condition with 95% bootstrap CI

### Text Normalization

Both reference transcripts and hypotheses are normalized using:
- NFC normalization
- Lowercase conversion
- Punctuation removal
- Devanagari digit to ASCII digit conversion
- Space collapsing

Edit the `normalize()` function in `eval_asr.py` to change normalization rules.

---

## Complete Workflow Example

```bash
# 1. Extract data
python extraction.py --out ../data --max-hours 3

# 2. Degrade with synthetic noise
python degrade.py --data ../data

# 3. Evaluate ASR
python eval_asr.py --data ../data --splits test --max-utts 100
```

---

## Troubleshooting

### ffmpeg AMR-NB encoder missing
```bash
# Check encoder availability
ffmpeg -encoders | grep -i amr

# If missing, install ffmpeg with libopencore_amrnb
# Ubuntu/Debian:
sudo apt-get install ffmpeg libopencore-amrnb

# macOS:
brew install ffmpeg --with-libopencore-amrnb
```

### Hugging Face authentication
```bash
# Login interactively
huggingface-cli login

# Or set token in .env file
echo "HF_TOKEN=hf_xxxxxxxxxxxx" > ../.env
```

### Out of memory errors
- Use smaller Whisper model (`--model tiny` or `--model base`)
- Reduce `--max-utts` in eval_asr.py
- Reduce `--workers` in degrade.py
- Use `--compute-type int8` for quantized models

### Slow processing
- Use GPU for ASR evaluation (`--device cuda`)
- Increase `--workers` in degrade.py (but not more than CPU cores / 2)
- Use sharded evaluation across multiple machines

---

## File Structure

```
../data/
├── clean/              # 16 kHz clean audio (gitignored)
├── clean8k/            # 8 kHz clean reference (gitignored)
├── degraded/           # 8 kHz degraded audio (gitignored)
├── manifest.csv        # Clean audio metadata
├── manifest_degraded.csv  # Degradation recipes
└── asr_*.jsonl         # ASR hypotheses (cached)
```

---

## Citation

If you use this code, please cite the IndicVoices dataset and the Whisper model appropriately.
