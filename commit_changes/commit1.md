# Commit Changes

**Title:** Initialize Hindi ASR telephone degradation study with data pipeline and evaluation scripts

**Changed made:**
1. Modified the .gitignore file to exclude additional data directories and environment configuration:
   - Added `data/clean` to prevent tracking of clean 16kHz audio files (large binary files)
   - Added `data/clean8k` to prevent tracking of 8kHz downsampled reference audio
   - Added `data/degraded` to prevent tracking of degraded telephone-channel audio
   - Added `.env` to prevent tracking of environment variables (Hugging Face tokens, API keys)

2. Created new project directory structure for Hindi ASR telephone degradation study:
   - Created `commit_changes/` directory to store documentation of commit changes
   - Created `data/` directory for audio data storage and processing:
     - Contains `manifest.csv` with metadata for clean Hindi speech from IndicVoices dataset
     - Contains `manifest_degraded.csv` with degradation recipes and metadata
     - Contains `asr_small.shard0of5.jsonl` with partial ASR transcription results
     - Subdirectories `clean/`, `clean8k/`, `degraded/` for audio files (excluded from git)
   - Created `scripts/` directory for project automation:
     - `extraction.py`: Extracts Hindi speech from IndicVoices Hugging Face dataset, converts to 16kHz WAV, creates speaker-level train/dev/test splits
     - `degrade.py`: Simulates 8kHz AMR-NB telephone channel with noise injection at various SNR levels, time-aligns degraded audio
     - `eval_asr.py`: Evaluates Whisper ASR model performance across degradation conditions (clean, clean8k, codec_only, SNR levels), calculates WER with bootstrap CI

**Changes:**
- Modified: .gitignore
- Untracked: commit_changes/, data/, scripts/

**Details:**
**Modified files:**
- .gitignore: Added 4 new entries:
  - data/clean (excludes clean data directory)
  - data/clean8k (excludes clean8k data directory)
  - data/degraded (excludes degraded data directory)
  - .env (excludes environment variables file)

**New directories (untracked):**
- commit_changes/ (for documenting commit changes)
- data/ (for data storage)
- scripts/ (for project scripts)
