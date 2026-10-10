# Commit Changes

## **Title:** Add crash recovery with progress logging and timestamped artifact directories

### **Modified files:**
- **scripts/degrade.py**: Added progress logging and resume capability for crash recovery
  - Added `.degrade_progress.jsonl` file to track progress of degradation operations
  - Added `--resume` argument to continue from previous run if interrupted
  - Implemented fingerprint-based validation to ensure resume only happens with identical settings
  - Progress is written immediately after each row completes, preventing data loss on crash/kill/Ctrl-C
  - Added `maxtasksperchild=200` to Pool to prevent worker memory leaks
  - Improved progress reporting with more frequent updates (every 100 rows instead of 200)
  - Added flush after each progress write to ensure data persistence

- **scripts/eval_asr.py**: Major refactoring to use timestamped artifact directories
  - Changed from fixed `data/artifacts/` to timestamped `artifacts/run_YYYYMMDD_HHMMSS/` directories
  - Added `get_project_root()` function to locate project root directory
  - Added `get_unique_artifact_dir()` function to create timestamped artifact directories
  - Added `get_artifact_dir()` function to handle both new and existing artifact directories
  - Added `--artifacts-dir` argument to specify existing artifact directory for merge mode
  - Updated `shard_files()` to use artifacts_dir instead of root
  - Updated all file paths to use artifacts_dir instead of root/artifacts
  - Added print statement showing artifacts directory location
  - Updated merge mode instructions to include `--artifacts-dir` argument
  - Updated shard completion message to show artifacts directory path


### **Deleted files:**
- **artifacts/shards/asr_small.shard0of5.jsonl**: Old shard file from previous run
- **artifacts/shards/asr_summary_whisper-large-v3.csv**: Old summary file from previous run

### **New files:**
- **artifacts/asr_summary_demand_whisper-large-v3.csv**: ASR evaluation summary for demand noise condition
- **artifacts/asr_summary_musan_whisper-large-v3.csv**: ASR evaluation summary for musan noise condition
- **artifacts/asr_summary_whisper-large-v3.csv**: ASR evaluation summary for baseline condition
- **artifacts/asr_whisper-large-v3-musan.jsonl**: Raw ASR hypotheses for musan noise condition
- **artifacts/asr_whisper-large-v3.jsonl**: Raw ASR hypotheses for baseline condition
- **commit_changes/commit3.md**: This commit documentation file
