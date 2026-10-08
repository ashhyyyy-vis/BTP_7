# Commit Changes

## **Title:** Add Groq backend support and refactor shard file storage for ASR evaluation

### **Modified files:**
- **scripts/eval_asr.py**: Major refactoring to add Groq API backend support and reorganize file storage structure
  - Added `--backend` argument to choose between "local" (Whisper) and "groq" backends
  - Added Groq-specific arguments for rate limiting: `--groq-rpm`, `--groq-ash`, `--groq-rpd`, `--groq-asd`, `--groq-margin`, `--groq-max-wait`
  - Refactored file storage: moved shard files from root-level `shards/` to `data/artifacts/shards/`
  - Changed `--shard-dir` argument to use `artifacts_dir` under data directory automatically
  - Added `RateLimitStop` exception handling for Groq API rate limits
  - Added `groq_budget_report()` function to estimate API quota usage
  - Updated all file paths to use `artifacts_dir` instead of `shard_root`
  - Added default model selection based on backend (small for local, whisper-large-v3 for groq)
  - Improved progress reporting with backend and model information
  - Added graceful stop when Groq rate limits exceed max wait time

- **scripts/__pycache__/eval_asr.cpython-312.pyc**: Python bytecode cache (auto-generated)

### **Deleted files:**
- **shards/asr_small.shard0of5.jsonl**: Old shard file location (moved to artifacts/shards/)

### **New files:**
- **artifacts/**: New directory structure for storing ASR evaluation artifacts
  - **artifacts/shards/**: New location for shard files (replaces root-level shards/)
- **commit_changes/commit2.md**: This commit documentation file