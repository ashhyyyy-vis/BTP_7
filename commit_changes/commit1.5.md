# Commit Changes

**Title:** Add documentation and refactor shard file storage for ASR evaluation

**Changed made:**
1. Modified scripts/eval_asr.py to refactor shard file storage:
   - Added `--shard-dir` argument to allow configurable shard file directory
   - Changed default shard directory from `data/shards/` to `shards/` at repository root
   - Updated `shard_files()` function to accept `shard_root` parameter instead of `root`
   - Modified shard file paths to use `shard_root` instead of `root` for all shard operations
   - This allows shard files to be committed to git while keeping the data directory gitignored

2. Created scripts/README.md with comprehensive documentation:
   - Detailed prerequisites (Python 3.8+, ffmpeg with AMR-NB encoder, Hugging Face authentication)
   - Complete pipeline overview (extraction.py → degrade.py → eval_asr.py)
   - Basic and advanced usage instructions for all three scripts
   - Sharded evaluation instructions for multi-machine setups
   - Troubleshooting section for common issues
   - File structure explanation

3. Created scripts/instructions.txt with step-by-step setup guide:
   - WSL setup instructions
   - Python 3.12 setup using uv virtual environment
   - Dependency installation commands
   - ffmpeg installation with AMR-NB encoder
   - Hugging Face authentication
   - Complete workflow commands in sequence
   - Additional options for customization

**Changes:**
- Modified: scripts/eval_asr.py
- Untracked: scripts/README.md, scripts/instructions.txt

**Details:**
**Modified files:**
- scripts/eval_asr.py:
  - Added `--shard-dir` argument (default: shards/)
  - Changed `shard_files(root, model)` to `shard_files(shard_root, model)`
  - Added `shard_root = Path(args.shard_dir) if args.shard_dir else Path("shards")`
  - Added `shard_root.mkdir(parents=True, exist_ok=True)`
  - Updated shard file paths to use `shard_root` instead of `root`

**New files:**
- scripts/README.md: Comprehensive documentation for the entire pipeline
- scripts/instructions.txt: Step-by-step setup and usage instructions
