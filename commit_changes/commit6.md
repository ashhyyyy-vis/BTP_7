# Commit Changes

## **Title:** Commit 6 : Gate 1/2 checkpoint, docs, runbook, Groq archive, and held-out DEMAND enhancement logs

### **Summary of Changes:**
1. **Repository & Audio Cleanup**:
   - Updated `.gitignore` to strictly exclude all audio (`*.wav`, `*.flac`), virtual environments (`.venv*`, `myenv/`, `denv/`), and bundle archives (`baseline_bundle.*`).
   - Untracked all previously committed audio `.wav` files from the git index.
2. **Groq Archive & Stale DEMAND Separation**:
   - Archived preliminary Groq results and pre-hold-out DEMAND outputs to `archive/groq_preliminary_2026-10-19/`.
   - Parked old 17-environment DEMAND enhancement outputs to `artifacts/Metrics/demand/Enhancement_OLD_17env/`.
3. **Documentation & Consolidated Checkpoint**:
   - Added `docs/local_baseline_runbook.md`, `docs/main_machine_checklist.md`, and `docs/project_checklist.md`.
   - Added consolidated results report: `Reports/2026-10-18_checkpoint1_consolidated_results.md`.
   - Marked earlier superseded reports with a clear deprecation banner.
   - Updated `scripts/README.md` documenting `hindi_norm.py` and `rescore.py`.
4. **Shell Script & Pipeline Alignment**:
   - Updated `shell_scripts/run_demand.sh` with explicit hold-out flags: `--test-noise DLIVING NPARK TCAR --dev-noise OOFFICE STRAFFIC`.
   - Moved one-off scripts to `scripts/_old/`.
5. **Held-Out DEMAND Test Enhancement Logs**:
   - Generated full 167-utterance enhancement audio for MUSAN and held-out DEMAND test sets.
   - Committed lightweight progress logs (`log.csv`) in `artifacts/Metrics/demand/Enhancement/` and `artifacts/Metrics/musan/Enhancement/`.
