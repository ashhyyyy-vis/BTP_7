# Commit Changes

## **Title:** Commit 5: Implement frozen v2 text normaliser, error decomposition, and DEMAND hold-out verification

### **Modified files:**
- **scripts/eval_asr.py**:
  - Replaced legacy text normalization with frozen v2 normalizer import (`from scripts.hindi_norm import normalize`).
  - Added CLI support for `--groq-key` and `--noise-type` arguments.
  - Hardened dependency imports (`dotenv`) for environments lacking optional packages.
- **scripts/eval_enh.py**:
  - Implemented Levenshtein dynamic programming error decomposition (`decompose_errors`), reporting Substitutions ($S$), Deletions ($D$), and Insertions ($I$).
  - Added `sub_pct`, `del_pct`, and `ins_pct` metric columns to `enh_summary_<model>.csv` export.
  - Supported evaluation of custom DFN variants (`dfn_atten12`, `dfn_mix50`).
  - Inherited frozen v2 text normalizer from `eval_asr.py`.
- **scripts/denoise.py**:
  - Added support for cheap DFN variants:
    - `dfn_atten12`: DeepFilterNet3 with maximum attenuation limited to 12 dB.
    - `dfn_mix50`: Sample-aligned 50/50 dual-path linear blending of DFN and degraded audio.
- **scripts/degrade.py**:
  - Supported explicit directory-level noise environment partitioning (`--test-noise`, `--dev-noise`, `--noise-group-by dir`).
  - Re-degraded DEMAND dataset with strict acoustic hold-out: test (`DLIVING`, `NPARK`, `TCAR`), dev (`OOFFICE`, `STRAFFIC`), and 12 unseen training environments.
- **scripts/extraction.py**:
  - Added `--hf-token` CLI parameter to override environment token.
  - Wrapped `dotenv` in safe import block.
- **scripts/enh_metrics.py**:
  - Added support for new front-end method names.
  - Fixed path routing for new `artifacts/Metrics/<dataset>/` hierarchy.
- **scripts/README.md**:
  - Updated documentation to reflect frozen v2 normaliser and updated evaluation workflow.
- **scripts/instructions.txt**:
  - Updated execution order and usage instructions.

### **New files:**
- **scripts/hindi_norm.py**:
  - Frozen v2 project-wide text normaliser (`normalize`, `normalize_v1`, `chars_nospace`).
  - Strips Nuktas (`़`), unifies Chandrabindu (`ँ` $\to$ `ं`), applies NFC canonical composition, and standardizes digits.
  - Provides space-free character representation (`chars_nospace`) for Character Error Rate (CER).
- **scripts/rescore.py**:
  - Offline re-scoring tool for cached transcripts.
  - Computes WER v1, WER v2, and space-free CER with paired bootstrap 95% confidence intervals.
  - Exports `rescore_summary_<model>.csv`.
- **shell_scripts/**:
  - `clean.sh`, `download_noise.sh`, `run_demand.sh`, `run_musan.sh`: Pipeline automation and dataset preparation scripts.
- **Reports/**:
  - `2026-10-17_speech_enhancement_and_error_decomposition_report.md`: Initial front-end evaluation and error decomposition report.
  - `2026-10-18_demand_leakage_audit_and_clean_wer_normalization_analysis.md`: Detailed leakage audit and 20 clean transcript pair analysis.
  - `2026-10-18_frozen_v2_normalisation_and_rescore_analysis_report.md`: Frozen v2 normalisation rescore results across MUSAN and DEMAND.
  - `2026-10-19_checkpoint_1_verification_and_cache_audit_report.md`: Verification and cache line audit for Checkpoint 1.
- **commit_changes/commit5.md** (and **commit5.md**):
  - Documentation of Commit 5 changes.

### **Deleted files:**
- **artifacts/experiments/data_musan/enh/wiener_dd/*.wav**: Reorganized into `artifacts/Metrics/` structure.
- **data/manifest_degraded.csv**: Replaced by per-dataset manifests (`data/degraded/<noise_type>/manifest_degraded.csv`).
