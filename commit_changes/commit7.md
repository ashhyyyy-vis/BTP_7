# Commit Changes

## **Title:** Commit 7 : Fine-tune local baseline runbook for autonomous agent execution and add runner script

### **Summary of Changes:**
1. **Local Baseline Runbook (`docs/local_baseline_runbook.md`)**:
   - Structured for autonomous AI agent execution with explicit directives, environment variables, and sequential validation assertions.
   - Added repository clone URL (`https://github.com/ashhyyyy-vis/BTP_7.git`) and branch (`WER-test`).
   - Added dedicated placeholders for Google Drive bundle links (`baseline_bundle.tar` and `baseline_bundle.md5`) alongside instructions for both CLI (`gdown`) and Windows host browser downloads.
   - Incorporated automatic VRAM detection logic (`>= 8000 MiB` $\to$ `float16`, `6000-7999 MiB` $\to$ `int8_float16`) in `baseline_env.sh`.
   - Documented exact file counts, expected outputs, smoke test verification, full 167-utterance evaluation, rescoring, and preliminary Groq sanity check benchmark table.
   - Added an agent completion checklist.

2. **Standalone Runner Script (`shell_scripts/run_local_baseline.sh`)**:
   - Added an end-to-end executable shell script that runs smoke test, untreated evaluation, enhanced evaluation, and rescoring sequentially with strict error handling (`set -euo pipefail`).
