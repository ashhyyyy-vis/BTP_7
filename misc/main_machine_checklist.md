# Main Machine Checklist (the data machine): git prep and the data bundle

Run everything from the project root (the folder that contains `scripts/`, `data/`, `artifacts/`).
After every step there is a **You should see** line. If what you see is different, **stop and send me the output**.

Order: **Part 1** (git) first, so the RTX machine can clone early. **Part 2** (data bundle) can run while the RTX machine is being set up. Part 3 is the hand-off.

---

# PART 1: before git commit

## 1.1 Update `.gitignore`
Rule: commit code, docs, metrics and results; never commit audio or `data/`.
```bash
cat .gitignore
```
Append only the lines that are missing:
```
data/
.env
.venv*/
__pycache__/
*.wav
*.flac
baseline_bundle.*
```
**You should see:** all seven patterns present in `.gitignore`.

## 1.2 Untrack any audio that is already committed
`.gitignore` does not untrack files that were committed earlier.
```bash
git ls-files | grep -E '\.(wav|flac)$' | xargs -r git rm --cached -q
git ls-files | grep -cE '\.(wav|flac)$'
du -sh .git
```
**You should see:** `0` on the second line, and `.git` under about 50 MB. If `.git` is much bigger, audio is already in the history; send me the number and do not push yet.

## 1.3 Archive the Groq results and park the stale DEMAND outputs
The Groq runs are slow and rate-limited, so keep them. The old DEMAND outputs are from the pre-hold-out audio (all 17 environments). Your new DEMAND audio reuses the same `deg_id` names, so old DEMAND transcripts or enhanced files would silently be treated as valid for the new audio. Move them out of the way:
```bash
mkdir -p archive/groq_preliminary_2026-10-19
cp -r artifacts/Metrics/musan/ASR archive/groq_preliminary_2026-10-19/musan_ASR
mv artifacts/Metrics/demand/ASR archive/groq_preliminary_2026-10-19/demand_ASR_pre_holdout
mkdir -p artifacts/Metrics/demand/ASR
echo "Groq whisper-large-v3, preliminary. demand_ASR_pre_holdout is from the pre-hold-out DEMAND audio (all 17 environments): do not reuse." > archive/groq_preliminary_2026-10-19/README.txt

mv artifacts/Metrics/demand/Enhancement artifacts/Metrics/demand/Enhancement_OLD_17env 2>/dev/null
[ -d artifacts/experiments/demand ] && mv artifacts/experiments/demand artifacts/experiments/demand_OLD
ls archive/groq_preliminary_2026-10-19 artifacts/Metrics/demand
```
**You should see:** `README.txt`, `demand_ASR_pre_holdout`, `musan_ASR` in the archive; and `ASR` plus `Enhancement_OLD_17env` under `artifacts/Metrics/demand`.
(The old DEMAND metrics CSVs now live in `Enhancement_OLD_17env`, clearly labelled, and get committed as evidence.)

## 1.4 Add the docs to the repo
Copy the files I sent you:

| File | Put it at |
|---|---|
| `local_baseline_runbook.md` (RTX machine steps) | `docs/local_baseline_runbook.md` |
| `main_machine_checklist.md` (this file) | `docs/main_machine_checklist.md` |
| `project_checklist.md` | `docs/project_checklist.md` |
| `results_checkpoint1.md` | `Reports/2026-10-18_checkpoint1_consolidated_results.md` |

Also add two lines to `scripts/README.md`: `hindi_norm.py` = the frozen text normaliser (v2, never edit), and `rescore.py` = offline re-scoring of cached transcripts.
**You should see:** `ls docs Reports` lists the files.

## 1.5 Mark the older reports as superseded
This keeps the repo from presenting claims we have since corrected. It does not rewrite them.
```bash
for f in Reports/2026-10-17_*.md Reports/2026-10-18_demand_*.md Reports/2026-10-18_frozen_*.md; do
  sed -i '1i > **SUPERSEDED:** see Reports/2026-10-18_checkpoint1_consolidated_results.md (section 8 lists corrections to this report).\n' "$f"
done
head -3 Reports/2026-10-17_*.md
```
**You should see:** the banner as the first line. The 10-19 audit report stays as it is.

## 1.6 Check the shell scripts and tidy up
```bash
grep -rn "/home/" shell_scripts scripts | head
cat shell_scripts/run_demand.sh
cat shell_scripts/clean.sh
```
**You should see:** no hard-coded personal paths in code; `run_demand.sh` containing `--test-noise DLIVING NPARK TCAR --dev-noise OOFFICE STRAFFIC` (this is the record of how your data was built); and `clean.sh` deleting only what you expect. It must never be run on the RTX machine.

Tidy-up:
- `commit5.md` exists twice (project root and `commit_changes/`): delete one copy.
- `reorganize_comprehensive.py` and `reorganize_structure.py` look like one-off scripts: delete them or move them to `scripts/_old/` so nobody runs them by accident.
- Leave `file:///home/...` links inside old reports alone.

## 1.7 Check exactly what will be committed
```bash
git add -A
git status --short | head -50
git ls-files -o -c --exclude-standard | xargs -d '\n' du -k 2>/dev/null | sort -n | tail -10
```
**You should see:** no audio files, no `data/`, no `.venv*`; the new `archive/`, `docs/` and updated `Reports/`; and the largest file well under 5 MB (the per-file metrics CSVs and Groq caches are the biggest). If any file is over about 5 MB, send me its name and do not commit.

## 1.8 Commit
Use your usual commit message file or write one. A suitable message:
`Gate 1/2 checkpoint: frozen v2 normaliser, rescore script, consolidated results, local-baseline runbook, Groq archive`

## 1.9 Test a fresh clone before pushing
```bash
rm -rf /tmp/clonetest && git clone . /tmp/clonetest
cd /tmp/clonetest && python -m scripts.hindi_norm && ls docs scripts && cd -
```
**You should see:** `hindi_norm self-test ok, version v2`, and `docs` plus `scripts` listings that include the runbook, `rescore.py`, `eval_asr.py`, `eval_enh.py`, `denoise.py`.

## 1.10 Push
```bash
git push
```
Note the repository's clone URL; the RTX machine needs it. If the repo is private, the runbook already tells the RTX machine to run `gh auth login`.
**You should see:** the push succeeds; the repo page shows `docs/`, `archive/`, `scripts/`.

---

# PART 2: build the data bundle (can run while the RTX machine is being set up)

## 2.1 Check the degraded data
```bash
find data/degraded/demand -name '*.wav' | wc -l
find data/degraded/musan  -name '*.wav' | wc -l
python - <<'EOF'
import csv, collections
r = list(csv.DictReader(open("data/degraded/demand/manifest_degraded.csv", encoding="utf-8")))
env = collections.defaultdict(set)
for x in r:
    if x["noise_file"]: env[x["split"]].add(x["noise_file"].replace("\\", "/").split("/")[0])
print({k: sorted(v) for k, v in env.items()})
EOF
```
**You should see:** `3716`, `3716`, and a dictionary with test = `['DLIVING', 'NPARK', 'TCAR']`, dev = `['OOFFICE', 'STRAFFIC']`, train = the other 12 environments. If a number or an environment list is different, stop and send me the output.

## 2.2 Make the enhancement audio (DeepFilterNet environment)
Activate the DeepFilterNet environment (Python 3.11, with `deepfilternet`), then run these three commands one after another:
```bash
python -m scripts.denoise --data data --noise-type musan  --methods wiener_dd dfn --splits test --max-utts 167 --snrs 0 5 10
python -m scripts.denoise --data data --noise-type demand --methods wiener_dd dfn --splits test --max-utts 167 --snrs 0 5 10
python -m scripts.denoise --data data --noise-type musan  --methods dfn_atten12 dfn_mix50 --splits test --max-utts 167 --snrs 0 5
```
Each prints progress every 50 files with a "min left" estimate. Leave it running; if it stops, run the same command again and it continues (it skips files that exist).
Then check:
```bash
find artifacts/Metrics/demand/Enhancement/dfn -name '*.wav' -newer data/degraded/demand/manifest_degraded.csv | wc -l
ls artifacts/Metrics/demand/Enhancement/wiener_dd | grep -c wav
ls artifacts/Metrics/musan/Enhancement/dfn | grep -c wav
ls artifacts/Metrics/musan/Enhancement/dfn_atten12 | grep -c wav
```
**You should see:** `501`, `501`, `501`, `334`. If the first is below 501, old files are still being reused: stop and send me the numbers.

## 2.3 Pack one bundle
```bash
tar -cf baseline_bundle.tar \
  data/manifest.csv data/clean data/clean8k \
  data/degraded/musan data/degraded/demand \
  artifacts/Metrics/musan/Enhancement artifacts/Metrics/demand/Enhancement
md5sum baseline_bundle.tar > baseline_bundle.md5
tar -tf baseline_bundle.tar | grep -c OLD
du -sh baseline_bundle.tar
```
**You should see:** `0` (no old files inside), and a size of roughly 1 to 2 GB. Send me the size.

## 2.4 Upload to your cloud drive
Upload **two files**: `baseline_bundle.tar` and `baseline_bundle.md5`. Never put them in git.
**You should see:** both files in your drive with the same sizes as on disk.

## 2.5 Commit the new small outputs (optional, after 2.2)
```bash
git add -A && git status --short | head
```
The new enhancement logs (tiny CSVs) show up here; audio does not. Commit them with a short message such as `Add enhancement logs for held-out DEMAND test`.

---

# PART 3: hand-off
Send me: the bundle size (2.3), the 2.1 and 2.2 counts, and the 1.7 size-check output. Then go to `docs/local_baseline_runbook.md` on the RTX machine.
