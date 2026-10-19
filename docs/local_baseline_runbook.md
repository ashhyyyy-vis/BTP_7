# Local Baseline Runbook (RTX machine), v5

This runbook is only for the **RTX machine**. Everything on the data machine (git prep, building the data bundle) is in `docs/main_machine_checklist.md`.

One path, no choices. Follow the steps in order. After each step there is a **You should see** line. If what you see is different, **stop and send me the output**; do not improvise.

**What we are doing:** run Whisper large-v3 locally over the same test audio you used with Groq, so we get numbers we can use for the rest of the project.

**Before you start, you must have:**
1. The repo pushed to GitHub (main machine checklist, Part 1).
2. `baseline_bundle.tar` and `baseline_bundle.md5` uploaded to your cloud drive (main machine checklist, Part 2).

**Decisions already made (do not change them):**
- Everything runs inside WSL2 (Ubuntu on Windows). Nothing is regenerated here: the audio comes in the bundle.
- Model: `large-v3`. Test set: all 167 test utterances.
- Windows steps are marked **[Windows]**; all other steps are typed in the **Ubuntu (WSL) window**.

---

## Step 1. [Windows] Set up WSL2 and check the GPU
1. Update the NVIDIA driver on Windows (GeForce Experience or nvidia.com). The driver goes on Windows, **not** inside Ubuntu.
2. Open PowerShell as Administrator and run (skip if Ubuntu is already installed):
```
wsl --install -d Ubuntu
wsl --update
```
Restart Windows if it asks. Open "Ubuntu" from the Start menu and create a username and password.
3. In the Ubuntu window:
```bash
nvidia-smi
```
**You should see:** a table showing your GPU name and its memory (for example `8192MiB`). **Send me this whole output.**
If it says "command not found" or "no devices", stop and send me what it says.

4. Stop Windows from sleeping while jobs run: Settings > System > Power > Screen and sleep > "when plugged in, put my device to sleep" = **Never**.

## Step 2. Get the code
```bash
sudo apt update && sudo apt install -y git curl zip gh
gh auth login          # only needed if the repo is private: choose GitHub.com, HTTPS, "Login with a web browser"
git clone <your repo URL> ~/project
cd ~/project
ls scripts docs
```
Use the clone URL from your repo page. Work in `~/project` **inside Ubuntu**, not under `/mnt/c` (that is much slower).
**You should see:** `scripts` listing `eval_asr.py`, `eval_enh.py`, `rescore.py`, `hindi_norm.py`, and `docs` listing this runbook.

## Step 3. Download, verify and unpack the bundle
1. In the Windows browser, download `baseline_bundle.tar` and `baseline_bundle.md5` from your cloud drive (they land in your Downloads folder).
2. In Ubuntu, find your Windows user name and copy the files:
```bash
ls /mnt/c/Users/
cp /mnt/c/Users/<YOUR_WINDOWS_NAME>/Downloads/baseline_bundle.tar  ~/project/
cp /mnt/c/Users/<YOUR_WINDOWS_NAME>/Downloads/baseline_bundle.md5  ~/project/
cd ~/project
md5sum -c baseline_bundle.md5
```
**You should see:** `baseline_bundle.tar: OK`. If you see FAILED, the download was damaged: download it again.

3. Unpack and count:
```bash
tar -xf baseline_bundle.tar
ls data/degraded/musan/*.wav  | wc -l
ls data/degraded/demand/*.wav | wc -l
ls artifacts/Metrics/demand/Enhancement/dfn | grep -c wav
```
**You should see:** `3716`, `3716`, `501`.

## Step 4. Install the speech recognizer
```bash
cd ~/project
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv venv --python 3.12 .venv-asr
source .venv-asr/bin/activate
uv pip install faster-whisper scipy soundfile numpy python-dotenv
uv pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"
```
Now decide the compute type from the memory number you saw in `nvidia-smi` (Step 1):
- **8000 MiB or more** (8 GB or bigger): keep `float16`.
- **6000 to 7999 MiB:** use `int8_float16`.
- **Under 6000 MiB:** stop and tell me; I will pick a smaller model.

Create the settings file once (change `float16` to `int8_float16` in the file if your memory is 6000 to 7999 MiB):
```bash
cat > ~/project/baseline_env.sh <<'EOF'
cd ~/project
source ~/project/.venv-asr/bin/activate
export COMPUTE=float16
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$(python -c "import os, nvidia.cublas.lib, nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ':' + os.path.dirname(nvidia.cudnn.lib.__file__))"):$LD_LIBRARY_PATH
EOF
source ~/project/baseline_env.sh
python -c "import ctranslate2; print(ctranslate2.get_cuda_device_count(), 'GPU(s) visible')"
```
**From now on, every time you open a new Ubuntu window, first run:** `source ~/project/baseline_env.sh`
**You should see:** `1 GPU(s) visible`. If it prints `0` or an error mentioning `libcublas` or `libcudnn`, stop and send me the error.

## Step 5. Smoke test (5 utterances, a few minutes)
```bash
python -m scripts.eval_asr --data data --noise-type musan --backend local --model large-v3 --device cuda --compute-type $COMPUTE --splits test --max-utts 5
```
The first run downloads the model (about 3 GB). Open a second Ubuntu window and run `nvidia-smi` while it works.
**You should see:** a WER table at the end; GPU memory in use in the second window; and a progress line with "s each". **Send me the "s each" number and the WER table.**
Then look at three transcripts:
```bash
head -3 artifacts/Metrics/musan/ASR/asr_large-v3.jsonl
```
**You should see:** Hindi text, not empty and not English.

## Step 6. Full run: untreated audio
```bash
python -m scripts.eval_asr --data data --noise-type musan  --backend local --model large-v3 --device cuda --compute-type $COMPUTE --splits test --max-utts 167
python -m scripts.eval_asr --data data --noise-type demand --backend local --model large-v3 --device cuda --compute-type $COMPUTE --splits test --max-utts 167
grep -c . artifacts/Metrics/musan/ASR/asr_large-v3.jsonl
grep -c . artifacts/Metrics/demand/ASR/asr_large-v3.jsonl
```
If the window closes or the machine restarts, run `source ~/project/baseline_env.sh` and run the same command again: it continues where it stopped.
**You should see:** `1336` and `1336` (167 utterances x 8 conditions).

## Step 7. Full run: enhanced audio, then the tables
```bash
python -m scripts.eval_enh --data data --noise-type musan  --backend local --model large-v3 --device cuda --compute-type $COMPUTE --splits test --max-utts 167 --front-ends wiener_dd dfn --snrs 0 5 10
python -m scripts.eval_enh --data data --noise-type demand --backend local --model large-v3 --device cuda --compute-type $COMPUTE --splits test --max-utts 167 --front-ends wiener_dd dfn --snrs 0 5 10
python -m scripts.eval_enh --data data --noise-type musan  --backend local --model large-v3 --device cuda --compute-type $COMPUTE --splits test --max-utts 167 --front-ends dfn_atten12 dfn_mix50 --snrs 0 5
python -m scripts.rescore --data data --noise-type musan  --model large-v3 --max-utts 167 --snrs 0 5 10
python -m scripts.rescore --data data --noise-type demand --model large-v3 --max-utts 167 --snrs 0 5 10
```
(The variants run separately because they exist for 0 and 5 dB only.)
**You should see:** each `rescore` run ends with "167 utterances scored" and a line `saved .../rescore_summary_large-v3.csv`.

## Step 8. Sanity check against Groq
```bash
python -m scripts.rescore --data data --noise-type musan --model large-v3 --max-utts 100 --snrs 0 5 10
```
Look at the first table ("Effect of the normaliser"), column **WER v2**. The Groq values for the same 100 utterances were: clean 25.7, clean8k 26.7, codec_only 30.3, snr_10 37.4, snr_5 42.1, snr_0 51.5.
**You should see:** each local value within about 3 points of those. If any differs by more than 5, stop and send me the table.

## Step 9. Save and send the results
```bash
pip freeze | grep -iE "faster-whisper|ctranslate2" > ~/project/versions.txt
cp artifacts/Metrics/musan/ASR/rescore_summary_large-v3.csv  /mnt/c/Users/<YOUR_WINDOWS_NAME>/Downloads/musan_rescore_large-v3.csv
cp artifacts/Metrics/demand/ASR/rescore_summary_large-v3.csv /mnt/c/Users/<YOUR_WINDOWS_NAME>/Downloads/demand_rescore_large-v3.csv
cp ~/project/versions.txt /mnt/c/Users/<YOUR_WINDOWS_NAME>/Downloads/
```
**Send me:** the `nvidia-smi` output (Step 1), the "s each" number (Step 5), the two `*_rescore_large-v3.csv` files, `versions.txt`, and the Step 8 table.

## Done when
- [ ] Step 3 checksum OK and counts 3716 / 3716 / 501
- [ ] Step 5 smoke test printed Hindi transcripts
- [ ] Step 6 counts 1336 and 1336
- [ ] Step 7 both `rescore` runs say 167 utterances
- [ ] Step 8 within about 3 points of the Groq values
- [ ] Results sent to me

## Rules that prevent bad data
- Never reuse a transcript folder or enhanced-audio folder after the audio changes.
- Do not edit `hindi_norm.py` or any ASR setting after the first full run.
- Do not run anything else on the RTX machine in parallel with Steps 6 and 7.
