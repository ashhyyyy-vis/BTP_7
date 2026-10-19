#!/usr/bin/env python3
"""
Gate 2, step 1: run front-end enhancers over the degraded 8 kHz telephone audio.

Reads   <data>/manifest_degraded.csv, <data>/degraded/*.wav          (from degrade.py)
Writes  <artifacts>/<data_name>/enh/<method>/<deg_id>.wav     8 kHz, sample-aligned with clean8k, same length
        <artifacts>/<data_name>/enh/<method>/log<_shardKofN>.csv   per file: deg_id, delay_samples
        (<artifacts> defaults to <projectroot>/artifacts; <data_name> is the last part of --data)

Methods
-------
  wiener_dd   classical baseline, no learning: STFT -> stationary noise PSD from the 15th
              percentile of each frequency bin over the clip -> decision-directed Wiener gain
              (floor -20 dB) -> ISTFT.  Pure numpy/scipy.
  dfn         pretrained DeepFilterNet3 (48 kHz, full-band).  Path: 8 kHz -> upsample x6 ->
              DeepFilterNet -> downsample x6 -> 8 kHz.   Needs torch + deepfilternet (see below).

Why 8 kHz output: a phone call has no content above 4 kHz, and eval_asr/eval_enh upsample every
8 kHz file to 16 kHz the same way, so the ASR sees exactly the same pipeline for the
baseline and for every enhancer. Anything an enhancer puts above 4 kHz is discarded.

Which files: split(s) --splits (default test), the first --max-utts utterances in manifest order
(same rule as eval_asr.py), and only noisy rows with SNR in --snrs (default 0 5 10; 15/20 dB have
almost nothing left to recover). codec-only rows are skipped (nothing to denoise).

Run
---
  python denoise.py --data data_musan --methods wiener_dd
  python denoise.py --data data_musan --methods dfn
  python denoise.py --data data_musan --methods wiener_dd dfn --shard 0/2     # machine A
  python denoise.py --data data_musan --methods wiener_dd dfn --shard 1/2     # machine B
Existing output files are skipped, so a stopped run just continues (use --force to redo).
After sharding, copy each machine's enh/<method>/ files into one artifacts/<data_name>/ folder.

DeepFilterNet install (use a SEPARATE environment from the ASR one)
-------------------------------------------------------------------
  pip install torch torchaudio        # CPU or CUDA build. If importing df fails with an error about
                                      # torchaudio.backend.common, use torch/torchaudio 2.0.x
                                      # (Python <= 3.11); newer torchaudio removed that module.
  pip install deepfilternet
The first run downloads the pretrained DeepFilterNet3 weights from GitHub into the df cache dir.
Uses the GPU automatically if torch sees one.
"""
import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.ndimage import uniform_filter1d
from scipy.signal import correlate, istft, resample_poly, stft

SR = 8000
PROJECT_ROOT = Path(__file__).resolve().parent.parent   # scripts/ -> project root


def get_dataset_name(data_path="data", noise_type=None):
    """Extract dataset name: explicit noise_type, or from data_path, or default 'synthetic'."""
    if noise_type:
        return noise_type
    data_path = Path(data_path)
    if data_path.name == "data":
        return "synthetic"
    if data_path.name.startswith("data_"):
        return data_path.name.replace("data_", "")
    return data_path.name


def get_manifest_path(data_path, dataset_name):
    """Get the manifest path based on dataset name and directory structure."""
    data_path = Path(data_path)
    if not data_path.exists() or data_path.name.startswith("data_"):
        data_path = data_path.parent / "data" if data_path.parent.name else Path("data")
    deg_manifest = data_path / "degraded" / dataset_name / "manifest_degraded.csv"
    if deg_manifest.exists():
        return deg_manifest
    if dataset_name == "synthetic" and (data_path / "manifest_degraded.csv").exists():
        return data_path / "manifest_degraded.csv"
    return deg_manifest

# ------------------------------------------------------------------ classical baseline
NFFT, HOP = 256, 64          # 32 ms window, 8 ms hop (75 % overlap)
NOISE_Q = 0.15               # noise PSD = this percentile (over the whole clip) of each frequency bin,
                             # converted with the exponential-distribution factor -ln(1-q): exact for
                             # stationary Gaussian noise, needs the bin to be speech-free >= 15 % of frames
FREQ_SMOOTH = 5              # bins: smooths the percentile estimate across frequency
DD_BETA = 0.98               # decision-directed smoothing
XI_MIN = 10 ** (-25 / 10)    # a-priori SNR floor
G_MIN = 10 ** (-20 / 10)     # gain floor (-20 dB) limits musical noise


def wiener_dd(x, sr=SR):
    """Offline, stationary-noise Wiener filter. A time-varying (non-stationary) noise is treated as
    its long-term average, which is the usual weakness of this classical baseline."""
    x = np.asarray(x, dtype=np.float64)
    _, _, X = stft(x, fs=sr, window="hann", nperseg=NFFT, noverlap=NFFT - HOP,
                   boundary="zeros", padded=True)
    P = np.abs(X) ** 2
    lam = np.percentile(P, 100 * NOISE_Q, axis=1) / (-np.log(1.0 - NOISE_Q))
    lam = np.maximum(uniform_filter1d(lam, size=FREQ_SMOOTH, mode="nearest"), 1e-12)
    G = np.empty_like(P)
    prev = P[:, 0].copy()                    # |clean amplitude|^2 estimate of the previous frame
    for k in range(P.shape[1]):
        gamma = P[:, k] / lam
        xi = DD_BETA * prev / lam + (1 - DD_BETA) * np.maximum(gamma - 1.0, 0.0)
        xi = np.maximum(xi, XI_MIN)
        g = np.maximum(xi / (1.0 + xi), G_MIN)
        prev = (g ** 2) * P[:, k]
        G[:, k] = g
    _, y = istft(G * X, fs=sr, window="hann", nperseg=NFFT, noverlap=NFFT - HOP, boundary=True)
    return y[: len(x)].astype(np.float32)


# ------------------------------------------------------------------ DeepFilterNet
class DFN:
    def __init__(self, atten_lim_db=None):
        try:
            import torch
            from df.enhance import enhance, init_df
        except Exception as e:  # noqa: BLE001
            sys.exit(f"Could not import torch / deepfilternet ({type(e).__name__}: {e}).\n"
                     "See the install notes at the top of this file.")
        self.torch, self._enhance = torch, enhance
        out = init_df(log_level="WARNING", log_file=None)   # default model: DeepFilterNet3
        self.model, self.state = out[0], out[1]              # newer versions also return suffix, epoch
        self.sr = int(self.state.sr())
        if self.sr % SR:
            sys.exit(f"DeepFilterNet sample rate {self.sr} is not a multiple of {SR}.")
        self.up = self.sr // SR
        self.atten = atten_lim_db
        print(f"DeepFilterNet loaded (sr={self.sr}, device={next(self.model.parameters()).device})")

    def __call__(self, x8):
        up = resample_poly(np.asarray(x8, dtype=np.float32), self.up, 1).astype(np.float32)
        y = self._enhance(self.model, self.state, self.torch.from_numpy(up)[None, :],
                          atten_lim_db=self.atten)
        y = y.detach().cpu().numpy().reshape(-1)
        return resample_poly(y, 1, self.up).astype(np.float32)


# ------------------------------------------------------------------ helpers
def estimate_delay(ref, deg, max_lag=400):
    """lag (samples) by which `deg` is delayed relative to `ref`"""
    n = min(len(ref), len(deg))
    c = correlate(deg[:n], ref[:n], mode="full", method="fft")
    mid = n - 1
    lo, hi = max(0, mid - max_lag), min(len(c), mid + max_lag + 1)
    return int(np.argmax(c[lo:hi]) + lo - mid)


def shift_fit(y, lag, n):
    if lag > 0:
        y = y[lag:]
    elif lag < 0:
        y = np.concatenate([np.zeros(-lag, dtype=y.dtype), y])
    if len(y) < n:
        y = np.concatenate([y, np.zeros(n - len(y), dtype=y.dtype)])
    return y[:n]


def select_rows(root, dataset_name, splits, max_utts, snrs):
    """Same utterance rule as eval_asr.py (manifest order, non-empty transcript, first N);
    keeps noisy rows at the requested SNRs."""
    manifest_path = get_manifest_path(root, dataset_name)
    with open(manifest_path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] in splits]
    by_utt = defaultdict(list)
    for r in rows:
        by_utt[r["utt_id"]].append(r)
    utts = [u for u, rs in by_utt.items() if rs[0]["transcript"].strip()][:max_utts]
    snrs = {float(s) for s in snrs}
    sel = [r for u in utts for r in by_utt[u] if r["snr_db"] != "" and float(r["snr_db"]) in snrs]
    return utts, sel


def parse_shard(text):
    k, n = (int(v) for v in text.split("/"))
    if not (n >= 1 and 0 <= k < n):
        raise argparse.ArgumentTypeError("use K/N with 0 <= K < N, e.g. 1/2")
    return k, n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--noise-type", choices=["synthetic", "musan", "demand"], default=None,
                    help="noise dataset type: synthetic, musan, or demand")
    ap.add_argument("--artifacts", default=str(PROJECT_ROOT / "artifacts"),
                    help="root folder for outputs; results go to <artifacts>/Metrics/<noise_type>/Enhancement/<method>/")
    ap.add_argument("--methods", nargs="+", default=["wiener_dd"],
                    choices=["wiener_dd", "dfn", "dfn_atten12", "dfn_mix50"])
    ap.add_argument("--splits", nargs="+", default=["test"])
    ap.add_argument("--max-utts", type=int, default=100)
    ap.add_argument("--snrs", type=float, nargs="+", default=[0, 5, 10])
    ap.add_argument("--shard", type=parse_shard, default=None, metavar="K/N",
                    help="process only files K, K+N, K+2N, ... of the selected list")
    ap.add_argument("--dfn-atten", type=float, default=None,
                    help="optional DeepFilterNet attenuation limit in dB (default: unlimited)")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    ap.add_argument("--force", action="store_true", help="redo files that already exist")
    args = ap.parse_args()

    root = Path(args.data)
    dataset_name = get_dataset_name(args.data, noise_type=args.noise_type)
    art_root = Path(args.artifacts) / "Metrics" / dataset_name
    utts, sel = select_rows(root, dataset_name, set(args.splits), args.max_utts, args.snrs)
    if not sel:
        sys.exit("No noisy rows selected (check --splits / --snrs against manifest_degraded.csv).")
    if args.shard:
        sel = sel[args.shard[0]:: args.shard[1]]
    tag = f"_shard{args.shard[0]}of{args.shard[1]}" if args.shard else ""
    print(f"[{dataset_name}] {len(utts)} utterances, {len(sel)} noisy files to enhance"
          + (f" (shard {args.shard[0]}/{args.shard[1]})" if args.shard else ""))

    for method in args.methods:
        outdir = art_root / "Enhancement" / method
        outdir.mkdir(parents=True, exist_ok=True)
        todo = [r for r in sel if args.force or not (outdir / f"{r['deg_id']}.wav").exists()]
        print(f"\n[{method}] {len(todo)} to do, {len(sel) - len(todo)} already done")
        if not todo:
            continue
        if method == "wiener_dd":
            fn = wiener_dd
        elif method == "dfn":
            if args.threads:
                import torch
                torch.set_num_threads(args.threads)
            fn = DFN(args.dfn_atten)
        elif method == "dfn_atten12":
            if args.threads:
                import torch
                torch.set_num_threads(args.threads)
            fn = DFN(atten_lim_db=12.0 if args.dfn_atten is None else args.dfn_atten)
        elif method == "dfn_mix50":
            if args.threads:
                import torch
                torch.set_num_threads(args.threads)
            dfn_model = DFN(args.dfn_atten)
            def mix50(x, m=dfn_model):
                y_enh = m(x)
                lag = estimate_delay(x, y_enh)
                y_enh = shift_fit(y_enh, lag, len(x))
                return 0.5 * y_enh + 0.5 * x
            fn = mix50

        log_path = outdir / f"log{tag}.csv"
        new_log = not log_path.exists()
        t0 = time.time()
        delays = []
        with open(log_path, "a", newline="", encoding="utf-8") as lf:
            lw = csv.writer(lf)
            if new_log:
                lw.writerow(["deg_id", "delay_samples"])
            for i, r in enumerate(todo, 1):
                x, sr = sf.read(str(root / r["degraded_path"]), dtype="float32")
                if sr != SR:
                    sys.exit(f"{r['degraded_path']} is {sr} Hz, expected {SR}.")
                y = fn(x)
                lag = estimate_delay(x, y)           # should be ~0; logged so a bad shift is visible
                y = shift_fit(y, lag, len(x))
                y = np.clip(y, -1.0, 1.0)
                sf.write(str(outdir / f"{r['deg_id']}.wav"), y, SR, subtype="PCM_16")
                lw.writerow([r["deg_id"], lag])
                delays.append(lag)
                if i % 50 == 0 or i == len(todo):
                    el = time.time() - t0
                    print(f"  {i}/{len(todo)}  ({el / i:.2f}s each, ~{el / i * (len(todo) - i) / 60:.0f} min left)",
                          flush=True)
        d = np.array(delays)
        print(f"[{method}] done. alignment shift applied (samples): median {np.median(d):.0f}, "
              f"min {d.min()}, max {d.max()}")
        if np.abs(d).max() > 40:
            print("  NOTE: some shifts are large (>5 ms); check enh/%s/log*.csv for those files." % method)


if __name__ == "__main__":
    main()
