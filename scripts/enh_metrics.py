#!/usr/bin/env python3
"""
Gate 2, step 2: signal-quality metrics for the front-ends (no ASR, runs in minutes on a CPU).

For every noisy test file at the chosen SNRs, compares against the clean 8 kHz reference
(<data>/clean8k/, sample-aligned by degrade.py):
    SI-SDR   dB, scale-invariant signal-to-distortion ratio            (own implementation)
    STOI     0..1, intelligibility                                      (pip install pystoi)
    PESQ-NB  1..4.5, narrowband perceptual quality, ITU-T P.862         (pip install pesq)
for the degraded audio ('none') and for each enhanced version in enh/<method>/.
STOI / PESQ are skipped with a notice when their package is not installed.

Reads   <data>/manifest_degraded.csv, clean8k/, degraded/
        <artifacts>/<data_name>/enh/<method>/          (written by denoise.py)
Writes  <artifacts>/<data_name>/enh_metrics_per_file.csv      one row per file and method
        <artifacts>/<data_name>/enh_metrics_summary.csv       mean and paired improvement over 'none' per SNR
(<artifacts> defaults to <projectroot>/artifacts; <data_name> is the last part of --data)

Run     python enh_metrics.py --data data_musan --front-ends wiener_dd dfn
Use the same --splits / --max-utts / --snrs as denoise.py.

How to read it: improvement over 'none' is the quantity that matters (paired mean difference
with a 95 % bootstrap CI over utterances; * = CI excludes 0). A front-end that makes SI-SDR or
PESQ worse is unlikely to help ASR; one that improves them may still not help ASR, because
metrics and recognisers do not weigh distortions the same way, so ASR (eval_enh.py) decides.
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

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


def si_sdr(ref, est, eps=1e-9):
    ref = ref - ref.mean()
    est = est - est.mean()
    a = float(est @ ref) / (float(ref @ ref) + eps)
    s = a * ref
    e = est - s
    return 10 * np.log10((float(s @ s) + eps) / (float(e @ e) + eps))


def load_metric_fns():
    fns = {"si_sdr": lambda ref, est: si_sdr(ref.astype(np.float64), est.astype(np.float64))}
    try:
        from pystoi import stoi
        fns["stoi"] = lambda ref, est: float(stoi(ref, est, SR, extended=False))
    except ImportError:
        print("NOTE: pystoi not installed -> STOI skipped   (pip install pystoi)")
    try:
        from pesq import pesq
        fns["pesq_nb"] = lambda ref, est: float(pesq(SR, ref, est, "nb"))
    except ImportError:
        print("NOTE: pesq not installed -> PESQ skipped     (pip install pesq; needs a C compiler)")
    return fns


def select_rows(root, dataset_name, splits, max_utts, snrs):
    manifest_path = get_manifest_path(root, dataset_name)
    with open(manifest_path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] in splits]
    by_utt = defaultdict(list)
    for r in rows:
        by_utt[r["utt_id"]].append(r)
    utts = [u for u, rs in by_utt.items() if rs[0]["transcript"].strip()][:max_utts]
    snrs = {float(s) for s in snrs}
    noisy = [r for u in utts for r in by_utt[u] if r["snr_db"] != "" and float(r["snr_db"]) in snrs]
    codec = [r for u in utts for r in by_utt[u] if r["snr_db"] == ""]
    return noisy, codec


def paired_ci(delta, boot=2000, seed=0):
    d = np.asarray(delta, float)
    d = d[~np.isnan(d)]
    if len(d) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(boot, len(d)))
    m = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--noise-type", choices=["synthetic", "musan", "demand"], default=None,
                    help="noise dataset type: synthetic, musan, or demand")
    ap.add_argument("--artifacts", default=str(PROJECT_ROOT / "artifacts"),
                    help="root folder for outputs; enhanced audio is read from and metrics are written to <artifacts>/Metrics/<noise_type>/Enhancement/")
    ap.add_argument("--front-ends", nargs="+", default=["wiener_dd", "dfn"])
    ap.add_argument("--splits", nargs="+", default=["test"])
    ap.add_argument("--max-utts", type=int, default=100)
    ap.add_argument("--snrs", type=float, nargs="+", default=[0, 5, 10])
    args = ap.parse_args()

    root = Path(args.data)
    dataset_name = get_dataset_name(args.data, noise_type=args.noise_type)
    art_root = Path(args.artifacts) / "Metrics" / dataset_name
    enh_dir = art_root / "Enhancement"
    enh_dir.mkdir(parents=True, exist_ok=True)
    sel, codec_rows = select_rows(root, dataset_name, set(args.splits), args.max_utts, args.snrs)
    if not sel:
        sys.exit("No noisy rows selected.")
    fns = load_metric_fns()

    # Ceiling: the same metrics for the codec-only files (no added noise). The AMR codec itself
    # distorts the waveform, so no denoiser can score better than this on these metrics.
    ceiling = {mt: [] for mt in fns}
    for r in codec_rows:
        ref = sf.read(str(root / r["clean8k_path"]), dtype="float32")[0]
        est = sf.read(str(root / r["degraded_path"]), dtype="float32")[0]
        n = min(len(ref), len(est))
        for name, fn in fns.items():
            try:
                ceiling[name].append(fn(ref[:n], est[:n]))
            except Exception:  # noqa: BLE001
                pass
    ceil_mean = {mt: (float(np.nanmean(v)) if v else float("nan")) for mt, v in ceiling.items()}
    if codec_rows:
        print("Codec-only ceiling (AMR-NB, no added noise), mean over %d files: " % len(codec_rows)
              + ", ".join(f"{k}={v:.2f}" for k, v in ceil_mean.items()))
    fes = [f for f in args.front_ends if f != "none"]
    methods = ["none"] + fes

    per_file = []  # dicts
    skipped = defaultdict(int)
    for i, r in enumerate(sel, 1):
        ref = sf.read(str(root / r["clean8k_path"]), dtype="float32")[0]
        for m in methods:
            if m == "none":
                p = root / r["degraded_path"]
            else:
                p = enh_dir / m / f"{r['deg_id']}.wav"
                if not p.exists():
                    fallback_p = art_root / "enh" / m / f"{r['deg_id']}.wav"
                    if fallback_p.exists():
                        p = fallback_p
            if not p.exists():
                skipped[m] += 1
                continue
            est, sr = sf.read(str(p), dtype="float32")
            if sr != SR:
                sys.exit(f"{p} is {sr} Hz, expected {SR}")
            n = min(len(ref), len(est))
            rec = dict(deg_id=r["deg_id"], utt_id=r["utt_id"], front_end=m, snr_db=float(r["snr_db"]),
                       noise_group=(r["noise_file"].replace("\\", "/").split("/")[0] if "/" in r["noise_file"].replace("\\", "/")
                                    else (r["noise_file"] and "file") or r["noise_type"]))
            for name, fn in fns.items():
                try:
                    rec[name] = fn(ref[:n], est[:n])
                except Exception:  # noqa: BLE001  (PESQ raises on too-quiet / too-short input)
                    rec[name] = float("nan")
            per_file.append(rec)
        if i % 100 == 0:
            print(f"  {i}/{len(sel)}", flush=True)
    for m, k in skipped.items():
        print(f"WARNING: {k} files missing for '{m}' (run denoise.py first); those utterances are not compared")

    metrics = list(fns)
    with open(enh_dir / "enh_metrics_per_file.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["deg_id", "utt_id", "front_end", "snr_db", "noise_group"] + metrics)
        wr.writeheader()
        wr.writerows(per_file)

    # index by (front_end, deg_id)
    idx = {(r["front_end"], r["deg_id"]): r for r in per_file}
    summary = []
    snr_list = sorted({r["snr_db"] for r in per_file}, reverse=True)
    print(f"\nMean metric per SNR, and improvement over 'none' (paired, 95% CI; * = excludes 0)")
    for fe in fes:
        print(f"\n== {fe} ==")
        print(f"{'SNR':>5} {'n':>4} " + "".join(f"{mt:>9}{'→ ' + fe[:6]:>9}{'Δ [95% CI]':>24}" for mt in metrics))
        for snr in snr_list + ["all"]:
            ids = [d for (m, d), r in idx.items() if m == fe and (snr == "all" or r["snr_db"] == snr)
                   and ("none", d) in idx]
            if not ids:
                continue
            line = f"{snr if snr == 'all' else format(snr, 'g'):>5} {len(ids):>4} "
            for mt in metrics:
                a = np.array([idx[("none", d)][mt] for d in ids])
                b = np.array([idx[(fe, d)][mt] for d in ids])
                dm, lo, hi = paired_ci(b - a)
                star = "*" if (lo > 0 or hi < 0) else " "
                line += f"{np.nanmean(a):>9.2f}{np.nanmean(b):>9.2f}{f'{dm:+.2f} [{lo:+.2f},{hi:+.2f}]{star}':>24}"
                summary.append(dict(front_end=fe, snr_db=snr, metric=mt, n=len(ids), none_mean=f"{np.nanmean(a):.3f}",
                                    fe_mean=f"{np.nanmean(b):.3f}", delta=f"{dm:.3f}", d_lo=f"{lo:.3f}",
                                    d_hi=f"{hi:.3f}"))
            print(line)
    for mt, v in ceil_mean.items():
        summary.append(dict(front_end="codec_only_ceiling", snr_db="ref", metric=mt, n=len(ceiling[mt]),
                            none_mean=f"{v:.3f}", fe_mean="", delta="", d_lo="", d_hi=""))
    with open(enh_dir / "enh_metrics_summary.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["front_end", "snr_db", "metric", "n", "none_mean", "fe_mean",
                                            "delta", "d_lo", "d_hi"])
        wr.writeheader()
        wr.writerows(summary)
    print(f"\nsaved {enh_dir / 'enh_metrics_summary.csv'} and {enh_dir / 'enh_metrics_per_file.csv'}")


if __name__ == "__main__":
    main()
