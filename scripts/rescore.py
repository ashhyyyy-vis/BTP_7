#!/usr/bin/env python3
"""
rescore.py -- re-score cached ASR transcripts under the frozen normaliser (hindi_norm.py).
NO audio and NO ASR calls: it only reads the manifest and the existing asr_<model>*.jsonl caches.

What it reports (95% bootstrap CIs over utterances; front-end differences are PAIRED):
  1. Normalisation effect on the untreated conditions:  WER v1 (old) vs WER v2 (frozen) vs CER
  2. For every condition and front-end found in the caches (none, wiener_dd, dfn, dfn_atten12,
     dfn_mix50, ...):  WER v2 and CER, with the paired difference to 'none'.
     -> use this to check that the Gate 2 conclusion survives the new normaliser.

Run (from the project root; same --data / --noise-type / --model you used for eval_asr.py):
  python -m scripts.rescore --data data --noise-type musan --model whisper-large-v3
  python -m scripts.rescore --data data --noise-type demand --model whisper-large-v3

Writes <artifacts>/Metrics/<dataset>/ASR/rescore_summary_<model>.csv

Cache discovery mirrors eval_enh.py (artifacts/Metrics/<dataset>/ASR, artifacts/shards, and the old
artifacts/experiments/<dataset>/asr/<model> folder). IMPORTANT: only point this at caches made by ONE
backend; Groq and local transcripts of the same model name share file names.
Front-ends are auto-detected from cache keys of the form  <deg_id>@<front_end>  (override with --front-ends).
"""
import argparse
import csv
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from scripts.hindi_norm import NORM_VERSION, chars_nospace, normalize, normalize_v1
except ImportError as e:
    if getattr(e, "name", None) not in ("scripts", "scripts.hindi_norm"):
        raise
    from hindi_norm import NORM_VERSION, chars_nospace, normalize, normalize_v1

COND_ORDER = ["clean", "clean8k", "codec_only", "snr_20", "snr_15", "snr_10", "snr_5", "snr_0"]
METRICS = {"wer_v1": (0, 1), "wer": (2, 3), "cer": (4, 5)}  # slots in the per-utterance vector


# ------------------------------------------------------------------ paths (same convention as eval_asr.py)
def get_dataset_name(data_path="data", noise_type=None):
    if noise_type:
        return noise_type
    data_path = Path(data_path)
    if data_path.name == "data":
        return "synthetic"
    if data_path.name.startswith("data_"):
        return data_path.name.replace("data_", "")
    return data_path.name


def get_manifest_path(data_path, dataset_name):
    data_path = Path(data_path)
    if not data_path.exists() or data_path.name.startswith("data_"):
        data_path = data_path.parent / "data" if data_path.parent.name else Path("data")
    deg = data_path / "degraded" / dataset_name / "manifest_degraded.csv"
    if deg.exists():
        return deg
    if dataset_name == "synthetic" and (data_path / "manifest_degraded.csv").exists():
        return data_path / "manifest_degraded.csv"
    return deg


# ------------------------------------------------------------------ scoring helpers
def edit_distance(ref, hyp):
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, w in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != w))
        prev = cur
    return prev[-1]


def score_pair(ref_raw, hyp_raw):
    """-> [edits_v1, nref_v1, edits_v2, nref_v2, char_edits, nref_chars]"""
    r1, h1 = normalize_v1(ref_raw).split(), normalize_v1(hyp_raw).split()
    r2, h2 = normalize(ref_raw).split(), normalize(hyp_raw).split()
    rc, hc = chars_nospace(ref_raw), chars_nospace(hyp_raw)
    return [edit_distance(r1, h1), len(r1), edit_distance(r2, h2), len(r2), edit_distance(rc, hc), len(rc)]


def agg(vecs, metric):
    a, b = METRICS[metric]
    e = np.array([v[a] for v in vecs], float)
    n = np.array([v[b] for v in vecs], float)
    return e, n


def boot_ci(vecs, metric, boot=1000, seed=0):
    e, n = agg(vecs, metric)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(e), size=(boot, len(e)))
    b = 100 * e[idx].sum(1) / n[idx].sum(1)
    return 100 * e.sum() / n.sum(), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def paired_delta(vecs_a, vecs_b, metric, boot=1000, seed=0):
    """metric(b) - metric(a) on the same utterances (lists aligned by utterance)."""
    ea, na = agg(vecs_a, metric)
    eb, nb = agg(vecs_b, metric)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ea), size=(boot, len(ea)))
    d = 100 * eb[idx].sum(1) / nb[idx].sum(1) - 100 * ea[idx].sum(1) / na[idx].sum(1)
    return (100 * eb.sum() / nb.sum() - 100 * ea.sum() / na.sum(),
            float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))


def fmt_delta(d, lo, hi):
    star = "*" if (lo > 0 or hi < 0) else " "
    return f"{d:+5.1f} [{lo:+5.1f},{hi:+5.1f}]{star}"


def load_cache(files):
    cache = {}
    for fp in files:
        if fp.exists():
            with open(fp, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # truncated last line of an interrupted run
                    cache[d["key"]] = d["hyp"]
    return cache


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--noise-type", choices=["synthetic", "musan", "demand"], default=None)
    ap.add_argument("--model", default="whisper-large-v3")
    ap.add_argument("--splits", nargs="+", default=["test"])
    ap.add_argument("--max-utts", type=int, default=100,
                    help="use the SAME value as the original run so the same utterances are scored")
    ap.add_argument("--snrs", type=float, nargs="+", default=[0, 5, 10])
    ap.add_argument("--front-ends", nargs="+", default=None,
                    help="default: auto-detect from the caches")
    ap.add_argument("--project-root", default=None,
                    help="folder that contains artifacts/ (default: parent of this script's folder)")
    args = ap.parse_args()

    root = Path(args.data)
    ds = get_dataset_name(args.data, args.noise_type)
    proj = Path(args.project_root).resolve() if args.project_root else Path(__file__).resolve().parent.parent
    asr_dir = proj / "artifacts" / "Metrics" / ds / "ASR"
    asr_dir.mkdir(parents=True, exist_ok=True)

    mpath = get_manifest_path(root, ds)
    if not mpath.exists():
        sys.exit(f"manifest not found: {mpath}")
    with open(mpath, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] in args.splits]
    by_utt = defaultdict(list)
    for r in rows:
        by_utt[r["utt_id"]].append(r)
    # same selection rule as eval_asr.py / eval_enh.py (old normaliser decides "usable")
    utts = [u for u, rs in by_utt.items() if normalize_v1(rs[0]["transcript"])][: args.max_utts]
    if not utts:
        sys.exit("No usable utterances.")
    snrs = {float(s) for s in args.snrs}

    files = [asr_dir / f"asr_{args.model}.jsonl"]
    files += [Path(p) for p in sorted(glob.glob(str(asr_dir / f"asr_{args.model}_shard*.jsonl")))]
    files += [Path(p) for p in sorted(glob.glob(str(asr_dir / f"asr_{args.model}_enh*.jsonl")))]
    shards = proj / "artifacts" / "shards"
    if shards.exists():
        files += [Path(p) for p in sorted(glob.glob(str(shards / f"asr_{args.model}*.jsonl")))]
    old = proj / "artifacts" / "experiments" / ds / "asr" / args.model
    if old.exists():
        files += [old / f"asr_{args.model}.jsonl"]
        files += [Path(p) for p in sorted(glob.glob(str(old / f"asr_{args.model}_enh*.jsonl")))]
    cache = load_cache(files)
    if not cache:
        sys.exit(f"no transcripts found (looked in {asr_dir}, {shards}, {old})")
    print(f"norm version {NORM_VERSION}; {len(cache)} cached transcripts from "
          f"{sum(1 for f in files if f.exists())} file(s); dataset={ds}")

    fes = args.front_ends or sorted({k.split("@", 1)[1] for k in cache if "@" in k})
    print("front-ends:", ["none"] + fes)

    def cond_label(r):
        return "codec_only" if r["snr_db"] == "" else f"snr_{float(r['snr_db']):g}"

    # per (front_end, condition): utt -> vector (summed if a utterance has several rows in a condition)
    per = defaultdict(dict)
    dropped = 0
    for u in utts:
        ref = by_utt[u][0]["transcript"]
        base_keys = [("clean", f"{u}::clean"), ("clean8k", f"{u}::clean8k")]
        for r in by_utt[u]:
            if r["snr_db"] == "":
                base_keys.append(("codec_only", r["deg_id"]))
            elif float(r["snr_db"]) in snrs:
                base_keys.append((cond_label(r), r["deg_id"]))
        if not all(k in cache for _, k in base_keys):
            dropped += 1
            continue
        for c, k in base_keys:
            v = score_pair(ref, cache[k])
            old_v = per[("none", c)].get(u)
            per[("none", c)][u] = v if old_v is None else [a + b for a, b in zip(old_v, v)]
        for fe in fes:
            by_c = defaultdict(list)
            for r in by_utt[u]:
                if r["snr_db"] != "" and float(r["snr_db"]) in snrs:
                    by_c[cond_label(r)].append(r["deg_id"])
            for c, dids in by_c.items():
                keys = [f"{d}@{fe}" for d in dids]
                if not all(k in cache for k in keys):
                    continue
                vec = None
                for k in keys:
                    v = score_pair(ref, cache[k])
                    vec = v if vec is None else [a + b for a, b in zip(vec, v)]
                per[(fe, c)][u] = vec
    if dropped:
        print(f"note: {dropped} of {len(utts)} utterances lack cached transcripts for the untreated ladder "
              f"and are excluded")
    n_scored = len(per.get(("none", "clean"), {}))
    if not n_scored:
        sys.exit("Nothing to score.")
    print(f"{n_scored} utterances scored\n")

    conds = [c for c in COND_ORDER if ("none", c) in per]
    out = []

    # ---- 1. normalisation effect, untreated audio
    print("1. Effect of the normaliser on untreated audio (no front-end):")
    print(f"{'condition':<12}{'WER v1':>8}{'WER v2':>8}{'CER':>7}   {'WER v2 95% CI':<16}")
    for c in conds:
        vecs = list(per[("none", c)].values())
        w1 = boot_ci(vecs, "wer_v1")[0]
        w2, lo, hi = boot_ci(vecs, "wer")
        cer = boot_ci(vecs, "cer")[0]
        print(f"{c:<12}{w1:>8.1f}{w2:>8.1f}{cer:>7.1f}   [{lo:5.1f}, {hi:5.1f}]")
    print("(v1 = old normaliser, v2 = frozen hindi_norm; CER ignores spaces, so compound splits do not count)\n")

    # ---- 2. front-end comparison per metric
    for metric, title in (("wer", "WER v2 (%)"), ("cer", "CER (%)")):
        print(f"2. {title}: front-end vs none, paired delta [95% CI], * = CI excludes 0")
        present = [fe for fe in fes if any((fe, c) in per for c in conds)]
        head = f"{'condition':<12}{'none':>7}"
        for fe in present:
            head += f"   {fe:<13}{'val':>6}  {'delta vs none':<21}"
        print(head)
        for c in conds:
            base = per[("none", c)]
            us = sorted(base)
            vb = [base[u] for u in us]
            w0, lo0, hi0 = boot_ci(vb, metric)
            out.append(dict(metric=metric, front_end="none", condition=c, value=f"{w0:.2f}", ci_lo=f"{lo0:.2f}",
                            ci_hi=f"{hi0:.2f}", delta="", d_lo="", d_hi="", n_utts=len(us)))
            line = f"{c:<12}{w0:>7.1f}"
            for fe in present:
                fb = per.get((fe, c))
                if not fb:
                    line += f"   {fe:<13}{'-':>6}"
                    continue
                us2 = [u for u in us if u in fb]
                if not us2:
                    line += f"   {fe:<13}{'-':>6}"
                    continue
                a = [base[u] for u in us2]
                b = [fb[u] for u in us2]
                w, lo, hi = boot_ci(b, metric)
                d, dlo, dhi = paired_delta(a, b, metric)
                out.append(dict(metric=metric, front_end=fe, condition=c, value=f"{w:.2f}", ci_lo=f"{lo:.2f}",
                                ci_hi=f"{hi:.2f}", delta=f"{d:.2f}", d_lo=f"{dlo:.2f}", d_hi=f"{dhi:.2f}",
                                n_utts=len(us2)))
                line += f"   {fe:<13}{w:>6.1f}  {fmt_delta(d, dlo, dhi):<21}"
            print(line)
        print()

    # also store the v1 column for the record
    for c in conds:
        vecs = list(per[("none", c)].values())
        w, lo, hi = boot_ci(vecs, "wer_v1")
        out.append(dict(metric="wer_v1", front_end="none", condition=c, value=f"{w:.2f}", ci_lo=f"{lo:.2f}",
                        ci_hi=f"{hi:.2f}", delta="", d_lo="", d_hi="", n_utts=len(vecs)))

    spath = asr_dir / f"rescore_summary_{args.model}.csv"
    with open(spath, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["metric", "front_end", "condition", "value", "ci_lo", "ci_hi",
                                           "delta", "d_lo", "d_hi", "n_utts"])
        wr.writeheader()
        wr.writerows(out)
    print(f"saved {spath}")


if __name__ == "__main__":
    main()
