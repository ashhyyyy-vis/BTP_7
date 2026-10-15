#!/usr/bin/env python3
"""
Gate 2, step 3: does a front-end (denoiser) lower ASR error on telephone-quality noisy speech?

Compares, on the SAME utterances and the SAME ASR:
    none        the degraded audio as is            (transcripts already in asr_<model>.jsonl)
    <front-end> the audio after enh/<front-end>/    (made by denoise.py)

Reads   <data>/manifest_degraded.csv, <data>/degraded/, <data>/enh/<front-end>/
        <data>/asr_<model>*.jsonl    existing transcripts of the degraded audio (never redone)
Writes  <data>/asr_<model>_enh[_shardKofN].jsonl   transcripts of the enhanced audio (cache)
        <data>/enh_summary_<model>.csv

Reports (95 % bootstrap CIs over utterances)
  1. WER per condition and front-end, with the PAIRED difference to 'none' (resampling the same
     utterances for both, so the CI of the difference is much tighter than the two separate CIs)
  2. "gap closed": share of the noise penalty (WER at that SNR minus codec-only WER) that the
     front-end removes.  100 % = as good as no noise at all, 0 % = no help, negative = made it worse.
  3. WER by noise group (the noise sub-folder: e.g. DLIVING / NPARK / TCAR for DEMAND,
     free-sound / sound-bible for MUSAN), pooled over SNRs, none vs each front-end.

Run  (use the same --model / --threads you used for eval_asr.py)
  python -m scripts.eval_enh --data data_musan --model whisper-large-v3 --front-ends wiener_dd dfn
  # several machines: same arguments everywhere, a different --shard on each, then --merge
  python -m scripts.eval_enh --data data_musan --model whisper-large-v3 --front-ends wiener_dd dfn --shard 0/3
  python -m scripts.eval_enh --data data_musan --model whisper-large-v3 --front-ends wiener_dd dfn --merge

Needs eval_asr.py next to this file (it reuses its text normalisation, WER code and Whisper
loader, so the numbers are computed exactly as before).
"""
import argparse
import csv
import glob
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

try:  # run as  python -m scripts.eval_enh   (package) ...
    from scripts.eval_asr import (COND_ORDER, edit_distance, get_transcriber, load_16k, normalize,
                                  parse_shard, get_dataset_name, get_project_root)
except ImportError:  # ... or as  python scripts/eval_enh.py  (flat)
    from eval_asr import COND_ORDER, edit_distance, get_transcriber, load_16k, normalize, parse_shard, get_dataset_name, get_project_root


# ------------------------------------------------------------------ small helpers
def cond_label(row):
    return "codec_only" if row["snr_db"] == "" else f"snr_{float(row['snr_db']):g}"


def noise_group(row):
    nf = row.get("noise_file", "")
    if not nf:
        return row.get("noise_type", "") or "none"
    return nf.replace("\\", "/").split("/")[0] if "/" in nf.replace("\\", "/") else "file"


def wer_of(pairs):
    e = sum(p[0] for p in pairs)
    n = sum(p[1] for p in pairs)
    return 100.0 * e / n


def boot_ci(pairs, boot=1000, seed=0):
    """pairs: [(edits, ref_len)] one per utterance -> (wer, lo, hi)"""
    e = np.array([p[0] for p in pairs], float)
    n = np.array([p[1] for p in pairs], float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(e), size=(boot, len(e)))
    b = 100 * e[idx].sum(1) / n[idx].sum(1)
    return 100 * e.sum() / n.sum(), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def paired_delta(pairs_a, pairs_b, boot=1000, seed=0):
    """WER(b) - WER(a) on the same utterances (lists aligned by utterance) -> (delta, lo, hi)"""
    ea = np.array([p[0] for p in pairs_a], float)
    eb = np.array([p[0] for p in pairs_b], float)
    n = np.array([p[1] for p in pairs_a], float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(n), size=(boot, len(n)))
    d = 100 * (eb[idx].sum(1) - ea[idx].sum(1)) / n[idx].sum(1)
    return 100 * (eb.sum() - ea.sum()) / n.sum(), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def fmt_delta(d, lo, hi):
    star = "*" if (lo > 0 or hi < 0) else " "
    return f"{d:+5.1f} [{lo:+5.1f},{hi:+5.1f}]{star}"


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--splits", nargs="+", default=["test"])
    ap.add_argument("--max-utts", type=int, default=100)
    ap.add_argument("--snrs", type=float, nargs="+", default=[0, 5, 10])
    ap.add_argument("--front-ends", nargs="+", default=["wiener_dd", "dfn"])
    ap.add_argument("--model", default="small")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--compute-type", default="int8")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--shard", type=parse_shard, default=None, metavar="K/N")
    ap.add_argument("--merge", action="store_true", help="no transcription: score what is in the caches")
    args = ap.parse_args()
    fes = [f for f in args.front_ends if f != "none"]

    root = Path(args.data)
    dataset_name = get_dataset_name(args.data)
    project_root = get_project_root()
    artifacts_root = project_root / "artifacts" / "experiments" / dataset_name
    
    with open(root / "manifest_degraded.csv", newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] in args.splits]
    by_utt = defaultdict(list)
    for r in rows:
        by_utt[r["utt_id"]].append(r)
    utts = [u for u, rs in by_utt.items() if normalize(rs[0]["transcript"])][: args.max_utts]
    if not utts:
        sys.exit("No usable utterances.")
    snrs = {float(s) for s in args.snrs}

    # utterances whose enhanced files are missing are dropped (and reported), not silently scored
    def noisy_rows(u):
        return [r for r in by_utt[u] if r["snr_db"] != "" and float(r["snr_db"]) in snrs]

    missing = defaultdict(int)
    keep = []
    for u in utts:
        ok = True
        for fe in fes:
            for r in noisy_rows(u):
                if not (artifacts_root / "enh" / fe / f"{r['deg_id']}.wav").exists():
                    missing[fe] += 1
                    ok = False
        if ok and noisy_rows(u):
            keep.append(u)
    for fe, k in missing.items():
        print(f"WARNING: {k} enhanced files missing for '{fe}' -> utterances with missing files are skipped "
              f"(run denoise.py first / finish all shards)")
    utts = keep
    if not utts:
        sys.exit("Nothing to evaluate: no utterance has all its enhanced files.")

    def items_for(ulist):
        """(cache key, utt, front_end, condition, noise_group, path)"""
        out = []
        for u in ulist:
            r0 = by_utt[u][0]
            out.append((f"{u}::clean", u, "none", "clean", "", root / r0["clean_path"]))
            out.append((f"{u}::clean8k", u, "none", "clean8k", "", root / r0["clean8k_path"]))
            for r in by_utt[u]:
                if r["snr_db"] == "":
                    out.append((r["deg_id"], u, "none", "codec_only", noise_group(r), root / r["degraded_path"]))
            for r in noisy_rows(u):
                c, g = cond_label(r), noise_group(r)
                out.append((r["deg_id"], u, "none", c, g, root / r["degraded_path"]))
                for fe in fes:
                    out.append((f"{r['deg_id']}@{fe}", u, fe, c, g, artifacts_root / "enh" / fe / f"{r['deg_id']}.wav"))
        return out

    # caches: transcripts of the degraded audio (eval_asr.py) + of enhanced audio (this script)
    cache = {}
    asr_dir = artifacts_root / "asr" / args.model
    files = [asr_dir / f"asr_{args.model}.jsonl"]
    files += [Path(p) for p in sorted(glob.glob(str(asr_dir / f"asr_{args.model}_shard*.jsonl")))]
    files += [Path(p) for p in sorted(glob.glob(str(asr_dir / f"asr_{args.model}_enh*.jsonl")))]
    for fp in files:
        if fp.exists():
            for line in open(fp, encoding="utf-8"):
                if line.strip():
                    d = json.loads(line)
                    cache[d["key"]] = d["hyp"]

    # ---------------- transcription
    if not args.merge:
        mine = utts[args.shard[0]:: args.shard[1]] if args.shard else utts
        items = items_for(mine)
        todo = [it for it in items if it[0] not in cache]
        out_name = (f"asr_{args.model}_enh_shard{args.shard[0]}of{args.shard[1]}.jsonl" if args.shard
                    else f"asr_{args.model}_enh.jsonl")
        n_enh = sum(1 for it in todo if it[2] != "none")
        print(f"{len(mine)} utterances, {len(items)} audio files, {len(todo)} to transcribe "
              f"({n_enh} enhanced, {len(todo) - n_enh} degraded/clean that are not cached yet)")
        if todo:
            transcribe = get_transcriber(args)
            t0 = time.time()
            with open(asr_dir / out_name, "a", encoding="utf-8") as out:
                for i, (key, u, fe, c, g, path) in enumerate(todo, 1):
                    hyp = transcribe(load_16k(str(path)))
                    cache[key] = hyp
                    out.write(json.dumps({"key": key, "hyp": hyp}, ensure_ascii=False) + "\n")
                    out.flush()
                    if i % 20 == 0 or i == len(todo):
                        el = time.time() - t0
                        print(f"  {i}/{len(todo)}  ({el / i:.1f}s each, ~{el / i * (len(todo) - i) / 60:.0f} min left)",
                              flush=True)
        if args.shard:
            print(f"\nshard done -> {asr_dir / out_name}\nCopy all asr_{args.model}_enh_shard*.jsonl into one "
                  f"artifacts folder, then run with --merge.")
            return

    # ---------------- scoring: only utterances with every item transcribed
    all_items = items_for(utts)
    by_item_utt = defaultdict(list)
    for it in all_items:
        by_item_utt[it[1]].append(it)
    complete = [u for u in utts if all(it[0] in cache for it in by_item_utt[u])]
    print(f"\n{len(complete)} of {len(utts)} utterances fully transcribed"
          + ("" if len(complete) == len(utts) else "  (the rest are NOT scored)"))
    if not complete:
        sys.exit("Nothing to score yet.")
    done = set(complete)

    # per (front_end, condition): utt -> (edits, nref);  per (front_end, group): utt -> summed
    per_cond = defaultdict(dict)
    per_group = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for key, u, fe, c, g, path in all_items:
        if u not in done:
            continue
        ref = normalize(by_utt[u][0]["transcript"]).split()
        hyp = normalize(cache[key]).split()
        e, n = edit_distance(ref, hyp), len(ref)
        per_cond[(fe, c)][u] = (e, n)
        if c.startswith("snr_"):
            acc = per_group[(fe, g)][u]
            acc[0] += e
            acc[1] += n

    snr_conds = [f"snr_{s:g}" for s in sorted(snrs, reverse=True)]
    cond_rows = [c for c in ["clean", "clean8k", "codec_only"] if ("none", c) in per_cond] + \
                [c for c in snr_conds if ("none", c) in per_cond]
    summary = []
    codec_wer = wer_of(list(per_cond[("none", "codec_only")].values())) if ("none", "codec_only") in per_cond else None

    print(f"\nWER (%) on {len(complete)} utterances, model={args.model}.   Δ = front-end minus none, paired 95% CI, "
          f"* = CI excludes 0")
    head = f"{'condition':<11}{'none':>8}"
    for fe in fes:
        head += f"   {fe:<9}{'WER':>5}  {'Δ vs none [95% CI]':<22}{'gap closed':>10}"
    print(head)
    for c in cond_rows:
        base = per_cond[("none", c)]
        us = sorted(base)
        w0, lo0, hi0 = boot_ci([base[u] for u in us])
        summary.append(dict(group="condition", front_end="none", name=c, wer=f"{w0:.1f}", ci_lo=f"{lo0:.1f}",
                            ci_hi=f"{hi0:.1f}", delta="", d_lo="", d_hi="", gap_closed_pct="", n_utts=len(us)))
        line = f"{c:<11}{w0:>8.1f}"
        for fe in fes:
            if (fe, c) not in per_cond:
                line += f"   {fe:<9}{'-':>5}"
                continue
            fb = per_cond[(fe, c)]
            a = [base[u] for u in us]
            b = [fb[u] for u in us]
            w, lo, hi = boot_ci(b)
            d, dlo, dhi = paired_delta(a, b)
            gc = ""
            if codec_wer is not None and c.startswith("snr_") and (w0 - codec_wer) >= 2.0:
                gc = f"{100 * (w0 - w) / (w0 - codec_wer):.0f}%"
            elif c.startswith("snr_"):
                gc = "n/a"
            summary.append(dict(group="condition", front_end=fe, name=c, wer=f"{w:.1f}", ci_lo=f"{lo:.1f}",
                                ci_hi=f"{hi:.1f}", delta=f"{d:.2f}", d_lo=f"{dlo:.2f}", d_hi=f"{dhi:.2f}",
                                gap_closed_pct=gc.rstrip("%") if gc != "n/a" else "", n_utts=len(us)))
            line += f"   {fe:<9}{w:>5.1f}  {fmt_delta(d, dlo, dhi):<22}{gc:>10}"
        print(line)
    if codec_wer is not None:
        print("(gap closed = (WER_none - WER_frontend) / (WER_none - WER_codec_only); n/a when the noise "
              "penalty is under 2 points)")

    groups = sorted({g for (fe, g) in per_group if fe == "none"})
    if groups:
        print("\nBy noise group, pooled over the SNRs above (each utterance has one noise file):")
        head = f"{'group':<14}{'n_utts':>6}{'none':>8}"
        for fe in fes:
            head += f"   {fe:<9}{'WER':>5}  {'Δ vs none [95% CI]':<22}"
        print(head)
        for g in groups:
            base = per_group[("none", g)]
            us = sorted(base)
            a = [tuple(base[u]) for u in us]
            w0, lo0, hi0 = boot_ci(a)
            summary.append(dict(group="noise_group", front_end="none", name=g, wer=f"{w0:.1f}", ci_lo=f"{lo0:.1f}",
                                ci_hi=f"{hi0:.1f}", delta="", d_lo="", d_hi="", gap_closed_pct="", n_utts=len(us)))
            line = f"{g:<14}{len(us):>6}{w0:>8.1f}"
            for fe in fes:
                fb = per_group.get((fe, g))
                if not fb:
                    line += f"   {fe:<9}{'-':>5}"
                    continue
                b = [tuple(fb[u]) for u in us]
                w, lo, hi = boot_ci(b)
                d, dlo, dhi = paired_delta(a, b)
                summary.append(dict(group="noise_group", front_end=fe, name=g, wer=f"{w:.1f}", ci_lo=f"{lo:.1f}",
                                    ci_hi=f"{hi:.1f}", delta=f"{d:.2f}", d_lo=f"{dlo:.2f}", d_hi=f"{dhi:.2f}",
                                    gap_closed_pct="", n_utts=len(us)))
                line += f"   {fe:<9}{w:>5.1f}  {fmt_delta(d, dlo, dhi):<22}"
            print(line)
        if len(groups) <= 3:
            print("(few utterances per group: treat group differences as indicative only)")

    spath = artifacts_root / "asr" / args.model / f"enh_summary_{args.model}.csv"
    with open(spath, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["group", "front_end", "name", "wer", "ci_lo", "ci_hi", "delta",
                                           "d_lo", "d_hi", "gap_closed_pct", "n_utts"])
        wr.writeheader()
        wr.writerows(summary)
    print(f"\nsaved {spath}")


if __name__ == "__main__":
    main()
