#!/usr/bin/env python3
"""
Gate 1: how much does the phone channel hurt ASR?

Transcribes, for each selected utterance, a ladder of conditions and reports WER for each:
    clean        original 16 kHz audio
    clean8k      same audio squeezed to 8 kHz only (isolates the bandwidth loss)
    codec_only   8 kHz + AMR-NB, no added noise (isolates the codec)
    snr_20 ... snr_0   noise at that SNR + 8 kHz + AMR-NB

Reads  <data>/manifest_degraded.csv   (from degrade.py)
Writes <data>/asr_<model>.jsonl       raw hypotheses (cache: re-runs skip finished items)
       <data>/asr_summary_<model>.csv per-condition WER with 95% bootstrap CI

Setup:   pip install faster-whisper scipy soundfile numpy python-dotenv
Run:     python eval_asr.py --data data                 # test split, first 100 utterances
         python eval_asr.py --data data --model medium  # bigger model (needs more RAM)

Only dev/test utterances have the full SNR grid, so use --splits dev or test.

SHARDING (several machines, one slice each)
    Every machine needs the same manifest and the SAME --splits / --max-utts / --model,
    because the utterance list is cut into slices by position. Then:

      machine 0:  python eval_asr.py --data data --num-shards 3 --shard-id 0
      machine 1:  python eval_asr.py --data data --num-shards 3 --shard-id 1
      machine 2:  python eval_asr.py --data data --num-shards 3 --shard-id 2

    Each writes its own  <data>/asr_<model>.shard<i>of<N>.jsonl  and does NOT score.
    Copy all shard files into one <data> folder, then merge + score (no model needed):

      python eval_asr.py --data data --merge

    Merge folds every shard file into asr_<model>.jsonl and scores. If some utterances are
    incomplete (a shard unfinished), they are dropped from scoring with a warning so every
    condition is still computed on the same utterances. Shards are resumable: re-running a
    shard skips whatever it already finished.
"""
import argparse
import csv
import json
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from dotenv import load_dotenv

load_dotenv()

COND_ORDER = ["clean", "clean8k", "codec_only", "snr_20", "snr_15", "snr_10", "snr_5", "snr_0"]


# ------------------------------------------------------------------ text normalisation
def normalize(text):
    """THE transcript convention. Reference and hypothesis both go through this.
    Edit it once, together with your teammates, and never per-experiment.
    Currently: NFC, lower-case Latin, drop punctuation and the danda, keep letters / combining
    marks / digits, Devanagari digits -> ASCII digits, collapse spaces.
    NOT handled yet: whether English words are written in Latin or Devanagari -- decide that
    rule and add it here (it can move WER by many points)."""
    out = []
    for c in unicodedata.normalize("NFC", text).lower():
        cat = unicodedata.category(c)
        if cat == "Nd":
            out.append(str(unicodedata.digit(c)))
        elif cat[0] in "LM":  # letters and combining marks (Devanagari matras are 'M')
            out.append(c)
        elif cat == "Cf":  # zero-width joiners etc.: delete, don't split the word
            continue
        else:
            out.append(" ")
    return " ".join("".join(out).split())


def edit_distance(ref, hyp):
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, w in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != w))
        prev = cur
    return prev[-1]


def corpus_wer(pairs, boot=1000, seed=0):
    """pairs: list of (edits, ref_len) per utterance -> (WER, ci_lo, ci_hi) in %"""
    e = np.array([p[0] for p in pairs], dtype=float)
    n = np.array([p[1] for p in pairs], dtype=float)
    wer = 100 * e.sum() / n.sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(e), size=(boot, len(e)))
    b = 100 * e[idx].sum(axis=1) / n[idx].sum(axis=1)
    return wer, float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


# ------------------------------------------------------------------ audio / ASR
def load_16k(path):
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr == 8000:
        x = resample_poly(x, 2, 1).astype(np.float32)  # what a 16 kHz ASR front end will see
    elif sr != 16000:
        raise ValueError(f"unexpected sample rate {sr} in {path}")
    return x


def get_transcriber(args):
    from faster_whisper import WhisperModel
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)

    def transcribe(audio):
        segs, _ = model.transcribe(audio, language="hi", beam_size=1,
                                   condition_on_previous_text=False, vad_filter=False)
        return " ".join(s.text.strip() for s in segs)

    return transcribe


def cond_label(row):
    return "codec_only" if row["snr_db"] == "" else f"snr_{float(row['snr_db']):g}"


# ------------------------------------------------------------------ cache helpers
def read_jsonl_cache(path):
    """key -> hypothesis. Tolerates a truncated last line (e.g. a job killed mid-write)."""
    cache = {}
    if not path.exists():
        return cache
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            cache[d["key"]] = d["hyp"]
    return cache


def shard_files(root, model):
    return sorted(root.glob(f"asr_{model}.shard*of*.jsonl"))


# ------------------------------------------------------------------ scoring
def score(root, args, items, utts, by_utt, cache):
    per_cond = defaultdict(list)  # cond -> [(edits, nref)]
    per_type = defaultdict(list)  # noise type (noisy conditions only) -> [(edits, nref)]
    for key, u, cond, nt, path in items:
        ref = normalize(by_utt[u][0]["transcript"]).split()
        hyp = normalize(cache[key]).split()
        pair = (edit_distance(ref, hyp), len(ref))
        per_cond[cond].append(pair)
        if cond.startswith("snr_"):
            per_type[nt].append(pair)

    summary = []
    print(f"\nWER (%) on {len(utts)} utterances, model={args.model}   [95% bootstrap CI over utterances]")
    print(f"{'condition':<12}{'WER':>7}   {'CI':<15}{'n':>5}")
    for c in [c for c in COND_ORDER if c in per_cond]:
        w, lo, hi = corpus_wer(per_cond[c])
        summary.append(dict(group="condition", name=c, wer=f"{w:.1f}", ci_lo=f"{lo:.1f}", ci_hi=f"{hi:.1f}",
                            n_utts=len(per_cond[c])))
        print(f"{c:<12}{w:>7.1f}   [{lo:5.1f}, {hi:5.1f}] {len(per_cond[c]):>5}")
    if per_type:
        print("\nNoisy conditions pooled over all SNRs, by noise type:")
        for nt, pairs in sorted(per_type.items()):
            w, lo, hi = corpus_wer(pairs)
            summary.append(dict(group="noise_type", name=nt, wer=f"{w:.1f}", ci_lo=f"{lo:.1f}",
                                ci_hi=f"{hi:.1f}", n_utts=len(pairs)))
            print(f"{nt:<12}{w:>7.1f}   [{lo:5.1f}, {hi:5.1f}] {len(pairs):>5}")

    spath = root / f"asr_summary_{args.model}.csv"
    with open(spath, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["group", "name", "wer", "ci_lo", "ci_hi", "n_utts"])
        wr.writeheader()
        wr.writerows(summary)
    print(f"\nsaved {spath}")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--splits", nargs="+", default=["test"])
    ap.add_argument("--max-utts", type=int, default=100)
    ap.add_argument("--model", default="small")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--compute-type", default="int8")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="total number of machines/slices (default 1 = no sharding)")
    ap.add_argument("--shard-id", type=int, default=0,
                    help="which slice this run handles, 0 .. num-shards-1")
    ap.add_argument("--merge", action="store_true",
                    help="combine all shard files in --data and score; no transcription")
    args = ap.parse_args()

    if args.num_shards < 1 or not (0 <= args.shard_id < args.num_shards):
        sys.exit("--shard-id must satisfy 0 <= shard-id < num-shards")
    if args.merge and args.num_shards > 1:
        sys.exit("--merge combines all shards itself; don't pass --num-shards with it")

    root = Path(args.data)
    with open(root / "manifest_degraded.csv", newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] in args.splits]

    by_utt = defaultdict(list)
    for r in rows:
        by_utt[r["utt_id"]].append(r)
    utts = []
    for u, rs in by_utt.items():
        if normalize(rs[0]["transcript"]):
            utts.append(u)
    utts = utts[: args.max_utts]  # truncate BEFORE sharding so every machine agrees on the list
    if not utts:
        sys.exit("No usable utterances in the chosen splits.")

    def build_items(utt_list):
        # work items: (cache key, utt, condition, noise_type, audio path)
        out = []
        for u in utt_list:
            r0 = by_utt[u][0]
            out.append((f"{u}::clean", u, "clean", "", root / r0["clean_path"]))
            out.append((f"{u}::clean8k", u, "clean8k", "", root / r0["clean8k_path"]))
            for r in by_utt[u]:
                nt = "" if r["snr_db"] == "" else r["noise_type"]
                out.append((r["deg_id"], u, cond_label(r), nt, root / r["degraded_path"]))
        return out

    main_cache_path = root / f"asr_{args.model}.jsonl"

    # ---------------------------------------------------------- merge mode: no transcription
    if args.merge:
        cache = read_jsonl_cache(main_cache_path)
        files = shard_files(root, args.model)
        if not files and not cache:
            sys.exit(f"No shard files (asr_{args.model}.shard*of*.jsonl) or cache found in {root}")
        for p in files:
            part = read_jsonl_cache(p)
            print(f"  {p.name}: {len(part)} hypotheses")
            cache.update(part)

        # persist the merged cache so later plain runs / re-merges see everything
        with open(main_cache_path, "w", encoding="utf-8") as out:
            for k, h in cache.items():
                out.write(json.dumps({"key": k, "hyp": h}, ensure_ascii=False) + "\n")
        print(f"merged cache: {len(cache)} hypotheses -> {main_cache_path}")

        # score only utterances whose full condition ladder is present
        complete = [u for u in utts if all(it[0] in cache for it in build_items([u]))]
        dropped = len(utts) - len(complete)
        if dropped:
            print(f"WARNING: {dropped} of {len(utts)} utterances are incomplete "
                  f"(missing shard output?) and are excluded from scoring.")
        if not complete:
            sys.exit("Nothing complete to score.")
        score(root, args, build_items(complete), complete, by_utt, cache)
        return

    # ---------------------------------------------------------- transcribe (whole set or one shard)
    sharded = args.num_shards > 1
    my_utts = utts[args.shard_id::args.num_shards] if sharded else utts
    items = build_items(my_utts)

    if sharded:
        cache_path = root / f"asr_{args.model}.shard{args.shard_id}of{args.num_shards}.jsonl"
        cache = read_jsonl_cache(main_cache_path)      # anything already merged counts as done
        cache.update(read_jsonl_cache(cache_path))     # plus this shard's own progress
        print(f"shard {args.shard_id}/{args.num_shards}: {len(my_utts)} of {len(utts)} utterances")
    else:
        cache_path = main_cache_path
        cache = read_jsonl_cache(cache_path)

    todo = [it for it in items if it[0] not in cache]
    print(f"{len(my_utts)} utterances, {len(items)} audio files, {len(todo)} still to transcribe")

    if todo:
        transcribe = get_transcriber(args)
        t0 = time.time()
        with open(cache_path, "a", encoding="utf-8") as out:
            for i, (key, u, cond, nt, path) in enumerate(todo, 1):
                hyp = transcribe(load_16k(str(path)))
                cache[key] = hyp
                out.write(json.dumps({"key": key, "hyp": hyp}, ensure_ascii=False) + "\n")
                out.flush()
                if i % 20 == 0 or i == len(todo):
                    el = time.time() - t0
                    print(f"  {i}/{len(todo)}  ({el / i:.1f}s each, ~{el / i * (len(todo) - i) / 60:.0f} min left)")

    if sharded:
        print(f"\nshard {args.shard_id}/{args.num_shards} done -> {cache_path}")
        print("Copy all shard files into one data folder, then run:  "
              f"python eval_asr.py --data {args.data} --model {args.model} "
              f"--splits {' '.join(args.splits)} --max-utts {args.max_utts} --merge")
        return

    score(root, args, items, my_utts, by_utt, cache)
    print(f"raw hypotheses in {cache_path}")


if __name__ == "__main__":
    main()
