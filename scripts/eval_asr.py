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
         pip install groq            # only for --backend groq (key in GROQ_API_KEY or .env)
Run:     python eval_asr.py --data data                 # test split, first 100 utterances
         python eval_asr.py --data data --model medium  # bigger model (needs more RAM)

Only dev/test utterances have the full SNR grid, so use --splits dev or test.

BACKENDS
    --backend local   (default) faster-whisper on this machine, default model "small".
    --backend groq    Groq Cloud Whisper, default model "whisper-large-v3".
                      Uses the SAME 16 kHz front end as local (load_16k, then an in-memory WAV),
                      so Groq's own resampler never touches your 8 kHz audio. The client paces
                      itself to the account limits (defaults = free plan: 20 requests/min,
                      7,200 audio-s/hour, 2,000 requests/day, 28,800 audio-s/day; requests
                      under 10 s are billed as 10 s). Override with --groq-rpm/--groq-ash/
                      --groq-rpd/--groq-asd if your limits page says otherwise. On a 429 it
                      waits for retry-after; if the wait is longer than --groq-max-wait (a daily
                      cap) it stops cleanly and you re-run later: the cache resumes where it left.
                      Caveats: Groq exposes no beam-size / VAD / condition_on_previous_text
                      controls, and its models differ from local ones, so WER is NOT comparable
                      with a local run. Cache and summary files are named by model, so they
                      never mix.

SHARDING (several machines, one slice each; for local models)
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
    (Sharding does NOT help with --backend groq: Groq limits are per organisation.)
"""
import argparse
import csv
import io
import json
import math
import os
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
GROQ_MIN_BILLED_S = 10.0  # Groq bills any request shorter than this as this many seconds
DEFAULT_MODEL = {"local": "small", "groq": "whisper-large-v3"}


class RateLimitStop(Exception):
    """Groq asked us to wait longer than --groq-max-wait (almost certainly a daily cap)."""


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


def wav_bytes_16k(x):
    """float32 16 kHz mono -> in-memory PCM16 WAV (what we upload to Groq)."""
    buf = io.BytesIO()
    sf.write(buf, np.clip(x, -1.0, 1.0), 16000, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def get_local_transcriber(args):
    from faster_whisper import WhisperModel
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)

    def transcribe(audio):
        segs, _ = model.transcribe(audio, language="hi", beam_size=1,
                                   condition_on_previous_text=False, vad_filter=False)
        return " ".join(s.text.strip() for s in segs)

    return transcribe


def get_groq_transcriber(args):
    try:
        import groq
    except ImportError:
        sys.exit("--backend groq needs the SDK:  pip install groq")
    if not os.environ.get("GROQ_API_KEY"):
        sys.exit("GROQ_API_KEY is not set (export it, or put it in a .env file next to this script).")

    # max_retries=0: the SDK's own silent retries would burn requests against the quota;
    # we handle 429s ourselves below.
    client = groq.Groq(max_retries=0)
    rpm_eff = args.groq_rpm * args.groq_margin
    ash_eff = args.groq_ash * args.groq_margin
    state = {"next": 0.0}

    def transcribe(audio):
        billed = max(GROQ_MIN_BILLED_S, len(audio) / 16000)
        interval = max(60.0 / rpm_eff, billed * 3600.0 / ash_eff)  # seconds between request starts
        wait = state["next"] - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        state["next"] = time.monotonic() + interval

        data = wav_bytes_16k(audio)
        for attempt in range(8):
            try:
                r = client.audio.transcriptions.create(
                    file=("audio.wav", data), model=args.model, language="hi",
                    temperature=0.0, response_format="json")
                return (r.text or "").strip()
            except groq.RateLimitError as e:
                ra = None
                try:
                    ra = float(e.response.headers.get("retry-after"))
                except (TypeError, ValueError, AttributeError):
                    pass
                wait = ra if ra is not None else 30.0 * (attempt + 1)
                if wait > args.groq_max_wait:
                    raise RateLimitStop(wait)
                print(f"    429 from Groq, waiting {wait:.0f}s ...", flush=True)
                time.sleep(wait + 1.0)
                state["next"] = time.monotonic() + interval
            except groq.APIConnectionError:
                time.sleep(min(2 ** attempt, 60))
            except groq.APIStatusError as e:
                if e.status_code >= 500:
                    time.sleep(min(2 ** attempt, 60))
                else:
                    raise
        raise RuntimeError("Groq request kept failing after 8 attempts")

    return transcribe


def get_transcriber(args):
    return get_groq_transcriber(args) if args.backend == "groq" else get_local_transcriber(args)


def groq_budget_report(todo, args):
    """Estimate quota use and wall time for the files still to transcribe (Groq billing rules)."""
    billed = []
    for it in todo:
        try:
            d = sf.info(str(it[4])).duration
        except Exception:
            d = GROQ_MIN_BILLED_S
        billed.append(max(GROQ_MIN_BILLED_S, d))
    n, tot = len(billed), sum(billed)
    rpm_eff = args.groq_rpm * args.groq_margin
    ash_eff = args.groq_ash * args.groq_margin
    hours = max(n / (rpm_eff * 60), tot / ash_eff)
    days = max(n / args.groq_rpd, tot / args.groq_asd)
    print(f"Groq budget: {n} requests, {tot / 3600:.2f} billed audio-hours "
          f"(10 s minimum per request); paced at ~{max(60 / rpm_eff, (tot / n) * 3600 / ash_eff):.1f}s/request "
          f"-> ~{hours:.1f} h of wall time")
    if days > 1:
        print(f"  WARNING: that exceeds the daily caps ({args.groq_rpd} requests / {args.groq_asd} audio-s): "
              f"expect ~{math.ceil(days)} days. The run stops when Groq asks for a long wait; "
              f"re-run later and it resumes from the cache.")


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
    shards_dir = root / "artifacts" / "shards"
    return sorted(shards_dir.glob(f"asr_{model}.shard*of*.jsonl"))


# ------------------------------------------------------------------ scoring
def score(root, artifacts_dir, args, items, utts, by_utt, cache):
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

    spath = artifacts_dir / f"asr_summary_{args.model}.csv"
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
    ap.add_argument("--backend", choices=["local", "groq"], default="local")
    ap.add_argument("--model", default=None,
                    help="default: 'small' (local) or 'whisper-large-v3' (groq)")
    ap.add_argument("--device", default="cpu", help="local backend only")
    ap.add_argument("--compute-type", default="int8", help="local backend only")
    ap.add_argument("--groq-rpm", type=float, default=20, help="requests/min limit (free plan: 20)")
    ap.add_argument("--groq-ash", type=float, default=7200, help="audio-seconds/hour limit (free plan: 7200)")
    ap.add_argument("--groq-rpd", type=float, default=2000, help="requests/day limit, for the estimate only")
    ap.add_argument("--groq-asd", type=float, default=28800, help="audio-seconds/day limit, for the estimate only")
    ap.add_argument("--groq-margin", type=float, default=0.9,
                    help="pace at this fraction of the minute/hour limits (default 0.9)")
    ap.add_argument("--groq-max-wait", type=float, default=1800,
                    help="if Groq says wait longer than this many seconds, stop cleanly (default 1800)")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="total number of machines/slices (default 1 = no sharding)")
    ap.add_argument("--shard-id", type=int, default=0,
                    help="which slice this run handles, 0 .. num-shards-1")
    ap.add_argument("--merge", action="store_true",
                    help="combine all shard files in --data and score; no transcription")
    args = ap.parse_args()
    if args.model is None:
        args.model = DEFAULT_MODEL[args.backend]

    if args.num_shards < 1 or not (0 <= args.shard_id < args.num_shards):
        sys.exit("--shard-id must satisfy 0 <= shard-id < num-shards")
    if args.merge and args.num_shards > 1:
        sys.exit("--merge combines all shards itself; don't pass --num-shards with it")
    if not (0 < args.groq_margin <= 1):
        sys.exit("--groq-margin must be in (0, 1]")
    if args.backend == "groq" and args.num_shards > 1 and not args.merge:
        print("note: Groq limits are per organisation, so sharding will not speed this up; "
              "running shards in parallel will just hit 429s sooner.")

    root = Path(args.data)
    artifacts_dir = root / "artifacts"
    shards_dir = artifacts_dir / "shards"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    shards_dir.mkdir(parents=True, exist_ok=True)
    
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

    main_cache_path = artifacts_dir / f"asr_{args.model}.jsonl"

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
        score(root, artifacts_dir, args, build_items(complete), complete, by_utt, cache)
        return

    # ---------------------------------------------------------- transcribe (whole set or one shard)
    sharded = args.num_shards > 1
    my_utts = utts[args.shard_id::args.num_shards] if sharded else utts
    items = build_items(my_utts)

    if sharded:
        cache_path = shards_dir / f"asr_{args.model}.shard{args.shard_id}of{args.num_shards}.jsonl"
        cache = read_jsonl_cache(main_cache_path)      # anything already merged counts as done
        cache.update(read_jsonl_cache(cache_path))     # plus this shard's own progress
        print(f"shard {args.shard_id}/{args.num_shards}: {len(my_utts)} of {len(utts)} utterances")
    else:
        cache_path = main_cache_path
        cache = read_jsonl_cache(cache_path)

    todo = [it for it in items if it[0] not in cache]
    print(f"{len(my_utts)} utterances, {len(items)} audio files, {len(todo)} still to transcribe "
          f"[backend={args.backend}, model={args.model}]")

    stopped = False
    if todo:
        if args.backend == "groq":
            groq_budget_report(todo, args)
        transcribe = get_transcriber(args)
        t0 = time.time()
        with open(cache_path, "a", encoding="utf-8") as out:
            for i, (key, u, cond, nt, path) in enumerate(todo, 1):
                try:
                    hyp = transcribe(load_16k(str(path)))
                except RateLimitStop as e:
                    print(f"\nGroq asked for a {float(e.args[0]) / 60:.0f} min wait (> --groq-max-wait): "
                          f"daily quota is probably spent. Stopping; {i - 1} of {len(todo)} done this run.")
                    stopped = True
                    break
                cache[key] = hyp
                out.write(json.dumps({"key": key, "hyp": hyp}, ensure_ascii=False) + "\n")
                out.flush()
                if i % 20 == 0 or i == len(todo):
                    el = time.time() - t0
                    print(f"  {i}/{len(todo)}  ({el / i:.1f}s each, ~{el / i * (len(todo) - i) / 60:.0f} min left)")

    if stopped:
        print("Progress is saved. Re-run the same command later to resume; scoring happens once everything is done.")
        return

    if sharded:
        print(f"\nshard {args.shard_id}/{args.num_shards} done -> {cache_path}")
        print("Copy all shard files into one data folder, then run:  "
              f"python eval_asr.py --data {args.data} --backend {args.backend} --model {args.model} "
              f"--splits {' '.join(args.splits)} --max-utts {args.max_utts} --merge")
        return

    score(root, artifacts_dir, args, items, my_utts, by_utt, cache)
    print(f"raw hypotheses in {cache_path}")


if __name__ == "__main__":
    main()
