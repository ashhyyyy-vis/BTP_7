#!/usr/bin/env python3
"""
Simulate an 8 kHz AMR-NB phone call from clean 16 kHz speech.

Per utterance and condition:
  clean 16k  ->  downsample to 8k  ->  + noise at target SNR (measured at 8k)
             ->  AMR-NB encode/decode (ffmpeg)  ->  time-align to clean  ->  degraded 8k WAV

Reads  <data>/manifest.csv   (written by extract_indicvoices.py)
Writes <data>/clean8k/*.wav       clean reference at 8 kHz (target for denoiser / metrics)
       <data>/degraded/*.wav      degraded 8 kHz audio, aligned and same length as clean8k
       <data>/manifest_degraded.csv   one row per degraded file, full recipe logged

Setup
-----
  pip install numpy scipy soundfile
  ffmpeg must have the AMR-NB *encoder*:   ffmpeg -encoders | grep -i amr
  (look for libopencore_amrnb; if it is missing, install an ffmpeg build that includes it)

Quick start (no noise download needed: synthetic white/pink/brown/babble noise)
  python degrade.py --data data --limit 20          # smoke test
  python degrade.py --data data                     # full run

With real noise (MUSAN noise folder, DEMAND, ...):
  python degrade.py --data data --noise-dir /path/to/musan/noise
  python degrade.py --data data --noise-dir /path/to/demand --noise-group-by dir

Design choices (all logged per row in the manifest)
  * dev/test utterances get the FULL GRID: every SNR (plus a codec-only row), same noise
    segment at every SNR, so results can be compared per utterance.
  * train utterances get --train-copies random conditions (SNR drawn uniformly in range).
  * Noise FILES are split train/dev/test by hashing, so test noise is never seen in training.
  * SNR is measured in the 8 kHz band, against speech-active power (frames within 40 dB of
    the loudest frame), i.e. what the codec actually receives.
  * The codec adds delay; it is measured by cross-correlation and removed, so degraded and
    clean8k line up sample-for-sample (needed for PESQ/STOI/SI-SDR and for training pairs).
  * Every row's randomness comes from a seed derived from (--seed, ids), so results do not
    depend on worker count or processing order.
"""
import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from functools import lru_cache
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import correlate, resample_poly

SR = 16000
SR_NB = 8000
FRAME = 320  # 20 ms at 16 kHz

OUT_COLS = [
    "deg_id", "utt_id", "split", "speaker_id",
    "clean_path", "clean8k_path", "degraded_path",
    "noise_type", "noise_file", "snr_db", "amr_bitrate", "seed", "delay_samples",
    "transcript", "duration_s", "scenario",
]

CFG = {}


# ----------------------------------------------------------------------------- helpers
def h(s):
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)


def derived_seed(base, key):
    return h(f"{base}:{key}") % (2 ** 32)


def split_of_key(key):
    v = h(key) % 100  # same rule as the extraction script
    return "train" if v < 80 else ("dev" if v < 90 else "test")


def _read16k(path):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        g = math.gcd(SR, sr)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    return x


@lru_cache(maxsize=128)
def load_audio16k(path):
    """speech clips: short, so a big cache is cheap"""
    return _read16k(path)


@lru_cache(maxsize=3)
def load_noise16k(path):
    """noise recordings can be minutes long (DEMAND ~5 min = ~19 MB each), so keep the cache tiny:
    128 of them per worker exhausted RAM on an 8 GB machine"""
    return _read16k(path)


def active_power(x, frame=FRAME):
    n = len(x) // frame
    if n == 0:
        return float(np.mean(x ** 2)) + 1e-12
    p = np.mean(x[: n * frame].reshape(n, frame) ** 2, axis=1)
    act = p[p > p.max() * 1e-4]
    return (float(act.mean()) if act.size else float(p.mean())) + 1e-12


def crop_or_tile(x, n, rng):
    if len(x) == 0:
        return np.zeros(n, dtype=np.float32)
    if len(x) < n:
        x = np.tile(x, int(math.ceil(n / len(x))))
    off = int(rng.integers(0, len(x) - n + 1))
    return x[off: off + n]


def colored_noise(n, beta, rng):
    """power spectrum ~ 1/f^beta : 0 white, 1 pink, 2 brown"""
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n)
    f[0] = f[1] if len(f) > 1 else 1.0
    x = np.fft.irfft(spec / (f ** (beta / 2.0)), n)
    return (x / (np.std(x) + 1e-12)).astype(np.float32)


def babble(n, pool, own_path, rng, k=5):
    cands = [p for p in pool if p != own_path]
    if not cands:
        return colored_noise(n, 1.0, rng)
    picks = rng.choice(len(cands), size=min(k, len(cands)), replace=False)
    mix = np.zeros(n, dtype=np.float64)
    for i in picks:
        x = crop_or_tile(load_audio16k(cands[i]), n, rng)
        mix += x / np.sqrt(active_power(x))
    return mix.astype(np.float32)


def estimate_delay(ref, deg, max_lag=800):
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


def run_ffmpeg(args):
    p = subprocess.run(["ffmpeg", "-v", "error", "-y"] + args, capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode(errors="ignore"))


def amr_roundtrip(x8, bitrate):
    with tempfile.TemporaryDirectory() as td:
        a, b, c = f"{td}/in.wav", f"{td}/x.amr", f"{td}/out.wav"
        sf.write(a, x8, SR_NB, subtype="PCM_16")
        run_ffmpeg(["-i", a, "-ac", "1", "-ar", "8000", "-c:a", "libopencore_amrnb",
                    "-b:a", str(bitrate), b])
        run_ffmpeg(["-i", b, "-ac", "1", "-ar", "8000", "-c:a", "pcm_s16le", c])
        y, sr = sf.read(c, dtype="float32")
    assert sr == SR_NB
    return y


# ----------------------------------------------------------------------------- worker
def init_worker(cfg):
    CFG.update(cfg)


def process(job):
    out = Path(CFG["out"])
    row = job["row"]
    clean_abs = str(out / row["clean_path"])
    clean = load_audio16k(clean_abs)

    # clean 8 kHz reference (written up front); mixing happens here, in the telephone band,
    # so the SNR label is the SNR the codec actually sees.
    clean8 = sf.read(str(out / row["clean8k_path"]), dtype="float32")[0]

    nt, snr = job["noise_type"], job["snr_db"]
    noise_name = ""
    if nt == "none":
        x8 = clean8.copy()
    else:
        nrng = np.random.default_rng(derived_seed(CFG["seed"], job["noise_key"] + ":noise"))
        n = len(clean)
        if nt == "file":
            pool = CFG["noise_pools"][row["split"]]
            path = pool[int(nrng.integers(len(pool)))]
            nz = crop_or_tile(load_noise16k(path), n, nrng)
            noise_name = str(Path(path).relative_to(CFG["noise_root"]))
        elif nt == "babble":
            nz = babble(n, CFG["clean_pools"][row["split"]], clean_abs, nrng)
        else:
            nz = colored_noise(n, {"white": 0.0, "pink": 1.0, "brown": 2.0}[nt], nrng)
        nz8 = resample_poly(nz, 1, 2).astype(np.float32)[: len(clean8)]
        if len(nz8) < len(clean8):
            nz8 = np.pad(nz8, (0, len(clean8) - len(nz8)))
        f8 = FRAME // 2
        gain = math.sqrt(active_power(clean8, f8) / (active_power(nz8, f8) * 10 ** (snr / 10.0)))
        x8 = clean8 + gain * nz8

    peak = float(np.abs(x8).max())
    if peak > 0.98:
        x8 = x8 * (0.98 / peak)  # avoid clipping; scale-invariant metrics are unaffected

    y8 = amr_roundtrip(x8, job["amr_bitrate"])
    delay = estimate_delay(x8, y8)
    y8 = shift_fit(y8, delay, len(clean8))

    deg_rel = f"degraded/{job['deg_id']}.wav"
    sf.write(str(out / deg_rel), y8, SR_NB, subtype="PCM_16")

    return {
        "deg_id": job["deg_id"], "utt_id": row["utt_id"], "split": row["split"],
        "speaker_id": row["speaker_id"], "clean_path": row["clean_path"],
        "clean8k_path": row["clean8k_path"], "degraded_path": deg_rel,
        "noise_type": nt, "noise_file": noise_name,
        "snr_db": "" if nt == "none" else f"{snr:.1f}",
        "amr_bitrate": job["amr_bitrate"], "seed": derived_seed(CFG["seed"], job["deg_id"]),
        "delay_samples": delay, "transcript": row["transcript"],
        "duration_s": row.get("duration_s", ""), "scenario": row.get("scenario", ""),
    }


# ----------------------------------------------------------------------------- planning
def plan_jobs(rows, args, noise_types):
    jobs = []
    lo, hi = min(args.snrs), max(args.snrs)
    for row in rows:
        utt = row["utt_id"]
        if row["split"] in args.grid_splits:
            trng = np.random.default_rng(derived_seed(args.seed, utt + ":type"))
            nt = noise_types[int(trng.integers(len(noise_types)))]
            for br in args.grid_bitrates:
                for snr in args.snrs:
                    jobs.append(dict(row=row, noise_type=nt, snr_db=float(snr), amr_bitrate=br,
                                     noise_key=utt, deg_id=f"{utt}__{nt}_snr{snr:g}_br{br}"))
                if not args.no_codec_only:
                    jobs.append(dict(row=row, noise_type="none", snr_db=0.0, amr_bitrate=br,
                                     noise_key=utt, deg_id=f"{utt}__none_br{br}"))
        else:
            for c in range(args.train_copies):
                key = f"{utt}:c{c}"
                prng = np.random.default_rng(derived_seed(args.seed, key + ":plan"))
                br = int(args.bitrates[int(prng.integers(len(args.bitrates)))])
                if prng.random() < args.codec_only_prob:
                    jobs.append(dict(row=row, noise_type="none", snr_db=0.0, amr_bitrate=br,
                                     noise_key=key, deg_id=f"{utt}__none_br{br}_c{c}"))
                else:
                    nt = noise_types[int(prng.integers(len(noise_types)))]
                    snr = round(float(prng.uniform(lo, hi)), 1)
                    jobs.append(dict(row=row, noise_type=nt, snr_db=snr, amr_bitrate=br,
                                     noise_key=key, deg_id=f"{utt}__{nt}_snr{snr:g}_br{br}_c{c}"))
    return jobs


def find_noise_files(root):
    exts = {".wav", ".flac"}
    return sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in exts)


def build_noise_pools(files, root, group_by, dev_noise=(), test_noise=()):
    """dev_noise / test_noise: explicit group names (e.g. DEMAND environment folders) forced into
    that split; when either is given, everything not listed goes to train. Otherwise the hash
    decides."""
    pools = {"train": [], "dev": [], "test": []}
    explicit = bool(dev_noise or test_noise)
    for p in files:
        rel = p.relative_to(root)
        key = rel.parent.as_posix() if group_by == "dir" and str(rel.parent) != "." else rel.as_posix()
        if explicit:
            top = key.split("/")[0]
            s = "test" if top in test_noise else "dev" if top in dev_noise else "train"
        else:
            s = split_of_key(key)
        pools[s].append(str(p))
    if explicit:
        found = {k.split("/")[0] for k in
                 ((p.relative_to(root).parent.as_posix() if group_by == "dir" else p.relative_to(root).as_posix())
                  for p in files)}
        missing = (set(dev_noise) | set(test_noise)) - found
        if missing:
            sys.exit(f"--dev-noise/--test-noise names not found under the noise dir: {sorted(missing)}")
    print("noise files per pool: " + ", ".join(f"{s}={len(v)}" for s, v in pools.items()))
    for s in pools:
        if not pools[s]:
            print(f"WARNING: no noise files landed in the '{s}' pool; using all noise files for "
                  f"'{s}' (noise is then NOT held out for that split).", file=sys.stderr)
            pools[s] = [str(p) for p in files]
    return pools


def check_ffmpeg():
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found on PATH.")
    enc = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    if "libopencore_amrnb" not in enc:
        sys.exit("This ffmpeg has no AMR-NB encoder (libopencore_amrnb). Install an ffmpeg build "
                 "that includes it (check with: ffmpeg -encoders | grep -i amr).")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--noise-dir", default=None)
    ap.add_argument("--noise-group-by", choices=["file", "dir"], default="file",
                    help="hold out noise per file or per folder (use 'dir' for DEMAND)")
    ap.add_argument("--test-noise", nargs="*", default=[], metavar="NAME",
                    help="folder names forced into the TEST noise pool, e.g. DLIVING NPARK TCAR")
    ap.add_argument("--dev-noise", nargs="*", default=[], metavar="NAME",
                    help="folder names forced into the DEV noise pool; all others go to train")
    ap.add_argument("--synthetic", nargs="*", default=None, choices=["white", "pink", "brown", "babble"],
                    help="synthetic noise types (default: all four if no --noise-dir, else none)")
    ap.add_argument("--snrs", type=float, nargs="+", default=[0, 5, 10, 15, 20])
    ap.add_argument("--grid-splits", nargs="+", default=["dev", "test"])
    ap.add_argument("--grid-bitrates", type=int, nargs="+", default=[12200])
    ap.add_argument("--no-codec-only", action="store_true", help="skip the noise-free codec-only row")
    ap.add_argument("--train-copies", type=int, default=2)
    ap.add_argument("--bitrates", type=int, nargs="+", default=[12200, 7950, 5900, 4750],
                    help="AMR-NB bitrates sampled for non-grid (train) rows")
    ap.add_argument("--codec-only-prob", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--workers", type=int, default=max(1, cpu_count() // 2))
    ap.add_argument("--limit", type=int, default=0, help="only the first N utterances (smoke test)")
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted run (same settings): skip rows already finished")
    args = ap.parse_args()

    check_ffmpeg()
    out = Path(args.data)
    with open(out / "manifest.csv", newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("clean_path")]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        sys.exit("manifest.csv has no usable rows.")

    # noise sources
    noise_files, noise_pools, noise_root = [], {}, str(out)
    synthetic = args.synthetic
    if args.noise_dir:
        noise_root = args.noise_dir
        noise_files = find_noise_files(args.noise_dir)
        if not noise_files:
            sys.exit(f"No .wav/.flac files under {args.noise_dir}")
        noise_pools = build_noise_pools(noise_files, Path(args.noise_dir), args.noise_group_by,
                                         set(args.dev_noise), set(args.test_noise))
        synthetic = synthetic or []
    else:
        synthetic = synthetic if synthetic is not None else ["white", "pink", "brown", "babble"]
    noise_types = (["file"] if noise_files else []) + list(synthetic)
    if not noise_types:
        sys.exit("No noise sources selected.")
    print(f"noise types: {noise_types}  ({len(noise_files)} noise files)")

    # clean 8 kHz references (written once, up front)
    (out / "clean8k").mkdir(exist_ok=True)
    (out / "degraded").mkdir(exist_ok=True)
    clean_pools = {"train": [], "dev": [], "test": []}
    for r in rows:
        src = str(out / r["clean_path"])
        clean_pools[r["split"]].append(src)
        r["clean8k_path"] = f"clean8k/{r['utt_id']}.wav"
        sf.write(str(out / r["clean8k_path"]),
                 resample_poly(load_audio16k(src), 1, 2).astype(np.float32), SR_NB, subtype="PCM_16")

    jobs = plan_jobs(rows, args, noise_types)
    print(f"{len(rows)} utterances -> {len(jobs)} degraded files, {args.workers} workers")

    cfg = dict(out=str(out), seed=args.seed, noise_pools=noise_pools, noise_root=noise_root,
               clean_pools=clean_pools)
    # progress log: every finished row is appended immediately, so a crash / kill / Ctrl-C does not
    # lose the run. --resume reuses rows from a previous run only if the settings are identical.
    fp = str(h(json.dumps(dict(
        seed=args.seed, noise_dir=args.noise_dir, group=args.noise_group_by, snrs=args.snrs,
        synth=sorted(synthetic), grid_splits=args.grid_splits, grid_br=args.grid_bitrates,
        no_codec_only=args.no_codec_only, copies=args.train_copies, br=args.bitrates,
        co_prob=args.codec_only_prob, limit=args.limit,
        pools={k: sorted(v) for k, v in noise_pools.items()}, n_rows=len(rows)), sort_keys=True)))
    prog_path = out / ".degrade_progress.jsonl"
    results, done_ids = [], set()
    if args.resume and prog_path.exists():
        with open(prog_path, encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except ValueError:
                    continue  # half-written last line from a crash
                if d.get("fp") == fp and (out / d["row"]["degraded_path"]).exists():
                    results.append(d["row"])
                    done_ids.add(d["row"]["deg_id"])
        print(f"resume: {len(done_ids)} of {len(jobs)} rows already done")
    else:
        prog_path.unlink(missing_ok=True)
    jobs = [j for j in jobs if j["deg_id"] not in done_ids]

    with open(prog_path, "a", encoding="utf-8") as prog, \
            Pool(args.workers, initializer=init_worker, initargs=(cfg,), maxtasksperchild=200) as pool:
        for i, res in enumerate(pool.imap_unordered(process, jobs, chunksize=4), 1):
            results.append(res)
            prog.write(json.dumps({"fp": fp, "row": res}, ensure_ascii=False) + "\n")
            prog.flush()
            if i % 100 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}", flush=True)

    results.sort(key=lambda r: r["deg_id"])
    mpath = out / "manifest_degraded.csv"
    with open(mpath, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=OUT_COLS)
        wr.writeheader()
        wr.writerows(results)

    delays = np.array([r["delay_samples"] for r in results])
    print(f"\nwrote {len(results)} rows -> {mpath}")
    print("per split:", dict(Counter(r["split"] for r in results)))
    print("per noise type:", dict(Counter(r["noise_type"] for r in results)))
    print(f"codec delay (samples @8 kHz): median {np.median(delays):.0f}, "
          f"min {delays.min()}, max {delays.max()}")


if __name__ == "__main__":
    main()
