#!/usr/bin/env python3
"""
Extract a Hindi slice of IndicVoices from Hugging Face (streaming) and write:
  <out>/clean/*.wav     16 kHz, mono, 16-bit WAV
  <out>/manifest.csv    one row per utterance (degraded_* columns left empty;
                        the degradation script fills them in later)

Usage
-----
  # 0) one-time: install deps and log in (dataset needs an HF access token)
  pip install datasets
  huggingface-cli login            # or: hf auth login   (or set HF_TOKEN)
  # ffmpeg must be on PATH

  # 1) look at the schema first (nothing is written)
  python extract_indicvoices.py --inspect

  # 2) extract ~3 hours of 2-15 s clips
  python extract_indicvoices.py --out data --max-hours 3

Speaker-level split: every speaker is hashed into train/dev/test (80/10/10),
so no speaker appears in more than one split.
"""
import argparse
import csv
import hashlib
import subprocess
import sys
import wave
from pathlib import Path

from datasets import Audio, load_dataset
from dotenv import load_dotenv

load_dotenv()

DATASET = "ai4bharat/IndicVoices"
TEXT_CANDIDATES = ["text", "normalized", "verbatim"]
SPEAKER_CANDIDATES = ["speaker_id", "speaker", "client_id"]
SCENARIO_CANDIDATES = ["scenario", "task_name"]

MANIFEST_COLS = [
    "utt_id", "split", "speaker_id", "clean_path", "degraded_path",
    "noise_file", "snr_db", "amr_bitrate", "seed", "transcript",
    "duration_s", "scenario",
]


def first_present(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None


def find_audio_col(features, example):
    if features:
        for name, feat in features.items():
            if isinstance(feat, Audio):
                return name
    for name, val in example.items():
        if isinstance(val, dict) and ("bytes" in val or "array" in val):
            return name
    return None


def to_wav16k(audio_bytes, out_path):
    p = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", "pipe:0",
         "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(out_path)],
        input=audio_bytes, capture_output=True,
    )
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode(errors="ignore"))


def wav_duration(path):
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def split_for(speaker):
    h = int(hashlib.md5(speaker.encode("utf-8")).hexdigest(), 16) % 100
    if h < 80:
        return "train"
    if h < 90:
        return "dev"
    return "test"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--language", default="hindi")
    ap.add_argument("--hf-split", default="train")
    ap.add_argument("--out", default="data")
    ap.add_argument("--max-hours", type=float, default=3.0)
    ap.add_argument("--max-utts", type=int, default=0, help="0 = no limit")
    ap.add_argument("--min-dur", type=float, default=2.0)
    ap.add_argument("--max-dur", type=float, default=15.0)
    ap.add_argument("--text-col", default=None)
    ap.add_argument("--audio-col", default=None)
    ap.add_argument("--speaker-col", default=None)
    ap.add_argument("--scenario", default=None,
                    help="keep only this scenario value, e.g. a conversational one "
                         "(check the values with --inspect)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--inspect", action="store_true",
                    help="print schema and one example, then exit")
    args = ap.parse_args()

    ds = load_dataset(DATASET, args.language, split=args.hf_split, streaming=True)

    # ---- inspect mode ----------------------------------------------------
    if args.inspect:
        print("features:", ds.features)
        ex = next(iter(ds))
        for k, v in ex.items():
            if isinstance(v, dict) and ("bytes" in v or "array" in v):
                size = len(v["bytes"]) if v.get("bytes") else "decoded array"
                print(f"{k}: <audio> ({size})")
            else:
                print(f"{k}: {v!r}")
        return

    # ---- work out columns ------------------------------------------------
    peek = next(iter(ds))
    cols = list(peek.keys())
    audio_col = args.audio_col or find_audio_col(ds.features, peek)
    text_col = args.text_col or first_present(cols, TEXT_CANDIDATES)
    spk_col = args.speaker_col or first_present(cols, SPEAKER_CANDIDATES)
    scen_col = first_present(cols, SCENARIO_CANDIDATES)
    if not audio_col or not text_col:
        sys.exit(f"Could not find audio/text columns in {cols}. "
                 "Run --inspect and pass --audio-col / --text-col.")
    if not spk_col:
        print("WARNING: no speaker column found; splitting by utterance "
              "(possible speaker leakage between splits).", file=sys.stderr)
    print(f"audio={audio_col} text={text_col} speaker={spk_col} scenario={scen_col}")

    # raw bytes out; ffmpeg does the decoding/resampling
    ds = ds.cast_column(audio_col, Audio(decode=False))
    # streaming shards can be ordered by speaker; shuffle for variety
    ds = ds.shuffle(seed=args.seed, buffer_size=1000)

    out = Path(args.out)
    (out / "clean").mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.csv"

    total_s, n, skipped = 0.0, 0, 0
    limit_s = args.max_hours * 3600.0

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=MANIFEST_COLS)
        wr.writeheader()
        for ex in ds:
            if total_s >= limit_s or (args.max_utts and n >= args.max_utts):
                break
            if scen_col and args.scenario and str(ex.get(scen_col)) != args.scenario:
                continue
            text = (ex.get(text_col) or "").strip()
            audio = ex.get(audio_col) or {}
            data = audio.get("bytes")
            if not text or not data:
                skipped += 1
                continue

            utt_id = f"hi_{n:06d}"
            wav_path = out / "clean" / f"{utt_id}.wav"
            try:
                to_wav16k(data, wav_path)
                dur = wav_duration(wav_path)
            except Exception as e:  # bad clip: skip, keep going
                print(f"skip {utt_id}: {e}", file=sys.stderr)
                wav_path.unlink(missing_ok=True)
                skipped += 1
                continue
            if not (args.min_dur <= dur <= args.max_dur):
                wav_path.unlink(missing_ok=True)
                skipped += 1
                continue

            speaker = str(ex.get(spk_col)) if spk_col else utt_id
            wr.writerow({
                "utt_id": utt_id,
                "split": split_for(speaker),
                "speaker_id": speaker,
                "clean_path": f"clean/{utt_id}.wav",
                "degraded_path": "", "noise_file": "", "snr_db": "",
                "amr_bitrate": "", "seed": "",
                "transcript": text,
                "duration_s": f"{dur:.2f}",
                "scenario": ex.get(scen_col, "") if scen_col else "",
            })
            n += 1
            total_s += dur
            if n % 100 == 0:
                f.flush()
                print(f"{n} utts, {total_s/3600:.2f} h (skipped {skipped})")

    print(f"done: {n} utts, {total_s/3600:.2f} h, skipped {skipped} -> {manifest_path}")


if __name__ == "__main__":
    main()
