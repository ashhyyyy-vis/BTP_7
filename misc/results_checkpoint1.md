# Checkpoint 1: Consolidated Results (Gate 1 and Gate 2)

**Project:** Real-time acoustic enhancement and RAG-based decision support for Hindi emergency telephony (Dial 112)
**Date:** 2026-10-18
**Status of every number in this document:** preliminary. All WER/CER values come from Whisper large-v3 served by the Groq API (not bit-repeatable at temperature 0, not comparable with local runs), 100 test utterances, scored with the frozen normaliser `hindi_norm.py` v2. A local re-baseline (Phase 2) supersedes them.

This document replaces the three separate reports dated 2026-10-17 and 2026-10-18 (enhancement/error decomposition, leakage/normalisation, v2 rescore). Where they disagree, this document is correct. Section 8 lists the corrections.

---

## 1. Summary

| Question | Answer | Confidence |
|---|---|---|
| Does the 8 kHz AMR-NB phone channel hurt Hindi ASR? | Yes. Clean 25.7% WER rises to 51.5% at 0 dB (MUSAN). Bandwidth costs about 1 point, the codec about 3.6 more, noise a further 4 to 21 | High within this setup |
| Does pretrained DeepFilterNet3 (DFN) used before ASR help? | No. It raises WER by about 7 to 14 points at 0, 5 and 10 dB on both corpora; all CIs exclude 0 | High within this setup |
| Does a classical Wiener filter help? | No. Neutral at 10 dB, +1.4 to +4.5 points at 0 to 5 dB (several CIs exclude 0) | Moderate (small effects) |
| Do gentle DFN variants (12 dB attenuation limit, 50/50 mix) help? | No detectable effect (within about +/-3 points of untreated audio). MUSAN only | Moderate |
| Does the result depend on the text normaliser? | No. Gate 2 deltas are unchanged between normaliser v1 and v2, and CER agrees | High |
| Is the train/dev/test split leak-free? | Yes: zero overlap in speakers, utterances, noise files and noise environments | High |
| Why does DFN hurt? | Unknown. Deletions and substitutions rise while insertions do not (MUSAN, v1 scoring); over-suppression is a hypothesis, not tested | Low |
| Does denoising help a human listener? | Not tested. Signal metrics are mixed (Section 5) | Open |

**Design consequence:** do not put a denoiser in front of ASR. Feed the raw call audio to an ASR adapted to noisy telephone speech. If a denoiser is used, it serves the human operator on a separate branch (Section 7).

---

## 2. Data and method

### 2.1 Speech data
- IndicVoices Hindi: 1,340 clean utterances, 185 speakers, split by speaker:

| Split | Utterances | Speakers |
|---|---|---|
| Train | 1,081 | 147 |
| Dev | 92 | 15 |
| Test | 167 | 23 |

- All ASR results here use the first 100 test utterances (the same 100 in every condition, corpus and front-end).

### 2.2 Telephone channel simulation (`degrade.py`)
16 kHz clean speech -> 8 kHz -> additive noise at a target SNR (measured at 8 kHz) -> AMR-NB 12.2 kbps round trip (ffmpeg `libopencore_amrnb`) -> codec delay removed by cross-correlation, so the degraded file is sample-aligned to `clean8k`. For ASR, every 8 kHz file is upsampled to 16 kHz the same way.

Conditions: `clean` (16 kHz original), `clean8k` (bandwidth only), `codec_only` (8 kHz + AMR, no noise), and noise at 10, 5 and 0 dB.

### 2.3 Noise corpora
- **MUSAN** noise (free-sound, sound-bible), held out by noise file.
- **DEMAND** (17 environments). Two versions exist:
  - *Pre-hold-out (used for ALL DEMAND results below):* test utterances drew noise from all 17 environments, because no environment hashed into the test pool and the script fell back to all files. Not a noise hold-out. Harmless for Gate 2 (no front-end was trained on it), but these numbers are not the held-out test.
  - *Re-degraded with hold-out (audited, not yet transcribed):* test = DLIVING, NPARK, TCAR (48 noise files, 1,002 clips); dev = OOFFICE, STRAFFIC (30 noise files, 552 clips); train = 12 other environments (192 noise files, 2,162 clips). Audit found zero overlap in noise environments, noise files, speakers and utterances. Note: training has 2 degraded copies per utterance; dev and test have 6.

### 2.4 Front-ends
| Name | Description |
|---|---|
| `none` | Degraded audio as is |
| `wiener_dd` | Offline decision-directed Wiener filter, stationary noise PSD from the 15th percentile, -20 dB gain floor |
| `dfn` | Pretrained DeepFilterNet3: 8 kHz -> x6 upsample -> 48 kHz DFN -> x6 downsample -> 8 kHz |
| `dfn_atten12` | DFN with attenuation limited to 12 dB (MUSAN only) |
| `dfn_mix50` | 50/50 sample-aligned mix of DFN output and the degraded input (MUSAN only) |

### 2.5 ASR and scoring
- Whisper large-v3 via Groq, `language=hi`, `temperature=0`.
- Normaliser `hindi_norm.py` v2 (frozen 2026-10-18): NFC, lower-case Latin, punctuation and danda removed, Devanagari digits -> ASCII, **nukta removed, chandrabindu -> anusvara**. Compound splitting/joining (पशु मेला vs पशुमेला) is not normalised; **CER with spaces removed** is reported next to WER to absorb it.
- 95% bootstrap CIs over utterances (1,000 resamples). Front-end differences are **paired** (same utterances resampled for both).
- Because 100 utterances give CIs of roughly +/-3 WER points, effects smaller than that cannot be detected.

---

## 3. Gate 1: the channel hurts ASR

Untreated audio (`none`), normaliser v2. WER % [95% CI] and CER %.

| Condition | MUSAN WER | MUSAN CER | DEMAND (pre-hold-out) WER | DEMAND CER |
|---|---|---|---|---|
| clean | 25.7 [21.7, 30.8] | 13.7 | 25.2 [21.4, 29.6] | 13.3 |
| clean8k | 26.7 [22.7, 30.7] | 13.5 | 26.2 [22.3, 30.0] | 13.1 |
| codec_only | 30.3 [25.9, 35.6] | 17.3 | 30.0 [25.8, 34.6] | 17.0 |
| 10 dB | 37.4 [32.3, 42.6] | 21.8 | 33.8 [29.1, 38.9] | 20.2 |
| 5 dB | 42.1 [37.7, 46.7] | 25.1 | 37.9 [33.1, 43.0] | 22.8 |
| 0 dB | 51.5 [45.6, 57.7] | 35.0 | 45.3 [39.8, 51.6] | 29.3 |

Reading: bandwidth reduction costs about 1 point; the AMR codec about 3.6 more; noise adds 7 / 12 / 21 points (MUSAN) and 4 / 8 / 15 points (DEMAND) over codec-only at 10 / 5 / 0 dB. Clean WER of about 25% is dominated by spelling conventions and genuine recognition errors in conversational Hindi, not by the channel.

Effect of the normaliser: clean WER fell from 27.4 to 25.7 (MUSAN) and from 26.9 to 25.2 (DEMAND), about 1.7 points. The effect shrinks to 0.7 to 1.2 points at 0 dB.

---

## 4. Gate 2: front-ends before ASR

### 4.1 WER (%), normaliser v2. Delta = front-end minus `none`, paired 95% CI, `*` = CI excludes 0

**MUSAN**

| SNR | none | wiener_dd | dfn | dfn_atten12 | dfn_mix50 |
|---|---|---|---|---|---|
| 10 dB | 37.4 | 39.2 (+1.7 [-0.6, +4.1]) | 47.2 (**+9.7 [+5.8, +14.0]***) | not run | not run |
| 5 dB | 42.1 | 46.6 (**+4.5 [+1.5, +7.6]***) | 55.4 (**+13.4 [+9.3, +17.2]***) | 40.9 (-1.2 [-3.4, +0.9]) | 41.6 (-0.5 [-3.0, +2.1]) |
| 0 dB | 51.5 | 55.1 (**+3.7 [+1.0, +6.5]***) | 65.3 (**+13.9 [+10.2, +17.8]***) | 52.6 (+1.1 [-1.7, +4.4]) | 52.8 (+1.3 [-1.7, +4.4]) |

**DEMAND (pre-hold-out noise)**

| SNR | none | wiener_dd | dfn |
|---|---|---|---|
| 10 dB | 33.8 | 33.8 (+0.1 [-1.9, +2.0]) | 40.7 (**+6.9 [+3.8, +10.3]***) |
| 5 dB | 37.9 | 39.3 (+1.4 [-1.0, +4.1]) | 47.3 (**+9.4 [+6.6, +12.6]***) |
| 0 dB | 45.3 | 48.6 (**+3.3 [+0.5, +6.6]***) | 56.7 (**+11.4 [+7.6, +15.5]***) |

### 4.2 CER (%), spaces ignored

**MUSAN:** none 21.8 / 25.1 / 35.0 at 10 / 5 / 0 dB. dfn +10.6* / +13.9* / +14.4*. wiener_dd +2.4* / +6.6* / +3.4*. dfn_atten12 +0.4 / +2.1 (n.s.), dfn_mix50 +0.9 / +1.8 (n.s.) at 5 / 0 dB.

**DEMAND (pre-hold-out):** none 20.2 / 22.8 / 29.3. dfn +6.0* / +8.9* / +12.7*. wiener_dd -0.4 / +2.5 / +4.3* (the 0 dB difference is significant).

CER shows the same pattern as WER, so the conclusions do not depend on how words are split.

### 4.3 Error decomposition (MUSAN, scored with normaliser v1; not re-run under v2)
WER = Sub% + Del% + Ins%.

| SNR | front-end | WER | Sub | Del | Ins |
|---|---|---|---|---|---|
| 10 | none | 38.5 | 27.5 | 9.0 | 1.9 |
| 10 | dfn | 47.7 | 32.7 | 13.7 | 1.4 |
| 5 | none | 42.9 | 32.2 | 8.2 | 2.4 |
| 5 | wiener_dd | 47.5 | 32.8 | 12.8 | 1.9 |
| 5 | dfn | 56.3 | 40.2 | 13.3 | 2.8 |
| 5 | dfn_atten12 | 42.0 | 31.6 | 8.0 | 2.4 |
| 5 | dfn_mix50 | 42.6 | 32.4 | 8.3 | 1.9 |
| 0 | none | 52.2 | 36.8 | 12.9 | 2.5 |
| 0 | wiener_dd | 55.8 | 39.8 | 13.2 | 2.8 |
| 0 | dfn | 66.0 | 44.1 | 19.3 | 2.6 |
| 0 | dfn_atten12 | 53.3 | 36.4 | 14.0 | 2.9 |
| 0 | dfn_mix50 | 53.5 | 34.8 | 15.8 | 2.9 |

Observations (what the data show):
- DFN's extra errors are deletions (+4.7 to +6.4 points) and substitutions (+5.2 to +8.0). Insertions stay about flat, so the extra errors are not inserted words.
- The attenuation-limited and mixed variants bring deletions back to the untreated level at 5 dB (8.0 and 8.3 vs 8.2) but not at 0 dB (14.0 and 15.8 vs 12.9).
- Wiener at 5 dB raises deletions from 8.2 to 12.8.
- Why this happens (for example, low-energy consonants treated as noise, or phase/spectral distortion) was **not tested**; treat it as a hypothesis.

---

## 5. Signal metrics (relevant to the operator branch)

Codec-only ceiling (clean8k reference): STOI 0.942, narrowband PESQ 4.14, SI-SDR 1.08 dB. SI-SDR is uninformative on AMR audio (ceiling about 1 dB), so STOI and PESQ are the headline metrics. Values below are the front-end minus `none`; `*` = paired CI excludes 0.

| Corpus | SNR | none STOI / PESQ | dfn dSTOI | dfn dPESQ | wiener dSTOI | wiener dPESQ |
|---|---|---|---|---|---|---|
| MUSAN | 10 dB | 0.794 / 2.41 | -0.016 | -0.07 | -0.020* | -0.46* |
| MUSAN | 5 dB | 0.730 / 2.03 | +0.021 | +0.17* | -0.020* | -0.35* |
| MUSAN | 0 dB | 0.653 / 1.74 | +0.052* | +0.27* | -0.020* | -0.26* |
| DEMAND (pre-hold-out) | 10 dB | 0.864 / 3.17 | -0.049* | -0.56* | -0.023* | -0.59* |
| DEMAND (pre-hold-out) | 5 dB | 0.810 / 2.81 | -0.024* | -0.35* | -0.023* | -0.48* |
| DEMAND (pre-hold-out) | 0 dB | 0.736 / 2.44 | +0.004 | -0.17* | -0.021* | -0.39* |

(DFN dSTOI at 5 dB on MUSAN has a CI of [0.000, 0.039], borderline.)

Reading: DFN improves PESQ/STOI only on MUSAN at 0 to 5 dB. On DEMAND it lowers PESQ at every SNR and STOI at 10 and 5 dB. Wiener lowers both everywhere. Signal metrics did not predict ASR benefit: on MUSAN at 0 to 5 dB DFN improves PESQ yet raises WER by about 13 points. Per-environment breakdowns (DEMAND) have only 2 to 11 utterances each and are not interpreted.

---

## 6. Established, preliminary, open

**Established (supported by the data here)**
1. The simulated phone channel degrades Whisper large-v3 on Hindi in a graded way (Section 3).
2. Pretrained DFN used before Whisper raises WER on both noise corpora at all tested SNRs (Section 4).
3. This is not a normalisation artifact (v1 vs v2 and CER agree).
4. Wiener and the gentle DFN variants show no detectable gain.
5. The train/dev/test partitions are leak-free (Section 2.3).

**Preliminary (needs confirmation)**
- Every absolute number: Groq-only, 100 utterances, not bit-repeatable. Confirm with the local baseline.
- DEMAND results: pre-hold-out noise. Re-run on the held-out test environments.
- Gentle-variant results: MUSAN only.

**Open**
- Mechanism behind DFN's damage.
- Whether denoising helps a human listener.
- Whether ASR adapted to degraded telephone audio closes the noise gap (main experiment; not started).
- How much of any adaptation gain comes from noise/channel adaptation vs learning the corpus's spelling conventions (needs a clean-adapted control).
- Generality beyond Whisper large-v3, simulated channels, and Hindi; real call audio.

**Scope of the negative result:** one classical filter, one pretrained neural denoiser and two inference-time variants of it, on one ASR, two noise corpora, simulated AMR-NB, 100 utterances. It does not show that no enhancement method could ever help, and it did not test a DFN fine-tuned on 8 kHz data.

---

## 7. Decisions taken

1. **No denoiser in front of ASR** (decision rule: no variant reached WER at or below untreated audio with a CI excluding zero at 0 and 5 dB).
2. **Two-branch design.** The incoming noisy call goes (a) through a denoiser to the operator's headset and (b) raw into an ASR adapted to noisy telephone audio, then to RAG triage. The ASR branch adds no denoiser latency. The operator branch is not yet validated (Phase 3b).
3. **Next main experiment:** adapt Whisper to degraded audio (LoRA if VRAM is tight), with a clean-audio-adapted control, evaluated on the held-out DEMAND test and on MUSAN.
4. Denoisers (`none`, `wiener_dd`, `dfn_atten12`) remain as frozen comparison conditions.

---

## 8. Corrections to the earlier reports

| Earlier statement | Correction |
|---|---|
| DEMAND results were on held-out test environments (DLIVING/NPARK/TCAR) | All DEMAND WER and signal-metric numbers come from the pre-hold-out run (all 17 environments). Held-out test audio is not yet transcribed |
| "Proves conclusively ... acoustic reality (over-suppression and non-linear spectral distortion)" | The result is not a normalisation artifact; the mechanism is a hypothesis |
| "Negative result firmly established" | Scoped as in Section 6 |
| "No benefit" | "No detectable benefit within about +/-3 points" |
| "Retraining DFN formally ruled out" | Not tested; not pursued for now |
| Clean WER "27.85%", orthographic share "5 to 8%", true acoustic error "18 to 20%" | Superseded by the full-run numbers (clean 25.7 / 25.2, CER 13.7 / 13.3). The orthographic share was an estimate from 20 pairs, not measured |
| Variants "return deletions to baseline" | At 5 dB only; at 0 dB deletions stay above baseline |
| Substitutions +7.4 at 0 dB | +7.3 (44.1 vs 36.8) |
| MUSAN partitioned "by utterance hash" | By noise file |
| 15 and 20 dB SNRs listed | Only 0, 5 and 10 dB were evaluated |
| "Semantic accuracy of 80-82%" | WER does not measure semantic accuracy; remove |

**Still to verify:** the reports quote 1,800 cached MUSAN and 1,400 DEMAND transcripts, while the conditions in the tables imply about 1,600 and 1,200. Check that no unexpected conditions are in the caches. Dev noise shows 30 distinct files where an earlier report said 32.

---

## 9. Reproducibility

| Item | Value |
|---|---|
| ASR | Whisper large-v3, Groq API, `language=hi`, `temperature=0` |
| Test utterances | first 100 of the 167 in the test split, same in every condition |
| Normaliser | `hindi_norm.py` v2 (`NORM_VERSION = "v2"`), imported by `eval_asr.py` |
| Re-scoring | `rescore.py` (offline, from cached transcripts) -> `artifacts/Metrics/<dataset>/ASR/rescore_summary_whisper-large-v3.csv` |
| Signal metrics | `enh_metrics.py` -> `<dataset>_enh_metrics_summary.csv` and `_per_file.csv` |
| Degradation | `degrade.py` (`--test-noise`, `--dev-noise`, `--resume`) |
| Front-ends | `denoise.py` (`wiener_dd`, `dfn`, `--dfn-atten`), mix variants as separate method names |
| Statistics | 1,000-resample bootstrap over utterances, paired for front-end comparisons |

To archive: the Groq caches (`asr_whisper-large-v3*.jsonl`), the summary CSVs, and the script versions above. Label them "Groq, preliminary".
