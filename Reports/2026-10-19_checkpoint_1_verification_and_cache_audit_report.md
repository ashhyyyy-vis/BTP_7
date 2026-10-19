# [2026-10-19] Checkpoint 1 Verification and Cache Audit Report

**Date:** October 19, 2026  
**Project:** Dial 112 Hindi Emergency Call Robust Automatic Speech Recognition (ASR)  
**Subject:** Empirical Verification of Checkpoint 1 Consolidated Results & Cache Audit  

---

## 1. Executive Summary

This report provides the formal empirical audit and verification for **Checkpoint 1: Consolidated Results (Gate 1 and Gate 2)**. 

Every numerical result in Checkpoint 1 has been verified against the underlying summary CSVs ([`rescore_summary_whisper-large-v3.csv`](file:///home/viseash/Ashley/BTP/artifacts/Metrics/musan/ASR/rescore_summary_whisper-large-v3.csv) and [`enh_metrics_summary.csv`](file:///home/viseash/Ashley/BTP/artifacts/Metrics/musan/Enhancement/enh_metrics_summary.csv)) and the JSONL transcript caches. All open verification questions raised in Section 8 of Checkpoint 1 are definitively resolved.

---

## 2. Audit of Cache Line Counts (1,800 MUSAN & 1,400 DEMAND)

### 2.1 The Discrepancy Question
Section 8 noted that the front-end comparison tables in Section 4.1 imply 1,600 transcripts for MUSAN and 1,200 for DEMAND, while `rescore.py` reported 1,800 and 1,400 cached transcripts respectively.

### 2.2 Empirical Audit Findings
An itemized audit of every key in the `.jsonl` files confirms:
- **Zero duplicate keys**.
- **Zero rogue or unexpected conditions**.
- The "extra" 200 transcripts in both caches correspond to the **untreated baseline at 15 dB (100 utterances) and 20 dB (100 utterances)**, which were generated during the initial full SNR grid in `eval_asr.py` but omitted from the front-end comparison tables (since enhancers were only tested at 0, 5, and 10 dB).

#### Itemized Breakdown:

| Cache Category | MUSAN Item Count | DEMAND Item Count | Notes |
| :--- | :---: | :---: | :--- |
| `clean` | 100 | 100 | Original 16 kHz |
| `clean8k` | 100 | 100 | Bandwidth-limited reference |
| `codec_only` (`none_br`) | 100 | 100 | 8 kHz + AMR-NB, no noise |
| `none::snr0` | 100 | 100 | Untreated degraded |
| `none::snr5` | 100 | 100 | Untreated degraded |
| `none::snr10` | 100 | 100 | Untreated degraded |
| **`none::snr15`** | **100** | **100** | *In baseline cache; omitted from front-end table* |
| **`none::snr20`** | **100** | **100** | *In baseline cache; omitted from front-end table* |
| **Subtotal `asr_whisper-large-v3.jsonl`** | **800** | **800** | **Exact match across both corpora** |
| `wiener_dd` (0, 5, 10 dB) | 300 | 300 | 100 utts $\times$ 3 SNRs |
| `dfn` (0, 5, 10 dB) | 300 | 300 | 100 utts $\times$ 3 SNRs |
| `dfn_atten12` (0, 5 dB) | 200 | — | 100 utts $\times$ 2 SNRs (MUSAN only) |
| `dfn_mix50` (0, 5 dB) | 200 | — | 100 utts $\times$ 2 SNRs (MUSAN only) |
| **Subtotal `asr_whisper-large-v3_enh.jsonl`** | **1,000** | **600** | **Exact match across both corpora** |
| **TOTAL CACHE COUNT** | **1,800** | **1,400** | **Reconciled exactly** |

---

## 3. Audit of Dev Noise File Counts (30 vs. 32 Files)

### 3.1 The Question
Section 8 noted that the dev split manifest contained 30 distinct noise files, whereas an earlier report stated 32 files.

### 3.2 Empirical Verification
- **Files physically present on disk**:
  - `data/noise/demand/OOFFICE/`: 16 channels (`ch01.wav` to `ch16.wav`)
  - `data/noise/demand/STRAFFIC/`: 16 channels (`ch01.wav` to `ch16.wav`)
  - **Total Dev Pool Capacity**: Exactly **32 files**.
- **Realized files in `data/degraded/demand/manifest_degraded.csv`**:
  - The dev split consists of 92 utterances.
  - During degradation, noise files were sampled uniformly at random with a fixed seed (`seed = 1337`).
  - Across the 92 draws, 30 distinct files were selected. Two channels (`STRAFFIC/ch03.wav` and `STRAFFIC/ch12.wav`) were simply not drawn in the 92 draws.
- **Verdict**:
  - **32** is the size of the *available dev noise pool*.
  - **30** is the number of *distinct noise files realized in the generated split*.
  - Both numbers are accurate in their respective contexts, and there is zero leakage into train or test.

---

## 4. Verification of Checkpoint 1 Tables

All numerical values, paired differences ($\Delta$), and 95% bootstrap confidence intervals in Checkpoint 1 were audited against the output artifacts:

### 4.1 Gate 1 Baseline (Untreated `none`, Normaliser v2)
- **MUSAN**: Clean 25.74% [21.68, 30.79] (CER 13.66%), Clean8k 26.67% (CER 13.54%), Codec-only 30.32% (CER 17.30%), 10 dB 37.44% (CER 21.79%), 5 dB 42.08% (CER 25.08%), 0 dB 51.49% (CER 35.01%).
- **DEMAND**: Clean 25.25% [21.35, 29.57] (CER 13.30%), Clean8k 26.18% (CER 13.12%), Codec-only 29.95% (CER 16.97%), 10 dB 33.79% (CER 20.19%), 5 dB 37.87% (CER 22.78%), 0 dB 45.30% (CER 29.35%).
- **Status**: **100% Verified** (exact match).

### 4.2 Gate 2 Front-End Comparisons
- **MUSAN**:
  - `dfn`: 10 dB +9.72% [5.78, 14.01]\*, 5 dB +13.37% [9.28, 17.25]\*, 0 dB +13.86% [10.16, 17.85]\*.
  - `wiener_dd`: 10 dB +1.73% [-0.56, 4.09], 5 dB +4.52% [1.47, 7.63]\*, 0 dB +3.65% [1.05, 6.45]\*.
  - `dfn_atten12`: 5 dB -1.18% [-3.43, 0.95], 0 dB +1.11% [-1.74, 4.35].
  - `dfn_mix50`: 5 dB -0.50% [-2.99, 2.12], 0 dB +1.30% [-1.73, 4.41].
- **DEMAND (Pre-hold-out)**:
  - `dfn`: 10 dB +6.87% [3.80, 10.26]\*, 5 dB +9.41% [6.59, 12.56]\*, 0 dB +11.45% [7.55, 15.51]\*.
  - `wiener_dd`: 10 dB +0.06% [-1.88, 2.04], 5 dB +1.42% [-0.97, 4.06], 0 dB +3.28% [0.52, 6.61]\*.
- **Status**: **100% Verified** (exact match).

---

## 5. Summary of Audit Confirmations

1. **Cache Integrity**: Confirmed. 1,800 (MUSAN) and 1,400 (DEMAND) lines are fully accounted for by 8 baseline conditions $\times$ 100 utterances plus the tested front-end conditions.
2. **Noise Split Integrity**: Confirmed. 32 files in pool, 30 realized in dev split.
3. **Reproducibility**: All values in Checkpoint 1 are verified and consistent with repository artifacts.

---

*Report prepared autonomously for the Dial 112 Emergency Call Speech Processing Project.*
