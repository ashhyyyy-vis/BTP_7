# [2026-10-18] Frozen v2 Normalisation Rescore and Front-End Robustness Report

**Date:** October 18, 2026  
**Project:** Dial 112 Hindi Emergency Call Robust Automatic Speech Recognition (ASR)  
**Corpora:** IndicVoices Hindi with MUSAN & DEMAND Additive Noise  
**ASR Architecture:** Whisper Large-v3 (Groq Backend, Identical Model across All Conditions)  
**Normaliser Version:** `hindi_norm.py` (Frozen v2)  

---

## 1. Executive Summary

Following the diagnostic discovery that orthographic diacritics and character representations in Hindi contributed to clean baseline errors, the text normalisation pipeline was formally standardized and frozen as **v2 (`scripts/hindi_norm.py`)**. 

All cached model hypotheses from previous experiments (1,800 transcripts across MUSAN and 1,400 transcripts across DEMAND) were re-scored offline without any new ASR or audio calls via [`scripts/rescore.py`](file:///home/viseash/Ashley/BTP/scripts/rescore.py).

### Core Findings:
1. **Clean Baseline Reduction**: Frozen v2 normalisation reduced clean Word Error Rate (WER) by **1.6 to 1.7 percentage points** across both corpora:
   - **MUSAN Clean WER**: dropped from 27.4% $\to$ **25.7%** (CER: **13.7%**).
   - **DEMAND Clean WER**: dropped from 26.9% $\to$ **25.2%** (CER: **13.3%**).
2. **Invariance of the DFN Degradation Penalty**: The catastrophic degradation caused by DeepFilterNet3 survived the new normaliser completely unchanged:
   - **MUSAN at 5 dB SNR**: $+13.4\%$ paired $\Delta$ under v2 (identical to $+13.4\%$ under v1).
   - **MUSAN at 0 dB SNR**: $+13.9\%$ paired $\Delta$ under v2 (was $+13.8\%$ under v1).
   - **DEMAND at 0 dB SNR**: $+11.4\%$ paired $\Delta$ under v2.
   This proves conclusively that the failure of deep neural enhancement is an **acoustic reality** (severe over-suppression and non-linear spectral distortion) rather than an evaluation or normalisation artifact.
3. **Cheap DFN Variants under v2**:
   - `dfn_atten12` reached **40.9%** WER at 5 dB ($-1.2\%$ vs. `none`, 95% CI $[-3.4, +0.9]$), recovering lost ground.
   - However, at 0 dB, `none` remained superior (**51.5%** vs. **52.6%** for `dfn_atten12` and **52.8%** for `dfn_mix50`). Because both 95% confidence intervals cross zero, the negative result stands.

---

## 2. Frozen v2 Normalisation Specification (`scripts/hindi_norm.py`)

The project text normalisation convention was upgraded from v1 to frozen v2:

| Component | v1 Convention (Legacy) | v2 Convention (Frozen) | Rationale |
| :--- | :--- | :--- | :--- |
| **Unicode Canonical Form** | NFC | NFC (via NFD decomposition) | Prevents decomposition mismatch. |
| **Punctuation & Danda** | Stripped (`Po`, `Pe`, `Pd`) | Stripped | ASR does not predict Indian dandas. |
| **Devanagari Digits** | Mapped to ASCII (`0-9`) | Mapped to ASCII (`0-9`) | Unified numerical representation. |
| **Nukta (`़` U+093C)** | Retained | **Stripped completely** (`ज़` $\to$ `ज`, `फ़` $\to$ `फ`, `ड़` $\to$ `ड`) | Annotators and Whisper disagree arbitrarily on Urdu/Persian loan consonants. |
| **Nasalization** | Retained separate | **Chandrabindu (`ँ`) $\to$ Anusvara (`ं`)** | Modern Hindi spelling freely interchanges both; prevents spurious errors on words like `पाँच`/`पांच`. |
| **Compound Spacing** | Preserved | Preserved (evaluated via CER) | Compound nouns (`पशु मेला` vs `पशुमेला`) are not forced; their effect is captured via space-free Character Error Rate (CER). |

### Code Integration:
In [`scripts/eval_asr.py`](file:///home/viseash/Ashley/BTP/scripts/eval_asr.py), the `normalize` function was replaced with:
```python
try:
    from scripts.hindi_norm import normalize
except ImportError:
    from hindi_norm import normalize
```
All future ASR and enhancement evaluations inherit this convention directly.

---

## 3. Rescore Results: MUSAN Dataset

- **Transcripts Scored**: 1,800 cached hypotheses on 100 standardized test utterances ($N_{\text{ref}} = 1,616$ words).
- **Summary File**: [`artifacts/Metrics/musan/ASR/rescore_summary_whisper-large-v3.csv`](file:///home/viseash/Ashley/BTP/artifacts/Metrics/musan/ASR/rescore_summary_whisper-large-v3.csv).

### 3.1 Normalisation Effect on Untreated Audio (`none`)

| Condition | WER v1 (old) | WER v2 (frozen) | $\Delta_{\text{norm}}$ | CER (%) | WER v2 [95% Bootstrap CI] |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **clean** | 27.4% | **25.7%** | **-1.7%** | 13.7% | [21.7, 30.8] |
| **clean8k** | 28.3% | **26.7%** | **-1.6%** | 13.5% | [22.7, 30.7] |
| **codec_only** | 31.9% | **30.3%** | **-1.6%** | 17.3% | [25.9, 35.6] |
| **snr_10** | 38.5% | **37.4%** | **-1.1%** | 21.8% | [32.3, 42.6] |
| **snr_5** | 42.9% | **42.1%** | **-0.8%** | 25.1% | [37.7, 46.7] |
| **snr_0** | 52.2% | **51.5%** | **-0.7%** | 35.0% | [45.6, 57.7] |

### 3.2 Front-End Comparison under Frozen Normaliser (WER v2)
*Paired difference against `none` computed across identical utterances with 95% bootstrap confidence intervals ($* = \text{CI excludes 0}$):*

| Condition | `none` | `dfn` (standard) | `dfn_atten12` | `dfn_mix50` | `wiener_dd` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **clean** | 25.7% | — | — | — | — |
| **clean8k** | 26.7% | — | — | — | — |
| **codec_only** | 30.3% | — | — | — | — |
| **snr_10** | 37.4% | **47.2%** (+9.7% [+5.8, +14.0]\*) | — | — | **39.2%** (+1.7% [-0.6, +4.1]) |
| **snr_5** | 42.1% | **55.4%** (+13.4% [+9.3, +17.2]\*) | **40.9%** (-1.2% [-3.4, +0.9]) | **41.6%** (-0.5% [-3.0, +2.1]) | **46.6%** (+4.5% [+1.5, +7.6]\*) |
| **snr_0** | 51.5% | **65.3%** (+13.9% [+10.2, +17.8]\*) | **52.6%** (+1.1% [-1.7, +4.4]) | **52.8%** (+1.3% [-1.7, +4.4]) | **55.1%** (+3.7% [+1.0, +6.5]\*) |

### 3.3 Front-End Comparison: Space-Free Character Error Rate (CER %)

| Condition | `none` | `dfn` (standard) | `dfn_atten12` | `dfn_mix50` | `wiener_dd` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **clean** | 13.7% | — | — | — | — |
| **clean8k** | 13.5% | — | — | — | — |
| **codec_only** | 17.3% | — | — | — | — |
| **snr_10** | 21.8% | **32.4%** (+10.6% [+7.0, +14.4]\*) | — | — | **24.2%** (+2.4% [+0.5, +4.1]\*) |
| **snr_5** | 25.1% | **39.0%** (+13.9% [+9.9, +17.7]\*) | **25.5%** (+0.4% [-1.4, +2.3]) | **26.0%** (+0.9% [-1.4, +3.5]) | **31.7%** (+6.6% [+3.3, +10.2]\*) |
| **snr_0** | 35.0% | **49.4%** (+14.4% [+10.8, +18.3]\*) | **37.1%** (+2.1% [-0.9, +5.0]) | **36.8%** (+1.8% [-1.1, +5.0]) | **38.4%** (+3.4% [+0.6, +6.4]\*) |

---

## 4. Rescore Results: DEMAND Dataset

- **Transcripts Scored**: 1,400 cached hypotheses on 100 standardized test utterances.
- **Summary File**: [`artifacts/Metrics/demand/ASR/rescore_summary_whisper-large-v3.csv`](file:///home/viseash/Ashley/BTP/artifacts/Metrics/demand/ASR/rescore_summary_whisper-large-v3.csv).

### 4.1 Normalisation Effect on Untreated Audio (`none`)

| Condition | WER v1 (old) | WER v2 (frozen) | $\Delta_{\text{norm}}$ | CER (%) | WER v2 [95% Bootstrap CI] |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **clean** | 26.9% | **25.2%** | **-1.7%** | 13.3% | [21.4, 29.6] |
| **clean8k** | 27.8% | **26.2%** | **-1.6%** | 13.1% | [22.3, 30.0] |
| **codec_only** | 31.5% | **30.0%** | **-1.5%** | 17.0% | [25.8, 34.6] |
| **snr_10** | 35.0% | **33.8%** | **-1.2%** | 20.2% | [29.1, 38.9] |
| **snr_5** | 38.9% | **37.9%** | **-1.0%** | 22.8% | [33.1, 43.0] |
| **snr_0** | 46.5% | **45.3%** | **-1.2%** | 29.3% | [39.8, 51.6] |

### 4.2 Front-End Comparison under Frozen Normaliser (WER v2)

| Condition | `none` | `dfn` (standard) | `wiener_dd` |
| :--- | :---: | :---: | :---: |
| **snr_10** | 33.8% | **40.7%** (+6.9% [+3.8, +10.3]\*) | **33.8%** (+0.1% [-1.9, +2.0]) |
| **snr_5** | 37.9% | **47.3%** (+9.4% [+6.6, +12.6]\*) | **39.3%** (+1.4% [-1.0, +4.1]) |
| **snr_0** | 45.3% | **56.7%** (+11.4% [+7.6, +15.5]\*) | **48.6%** (+3.3% [+0.5, +6.6]\*) |

### 4.3 Front-End Comparison: Space-Free Character Error Rate (CER %)

| Condition | `none` | `dfn` (standard) | `wiener_dd` |
| :--- | :---: | :---: | :---: |
| **snr_10** | 20.2% | **26.2%** (+6.0% [+3.1, +9.3]\*) | **19.8%** (-0.4% [-2.0, +1.1]) |
| **snr_5** | 22.8% | **31.6%** (+8.9% [+6.2, +11.8]\*) | **25.2%** (+2.5% [-0.2, +5.4]) |
| **snr_0** | 29.3% | **42.1%** (+12.7% [+8.8, +16.7]\*) | **33.7%** (+4.3% [+1.5, +7.6]\*) |

---

## 5. Synthesis & Methodological Conclusions

```
Performance Trajectory Across Normalisation and Front-Ends (MUSAN 0 dB SNR):
┌────────────────────────────────────────────────────────┐
│ none (v1 Normaliser):           52.2% WER              │
│ none (v2 Frozen Normaliser):     51.5% WER  (CER: 35.0%)│
│ dfn_atten12 (12 dB limit):      52.6% WER  (CER: 37.1%)│
│ dfn_mix50 (50/50 linear blend): 52.8% WER  (CER: 36.8%)│
│ wiener_dd (Decision-directed):  55.1% WER* (CER: 38.4%)│
│ dfn (Standard DeepFilterNet3):  65.3% WER* (CER: 49.4%)│
└────────────────────────────────────────────────────────┘
```

1. **Normalisation Fixes Superficial Orthography, Not Front-End Loss**:
   Freezing v2 normalisation recovered 1.6–1.7% in clean WER by neutralizing diacritic mismatches (Nuktas and Chandrabindus). However, it did not alter the fundamental finding: **enhancement front-ends do not improve recognition accuracy on narrow-band telephone audio**.
2. **DeepFilterNet3 Degrades Phonetic Fidelity**:
   Standard DFN causes an absolute increase of **+11.4% to +13.9% in WER** and **+12.7% to +14.4% in CER**. Because CER ignores word segmentation entirely, this confirms that DFN damages character-level phonetic information.
3. **Cheap Variants Eliminate Catastrophic Harm but Do Not Surpass Untreated Audio**:
   Both `dfn_atten12` and `dfn_mix50` averted the severe ~14% error explosion, matching untreated audio at 5 dB (40.9% vs 42.1%). However, under adverse 0 dB noise, the untreated AMR-NB audio remains the most accurate representation for the recognizer (51.5% vs 52.6%).
4. **Final Protocol Status**:
   With Gate 2 comprehensively evaluated and verified across both MUSAN and DEMAND under frozen v2 text normalisation, the negative result for pre-enhancement front-ends is firmly established.

---

*Report prepared autonomously for the Dial 112 Emergency Call Speech Processing Project.*
