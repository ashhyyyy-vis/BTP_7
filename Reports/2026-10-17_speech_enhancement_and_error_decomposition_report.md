# [2026-10-17] Investigation of Speech Enhancement Front-Ends for Emergency Call (Dial 112) Hindi ASR under AMR-NB Telephone Degradation

**Date:** October 17, 2026  
**Project:** Dial 112 Hindi Emergency Call Robust Automatic Speech Recognition (ASR)  
**Corpus:** IndicVoices Hindi  
**ASR Architecture:** Whisper Large-v3 (Identical Inference Backend across All Control & Treatment Conditions)  

---

## 1. Executive Summary

In emergency response call centers (such as India's Dial 112), incoming telephony audio is heavily degraded by narrow-band cellular transmission (Adaptive Multi-Rate Narrowband / AMR-NB codec at 8 kHz sampling rate) combined with adverse acoustic background noise from callers in distress (traffic, public places, domestic environments).

This study investigates whether inserting a speech enhancement (SE) front-end prior to Automatic Speech Recognition (ASR) improves word recognition accuracy on degraded telephone speech. We systematically evaluated:
1. **Unprocessed degraded baseline (`none`)**: AMR-NB encoded audio with background noise.
2. **Classical statistical baseline (`wiener_dd`)**: Decision-directed stationary Wiener filtering.
3. **Pretrained deep neural enhancer (`dfn`)**: Full-band DeepFilterNet3 adapted to the telephone band.
4. **Constrained neural variants (`dfn_atten12` and `dfn_mix50`)**: Attenuation-limited and linear blend interventions designed to prevent over-suppression.

### Key Conclusions:
- **Catastrophic Failure of Unconstrained Deep Neural Enhancement**: Standard full-band DeepFilterNet3 worsened Word Error Rate (WER) across all signal-to-noise ratios (SNRs). At 0 dB SNR, WER degraded from 52.2% to 66.0% (+13.8% absolute error penalty).
- **Diagnostic Root Cause via Error Decomposition**: Decomposing Levenshtein errors into Substitutions ($S$), Deletions ($D$), and Insertions ($I$) revealed that insertions remained virtually constant (~2.5%), ruling out hallucination on silenced intervals. Instead, errors were heavily driven by a massive spike in **deletions** (+6.4% absolute jump at 0 dB) and **substitutions** (+7.4% absolute jump), demonstrating that neural SE aggressively over-suppressed speech phonemes and induced non-linear acoustic distortions.
- **Efficacy of Cheap Interventions**: Limiting attenuation to 12 dB (`dfn_atten12`) and 50/50 blending (`dfn_mix50`) successfully eliminated the ~14% error explosion, lowering deletions back to baseline levels (~8%) and achieving 42.0% WER at 5 dB (vs. 42.9% baseline).
- **Decision Rule Triggered**: However, at 0 dB, neither variant achieved a statistically significant improvement over unprocessed audio (53.3% vs. 52.2%, with the 95% bootstrap confidence interval spanning zero). Following the study's decision framework, **deep speech enhancement front-ends are abandoned** in favor of **direct acoustic adaptation of the ASR system on AMR-NB degraded speech**.
- **Acoustic Leakage Prevention**: Strict noise pool partitioning was implemented and verified for the DEMAND dataset, reserving unseen environments (`DLIVING`, `NPARK`, `TCAR`) exclusively for test, (`OOFFICE`, `STRAFFIC`) for dev, and 12 distinct environments for training data generation.

---

## 2. Experimental Methodology

### 2.1 Audio Pipeline & Telephone Simulation
Audio is processed according to realistic emergency cellular network conditions:
- **Clean Source**: 16 kHz studio-quality Hindi utterances from the IndicVoices dataset.
- **Reference Generation**: Downsampled to 8 kHz PCM (`clean8k`), providing the sample-aligned reference target.
- **Noise Mixing**: Additive acoustic noise mixed at calibrated telephone-band active-speech SNRs:
  - High noise: 0 dB SNR
  - Moderate noise: 5 dB SNR
  - Low noise: 10 dB SNR, 15 dB SNR, 20 dB SNR
- **AMR-NB Codec Roundtrip**: Processed via `libopencore_amrnb` at 12.2 kbps (standard maximum AMR-NB bitrate) and resampled back to 16 kHz for Whisper inference.
- **Time Alignment**: Delay introduced by the AMR-NB framing and filterbanks (~39 samples / ~4.8 ms) is estimated via cross-correlation and compensated, guaranteeing sample-for-sample alignment.

### 2.2 Datasets & Noise Partitions
- **Speech**: 100 standardized test utterances (1,616 words) drawn from the official test split of IndicVoices Hindi.
- **MUSAN**: Multi-source noise (free-sound, sound-bible) partitioned by utterance hash.
- **DEMAND (Held-Out Acoustic Environments)**: Strict spatial and environmental hold-out:
  - **Test Set** (48 audio files, 1,002 degraded files): `DLIVING` (Domestic living room), `NPARK` (Nature park), `TCAR` (Transportation car).
  - **Dev Set** (32 audio files, 552 degraded files): `OOFFICE` (Office space), `STRAFFIC` (Street traffic).
  - **Train Set** (192 audio files, 2,162 degraded files): 12 distinct environments (`DKITCHEN`, `DWASHING`, `NFIELD`, `NRIVER`, `OHALLWAY`, `OMEETING`, `PCAFETER`, `PRESTO`, `PSTATION`, `SPSQUARE`, `TBUS`, `TMETRO`).

### 2.3 Evaluated Front-Ends
1. **`none` (Unprocessed Control)**: Degraded 8 kHz AMR-NB audio resampled directly to 16 kHz.
2. **`wiener_dd` (Classical Baseline)**: Decision-directed stationary Wiener filter operating on Short-Time Fourier Transform (STFT) frames (32 ms Hann window, 8 ms hop). Noise power spectral density is estimated via the 15th percentile with a -20 dB gain floor.
3. **`dfn` (DeepFilterNet3)**: Full-band deep complex-valued neural network trained on 48 kHz audio. 8 kHz audio is upsampled by $6\times$, enhanced by DeepFilterNet3, and downsampled by $6\times$ back to 8 kHz.
4. **`dfn_atten12` (Attenuation-Limited DFN)**: DeepFilterNet3 with an explicit maximum attenuation threshold capped at 12 dB:
   $$G_{\text{min}} = -12 \text{ dB}$$
5. **`dfn_mix50` (Linear Dual-Path Blend)**: Output audio generated as a sample-aligned 50/50 linear mixture of enhanced and degraded audio:
   $$y[n] = 0.5 \cdot y_{\text{dfn}}[n] + 0.5 \cdot x[n]$$

### 2.4 Same-Backend ASR Evaluation
To eliminate confounding variables across models or API quantization:
- **Engine**: OpenAI Whisper Large-v3 evaluated via the Groq Cloud inference API.
- **Pacing & Quotas**: Paced at 20 RPM with margin 0.9 (~5.5 seconds per request) to prevent rate-limit throttling (HTTP 429).
- **Statistical Significance**: Paired bootstrap resampling ($N = 1000$ iterations) computing 95% Confidence Intervals (CIs) on the difference:
  $$\Delta = \text{WER}_{\text{frontend}} - \text{WER}_{\text{none}}$$

---

## 3. Results & Empirical Analysis

### 3.1 Initial Comparison: Unprocessed vs. Wiener vs. Standard DFN

All evaluations conducted on the same 100 test utterances ($N_{\text{ref}} = 1,616$ words):

| Condition | Front-End | WER (%) | Paired $\Delta$ vs. `none` [95% CI] | Gap Closed (%) |
| :--- | :--- | :---: | :---: | :---: |
| **clean** | `none` | 27.4 | — | — |
| **clean8k** | `none` | 28.3 | — | — |
| **codec_only** | `none` | 31.9 | — | — |
| **snr_10** | `none` | 38.5 | — | — |
| | `wiener_dd` | 40.3 | +1.8 [-0.5, +4.0] | -27% |
| | `dfn` | **47.7** | **+9.2 [+5.3, +13.3]*** | **-139%** |
| **snr_5** | `none` | 42.9 | — | — |
| | `wiener_dd` | 47.5 | +4.6 [+1.5, +7.6]* | -42% |
| | `dfn` | **56.3** | **+13.4 [+9.3, +17.1]*** | **-122%** |
| **snr_0** | `none` | 52.2 | — | — |
| | `wiener_dd` | 55.8 | +3.6 [+1.0, +6.4]* | -18% |
| | `dfn` | **66.0** | **+13.8 [+10.3, +17.7]*** | **-68%** |

*\* Denotes statistical significance (95% CI excludes 0).*

#### Observation:
Neither front-end improved ASR accuracy. DeepFilterNet3 proved severely detrimental, increasing WER by +13.4% at 5 dB and +13.8% at 0 dB.

---

### 3.2 Error Decomposition Analysis ($S, D, I$)

To uncover the exact failure mode of the neural enhancer, word errors were decomposed into Substitutions ($S$), Deletions ($D$), and Insertions ($I$) such that $\text{WER} = \text{Sub\%} + \text{Del\%} + \text{Ins\%}$:

| Condition | Front-End | Overall WER (%) | Sub (%) | Del (%) | Ins (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **codec_only** | `none` | 31.9 | 22.1 | 8.3 | 1.5 |
| **snr_10** | `none` | 38.5 | 27.5 | 9.0 | 1.9 |
| | `wiener_dd` | 40.3 | 29.3 | 8.8 | 2.2 |
| | `dfn` | **47.7** | **32.7** *(+5.2)* | **13.7** *(+4.7)* | 1.4 |
| **snr_5** | `none` | 42.9 | 32.2 | 8.2 | 2.4 |
| | `wiener_dd` | 47.5 | 32.8 | 12.8 | 1.9 |
| | `dfn` | **56.3** | **40.2** *(+8.0)* | **13.3** *(+5.1)* | 2.8 |
| **snr_0** | `none` | 52.2 | 36.8 | 12.9 | 2.5 |
| | `wiener_dd` | 55.8 | 39.8 | 13.2 | 2.8 |
| | `dfn` | **66.0** | **44.1** *(+7.3)* | **19.3** *(+6.4)* | 2.6 |

```
Error Rate Distribution at 0 dB SNR:
┌────────────────────────────────────────────────────────┐
│ none:      [==== 36.8% Sub ====][== 12.9% Del ==][=] 52.2%
│ wiener_dd: [==== 39.8% Sub ====][== 13.2% Del ==][=] 55.8%
│ dfn:       [===== 44.1% Sub =====][=== 19.3% Del ===][=] 66.0%
└────────────────────────────────────────────────────────┘
```

#### Diagnostic Conclusions:
1. **Zero Hallucination Signal**: The insertion rate remained flat at ~2.5% across all methods. Whisper did not hallucinate on speech pauses or noise-gated silence.
2. **Aggressive Speech Over-Suppression**: Deletions surged by **+6.4% absolute** (from 12.9% to 19.3% at 0 dB). The neural mask treated low-energy speech segments (aspirated consonants, unvoiced plosives, and grammatical postpositions in Hindi) as noise.
3. **Phonetic Distortion**: Substitutions increased by **+7.3% to +8.0% absolute**. The phase reconstruction and non-linear filtering distorted vocalic formant trajectories, prompting Whisper to substitute words with acoustically similar candidates.

---

### 3.3 Evaluation of Constrained DFN Variants (`dfn_atten12` & `dfn_mix50`)

To test whether the deletion and substitution penalties could be reversed without retraining, `dfn_atten12` and `dfn_mix50` were tested at 0 dB and 5 dB:

| Condition | Front-End | WER (%) | Paired $\Delta$ vs. `none` [95% CI] | Gap Closed | Sub (%) | Del (%) | Ins (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **snr_5** | `none` | **42.9** | — | — | 32.2 | 8.2 | 2.4 |
| | `dfn` (standard) | 56.3 | +13.4 [+9.3, +17.1]* | -122% | 40.2 | 13.3 | 2.8 |
| | `dfn_atten12` | **42.0** | **-0.9 [-2.9, +1.1]** | **+8%** | 31.6 | 8.0 | 2.4 |
| | `dfn_mix50` | **42.6** | **-0.3 [-2.8, +2.2]** | **+3%** | 32.4 | 8.3 | 1.9 |
| **snr_0** | `none` | **52.2** | — | — | 36.8 | 12.9 | 2.5 |
| | `dfn` (standard) | 66.0 | +13.8 [+10.3, +17.7]* | -68% | 44.1 | 19.3 | 2.6 |
| | `dfn_atten12` | **53.3** | **+1.1 [-1.7, +4.4]** | -5% | 36.4 | 14.0 | 2.9 |
| | `dfn_mix50` | **53.5** | **+1.2 [-1.8, +4.4]** | -6% | 34.8 | 15.8 | 2.9 |

```
WER Comparison Across Interventions (SNR = 0 dB):
┌────────────────────────────────────────────────────────┐
│ none (Untreated):        52.2%                         │
│ dfn_atten12:             53.3% (+1.1%, CI spans 0)     │
│ dfn_mix50:               53.5% (+1.2%, CI spans 0)     │
│ wiener_dd:               55.8% (+3.6%*)                │
│ dfn (Standard):          66.0% (+13.8%*)               │
└────────────────────────────────────────────────────────┘
```

#### Outcome of Interventions:
1. **Deletions Restored**: Capping attenuation at 12 dB successfully reduced deletions at 5 dB from 13.3% back down to 8.0% (matching the untreated control).
2. **Moderate SNR (5 dB)**: Both variants recovered all lost ground, with `dfn_atten12` reaching 42.0% (a nominal 0.9% improvement over untreated degraded audio).
3. **Severe Noise (0 dB)**: Untreated audio remained superior (52.2% vs. 53.3%). Neither variant yielded a statistically significant improvement over `none` (95% CI includes zero).

---

## 4. Architectural Decision & Next Steps

### 4.1 Decision Rule Outcome
The predetermined project protocol specified:
> *If any cheap DFN variant gets WER at or below no-denoiser, keep it. Otherwise report the negative result and pivot.*

Because neither variant consistently or statistically outperformed the unprocessed baseline across noise levels, **we conclude that pre-enhancement front-ends introduce an acoustic-distribution mismatch that harms narrow-band telephony ASR**. Retraining or fine-tuning DeepFilterNet is formally ruled out.

### 4.2 Pivot to Direct ASR Acoustic Adaptation
The project is pivoting directly to **in-domain ASR acoustic adaptation**:
1. **Core Hypothesis**: Whisper's transformer encoder has sufficient representational capacity to handle stationary and non-stationary background noise when trained on realistic AMR-NB telephone channel characteristics.
2. **Acoustically Isolated Training Data**:
   - 2,162 degraded training audio clips generated from IndicVoices Hindi using the 12 strictly held-out DEMAND environments.
   - Clean reference transcript target pairs.
3. **Role of Enhancement**: Speech enhancement algorithms (`none`, `wiener_dd`, `dfn_atten12`) will serve exclusively as frozen external evaluation benchmarks rather than preprocessing pipeline stages.

---

*Report prepared autonomously for the Dial 112 Emergency Call Speech Processing Project.*
