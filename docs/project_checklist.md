# Project Checklist: Hindi Emergency-Call (Dial 112) Triage under Telephone Degradation

Last updated: 2026-10-18

## 0. Objective (one paragraph)

Given a noisy 8 kHz AMR-NB emergency call in Hindi, produce an accurate transcript and then a correct triage suggestion (category, priority, key facts) through a retrieval-augmented layer. Denoising was a candidate way to improve ASR accuracy. It was tested and did not help, so the main contribution is now **adapting the ASR to degraded telephone audio**, with the denoising work reported as a rigorous comparison.

Pipeline: `audio -> (optional front-end) -> Hindi ASR -> RAG triage -> decision support`

## 1. Status snapshot

### Done
- [x] Hindi speech extracted from IndicVoices (`extraction.py`)
- [x] Telephone-channel simulator (`degrade.py`): 16 kHz -> 8 kHz, noise mixed at target SNR, AMR-NB codec round-trip, delay-aligned against `clean8k`. Run on MUSAN and DEMAND noise
- [x] Gate 1: the channel hurts ASR (Whisper large-v3 via Groq, MUSAN: clean 27.4%, codec-only 31.9%, 10/5/0 dB = 38.5/42.9/52.2%)
- [x] Gate 2: pretrained DeepFilterNet3 raises WER by +9.2 / +13.4 / +13.8 points at 10/5/0 dB (CIs exclude 0). Wiener baseline gives no gain (+1.8 / +4.6 / +3.6)
- [x] Error decomposition: DFN damage is deletions and substitutions, not insertions (hallucination is not the main cause)
- [x] Cheap variants (`dfn_atten12`, `dfn_mix50`): about equal to no-denoiser at 5 dB (42.0 / 42.6 vs 42.9) and slightly worse at 0 dB (53.3 / 53.5 vs 52.2). None is significant
- [x] Decision: no denoiser as a preprocessing stage; pivot to ASR adaptation
- [x] DEMAND re-degraded with a noise hold-out (per report): test DLIVING/NPARK/TCAR, dev OOFFICE/STRAFFIC, train = 12 other environments
- [x] Findings note and experiment report saved in the project

### Not done
- [ ] Local (non-Groq) baseline
- [ ] ASR adaptation (main contribution)
- [ ] RAG triage layer and its evaluation
- [ ] Realistic call test set
- [ ] Latency benchmarks
- [ ] Final write-up

### Known caveats (must be reflected in the report)
- All WER so far is from the Groq API. Groq is not bit-repeatable (clean WER 26.9 vs 27.4 for identical audio) and cannot be fine-tuned. Groq and local WER are not comparable
- Variant results (`dfn_atten12`, `dfn_mix50`) exist for MUSAN only
- Old DEMAND results used all 17 environments (no hold-out); they are superseded by the re-degraded set
- 100 test utterances give CIs of about +/-3 WER points. Smaller effects cannot be detected
- Mechanism claims (why DFN deletes speech) are hypotheses, not tested
- Only pretrained DFN was tested. A DFN fine-tuned on 8 kHz telephone data has not been tried
- Hindi only; channel is simulated, not real calls

## 2. Checklist

### Phase 1: Pre-training checks (do first; about half a day)

**1.1 Verify the DEMAND hold-out actually applied**
- [ ] Find the `noise files per pool: ...` line from the re-degrade run (or re-run with `--resume` off and check it) and confirm test = DLIVING/NPARK/TCAR, dev = OOFFICE/STRAFFIC, train = the other 12
- [ ] Confirm no "no noise files landed in the pool" warning appeared
- Done when: pool sizes are recorded in the report

**1.2 Verify no data leakage between train and test**
- [ ] Compare `utt_id` and speaker columns between the training manifest and the 100-utterance test manifest. Overlap must be zero
- [ ] Confirm training utterances come from the IndicVoices train split and test from its test split
- [ ] Check how many unique clean utterances sit behind the 2,162 degraded training clips (each utterance may be degraded several times). Split train/dev by utterance and speaker, never by degraded file
- [ ] Compute total training hours
- Done when: a short leakage table (overlap counts = 0) is in the report

**1.3 Correct the experiment report**
- [ ] State clearly that section 3 results are MUSAN only
- [ ] Reword deletion claim: variants return deletions to baseline at 5 dB only; at 0 dB they stay above (14.0 / 15.8 vs 12.9). Mention Wiener's deletion rise at 5 dB (8.2 -> 12.8)
- [ ] Reword mechanism as "consistent with over-suppression", not established
- [ ] Remove "retraining DFN is formally ruled out" and replace with "not pursued; pretrained and inference-time variants tested"
- [ ] Add "no detectable benefit within about +/-3 points", not "no benefit"
- [ ] Fix small mismatches: +7.4 vs +7.3 substitution rise; MUSAN partitioned by noise file (not utterance hash); remove 15/20 dB SNRs unless they were run
- Done when: every number in the report matches the CSVs

### Phase 2: Local baseline (needs the RTX machine; 1-2 days)

**2.1 Hardware**
- [ ] On the RTX machine run `nvidia-smi`; record GPU model and VRAM
- [ ] Install Linux (training data loaders are Linux-only) and the NVIDIA driver, then confirm with `nvidia-smi`
- [ ] Rough model choice by VRAM: >=24 GB large-v3 with LoRA; 12-16 GB large-v3 with 8-bit/gradient checkpointing, or medium; <=8 GB small or medium
- Done when: VRAM and chosen model size are written down

**2.2 Local environment**
- [ ] Python 3.12 `uv` venv for ASR, separate from the DFN environment
- [ ] Install faster-whisper with CUDA support; confirm it runs on a 5-file test
- [ ] Download the Whisper model you will fine-tune (local directory) and note exact name/revision
- [ ] Copy the project scripts and the degraded test sets (MUSAN test, DEMAND test)

**2.3 Local baseline runs**
- [ ] Use `eval_asr.py` with the local backend and a **separate artifacts directory** (cache files are named by model only, so never share a directory with Groq results)
- [ ] Freeze decoding settings now: `language=hi`, `beam_size`, `temperature`, and the `normalize()` function. Do not change them after this point
- [ ] Run on the new DEMAND test set and on MUSAN test: clean, clean8k, codec-only, 0/5/10 dB for `none`
- [ ] Run `wiener_dd` and `dfn_atten12` as frozen benchmarks on the same conditions (outputs exist from `denoise.py`; re-run `denoise.py` for any regenerated test files)
- [ ] Compute bootstrap CIs with `eval_enh.py`
- [ ] Baseline the **same Whisper size you will fine-tune** (adaptation must compare same model before vs after)
- [ ] Record a local-vs-Groq comparison on the clean and none conditions
- Done when: a local WER table (with CIs) exists for every test condition

**2.4 Normalisation check (cheap; before trusting any WER)**
- [ ] Print 20 clean-condition hypothesis/reference pairs side by side
- [ ] Check whether part of the 27% clean WER comes from normalisation (digits vs words, Latin vs Devanagari English words, punctuation, spelling variants)
- [ ] If so, fix `normalize()` once, then re-score all caches (no new ASR calls needed)
- Done when: you can say how much of the clean WER is normalisation vs real errors

### Phase 3: ASR adaptation (main contribution; 4-7 days)

**3.1 Training data**
- [ ] Use the degraded training clips (DEMAND train environments) paired with clean reference transcripts
- [ ] Hold out a dev subset by speaker/utterance. Dev noise environments (OOFFICE/STRAFFIC) stay dev only
- [ ] Optionally mix in clean and codec-only clips (about 10-20%) so clean-speech performance does not collapse
- [ ] If hours are small, consider generating more degraded copies of training utterances with different SNRs and noise files (training split only)
- [ ] Never put test environments or test utterances in training

**3.2 Training**
- [ ] LoRA (PEFT) on the Whisper model from 2.3, unless VRAM allows full fine-tuning
- [ ] Hindi language/task tokens set; same normalisation for targets as for scoring
- [ ] Start with small learning rate (about 1e-5 full / 1e-4 to 1e-3 LoRA), few epochs, evaluate on dev every epoch, keep the best checkpoint
- [ ] Log train/dev loss and dev WER; stop on dev WER rise (small data overfits quickly)
- [ ] Save adapters and merge into the base model

**3.3 Deployment format**
- [ ] Convert merged model to CTranslate2 (`ct2-transformers-converter`) so `eval_asr.py` can run it as a local model
- [ ] Sanity check: transcribe 5 files with both HF and CTranslate2 versions; transcripts should be near-identical

**3.4 Evaluation (run test once)**
- [ ] Evaluate adapted model on DEMAND test (unseen environments) and MUSAN test (unseen corpus), at clean, codec-only, 0/5/10 dB
- [ ] Ablation table: front-end (none / wiener_dd / dfn_atten12) x ASR (base / adapted)
- [ ] Paired bootstrap CIs for adapted vs base on each condition
- [ ] Clean-audio WER check (adaptation must not hurt clean speech much)
- [ ] Error decomposition (S/D/I) for adapted vs base
- Done when: you can state, with CIs, how many WER points adaptation recovers on unseen noise, and the share of the noise gap closed

**3.5 Fallbacks**
- If adaptation gives no significant gain: report it, and check training data size/quality and learning rate before concluding anything
- If clean WER collapses: add clean/codec-only data and lower LoRA rank/learning rate

### Phase 4: Triage layer (can start now on another machine; 5-8 days)

**4.1 Knowledge base**
- [ ] Collect triage protocol sources (emergency dispatch protocols, first-aid and fire/police/ambulance guidance, Dial 112 / ERSS public documents). Record source URL, licence and date for each
- [ ] Convert to clean text; keep Hindi and English versions where available
- [ ] Chunk with metadata (category, urgency, source)

**4.2 Pipeline**
- [ ] Choose a multilingual embedding model (supports Hindi) and a vector store (FAISS or Chroma)
- [ ] Retrieval: top-k protocol chunks for a transcript
- [ ] Agentic triage step: an LLM extracts incident type, location cues, victim count, urgency; retrieves protocols; outputs a structured result (category, priority, recommended dispatch, confidence, supporting excerpts)
- [ ] Define the output schema and keep it fixed

**4.3 Test set and metrics**
- [ ] Build 30-60 labelled emergency scenarios in Hindi (text first, then synthesised or recorded audio); label each with category and priority. Include a few real or realistic recordings if you can
- [ ] Check whether IndicVoices `scenario` metadata gives usable coarse labels (probably not emergency-specific)
- [ ] Metrics: category accuracy, priority accuracy, key-fact extraction rate
- [ ] Run triage on ground-truth text, base-ASR transcripts and adapted-ASR transcripts to measure error propagation from ASR to triage

### Phase 5: System, latency, write-up (4-6 days)

**5.1 Latency**
- [ ] Time each stage (ASR, retrieval, LLM) per call on available hardware, report mean/p95
- [ ] Record hardware specs next to every number

**5.2 Optional extras (only if time remains, in this priority)**
- [ ] Fine-tune DFN on 8 kHz telephone training data and test it with the same WER protocol
- [ ] Tune Wiener on the dev split
- [ ] Edge deployment test on the Ryzen machines

**5.3 Write-up**
- [ ] Fix "Agrawal et al." vs "Agro et al." in the proposal
- [ ] Sections: channel simulation; Gate 1 and Gate 2 (incl. negative DFN result and error decomposition); local baseline; ASR adaptation ablation; triage results; latency; limitations (Hindi only, simulated channel, small test set, Groq non-determinism)
- [ ] Every number traceable to a CSV/script output; scripts and seeds listed
- [ ] Final consistency pass: report numbers vs CSVs, figure/table labels, units

## 3. Cutting order if time runs short

Drop in this order: edge deployment, DFN fine-tuning, Wiener tuning, real-call test set (use realistic scenarios only). Phases 1-4 and the latency benchmark are the core.

## 4. Next steps (in order)

1. **Today: leakage and hold-out checks (Phase 1.1 and 1.2).** Compare utterance and speaker IDs between the training and test manifests, and confirm the DEMAND pool sizes. Nothing else is safe until this is done.
2. **Today: 20-pair normalisation check (2.4).** It only needs the existing caches and tells you how much of the 27% clean WER is real.
3. **Fix the report (1.3)** using the list above, so you are not carrying wrong claims forward.
4. **Run `nvidia-smi` on the RTX machine and send me the output.** The GPU model and VRAM decide the Whisper size and whether LoRA is needed.
5. **Move to Linux and set up the local ASR environment (2.1-2.2) on the RTX machine**, with a separate artifacts directory for local runs.
6. **Run the local baselines (2.3)** with frozen decoding settings, including the model size you will fine-tune.
7. **In parallel, on another machine, start collecting triage protocol documents (4.1).**
8. **Then I write the fine-tuning script** (LoRA, dev-based stopping, CTranslate2 conversion) sized to your VRAM, once steps 1-4 are done.

## 5. Open questions to settle

- Does your guide expect a signal-enhancement contribution given the title? If yes, plan the optional DFN-fine-tuning experiment (5.2) rather than dropping it
- What triage categories and priorities should the system output (use Dial 112 / ERSS definitions if you can find them)?
- Is there any real or realistic Hindi emergency-call audio you are allowed to use for testing?
