## Branch and scope

Implementation/review branch: [`medical-dominic`](https://github.com/csongorszasz/nordic-ai-cup-2026-amigos/tree/medical-dominic) at [`8487686`](https://github.com/csongorszasz/nordic-ai-cup-2026-amigos/commit/8487686bc824e331ba43062fee538f6b74b3ce7e).

Investigation completed **September 18, 2026**. Inference must remain local, finish within 60 seconds per conversation, and fit an 8 GB laptop GPU; A100-class hardware is available for training.

## Executive decision

Stop tuning the current pipeline as the final architecture:

```text
ASR -> clause windows -> rule-based proposition -> generic MNLI
    -> NLI-scored word ranges -> greedy trimming
```

Keep local ASR and timestamp mapping, but replace clause-anchored verification/localization with a **task-trained full-transcript model** that jointly predicts:

- `SUPPORT`, `REFUTE`, or `NOT_MENTIONED`;
- global evidence start/end tokens;
- evidence-token saliency;
- expected span tIoU;
- structured agreement for medication, dose, unit, duration, route, anatomy, polarity, temporality, and speech act.

Recommended deployment shape:

```text
ASR + accurate word alignment
-> timestamped full transcript, preferably with speaker turns
-> batched full-context encoder for all 10 questions
   |- support/refute/not-mentioned head
   |- start/end + evidence-token heads
   |- expected-tIoU head
   `- structured slot-relation heads
-> optional legacy NLI verifier for uncertain cases only
-> score-aware decoder
-> optional small frame-level boundary refiner
```

## Findings

1. **The only proven service score is 0.457.** T023 used the actual local turbo/base/capped configuration. T025 reports 0.579, but it is evaluated on the same 39 training conversations used for architecture and threshold tuning, with large-v3 transcripts and large NLI rather than the served configuration. It is not an unbiased deployment estimate. See [`docs/tries.md`](https://github.com/csongorszasz/nordic-ai-cup-2026-amigos/blob/medical-dominic/medical-appointment/docs/tries.md).

2. **Localization, not NLI capacity, dominates the loss.** T025 reaches 0.938 answer accuracy but only 0.340 mean tIoU. The global word-range oracle is 0.894, while the NLI-selected neighbourhood oracle is about 0.656. A perfect ranker restricted to the current neighbourhood still cannot recover the anchor-selection loss.

3. **ADR-0003 is too narrow as the final solution.** A supervised ranker is directionally correct, but confining it to clause +/- 1 around the most-entailing clause preserves roughly 0.24 tIoU of avoidable search loss. Localization must be global over the transcript or jointly learned with answerability.

4. **The served candidate cap is structurally biased.** [`answer.py`](https://github.com/csongorszasz/nordic-ai-cup-2026-amigos/blob/medical-dominic/medical-appointment/answer.py#L95-L97) sorts candidates by shortest length before truncating. In a 40-word unfiltered region, `MAX_CANDIDATES=120` retains almost entirely one- to four-word spans. Gold spans have median duration 2.88 seconds, maximum 14.2 seconds, and 50/195 exceed four seconds. The served `MAX_RANGE_WORDS=10`/`MAX_CANDIDATES=120` configuration has no reported oracle.

5. **The rule-based propositioniser is brittle.** A direct corpus audit found roughly 50 clearly malformed rewrites among 390 questions, including forms such as `The pains in have the hands...` and `The patient ask for penicillin did.` It improved generic MNLI relative to raw questions, but it should not survive into a task-trained model.

6. **The numeric guard is effectively dead.** Existing trials show no decision changes when it is disabled. It also cannot link a number to its medication/unit, distinguish historical from current values, or handle the many non-numeric hard negatives.

7. **Question wording leaks useful priors.** A character n-gram logistic model using only question text reached 0.726 leave-one-conversation-out accuracy. At a conservative threshold it identified 47/53 off-topic questions with zero false positives. This is useful as a weak ensemble feature and robust CPU fallback, not as the main answerer.

8. **The real data is small enough to curate.** The supplied audio totals about 79.5 minutes. Unique positive evidence occupies about 9.2 minutes, or 23.4 minutes with +/- 3 seconds of context. Correcting evidence-region transcripts, speaker roles, and contradiction passages is likely higher value than another generic model sweep.

9. **Serving/reproducibility has gaps.** The current Dockerfile copies only four Python files despite imports from several other modules and omits model dependencies. `local/setup_env.sh` downloads large-v3 and base NLI, while the runbook claims turbo and potentially large NLI are ready. The filename-based transcript cache can also reuse stale audio if a filename is repeated; serving should disable it or key by audio SHA-256 plus the complete decoding configuration.

## Component decisions

| Component | Decision | Replacement |
|---|---|---|
| Whisper-only ASR selection | Reopen | Bake off Qwen3-ASR-0.6B + Qwen forced aligner, Parakeet-TDT-0.6B, large-v3-turbo, and distil-large-v3 using downstream score, critical-slot errors, timestamp oracle, latency, and VRAM. |
| Word timestamps | Keep and improve | Forced alignment/native timestamps plus learned boundary offsets. |
| Clause windows | Remove as semantic/search unit | Process the complete consultation in a long-context bidirectional encoder. |
| Proposition rules | Remove from main path | Train directly on original questions. Keep only as an optional legacy-NLI ensemble input. |
| Generic MNLI | Demote | Use as teacher or verifier for low-margin predictions, exposing entailment/contradiction/neutral logits. |
| Numeric guard | Replace | Learned/structured linked-slot agreement with unit normalization and discourse state. |
| Candidate enumeration and greedy trim | Remove | Direct global start/end or biaffine span prediction with tIoU-aware supervision. |
| Per-question sequential inference | Replace | Batch all ten question/transcript pairs; later add lightweight cross-question consistency. |
| All-true fallback | Replace | Tiny question prior plus lexical/slot retrieval and a hard internal deadline returning accumulated valid results. |
| Direct 4B-10B audio LLM | Teacher/ablation only | Too risky for exact boundaries, latency variance, and 8 GB deployment; use audio grounding only as a local boundary refiner initially. |

## Recommended model and training objective

Start with **ModernBERT-base** and bake off ModernBERT-large after the pipeline works. The full consultation fits its 8,192-token context, avoiding retrieval gating and repeated clause encoding.

```text
L = three-way classification
  + start/end span loss
  + evidence-token saliency loss
  + soft-IoU or expected-tIoU loss
  + structured slot-relation loss
  + negative-span suppression
  + same-conversation minimal-pair contrastive loss
```

For hard negatives, annotate the **contradicting passage** even though the API returns null for negative answers. This can expand useful rationale supervision from 195 positive spans toward all 337 positive/hard-negative questions and directly teaches why a near-miss is false.

Generate minimal counterfactuals that alter exactly one causal field: drug, vaccine, dose, unit, frequency, duration, route, lab value, anatomy, left/right, normal/abnormal, speaker, historical/current, suggested/agreed/refused. Also train evidence-deletion examples, wrong-neighbour spans, corrections, hypotheticals, and superseded instructions.

## Score-aware decoding

Accuracy-optimal thresholding is not score-optimal because a missed positive loses both accuracy and tIoU. If `p` is the calibrated positive probability and `q` is expected tIoU for the proposed span, the idealized decision is:

```text
predict yes when p > 0.4 / (0.8 + 1.2q)
```

At current span quality `q ~= 0.36`, the threshold is about 0.325, explaining why tau near 0.3 helped. At `q ~= 0.70`, the threshold falls to about 0.244. Learn/calibrate both quantities from grouped out-of-fold predictions and optimize the official score directly.

## Prioritized work

- [ ] **P0: establish an honest baseline.** Serve the exact best candidate on the 8 GB machine; report ASR model, alignment, verifier, all caps, accuracy by type, mean tIoU, P50/P95/max latency, and VRAM. Do not mix large-v3 cached transcripts with a turbo serving claim.
- [ ] **P0: add conversation-grouped evaluation.** Use five-fold GroupKFold for iteration and LOCO for finalists. Persist out-of-fold probabilities and spans for calibration. Never tune thresholds on in-fold predictions.
- [ ] **P1: curate the real corpus.** Correct speaker-attributed evidence-region transcripts and annotate contradiction passages for hard negatives.
- [ ] **P1: implement the full-transcript three-way span model.** Direct global answerability and start/end prediction; no clause anchor.
- [ ] **P1: add score-aware calibration.** Optimize the exact `0.4 * accuracy + 0.6 * mean_tIoU` objective on grouped out-of-fold predictions.
- [ ] **P2: run the ASR/alignment bake-off.** Compare Qwen3-ASR/aligner, Parakeet, turbo, and distil by downstream metrics rather than generic WER alone.
- [ ] **P2: add linked-slot and discourse features.** Cover medication/dose/unit/frequency/duration, polarity, role, historical/current, suggested/agreed/refused, and corrections.
- [ ] **P2: generate minimal-edit counterfactual training data.** Train on clean and realistic ASR-corrupted forms.
- [ ] **P2: batch all ten questions and add a conservative question-prior ensemble/fallback.**
- [ ] **P3: add selective local re-decoding.** Re-run only ambiguous 5-15 second evidence regions with a stronger beam/model.
- [ ] **P3: add a query-conditioned frame-level boundary refiner.** Predict offsets around the text-selected region to exceed the current word-time ceiling.
- [ ] **P0 serving hardening:** real dummy inference warm-up, 50-second internal deadline, partial valid fallback, audio-content cache keys or no serving cache, pinned/pre-downloaded weights, and a functional deployment image.

## Acceptance criteria

- Evaluation is strictly conversation-grouped and produces reproducible out-of-fold predictions.
- Positive recall remains at least 0.93 and hard-negative accuracy at least 0.92 out of fold.
- Mean tIoU reaches at least 0.60 out of fold before pursuing direct audio-LLM complexity.
- First redesign target: approximately 0.94 accuracy / 0.65 mean tIoU / **0.766 score**.
- P95 end-to-end latency is below 50 seconds and maximum below 58 seconds on the target 8 GB GPU, with no timeouts.
- `/predict` never raises and always returns a valid response of the required length.
- The complete served configuration improves materially over the proven validation score of 0.457 before the one-shot evaluation.

## Primary references

- ModernBERT: https://arxiv.org/abs/2412.13663
- Qwen3-ASR and forced aligner: https://arxiv.org/abs/2601.21337
- Parakeet-TDT: https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3
- WhisperX alignment: https://arxiv.org/abs/2303.00747
- VSLNet span-based temporal localization: https://arxiv.org/abs/2004.13931
- QD-DETR query-dependent temporal localization: https://arxiv.org/abs/2303.13874
- Frame-level audio grounding: https://arxiv.org/abs/2609.15215
- SQuAD 2.0 no-answer supervision: https://arxiv.org/abs/1806.03822