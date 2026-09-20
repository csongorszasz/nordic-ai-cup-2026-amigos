# Prior work: medical-appointment

Snapshot: September 20, 2026. This is a handoff summary, not a replacement for
the detailed experiment ledger in `docs\tries.md`.

## Bottom line

The best **qualified local** result remains **0.8007415 composite**:
accuracy **389/390 = 0.997436**, mean tIoU **0.669612**.

The configuration is **Gemma 4 26B-A4B, FP16, base L1 prompt, Whisper
large-v3-turbo int8**, with a cross-fitted **+0.2-second start / unchanged end**
correction. Full uncached HTTP replay and external prediction parity were
verified before publication. This improved the matched uncorrected baseline
from **0.7886**.

No subsequent candidate has established a sufficiently reliable improvement
and passed the complete promotion gates. Several experiments produced higher
point estimates, but those are not additional qualified best scores.

The repository's best documented historical **official validation** result is
**0.744 with Gemma E4B**. Do not confuse that with the larger model's local
0.8007 result. The continuing improvement loop has not queued official
validation or evaluation attempts.

## How to compare the numbers

The metric is `0.4 * accuracy + 0.6 * mean_tIoU`. Mean tIoU includes every
annotated positive, including missed positives and failed citations.

| Evaluation set | Size | Relevant incumbent |
| --- | --- | --- |
| Entire supplied corpus | 39 conversations, 390 questions, 195 positives | 0.8007415 composite |
| Demonstration-disjoint cohort | 35 conversations, 350 questions, 178 positives | 0.7978423 composite |
| Earlier fixed localizer pilot | 7 conversations, 70 questions, 35 positives | 0.8362 composite |
| Current prompt pilot | 3 conversations, 30 questions, 11 positives | Qualified Gemma reference: 0.888534 |

The four excluded demonstration-source conversations are `sample_10`,
`sample_17`, `sample_18`, and `sample_19`. Never compare a pilot or 350-question
OOF point score directly with the full-corpus 0.8007.

An **oracle** chooses using reference spans. It diagnoses representation or
candidate coverage; it is not an achieved prediction score.

## 1. Early baselines and model progression

| Approach | Recorded outcome | What it established |
| --- | --- | --- |
| Shipped all-yes baseline | 0.200 composite | Protocol floor, not a useful solution. |
| Clause-level NLI | Initial local score 0.469; extensive threshold, search, and trimming sweeps | Topical entailment is not a reliable citation-extent objective. |
| Merged clause context | Large NLI local 0.579; served base configuration official validation 0.545 | More decision context substantially improved positive recall. |
| ModernBERT learned span model | Historical grouped local result about 0.610; official validation 0.606 | Learned localization beat NLI-scored subranges, but decisions remained weaker than the later LLM. |
| Legacy/ModernBERT hybrid | Historical offline 0.647-0.661 | Stronger decisions and better spans can help, but this was not a qualified served result. |
| Qwen2.5-7B, transcript-only LLM | L1 few-shot local 0.665; L0 0.588, L2 0.611 | Whole-transcript LLM decisions were much stronger. |
| Qwen/ModernBERT hybrid | Historical offline 0.705 | A useful diagnostic, not a deployment qualification. |
| Gemma E4B L1 | Local 0.729; official validation 0.744 | Strong local inference with verbatim evidence quotes. |
| Gemma 26B-A4B | Earlier local 0.755; later deployment-matched baseline 0.7886 | Transcript/configuration provenance materially affects comparisons. |
| Qualified boundary correction | Full local HTTP 0.8007415 | The only retained improvement over the matched 26B/turbo baseline. |

Some early NLI metrics were recovered from logs after an old synchronization
command deleted result files. Historical ModernBERT OOF results also predate
repairs to question/passage-source grouping. Keep these caveats when citing
them; do not treat all historical numbers as equally strong evidence.

## 2. Prompt, decoding, and model-capacity experiments

We tested evidence-first output, shortest quotes, wording-matched occurrences,
segment-scoped citations, final-statement preference, more positive examples,
full-context demonstrations, removal of numeric timestamps, local quote
refinement, blind local extraction, and word-index pointers.

Representative full-corpus outcomes with the retained timing correction:

| Experiment | Composite / outcome |
| --- | --- |
| Full-context positive demonstrations | 0.8005; no reliable gain over 0.8007 |
| Additional positive demonstration | 0.7876 |
| Remove numeric timestamps | 0.7939 |
| Prefer the final supporting statement | 0.7938 |
| Anchored localization-only refinement | 0.8003; mostly copied the incumbent |
| Blind local extraction | 0.7825 despite 107 edits |
| Local word pointers | 0.7347 |
| Dense Gemma 31B | 0.7978, with additional decision/grounding loss |
| Native BF16 for the incumbent | 0.7924 |
| Beam width two | Exact score/span tie with greedy; slower |
| FP16 rather than int8 turbo ASR | 0.8062 point estimate, but no reliable demonstration-disjoint gain |

Simple two-citation blending, shorter/longer choices, earlier/later choices,
fixed agreement rules, and quote-completion likelihood reranking also failed
to qualify. More context, more output, or a larger model did not consistently
identify the annotated occurrence.

## 3. Learned localization and reward optimization

| Method | Evaluation and outcome |
| --- | --- |
| Anchored delta-tIoU ridge ranker | 350-question grouped scores 0.7958 / 0.7964 versus 0.7978; rejected |
| Fine boundary-offset grid | Grouped score about 0.7977; no improvement over the simpler +0.2/0 correction |
| E4B quote-localizer LoRA | Full 350-question OOF: adapted 0.7976 versus incumbent 0.7978; substantially better than unadapted E4B 0.7709, but not the incumbent |
| Temporal-reward GRPO | Full 350-question OOF: 0.8040 versus 0.7978; delta +0.006181, conversation-bootstrap interval [-0.022475, +0.034658]; inconclusive |
| Frozen ModernBERT local span-risk head | 70-question pilot 0.6714 versus 0.8362; rejected |
| Fine-tuned encoder plus span-risk head | Same pilot 0.7897; rejected |
| Explicit incumbent-anchor markers | Tied 0.8362 by making no span changes |
| Supervised warm-start followed by span risk | Same pilot 0.8083; rejected |
| Duration-aware boundary correction | Both grouped seeds about 0.7977 versus 0.7978; rejected |

GRPO was a real training experiment: matching-fold SFT warm starts, grounded
validity penalties, exact tIoU, temporal Wasserstein shaping, a frozen KL
reference, verified gradients, and complete OOF coverage. It used an isolated
torch/TRL overlay rather than upgrading serving. Its positive point estimate
is not a proven gain, and one replay change from the newer runtime was not
attributed to reinforcement learning.

These results argue against repeating local span-head or offset sweeps without
a new mechanism. They do not prove that supervised localization is impossible.

## 4. ASR and acoustic alignment

**Medical vocabulary and model replacement.** Cached transcripts showed
spellings such as Panodil/panadil, Activelle/Activel, and Ibumetin/Ibumedin.
The nine examined medication questions already had correct booleans.
An initial Whisper prompt did not fix the examined discrepancies. Repeated
hotwords corrected Panodil once but also omitted a repeated symptom clause.

A pinned medical Whisper large-v3 checkpoint was safely converted and compared
with generic large-v3, retaining turbo controls. It did not establish a useful
medication-recognition advantage and introduced other spelling differences.
No human-reference WER or downstream score gain was claimed. Turbo remains
the incumbent; hints default to off.

**Qwen3 forced alignment.** Exact word-occurrence mapping was preserved.
Invalid timestamp classes beyond the real audio were fixed before evaluation.
The bounded full run scored **0.7838**, below **0.8007**. Separate grouped
calibration reached **0.7935** on the 350-question cohort versus **0.7978**.
Rejected.

**Conditional wav2vec2 CTC alignment.** We aligned bounded audio crops around
the incumbent's exact word occurrences, using an explicit 16 kHz sample clock,
20 ms frame spacing, reversible word ownership for spoken-number expansions,
and scored fallbacks for unsupported inputs. Fixing ordinary split hyphenated
words raised coverage to 191 aligned citations; three `A1c` cases retained
the incumbent.

The complete CTC result was **0.796676 composite / 0.662837 mIoU**, below
**0.800742 / 0.669612**. Its local endpoint-proposal oracle was only
**0.688711 mIoU**. A final four-policy old/new start/end cross-fit scored
**0.797665 / 0.796693** under two grouped seeds, versus **0.797842**.
The timing family is closed for now; no replacement was deployed.

## 5. Global source units and inverse-question ranking

The source-event direction separated:

1. Which occurrence establishes the question's claim.
2. Whether the evidence unit is a short statement, confirmation, or episode.
3. Where its acoustic endpoints fall.

A hierarchy of unmerged clauses/sentences and bounded contiguous episodes
kept interpretation context separate from the cited span. Its full-pool
oracle was **0.882069 mIoU**, falling to **0.838940** after a fixed 32-unit
shortlist plus the incumbent. That was insufficient near-perfect headroom,
not a new achieved score.

Pinned FLAN-T5-base then tested `P(question | source unit, context)` without
putting the target question into the conditioning prompt. Cached/batched
likelihood was checked against direct teacher forcing. On all 390 questions:

| Frozen ranking policy | Composite |
| --- | --- |
| Context-conditioned question likelihood | 0.543352 |
| Source-only question likelihood | 0.599985 |
| Context minus masked-source likelihood | 0.580439 |
| Qualified incumbent | 0.800742 |

All policies regressed, including on the demonstration-disjoint cohort.
No larger fitted selector or synthetic-data expansion was justified by this
small-model experiment.

## 6. Current prompt-focused round

The latest direction prioritizes semantically valid answers and evidence over
imitating questionable reference areas. We reconstructed all 39 recorded
incumbent prompt hashes, preserving historical large-v3 demonstrations and
frozen turbo target transcripts.

The round has **14 development conversations / 140 questions** and **21
confirmation conversations**, with all demonstration sources excluded.
The corpus has already been inspected extensively; this is not a virgin
holdout. Confirmation must not feed another round of prompt tuning.

Development review found genuine status/scope problems, but also valid
alternative occurrences. For example, an intention is weaker evidence than
an available confirmation of completion; conversely, an earlier patient
preference can be perfectly valid even if the reference selects a later one.
Do not impose universal shortest, latest, full-sentence, or clinician-only rules.

With GPU capacity unavailable under the existing policy, we measured CPU
feasibility. Gemma BF16 needed **254.90 s for 64 tokens**, so a pinned
**Qwen3-4B-Instruct-2507, FP32** control was used for a practical pilot.
Its 64-token smoke took **32.25 s**; neither figure is an HTTP latency gate.
The larger cached Qwen3.8-27B was not tested in this matched round.

The complete Qwen pilot used the same 30 questions for every arm:

| Arm | Composite | mIoU | Correct answers |
| --- | --- | --- | --- |
| Base | 0.702600 | 0.504333 | 30/30 |
| Consistent evidence-first `v1` | 0.709789 | 0.560760 | 28/30 |
| Evidence-first plus claim rules, `v1_claim` | 0.656664 | 0.472219 | 28/30 |
| Qualified Gemma reference, different model/runtime | 0.888534 | 0.814223 | 30/30 |

The apparent `v1` gain was rejected: it answered yes to fever while citing
"No fever", and yes to severe tenderness while citing slight tenderness.
`v1_claim` also joined noncontiguous statements, causing a correctly recorded
grounding failure. A higher point score does not excuse those errors.

**Prepared, not yet measured:** one explicitly documented answer-order control
(`claim`) uses the identical claim rules with the original answer-first schema
and demonstrations. It is a post-pilot development extension, not a
preregistered winner. No new medical wording or confirmation feedback is used.

## 7. Dataset and evaluation findings

- The qualified baseline has 21 disjoint positive citations, 17 separated from
  the reference by at least two seconds. Fourteen citations are at least twice
  the reference duration and 22 at most half; these categories overlap.
- All 390 reference endpoints lie on a 20 ms grid. This does not identify the
  annotation generator or guarantee that a 20 ms aligner solves the task.
- The unrestricted raw word-boundary oracle is **0.927250 mIoU** with perfect
  decisions. With the qualified correction and frozen decisions it is
  **0.925339**. These are representation ceilings, not achieved predictions.
- Two references point to opening greetings although both independently cached
  ASR variants place the medical evidence much later: `sample_63_yes_q02`
  and `sample_64_yes_q02`. This is tracked in **#16**, mentioning **@Domynis**.
  No official labels or denominators were changed. Even perfect reference
  matches on those two cases would recover only about 0.00615 composite points;
  they do not explain the entire plateau.
- Exact semantic support, the chosen annotated occurrence, and boundary
  precision are different questions. A semantically correct citation can
  disagree with the one annotated span.

## 8. Engineering and collaboration work retained

Experiments moved to immutable IDUN snapshots with audio/transcript/configuration
hashes, complete question coverage, grouped comparisons, explicit failure
accounting, and isolated dependency overlays. The actual serving model was
identified as 26B rather than assumed to be E4B.

The serving path gained quote-occurrence grounding, provenance-bound calibration,
monotonic deadlines, a preloaded owned worker, real warm-up, recovery, and
uncached HTTP acceptance/publication checks. In-memory MP3 decoding removed a
temporary-file lifecycle hazard and was verified bit-identical on all 39 clips.
`api.py` and `dtos.py` stayed unchanged; request-path inference stays local.

The read-only review of `medical-dominic-add-ngrok` produced issues **#10**
(A/B isolation and double calibration), **#11** (speaker-sidecar provenance),
and **#12** (thinking-control forwarding), each mentioning **@Domynis**.
A later read-only review found corresponding fixes on that branch.
No changes were pushed to the protected branch.

That branch also explored diarization, confidence diagnostics, multi-quote
selection, soft span targets, clinician-authority prompts, complete-sentence
quotes, similar/occurrence demonstrations, explicit reasoning, and thinking
modes. These were inspected rather than silently repeated or treated as
qualified improvements. Its Qwen3.8-27B raw probe was not established as a
transcript-matched comparison with the 0.8007 control.

## What not to repeat without new evidence

- Do not keep sweeping quote lengths, timestamps, offsets, or model size alone.
- Do not infer a whole-corpus win from a favorable pilot or disjoint-only subset.
- Do not use gold overlap to decide when a runtime refinement applies.
- Do not double-apply calibration or mix ASR caches, demonstration sets, and
  model/runtime versions while calling the change "prompt only".
- Do not promote semantic regressions, reinterpret valid alternatives as false,
  or learn to cite unrelated greetings to imitate suspect references.
- Do not use captured evaluation requests as training or calibration data.
- Do not bypass the GPU limits, modify existing services for experimental
  capacity, push to the protected branch, or queue official competition attempts.

## Where to find the evidence

| Material | Location |
| --- | --- |
| Detailed trial chronology, run IDs, uncertainty, and negative results | `docs\tries.md` |
| Verified serving and experiment procedures | `docs\local-serving.md` |
| Domain/architecture definitions | `CONTEXT.md`, `docs\adr\` |
| Qualified release | `offset-release-dns-2732ab44` |
| Full GRPO OOF | `grpo-oof-bc86d18e` |
| Source-event annotation audit | `source-event-audit-31e34870` |
| Global source-unit geometry | `source-unit-geometry-9feabc68` |
| Full inverse-question ranking | `inverse-question-full-20d667a2` |
| Full compound-complete CTC comparison | `ctc-full-compounds-8f14e387` |
| Endpoint-source cross-fit | `endpoint-source-cv-1eace62d` |
| Exact prompt-input freeze | `prompt-round-freeze-bf75822a` |
| Complete first prompt pilot | `prompt-qwen4b-pilot-e6075a85` |

Pulled run artifacts live under `results\idun_runs\<run-id>\artifacts`.
They, model weights, transcripts, and captures are gitignored. New substantial
verified changes are committed and pushed on `csongor/medical-dominic-llm`.
