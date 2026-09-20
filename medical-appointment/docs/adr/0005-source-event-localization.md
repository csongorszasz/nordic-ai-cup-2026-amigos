# ADR-0005: Separate source-unit selection, context, and acoustic boundaries

- **Status:** Experimental; not enabled in serving
- **Date:** 2026-09-19
- **Related:** ADR-0001, ADR-0003, ADR-0004

## Context

The qualified 26B/turbo pipeline scores 0.8007415 locally with 389/390 decisions
correct but mean tIoU 0.6696119. The reproduced annotation audit finds 21
disjoint positive citations, including 17 separated by at least two seconds.
Both overlong and overshort citations occur. In the same consultation, a
normal-result question cites a short result statement while an examination
question cites a multi-turn episode.

This contradicts the general interpretation of ADR-0003's claim that every
reference is the minimal entailing passage. It does not invalidate or change
the historical NLI implementation. ADR-0004's distinction between decision
context and cited words remains useful.

Every supplied reference endpoint lies on a 20 ms grid, but that does not
identify the annotation generator or prove acoustic precision. The qualified
word-offset representation has a perfect-selection, frozen-decision ceiling
of 0.925339 mean tIoU in the audit, not a near-perfect achieved score.

## Experimental decision

Keep source occurrence, evidence extent, and endpoint timing separate:

- Build source units throughout the conversation, not only around the existing
  quote. Preserve one-word replies and unmerged clause/sentence atoms.
- Build bounded episode candidates from contiguous sentence sequences. These
  are possible extents, not claims that a clinical action actually occurred.
- Attach neighboring sentence context separately from the citation. Do not
  enlarge the scored span merely because its interpretation needs more text.
- Keep repeated wording as distinct occurrences with exact word indices.
- Preserve the incumbent as an already calibrated candidate, applying its
  boundary correction exactly once.

The initial model-free geometry recipe uses at most eight sentence atoms,
64 words, and 30 seconds per episode, plus one context sentence on each side.
A 32-unit shortlist is drawn round-robin across clause, sentence, episode, and
incumbent-neighborhood queues, with the incumbent additional and retained first.
The initial lexical retrieval keeps short clinical tokens, numbers, and
negation. It is a coverage diagnostic, not an entailment or decision rule.

Future source ranking tests the likelihood of generating the supplied question
from a candidate unit and its necessary context. The question is a target,
not a prompt shortcut. Conditional acoustic alignment is a separate arm;
neither a large candidate oracle nor a small-model feasibility pass establishes
an end-to-end gain.

## Gates and consequences

`probe_source_localization.py` requires the provenance-bound annotation audit,
reproduces its incumbent, and freezes its candidate recipe before measuring
coverage. Runtime candidates contain no labels, reference spans, or oracle
features. Reference-assisted diagnostics are written separately, with every
positive retained under both perfect-decision and frozen-decision denominators.

All learned policies remain conversation-grouped and demonstration-disjoint.
New model work uses isolated IDUN allocations within existing resource limits.
The current endpoint, legacy window behavior, `api.py`, and `dtos.py` stay
unchanged. Only a matched quality winner passing the full uncached HTTP and
failure-recovery gates can become a serving option.
