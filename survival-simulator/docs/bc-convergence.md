# Offline BC convergence

The purpose is faithful teacher copying, not merely exhausting an update budget.
The BC control stays BC-only. A separately approved corrected DAgger challenger is
described below; PPO is not part of this comparison.

## Flow

1. Freeze the existing short demonstration prefix and establish a copying control.
2. Collect complete episodes from `controller-turnaway-wall-aware.json` into immutable
   train/validation shards. Split by world, never by adjacent frames.
3. Repeatedly fit the fixed data, counting actual optimizer steps. Recompute recurrent
   states from current weights; never reuse an obsolete collection model's hidden states.
4. Diagnose a plateau using distance, stopping energy, travel/turn angles and reproduction
   precision/recall. Missing positive examples are unassessed, not perfect classification.
5. Make one demonstrated correction at a time and repeat the same-data control.
6. Fit the full corpus with sequence minibatches, recurrent burn-in and whole-history
   validation. Evaluate frozen checkpoints on full games separately.
7. Qualify teacher-like native score before calling the BC phase successful. A resource
   cap is a pause; a poor plateau is a diagnosis, not convergence.

## Decisions supported by the controls

The original trainer took one optimizer step per complete dataset epoch. The first
diagnostic therefore made only five steps. Offline fitting removes repeated simulation
and batches encoding across frames while keeping independent team pooling. Tests compare
both outputs and gradients with ordinary frame-by-frame forward passes.

Longer optimization substantially improves the current model: the 16-frame control
reached loss around `1e-5`. The 160-frame control instead plateaued near `0.074`, dominated
by 30 badly wrong headings with tanh angular latents near saturation.

`model.angle_head="vector_bc"` is the isolated correction for that measured failure.
It regresses unit-circle sine/cosine targets directly, with half squared vector error
(equal to the prior `1-cos(error)` when both vectors are unit length). `atan2` is used
only to decode deterministic actions. The encoder, GRU, distance and reproduction
weights are preserved during explicit migration; the new head begins facing forward
with no turn. A new optimizer is used for that correction.

The checkpoint action version is `bc-vector-angles-v2`. Legacy bounded checkpoints
remain supported. Stochastic collection/PPO with this BC-specific head is rejected,
not silently assigned an incorrect action distribution. This does not introduce a
stop head, hidden-state teacher, or teacher-action fallback.

## Commands

These commands launch real work. Use IDUN for the experiments and a frozen source release.
The short controls are **not** full-game or held-out evidence.

```powershell
# Complete teacher episodes, preserving all expert data.
python -m src.training.demonstrations --config .\configs\imitation-gru.json --output .\training-results\bc-corpus --train-worlds 8 --validation-worlds 4 --workers 4

# Existing small-data control, with the unchanged bounded model.
python -m src.training.offline_bc --checkpoint .\training-results\bc-authorized-25405107\checkpoint.pt --output .\training-results\bc-fixed --steps 1000 --device cuda

# Continue the same frozen data; steps is a total target, not an additional count.
python -m src.training.offline_bc --checkpoint .\training-results\bc-authorized-25405107\checkpoint.pt --resume .\training-results\bc-fixed\best.pt --output .\training-results\bc-fixed-continued --steps 10000 --device cuda

# Explicit, versioned correction after reproducing the angular saturation failure.
python -m src.training.offline_bc --checkpoint .\training-results\bc-authorized-25405107\checkpoint.pt --warm-start .\training-results\bc-fixed-continued\best.pt --angle-head vector_bc --output .\training-results\bc-vector-control --steps 10000 --device cuda
```

Each control writes metrics, copying curves, resumable checkpoints and runtime-compatible
policy descriptors. `training_fit` means only that the available fixed labels meet the
declared diagnostic tolerances; it cannot certify unobserved reproduction or predators.

For the complete corpus:

```powershell
python -m src.training.bc_corpus_fit --corpus .\training-results\bc-corpus --checkpoint .\training-results\bc-vector-control\best.pt --output .\training-results\bc-corpus-fit --steps 1000 --sequence-length 64 --burn-in 256 --batch-sequences 4 --validate-every 250 --device cuda
```

The corpus fitter logs sequence-minibatch metrics and evaluates full validation histories
from reset. It preserves data, sampler, optimizer and scheduler state. Native acquisition
ticks are counted once, separately from repeated example exposures. It publishes
immutable score-evaluation snapshots for the existing IDUN watcher. Keep the evaluator
separate from the GPU job; a full pending queue pauses fitting instead of dropping models.

The first corpus experiment uses 256 burn-in ticks: the teacher's slow score EMA retains
`0.98**256`, less than 1% of a preceding initial value. That motivates a starting history
window, not a guarantee that the GRU learns the same memory. The 64 scored ticks cover
the corresponding approximate EMA time scale; full-history validation detects mismatch.
Four sequences per optimizer minibatch reduce single-trajectory noise while retaining
bounded sequential backpropagation memory. These settings remain explicit experiment
choices, and measured validation behavior determines continuation or revision.

Validation patience is counted in complete data passes. The current scheduler halves
the learning rate after a documented plateau; a plateau after lower-rate continuations
is reported as `plateau_needs_diagnosis`, never teacher-level qualification.
Presets are measured starting hypotheses, not optimality proofs.

Read `manifest.json`, `summary.json`, `copying-metrics.jsonl`, `copying-curve.png` and
the separate `progress/learning-curve.png`. A low supervised loss does not replace
full-horizon native-score and real endpoint checks.

## Corrected DAgger versus continued BC

The first diagnostic uses one common checkpoint and the same model/optimizer state
for both arms. Optional peer context preserves public teammate ID/energy links;
its migration zeros the previously unused Agent-input columns and their Adam
moments, preserving initial predictions. No teacher action is used as an input.

`src.training.dagger_collection` provides three explicit operations:

- `anchor`: freeze the shared starting model, optimizer, RNG and source/data identity.
- `collect`: query the teacher at every actual state, advance the student's recurrent
  state every tick, and execute deterministic student actions or the **entire team's**
  teacher actions. Save teacher labels and executed actions separately.
- `aggregate`: retain every original expert/validation shard and append immutable
  recovery episodes. Reconstruct previous-action features from executed actions,
  never from counterfactual teacher proposals.

The initial round collects two new, disjoint worlds with teacher probability 0.5.
This is a declared intervention/coverage choice, not an optimal value. Even two
maximum-horizon worlds cannot exceed the existing expert-frame count, preserving an
expert majority. The aggregator verifies this condition and does not silently evict data.
Validation and benchmark worlds are excluded from collection.

Both fitting arms use `--data-fork` from the **same anchor**. Unlike exact `--resume`,
this experiment preserves weights and Adam state but deliberately resets the data
sampler and scheduler in both arms and holds the learning rate fixed. Model, loss,
sequence length, burn-in, batch size, update budget and evaluation protocol match.
Only the DAgger arm gets the extra learner-state labels. Added collection and
actual frame/agent exposures are reported separately, not called identical compute.

Ordinary `--fork` still rejects corpus changes; only the explicit data-comparison
path accepts a checksum-verified expert-plus-recovery corpus with unchanged validation.
`--data-fork` cannot simultaneously migrate model inputs. Prepare the common anchor first.

Use the checked-in `job_corrected_dagger_collect.slurm`, `job_corrected_dagger_fit.slurm`
and `job_corrected_dagger_evaluate.slurm` only after reviewing their bounded parameters.
The first comparison adds 500 optimizer steps to each arm. It is a diagnostic round,
not a claim of DAgger convergence or permission to promote a model based on training loss.
