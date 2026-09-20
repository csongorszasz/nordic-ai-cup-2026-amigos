# Policy pipeline

## Decisions

| Decision | Reason | Alternative / when to reconsider |
| --- | --- | --- |
| Keep the reference benchmark engine unchanged | Rendering consumes world RNG; cached observations and mutation edge cases are part of the contract. | Training uses an RNG-equivalent headless core that consumes the historical render draws without allocating/painting surfaces. The ordinary core remains authoritative for selection. |
| Hierarchical controller before learning | Direct controls, shared target assignment, patch camping, and population-aware breeding are substantially cheaper than per-agent candidate lattices and remain independently competitive. | Scalar/vectorized backends remain regression oracles. A learned policy must beat the hierarchical controller on score and HTTP budget. |
| Privileged teacher -> DAgger -> optional learned residual | A full-state planner can supply coordination and long-horizon labels that the old reactive teacher does not contain. | Do not start a large neural run until simulator throughput and teacher quality are measured. |
| Structured, ragged entity features | DTOs already describe objects. Per-type encoding and pooling handle arbitrary populations without pixels or dropping agents. | Configurable attention; require score evidence before adding more model complexity. |
| Shared actor, identity-keyed GRU, pooled team critic | Agents share controls and the controller receives all observations. Births/deaths change actor state, not the ongoing team objective. | `model.memory=none`, `model.team_context=false`, or `model.critic=local` are ablations. This is centralized control, not strict decentralized MAPPO. |
| Native team reward, no reward clipping | Score is elapsed time + eaten energy/1000 - energy lost to predation/100. Population is not a reward. | Any future shaping must be named, logged, and selected against the unmodified native score. |
| Serialized stateful HTTP default | The controller benefits from history, but the public DTO can omit `sim_time`. Retry identity and first-tick/reset detection therefore cannot depend on time alone. | `SURVIVAL_SINGLE_STREAM=1` additionally rejects explicit out-of-order timestamps. Use one server worker. |
| No world model by default | Existing CPU simulation is expensive, but a world model adds observation/action/population adaptation and hardware risks. | Consider one small decoder-free recurrent challenger only after the headless/macro-action collector and privileged teacher are measured. |

## Component map

```text
public StepResponse
    -> policies/features + geometry
    -> hierarchical scene/assignment controller OR legacy heuristic OR neural policy
    -> actions -> validated ActionRequest[]
          ^                     |
          |                     v
    training/env <--------- reference SimulationCore

benchmark factory ----- policies/runtime ----- serving/session + /predict
                             |
                 versioned config + checkpoint
```

| Module | Owns | Does not own |
| --- | --- | --- |
| `policies/config.py` | Strict experiment/runtime settings and typed overrides | Reading model weights or engine state |
| `policies/features.py`, `geometry.py` | Public observations, coordinate/cost semantics | Hidden map/seed, omniscient planning |
| `policies/hierarchical.py` | Shared-frame reconstruction, target reservation, direct movement, threat response, and breeding roles | Hidden simulator state or expensive action lattices |
| `policies/heuristic.py`, `vectorized.py` | Legacy scalar/vectorized candidate scoring and regression oracle | Default submission decisions |
| `policies/networks.py`, `actions.py`, `memory.py` | Ragged encoding, hybrid controls, agent identities | HTTP sessions or world stepping |
| `training/env.py`, `workers.py`, `seeds.py` | RNG-equivalent headless stepping, optional macro actions, isolated CPU workers, disjoint worlds | Changing reference benchmark physics |
| `training/rollout.py`, `imitation.py`, `ppo.py`, `learner.py` | Collection, sequence masks, losses and updates | Selecting a competition submission |
| `training/artifacts.py` | Resolved configurations, provenance, checkpoints, failures | Replacing benchmark comparisons |
| Existing `benchmarking/` | Full-horizon score measurements and paired comparisons | Training reward changes |
| `serving/session.py`, `agent_server.py` | Serialization, retry handling, explicitly scoped state | Automatic model fallback or concurrent-game inference |

## Rule-only turnaway policy

`configs/controller-turnaway-rules.json` selects `heuristic.backend=turnaway`.
This backend uses no heuristic tuning fields. Each agent takes the first applicable rule:

1. Spawn without moving or turning whenever the engine permits it. Spawning has
    exclusive priority, even during danger; there are no age, reserve, role, or population gates.
2. Retreat from the nearest sensed predator at the currently permitted movement limit,
    turning to face its last observed position from the chosen destination. "Opposite"
    refers to the predator's observed bearing, not its own heading. There is no danger-distance cutoff.
3. Walk toward the nearest sensed fruit, capped at its distance and adjusted for terrain.
4. Approach a random point around the nearest sensed tree. The sampling disk is one
    available walking step in radius, adjusted for terrain; there is no fixed patch radius.
5. Walk in a uniformly random direction otherwise, facing the direction of movement.

All movement checks the swept path against observed walls, including the engine's
body-size buffer at corners. Blocked paths try wall-derived tangents or outward normals
without reversing the intended direction, then shorten the original move if necessary.
Unobserved walls and future predator motion cannot be guaranteed safe from the public DTO.
Random exploration is seeded, resettable, and independent of input agent ordering.

The remaining numbers are game mechanics and geometry, not strategy settings:
reproduction eligibility, movement costs and limits, terrain multipliers, the reference
agent body radius, and angles defining a circle. Aggressive spawning can leave a parent
nearly out of energy; that is intentional under this priority order, not a claim of better score.

Run from `survival-simulator`:

```powershell
python benchmark.py run --policy src.policies.runtime:create_policy --config .\configs\controller-turnaway-rules.json --suite quick --output .\benchmark-results\turnaway-rules-quick
```

The teacher-anchored learning presets pin `configs\controller-turnaway-wall-aware.json`
by hash. Inline teacher overrides are rejected when a descriptor is pinned. The simple
TurnAway policy remains a benchmark ablation, not a substitute for the requested teacher.

## Contracts that isolate bugs

Movement is body-relative **before** turning. Energy is charged before terrain and collision effects.
Turning costs `min(pi, abs(turn))/(2*pi)`. Breeding requires energy **strictly greater**
than 100 after movement/turning. Mutated walking speed may exceed sprint speed.
The engine's `rel_dir` is an object-to-observer bearing, not a simple heading difference.
Predator and fruit observations can be stale. Fruit observations already inside the
minimum engine contact radius are ignored because collection occurs before the response
is returned.

Policies use `act(StepResponse) -> list[ActionRequest]` and cover every living agent.
At time/score zero, the client may announce the configured initial population before
observations exist; this bootstrap returns no actions. Its count is not hard-coded to
five because `SimulationCore` supports other initial populations.
Features are versioned, finite, and permutation-insensitive. Memory is keyed by agent ID,
cleared for departed agents, and initialized for births. Previous-action features always
describe what was actually executed, including teacher/learner mixtures.

The opt-in `structured-public-v2` feature schema adds public score, log agent identity,
and stable within-team identity rank. A synthetic symmetry fixture demonstrates that the
old inputs can be identical for agents receiving different teacher actions; the new inputs
distinguish them without depending on list order. Score history can be learned through
recurrence. This does not guarantee exact cloning of a geometric assignment algorithm.
Legacy `structured-v1` weights retain their original input shape. Changing the feature
flag is an architecture change, not a compatible checkpoint override.

Action log-standard-deviation bounds/initial value and the PPO actor divisor are named
configuration settings. Defaults preserve legacy behavior; `actor_divisor=5` scales the
summed actor surrogate against the initial team size, not current population. This is a
MAPPO-style factorized surrogate, not JointPPO's clipped joint likelihood ratio.

The critic receives one native team reward per tick. Agent death ends that agent's recurrent
sequence, not the team's value target. Extinction/horizon termination does not bootstrap;
a collector segment boundary does. Recurrent training is truncated BPTT, not full-history
backpropagation. Action likelihoods retain the sampled continuous latent and conditional
reproduction mask; inference and training share the codec.

## Experiments and artifacts

Run from `survival-simulator`. Architecture and parameter changes use a JSON preset and
`--set path=value`; unknown or incompatible settings fail rather than being ignored.
Add a new architecture behind the network interface, not by editing the collector and API.
Controller search batches independent candidate/world evaluations, while CMA adaptation
waits for the complete population. The checked-in hierarchical search preset contains
enough candidates for multiple CMA generations; it is configuration only and does not
start training. Results keep candidate/world identities despite out-of-order completion.

Each new output directory contains the resolved config, provenance manifest, incremental
events, and summary. Learning also writes `checkpoint.pt`, its checksum/version sidecar,
and a benchmark-compatible `policy.json`. Keep the sidecar with the weights.
Imitation also exports the bounded teacher/learner replay as `dataset.json.gz`;
the weights-only-safe replay inside the checkpoint is the authoritative resume state.
Periodic evaluation uses separate slim immutable snapshots, published with a ready manifest.
An independent evaluator owns episodes and learning curves; the optimizer never does.
`evaluation.max_pending` applies backpressure rather than dropping checkpoints.
Checkpoint paths in descriptors are relative to the use-case working directory.
`--resume` restores learning state and targets the configured total update count, but
restarts reference worlds; it does not promise an exact Pygame-state continuation.

Training worlds exclude `standard` and `holdout`. The training adapter skips rendering
surface work while preserving the reference render RNG stream. `resources.action_repeat`
can suppress intermediate agent observations for explicit macro-action experiments;
reproduction is applied only on the first repeated tick. The reviewed learning pipeline
requires `action_repeat=1` until equivalent serving semantics exist; actual native ticks,
including reset/bootstrap work, are recorded. Use standard for development, freeze a
candidate, then use holdout without further tuning. Score uncertainty is over worlds,
not agents or ticks. Windows results do not certify Linux/evaluator equivalence.
Training completion and low imitation loss are not evidence that a model should replace
the controller. Compare complete native-score runs first.

The endpoint budget is 10 seconds individually and 600 seconds accumulated per run.
At the full horizon that is roughly 20 ms per complete HTTP request, including networking
and serialization. `src.benchmarking.http:create_policy` enforces both limits against the
actual endpoint. Per-agent GPU timing is not a submission latency measurement.

## Primary method references

- MAPPO: *The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games* (NeurIPS 2022), `https://arxiv.org/abs/2103.01955`.
- DAgger: *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning* (AISTATS 2011), `https://proceedings.mlr.press/v15/ross11a.html`.
- CMA-ES: *The CMA Evolution Strategy: A Tutorial* (2016), `https://arxiv.org/abs/1604.00772`.
- Set pooling: *Deep Sets* (NeurIPS 2017), `https://arxiv.org/abs/1703.06114`.
- Deferred world-model reference: *Mastering diverse control tasks through world models* (Nature 2025), `https://www.nature.com/articles/s41586-025-08744-2`.

The implementation uses original task-specific code and pinned PyTorch/CMA-ES packages,
not a copied research training stack. Retain package licenses and attribution when redistributing.
