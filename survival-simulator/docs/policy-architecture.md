# Policy pipeline

## Decisions

| Decision | Reason | Alternative / when to reconsider |
| --- | --- | --- |
| Keep the reference engine unchanged | Rendering consumes world RNG; "removing graphics" changes seeds. Cached observations and mutation edge cases are part of the contract. | A faster kernel only after behavior and distribution comparisons; never train against a silently repaired game. |
| Controller -> imitation/DAgger -> recurrent PPO | A small, understandable controller is useful on its own and supplies recovery labels for the learner. | Keep the controller if learning fails to improve held-out native scores. |
| Vectorized candidate scoring, scalar adjudication | Full scalar-controller episodes exceeded the local cumulative policy-time budget. NumPy scores all candidates together, then the scalar oracle settles close contenders and numerical boundaries. | `heuristic.backend=scalar` remains the oracle. Differential tests and real HTTP measurements are required before accepting a faster backend. |
| Structured, ragged entity features | DTOs already describe objects. Per-type encoding and pooling handle arbitrary populations without pixels or dropping agents. | Configurable attention; require score evidence before adding more model complexity. |
| Shared actor, identity-keyed GRU, pooled team critic | Agents share controls and the controller receives all observations. Births/deaths change actor state, not the ongoing team objective. | `model.memory=none`, `model.team_context=false`, or `model.critic=local` are ablations. This is centralized control, not strict decentralized MAPPO. |
| Native team reward, no reward clipping | Score is elapsed time + eaten energy/1000 - energy lost to predation/100. Population is not a reward. | Any future shaping must be named, logged, and selected against the unmodified native score. |
| Stateless HTTP default | The competition DTO has no episode ID and does not establish whether games can interleave. | Neural serving requires an explicitly agreed single stream and `SURVIVAL_SINGLE_STREAM=1`; do not assume a final terminal request arrives. |
| No world model by default | Existing CPU simulation is expensive, but a world model adds its own observation/action/population adaptation and hardware risks. | Consider one small vector-input Dreamer-style challenger only after measured PPO/collection results justify it. No claim of a task-specific optimal policy. |

## Component map

```text
public StepResponse
    -> policies/features + geometry
    -> heuristic OR networks + memory
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
| `policies/heuristic.py` | Stateless food/threat/wall/exploration/breeding decisions | Training or global simulation RNG |
| `policies/vectorized.py` | Float64 batch scoring with conservative contender selection | Candidate generation, changed physics, or dropping observations/agents |
| `policies/networks.py`, `actions.py`, `memory.py` | Ragged encoding, hybrid controls, agent identities | HTTP sessions or world stepping |
| `training/env.py`, `workers.py`, `seeds.py` | Reference stepping, isolated CPU workers, disjoint training worlds | GPU actor construction in children |
| `training/rollout.py`, `imitation.py`, `ppo.py`, `learner.py` | Collection, sequence masks, losses and updates | Selecting a competition submission |
| `training/artifacts.py` | Resolved configurations, provenance, checkpoints, failures | Replacing benchmark comparisons |
| Existing `benchmarking/` | Full-horizon score measurements and paired comparisons | Training reward changes |
| `serving/session.py`, `agent_server.py` | Serialization, retry handling, explicitly scoped state | Automatic model fallback or concurrent-game inference |

## Contracts that isolate bugs

Movement is body-relative **before** turning. Energy is charged before terrain and collision effects.
Turning costs `min(pi, abs(turn))/(2*pi)`. Breeding requires energy **strictly greater**
than 100 after movement/turning. Mutated walking speed may exceed sprint speed.
The engine's `rel_dir` is an object-to-observer bearing, not a simple heading difference.
Predator and fruit observations can be stale.

Policies use `act(StepResponse) -> list[ActionRequest]` and cover every living agent.
At time/score zero, the client may announce the configured initial population before
observations exist; this bootstrap returns no actions. Its count is not hard-coded to
five because `SimulationCore` supports other initial populations.
Features are versioned, finite, and permutation-insensitive. Memory is keyed by agent ID,
cleared for departed agents, and initialized for births. Previous-action features always
describe what was actually executed, including teacher/learner mixtures.

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
waits for the complete population. Results keep candidate/world identities despite
out-of-order completion; progress logging never changes the reference physics.

Each new output directory contains the resolved config, provenance manifest, incremental
events, and summary. Learning also writes `checkpoint.pt`, its checksum/version sidecar,
and a benchmark-compatible `policy.json`. Keep the sidecar with the weights.
Imitation also exports the bounded teacher/learner replay as `dataset.json.gz`;
the weights-only-safe replay inside the checkpoint is the authoritative resume state.
Checkpoint paths in descriptors are relative to the use-case working directory.
`--resume` restores learning state and targets the configured total update count, but
restarts reference worlds; it does not promise an exact Pygame-state continuation.

Training worlds exclude `standard` and `holdout`. Use standard for development, freeze a
candidate, then use holdout without further tuning. Score uncertainty is over worlds,
not agents or ticks. Windows results do not certify Linux/evaluator equivalence.
Training completion and low imitation loss are not evidence that a model should replace
the controller. Compare complete native-score runs first.

The endpoint budget is 10 seconds individually and 600 seconds accumulated per run.
At the full horizon that is roughly 20 ms per complete HTTP request, including networking
and serialization. Per-agent GPU timing is not a submission latency measurement.

## Primary method references

- MAPPO: *The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games* (NeurIPS 2022), `https://arxiv.org/abs/2103.01955`.
- DAgger: *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning* (AISTATS 2011), `https://proceedings.mlr.press/v15/ross11a.html`.
- CMA-ES: *The CMA Evolution Strategy: A Tutorial* (2016), `https://arxiv.org/abs/1604.00772`.
- Set pooling: *Deep Sets* (NeurIPS 2017), `https://arxiv.org/abs/1703.06114`.
- Deferred world-model reference: *Mastering diverse control tasks through world models* (Nature 2025), `https://www.nature.com/articles/s41586-025-08744-2`.

The implementation uses original task-specific code and pinned PyTorch/CMA-ES packages,
not a copied research training stack. Retain package licenses and attribution when redistributing.
