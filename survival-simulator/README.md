> **Concluded problem categorization: The simulator is a cooperative, partially observed, continuous-space foraging-and-survival task with shared control, energy economics, predators, and a variable population.**

# Survival simulator

Improvise, adapt, overcome!

You are the hivemind of an entire species of herbivores. Ensure their survival by eating fruits and conserving energy, but beware of the predators roaming the territory.

## About the game
You are in control of all members of your species (agents) simultaneously. At every tick you will recieve a list of each agent's observations and status. The agents have a hearing/smelling radius and a vision cone. Any object within the vision cone is added to observations with an object type and data depending on object type. Note that creatures cannot hear/smell walls, only see them and vision can be blocked by walls.

![Entities](/images/Entity_overview.png)

After receiving an observation for each agent you will have to respond with the action for each agent. This includes moving, turning and even spawning new agents.
Agents can move in any direction but are limited by their speed/sprint_speed with sprinting having a higher energy cost per unit traveled. They can also turn at a low energy cost and even spawn new agents at a very high cost.

A newly spawned agent will inherit the parent's traits with a small chance of mutations happening in each trait. These mutations can affect the agent both positively and negatively.

During the simulation predators will spawn in to hunt the agents.

The simulation will run until all agents have died or the environment has simulated 3000 seconds corresponding to 30000 ticks.


## Environment
Running the simulation creates a random environment with different biomes which affect fruit spawn rates and creature movement. The environment will also have a number of obstacles, obstructing vision and movement.
5 agents will spawn in at the start of the simulation as well as some fruits and fruit trees.
During the simulation fruit will spawn around trees as a source of energy.
Both trees and fruit have a life cycle growing/rotting over time. 

Predators will spawn in during the simulation with increased odds over time. The predators will hunt down agents, killing any agent they touch and stealing their remaining energy. This will decrease your score based on the agent's remaining energy!


## Your goal
Your main goal is to keep your species alive for as long as possible.
If your species can consistently survive the entire simulation time, your score can be increased further by eating fruit and avoiding getting eaten by predators.

## Status and Observations

At every environment step you will receive game_status, score and a list of agent_status objects.
The game_status will be "ok" as long as the simulation is running.
The score is the current accumulated score since simulation start.
The content of each entry in the agent_status list can be seen in the table below:
| Name              | Explanation                                                   |
|-------------------|---------------------------------------------------------------|
| agent_id          | ID to keep track of agents                                    |
| observations      | List of observations for the agent                            |
| energy            | Agent's remaining energy                                      |
| biome             | The biome type that the agent is currently in                 |
| age               | How many simulated seconds the agent has been alive           |
| speed             | The maximum speed the agent can move at no additional cost    |
| sprint_speed      | The maximum speed the agent can move (higher energy cost)     |
| hearing_radius    | How far the agent can hear/smell entities                     |
| vision_angle      | The angle of the vision cone (radians)                        |
| vision_range      | How far the agent can see                                     |
| max_energy        | How much energy the agent can store                           |

The speed, sprint_speed, hearing_radius, vision_angle, vision_range, and max_energy describes static agent traits/attributes that can mutate when spawning new agents.

The sense traits/attributes are shown in the following figure:
![Traits](/images/agent_trait_ref.png)

The observations have the following format based on what is being observed:

| Observation type  | Data                                                                  |
|-------------------|-----------------------------------------------------------------------|
| Fruit             | Type, Distance, Angle (radians)                                       |
| Agent             | Type, Distance, Angle (radians), Relative looking direction           |
| Predator          | Type, Distance, Angle (radians), Relative looking direction           |
| Tree              | Type, Distance, Angle (radians)                                       |
| Edge              | Type, Coordinates (start, end)                                        |


Edges are only observed if within the vision cone. Other observations are also observed in the hearing/smell range:
![Sensing](/images/Agent_senses.png)

## Controls
After receiving the list of agent statuses and observations from the environment, your controller must decide what each agent should do during the next simulation step.
This decision should be returned as a list of ActionRequests, one for each agent (See [DTOs.py](src/utils/DTOs.py)).

Each ActionRequest must include the following fields:

|Field	            | Type	| Description                                                               |
|-------------------|-------|---------------------------------------------------------------------------|
|agent_id	        | int	| The ID of the agent this action applies to.                                   |
|move_distance	    | float | Distance to move (capped by speed or sprint_speed).                       |
|move_direction     | float | Movement direction relative to the agent's heading before turning (radians). |
|turn_angle	        | float | Rotation applied this step (radians).                                     |
|spawn_agent	    | bool  | Whether the agent should attempt to spawn a new agent (high energy cost).   |

For a full example of how actions are used in practice, see [dummy_agent_policy.py](src/utils/controllers/dummy_agent_policy.py) and [agent_server.py](agent_server.py).

## Energy costs
|Action                                               | Energy cost                                     |
|-----------------------------------------------------|-------------------------------------------------|
| Walking (move_distance <= speed                     | move_distance * 0.05                            |
| Sprinting (speed <= move_distance <= sprint_speed)  | speed * 0.05 + (move_distance - speed) * 0.5    |
| Turning                                             | min(pi, abs(turn_angle)) / (2 * pi)              |
| Spawning                                            | 100                                             |
| Living (passive cost over time)                     | 1 / 10 * biome_energy_modifier                  |

When agents are older than a randomly chosen age between 60 and 120, their living cost will increase by 0.01 * agent.age.


## Scoring
Your score is mainly determined by how long your species survive. However, eating a fruit will increase your score by a small amount and getting eaten by predators will decrease your score based on how much energy the agent had remaining.

## Validation and Evaluation
To test your model and server connection, start a validation attempt. You can only have one attempt going at once, but attempts are unlimited. Your attempt will be put into a queue, and run when it's your turn. The validation attempts will use random seeds. 

Once you are ready to evaluate your final model, start your evaluation attempt. You only have **ONE** try, so make sure the model is ready for the final test. Your score from the evaluation is the one you will be judged on. 

Note that the evaluation attempt will run three attempts in a row and your score will be the average result, so you should ensure your agent server keeps running through all three simulations.

The evaluation will have preset seeds.

## Quickstart

Clone the repository and enter the use case folder:

```cmd
git clone https://github.com/amboltio/Nordic-AI-Cup-2026
cd Nordic-AI-Cup-2026/survival-simulator
```

### Install
The simulator requires **Python 3.10-3.13**. Python 3.14 is not supported by the pinned Pygame version. We recommend installing the dependencies in a virtual environment so they do not interfere with your other projects.

Linux / macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows:
```cmd
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Remember to activate the environment (`source .venv/bin/activate` or `.venv\Scripts\activate`) in every new terminal before running any of the scripts below.

To verify that the installation works, run a quick headless simulation:

```cmd
python -c "from local_playground import local_simulation; local_simulation(verbose=False)"
```

You should see a stream of `Score | Agents alive | Time` lines, ending with a `Game over!` message and the seed that was used.

# Testing locally
To test the simulation locally you can run [local_playground.py](local_playground.py). This can be used to get an idea of how the simulation works. It is recommended to use this file for any potential training with "verbose" set to False to run simulations faster.

Run a rendered game from `survival-simulator` with the appropriate environment active:

```powershell
python local_playground.py --config .\training-results\ppo-gru-full\policy.json --seed 1 --fps 60
python local_playground.py --config .\configs\controller-turnaway-rules.json --seed 1
```

Use the exported runtime `policy.json`, not the training experiment config. Neural
policies require the training dependencies, the referenced checkpoint, and its
`checkpoint.pt.json` sidecar. Checkpoint paths resolve from the working directory,
as they do in benchmarks. Close the window or press Escape to stop.

Omit `--config` for the original random policy. `--seed` controls both world and
policy randomness; omitting it generates a random seed. `--fps` caps playback speed
without changing simulation time steps. For a short check without a window:

```powershell
python local_playground.py --config .\training-results\ppo-gru-full\policy.json --seed 1 --headless --max-steps 10
```

The step cap includes the initial empty-action tick. It is a diagnostic run, not
a complete benchmark result. The existing `local_simulation(verbose=False)` call
remains supported.

## Comparing policies

The benchmark evaluates **local Python policies** against the existing random policy
or against any other saved policy run. It does not change the simulator, scoring,
playground, or HTTP server.

Run these commands from `survival-simulator` with the virtual environment active:

```powershell
python -m pip install -r requirements-benchmark.txt
python benchmark.py run --policy random --suite quick --output .\benchmark-results\random-quick
python benchmark.py run --policy random --suite standard --output .\benchmark-results\random-standard
```

The `random` alias identifies `random-local-v1`: the original `action_decision`
function with one persistent policy RNG per episode, as in `local_playground.py`.
It is **not** the HTTP server's variant, which reseeds on every request.

| Suite | World seeds | Purpose |
| --- | --- | --- |
| `quick` (default) | 3, drawn from standard | Exploratory iteration with fewer episodes |
| `standard` | 20 | Routine comparisons and the saved random baseline |
| `holdout` | 20, disjoint from standard | Explicit final comparisons, not routine tuning |

The versioned seed lists live in `benchmarks\suites`. Quick runs still use the full
episode horizon. Every episode begins with the existing empty-action tick and
stops at extinction or when engine time is **strictly greater than 3000 seconds**,
matching the current simulator scripts rather than assuming exactly 30000 ticks.

### Adding a policy

Create an importable factory, for example
`src\utils\controllers\my_policy.py`. The benchmark calls it with `seed` and
`config` keyword arguments for each episode. Its returned object must implement
`act(step: StepResponse)`, returning a sequence of `ActionRequest` objects:

```python
import random
from src.utils.DTOs import ActionRequest, StepResponse


class MyPolicy:
    def __init__(self, seed, config):
        self.rng = random.Random(seed)
        self.config = config
        self.memory = {}

    def act(self, step: StepResponse) -> list[ActionRequest]:
        return [
            ActionRequest(
                agent_id=agent.agent_id,
                move_distance=0.0,
                move_direction=0.0,
                turn_angle=0.0,
                spawn_agent=False,
            )
            for agent in step.agent_status
        ]


def create_policy(seed, config):
    return MyPolicy(seed, config)
```

This deliberately simple example is an interface illustration, not a proposed
competitive policy. The batch API exposes every current agent together, enabling
shared memory and coordination. State must reset in the factory; use private
RNGs initialized from the supplied policy seed, not global random state.

```powershell
python benchmark.py run --policy src.utils.controllers.my_policy:create_policy --label my-policy --suite standard --output .\benchmark-results\candidate-standard
python benchmark.py compare --reference .\benchmark-results\random-standard --candidate .\benchmark-results\candidate-standard --output .\benchmark-results\comparison
```

Use any saved run as `--reference` to compare solutions with each other. Repeat
`--candidate` to compare several candidates together. Each output path must be new;
existing directories are never overwritten. Compare the same baseline directory
against itself to exercise reporting without additional simulation.

Pass `--config .\policy-config.json` for a JSON object of policy parameters.
Use repeatable `--artifact .\model-file` arguments to fingerprint model weights or
additional source/config files, especially dependencies outside the local policy
source tree. Policies receive observations only, not the simulator or its RNG;
these are trusted local plugins, not sandboxed programs.

Use `--fixed-policy-seed 1` when measuring the exact deterministic policy instance
served by the default endpoint. Without it, repeat zero intentionally uses each
world seed as the policy seed for stochastic-policy comparisons.

To make the documented 10-second/request and 600-second cumulative wait limits a
hard benchmark gate, start the endpoint and run it through the HTTP policy:

```powershell
python benchmark.py run --policy src.benchmarking.http:create_policy --config .\configs\http-loopback.json --fixed-policy-seed 1 --suite quick --output .\benchmark-results\controller-http-quick
```

The run fails immediately on a non-200 response, malformed action payload, individual
timeout, or cumulative wait-budget overrun. Replace the URL in a copied config when
measuring the public tunnel rather than loopback.

### Measurements and saved reports

**Mean native final score is the primary ranking metric.** Reports also include
median, spread and extrema; capped survival time; time-limit completion fraction;
initial/final/peak/mean population; and wall-clock diagnostics. Raw engine time and
tick counts are retained. Extinction is a successfully evaluated episode, but is
not a time-limit completion.

Initialization, policy construction, simulation stepping, policy execution, and
episode wall time are measured separately. Latency is for a **batch call across
all living agents**, includes cold calls, and is summarized per episode as
mean/p50/p95/max. It is not per-agent inference latency or an HTTP timeout guarantee.
Episode timings exclude CLI startup, policy module import, provenance hashing, and
report generation.
Different machines or timing contexts are flagged rather than treated as controlled
speed comparisons.

Each run stores `manifest.json`, incremental `episodes.jsonl`, `episodes.csv`,
`summary.json`, and `report.md`. The manifest records seeds, settings, policy
configuration, source/artifact fingerprints, Git state, and runtime provenance.
Generated runs and comparison directories under `benchmark-results` are ignored
by Git; retain or share them explicitly when comparing work across machines.

Comparison reports are generated entirely from saved records and include
paired-case data, score distributions, per-seed score differences, and
score-versus-policy-latency PNG plots. Simulation and numerical reporting do not
require Matplotlib; install `requirements-benchmark.txt` for plots, or use
`compare --no-plots` for numerical output only.

```powershell
python benchmark.py inspect --run .\benchmark-results\random-standard
```

Comparisons require the exact same suite/cases, simulator settings and source,
runner protocol, and compatible Python/OS/runtime dependencies. Policy code,
parameters, and Git revisions may differ. Missing, failed, duplicated, truncated,
or otherwise incompatible trials cannot silently enter a complete ranking.

### Repeats, uncertainty, and failures

Use `--repeats N` to evaluate multiple stochastic policy trials per world seed.
Repeat zero uses the world seed as its policy seed. Further policy seeds use the
first four bytes of SHA-256 over
`sha256-policy-seed-v1:<world_seed>:<repeat_index>`, interpreted big-endian.
Every resolved case and seed is saved; policies are compared on matching cases,
never by row position.

Reports show absolute score deltas and win/tie/loss counts with a `1e-9` absolute
tie tolerance, not percentages over potentially zero or negative baselines.
The 95% paired percentile bootstrap interval uses 10000 resamples and an independent
fixed report RNG. Repeats are averaged within a world seed before resampling seeds;
agents, ticks, and repeated trials of one world are not independent environments.
One-seed intervals are unavailable, and constant differences have point intervals.
Quick-suite results are exploratory; an interval crossing zero does not establish
a clear improvement. Holdout seeds are version-controlled, not secret.

The simulator's observation ordering is not fully deterministic, even for matching
seeds on the same platform. Seeds and provenance make experiments comparable but
do **not** guarantee identical replay. Different policies also consume different
amounts of environment randomness through their actions. No observation sorting,
physics changes, or RNG-consuming graphics optimizations are applied by the harness.

Failures stop the run with a nonzero exit code and explicit error details; missing
scores are never replaced by zero or random actions. Completed episodes survive
interruptions. Use `inspect` for partial runs, and a new output directory to rerun.
There is no preemptive timeout for a local policy that blocks indefinitely.

For a bounded integration probe, add `--max-steps 2`. The initial empty-action tick
counts toward this cap. Such diagnostic runs are never eligible for full-horizon
comparisons, even if an episode becomes extinct before the cap.

### Benchmark development

The benchmark uses standard-library `unittest`, without another test-runner dependency:

```powershell
python -m unittest discover -s tests -p "test_benchmark_*.py"
$env:BENCHMARK_INTEGRATION = "1"
python -m unittest discover -s tests -p "test_benchmark_runner.py"
Remove-Item Env:BENCHMARK_INTEGRATION
```

## Developing a policy

For the current **BC convergence** work, see `docs\bc-convergence.md`: immutable complete
teacher demonstrations, offline optimizer steps, current-weight recurrent validation,
and copying/score gates. It does not advance to DAgger or PPO merely because a budget ends.

The primary submission candidate is now the **stateful hierarchical controller**:
shared teammate-relative scene reconstruction, distinct fruit/tree assignments,
patch camping, facing-aware predator escape, and population/trait-aware breeding.
The older scalar/vectorized action lattice remains available as an oracle and ablation.
Imitation/DAgger and recurrent PPO are experimental challengers and must beat the
controller on native score and the real HTTP budget before selection. See
`docs\policy-architecture.md` for component boundaries and decision rationale.

### TurnAway escape ablations

The hierarchical policy supports three predator escape strategies while keeping its
foraging, scouting, team assignments, and population logic identical:

| Configuration | Escape behavior |
| --- | --- |
| `configs\controller-turnaway-direct.json` | Move directly opposite the nearest observed predator, even when that route crosses a wall. Turn to face the escape heading. |
| `configs\controller-turnaway-wall-aware.json` | Start with direct-away movement, then try ±60°, ±90°, and reverse alternatives when the path is blocked. Turn to face the selected escape heading. |
| `configs\controller-turnaway-predictive.json` | Predict the predator two ticks ahead, select a wall-aware separation/cover route, and turn to keep the predator visible. This is the default in `configs\controller.json`. |

Run quick matched-seed comparisons from `survival-simulator`:

```powershell
python benchmark.py run --policy src.policies.runtime:create_policy --config .\configs\controller-turnaway-direct.json --label turnaway-direct --suite quick --output .\benchmark-results\turnaway-direct-quick
python benchmark.py run --policy src.policies.runtime:create_policy --config .\configs\controller-turnaway-wall-aware.json --label turnaway-wall-aware --suite quick --output .\benchmark-results\turnaway-wall-aware-quick
python benchmark.py run --policy src.policies.runtime:create_policy --config .\configs\controller-turnaway-predictive.json --label turnaway-predictive --suite quick --output .\benchmark-results\turnaway-predictive-quick
python benchmark.py compare --reference .\benchmark-results\turnaway-direct-quick --candidate .\benchmark-results\turnaway-wall-aware-quick --candidate .\benchmark-results\turnaway-predictive-quick --output .\benchmark-results\turnaway-quick-comparison
```

Use `--suite standard` with new output directory names after the quick runs complete.
To exercise one strategy through the local HTTP server, select its JSON before startup:

```powershell
$env:SURVIVAL_POLICY_CONFIG = ".\configs\controller-turnaway-direct.json"
$env:SURVIVAL_SINGLE_STREAM = "1"
python .\agent_server.py
```

In another terminal, run `python .\simulation_server.py`. Stop the server before
changing strategies because the configuration is loaded once at startup.

Run all commands below from `survival-simulator`, using the existing virtual environment.
Controller inference and ordinary benchmarks only need `requirements.txt`. Training/search
adds pinned optional dependencies:

```powershell
python -m pip install -r requirements-training.txt
```

For the supported NVIDIA CUDA build, install PyTorch from its official index before the
training requirements:

```powershell
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-training.txt
```

Experiments use strict JSON configurations and typed overrides, not edits to Python.
**All real runs are user-launched.** The assistant may implement and run synthetic checks
or regenerate saved plots, but must not launch training, simulator evaluation, or cluster jobs.
The BC/PPO presets are small **provisional diagnostic plans**, not proven best parameters.

```powershell
# Pure preflight: no simulation, optimizer, GPU initialization or job submission.
python train.py --config .\configs\imitation-gru.json --preflight --evidence .\configs\evidence-diagnostic.json --output .\plans\bc
# User reviews the resolved settings/evidence/budgets, then launches:
python train.py --config .\configs\imitation-gru.json --run-plan .\plans\bc\run-plan.json --output .\training-results\bc
# Separate user-launched evaluator; --once drains currently published requests only.
python -m idun.watch_checkpoints --run .\training-results\bc --once
# This command reads saved results only and never runs an episode.
python -m src.training.progress --run .\training-results\bc
```

The required imitation/DAgger teacher is **`configs\controller-turnaway-wall-aware.json`**,
identified by its descriptor and source hashes. Change the teacher only as a deliberate
new experiment. The current BC preset uses teacher-only collection; enable DAgger explicitly
with `--set imitation.dagger_rounds=1` in both preflight and launch commands.

Use `configs\ppo-gru.json` for a teacher-free scratch control. For warm-start PPO, add
`--set checkpoint=training-results\bc\checkpoint.pt` to both commands and deliberately choose
whether to use decaying `ppo.imitation_weight`. Learning-rate/batch/architecture overrides
must be repeated identically at launch; any change invalidates the plan.
Extended runs require exact effective values and hashed measured-pilot evidence in every
parameter group. Existing citations/defaults are hypotheses, not proof of optimality.

Evaluation snapshots are published initially, every five completed collect/fit updates,
and at the final update by default. The user-launched evaluator measures complete native
episodes on the fixed development suite and refreshes `progress\learning-curve.png`,
the native-tick view, CSV, JSONL, and `status.json`. Teacher mean and world-seed spread are
shown; losses and partial-rollout rewards are not substituted for scores. Evaluation lag,
failures, and missing points are explicit. Holdout is reserved for frozen final candidates.
Install `requirements-benchmark.txt` on the evaluator for plots.

If the pending queue fills, the trainer preserves its completed-update checkpoint and
pauses with exit code 3. Drain that run's queue before resuming. Each episode is attempted
once by default; explicit `python -m src.training.evaluation retry --run ... --update N
--reason "..."` is permitted only if the reviewed `evaluation.max_attempts` and episode
budget reserved another attempt. It does not silently create extra work.

For example, `--set model.encoder=attention`, `--set model.memory=none`,
`--set optimizer.learning_rate=0.0001`, and `--set resources.device=cpu` change supported
architectures or parameters without touching the collector, policy factory, or API.
Choose worker counts using measured throughput and memory, not the logical CPU count.
Rollouts stay in host memory; neural training enforces its configured VRAM/token budget.
Search evaluates independent candidate/world jobs across those workers, including whole
CMA populations. It records completed episodes immediately and prints periodic native-tick
progress; interrupted batches never become a complete candidate ranking.

Use `--resume .\training-results\ppo\checkpoint.pt` with a **new** output directory to
resume optimizer/RNG/update state. `updates` is the target total, not an additional count.
Resume restarts the reference worlds; it does not serialize Pygame or promise exact
mid-episode continuation. A warm-start `checkpoint` loads weights without optimizer state.
Keep `checkpoint.pt.json` alongside the weights for checksum/schema validation. Generate
a new preflight with the same `--resume` source before launching the resumed run.
Resume also requires the original `manifest.json` and matching simulator, runtime,
PyTorch, and seed-generation/suite provenance. Use a warm start when intentionally
changing those; resuming must not silently change the training-world sequence.
Teacher/cadence/evaluation compatibility is also checked. Historical curve points retain
their update numbers; warm-start collection costs are included in native-tick accounting.

IDUN A100/H100 training may use explicitly reviewed budgets above 8 GB; the application
checks actual device memory and allocated CPUs instead of silently choosing larger batches.
See `docs\pipeline-overview.md` for the staged flow and `idun\HANDOFF.md` for manual launch.

Compare exported policies through the existing benchmark:

```powershell
python benchmark.py run --policy src.policies.runtime:create_policy --config .\configs\controller.json --suite standard --output .\benchmark-results\controller-standard
python benchmark.py run --policy src.policies.runtime:create_policy --config .\training-results\ppo\policy.json --artifact .\training-results\ppo\checkpoint.pt --suite standard --output .\benchmark-results\ppo-standard
python benchmark.py compare --reference .\benchmark-results\controller-standard --candidate .\benchmark-results\ppo-standard --output .\benchmark-results\ppo-comparison
```

Generated outputs under `training-results` are ignored by Git. They contain resolved
configurations, source/runtime provenance, incremental events, summaries, and any trained
weights. Preserve them explicitly when moving machines. Training seeds exclude the
standard/holdout suites; freeze a candidate before consulting holdout.

Training adapters use an RNG-equivalent headless core: rendering surfaces and pixel writes
are skipped while the historical render RNG draws are still consumed. The reference
benchmark continues to use the ordinary rendering-inclusive core. An optional
`resources.action_repeat` setting (1-10, default 1) can suppress intermediate agent
observation generation for macro-action experiments; reproduction is executed only on
the first repeated tick. Treat action repeat as a separate experiment because it changes
the policy decision cadence.

# Run on server
You can serve your endpoint locally and test that everything starts without errors by running [agent_server.py](agent_server.py). Then open a browser and navigate to [http://localhost:9052](http://localhost:9052). You should see a message stating that the agent server is running. 
Feel free to change the `HOST` and `PORT` settings in [agent_server.py](agent_server.py).

The server loads `configs\controller.json` once at startup and uses the same policy
factory as the benchmark. The default is the hierarchical controller; the previous
vectorized lattice remains in `configs\controller-vectorized.json`. To select another
exported policy, set `SURVIVAL_POLICY_CONFIG` to its JSON descriptor. Paths inside a
descriptor are relative to this use-case working directory. Restart the server after
changing the selected configuration.

The HTTP boundary accepts both the official verifier payload (`game_status="running"`,
omitted `sim_time`/`n_agents`, lowercase observation types) and the richer local
simulator payload. Inputs are normalized before they reach the policy.

The default controller and neural policies retain state behind one process lock. They reset
on bootstrap, explicit time rollback, or an observed first tick, including verifier payloads
that omit `sim_time`; identical retries are cached. Set `SURVIVAL_SINGLE_STREAM=1` to enable
strict out-of-order rejection when explicit times are present. Use one server worker because
the public DTO has no episode identifier. Missing/bad checkpoints fail startup and never
silently fall back to random actions.

To run a simulation on the server, you can run [simulation_server.py](simulation_server.py) while the endpoint is running.

The default movement logic for agents can be found in [dummy_agent_policy.py](src/utils/controllers/dummy_agent_policy.py).


### OBS
For comparisons with the validation/evaluation server, use a Linux machine. Matching
OS, versions, and seeds does not guarantee exact replay: observation ordering can
still vary. The benchmark records provenance and documents these limitations.

To avoid bottlenecking the system, the server will wait for responses for up to 10 seconds. If no responses are received from the agent server within that time or if the accumulated wait time reaches 600 seconds, the run will end.
