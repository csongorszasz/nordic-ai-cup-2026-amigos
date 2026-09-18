# Policy pipeline overview

```mermaid
flowchart TD
    A[Survival Simulator<br/>seeded world] --> B[StepResponse<br/>all living agents]
    B --> C[Validation and normalization]
    C --> D[Entity features<br/>geometry and memory]

    D --> H{Policy path}

    %% Controller path
    H -->|Primary baseline| HC[Hierarchical controller]
    HC --> HS[Shared scene reconstruction]
    HS --> HA[Target assignment<br/>foraging, escape, breeding]
    HA --> ACT[Validated ActionRequest<br/>for every living agent]

    %% Controller search
    CS[Controller search<br/>CMA-ES or random] --> CP[Candidate parameter sets]
    CP --> CW[Evaluate across training worlds]
    CW --> CO[Mean + lower-tail objective]
    CO -->|next CMA generation| CP
    CO -->|best parameters| HC

    %% Learning path
    H -->|Imitation| T[Privileged teacher]
    T --> DS[Demonstration dataset]
    DS --> IL[Behavioral cloning]
    IL --> DG[DAgger rollouts]
    DG -->|teacher corrections| DS
    DG --> NP[Neural shared policy]

    H -->|Reinforcement learning| NP
    NP --> RO[Multi-world rollout collection]
    RO --> RW[Native team rewards]
    RW --> PPO[PPO update<br/>actor + team critic]
    PPO --> NP

    %% Environment loop
    ACT --> E[Environment step]
    NP --> ACT
    E --> A

    %% Selection
    HC --> SB[Standard benchmark]
    NP --> SB
    SB --> COMP[Paired score and latency comparison]
    COMP --> G{Candidate beats controller?}
    G -->|No| DIAG[Inspect failures and metrics]
    DIAG --> HC
    DIAG --> NP
    G -->|Yes| FREEZE[Freeze policy, config and weights]
    FREEZE --> HB[One-time holdout benchmark]
    HB --> SG{Score and HTTP budget acceptable?}
    SG -->|No| REJECT[Reject candidate<br/>do not tune on holdout]
    SG -->|Yes| DEPLOY[Serve through agent_server]

    %% Production
    DEPLOY --> API["/predict endpoint"]
    API --> SESSION[Serialized stateful session]
    SESSION --> C

    %% Artifacts
    CW -.-> AR[(Search artifacts)]
    DS -.-> AR2[(Dataset and provenance)]
    PPO -.-> AR3[(Checkpoint and training metrics)]
    SB -.-> AR4[(Benchmark records)]
    FREEZE -.-> AR5[(Versioned policy descriptor)]
```

## Oversight gates

| Gate | Decision |
| --- | --- |
| Controller search | Does tuning improve both average and lower-tail world performance? |
| Teacher quality | Does the teacher outperform the existing controller consistently? |
| Imitation | Does the learned policy reproduce useful behavior, rather than merely achieve low training loss? |
| PPO | Is native benchmark score improving without excessive action latency? |
| Standard benchmark | Does the candidate beat the hierarchical controller on paired seeds? |
| Freeze | Are configuration, code, weights, checksum, and provenance fixed? |
| Holdout | Does the frozen candidate generalize without further tuning? |
| Deployment | Does the real endpoint remain within individual and cumulative HTTP budgets? |

The principal selection rule is:

```text
Controller search/training metrics
             |
             v
        diagnostic only
             |
             v
Full native-score benchmark + endpoint latency
             |
             v
      deployment decision
```

Low imitation loss, PPO reward, or controller-search score alone should never promote a
policy. The final authority is the unchanged simulator benchmark against the hierarchical
controller.
