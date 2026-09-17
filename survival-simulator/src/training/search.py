"""Budgeted maximization over validated heuristic settings, with no world selection."""

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Literal

from src.policies.config import HeuristicConfig, SearchConfig


@dataclass(frozen=True)
class SearchTrial:
    index: int
    config: HeuristicConfig
    score: float

    @property
    def parameters(self) -> dict[str, float | int | str]:
        """A fresh full parameter dictionary, including settings not searched."""
        return self.config.model_dump(mode="json")


@dataclass(frozen=True)
class SearchResult:
    best_config: HeuristicConfig
    best_score: float
    trials: tuple[SearchTrial, ...]
    method: Literal["random", "cma"]


def _evaluate(
    config: HeuristicConfig, index: int, evaluate: Callable[[HeuristicConfig], float],
) -> SearchTrial:
    try:
        score = evaluate(config)
    except Exception as error:
        error.add_note(f"Controller search candidate {index}: {config.model_dump(mode='json')}")
        raise
    if isinstance(score, bool) or not isinstance(score, Real) or not math.isfinite(score):
        raise ValueError(
            f"Controller search candidate {index} must return a finite real score, got {score!r}; "
            f"parameters={config.model_dump(mode='json')}"
        )
    return SearchTrial(index, config, float(score))


def optimize_controller(
    base: HeuristicConfig, search: SearchConfig, seed: int,
    evaluate: Callable[[HeuristicConfig], float],
    *,
    evaluate_many: Callable[[Sequence[HeuristicConfig]], Sequence[float]] | None = None,
) -> SearchResult:
    """Evaluate the unmodified baseline first, then maximize within the evaluation budget.

    Bounds apply to proposals, not the baseline. CMA works in normalized coordinates,
    adapting only after complete populations of at least three. A final partial
    population is evaluated without ``tell``; budgets of two or three only sample
    the initial distribution. A budget of one never imports CMA. ``method`` names
    the requested sampler, not a claim of convergence or completed generations.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Controller search seed must be an integer.")
    if not isinstance(base, HeuristicConfig) or not isinstance(search, SearchConfig):
        raise TypeError("Controller search requires validated HeuristicConfig and SearchConfig values.")
    bounds = tuple(sorted(search.parameters.items()))
    parameters = base.model_dump(mode="json")
    trials = []
    if search.method == "random":
        rng = random.Random(seed)
        proposals = [base]
        while len(proposals) < search.candidates:
            proposal = parameters | {
                name: rng.uniform(lower, upper) for name, (lower, upper) in bounds
            }
            config = HeuristicConfig.model_validate(proposal)
            proposals.append(config)
        trials.extend(_evaluate_group(proposals, 0, evaluate, evaluate_many))
    else:
        if evaluate_many is None or search.candidates == 1:
            trials.append(_evaluate(base, 0, evaluate))
        if len(trials) < search.candidates:
            _cma_trials(base, search, seed, bounds, parameters, evaluate, trials, evaluate_many)
    best = max(trials, key=lambda trial: (trial.score, -trial.index))
    return SearchResult(best.config, best.score, tuple(trials), search.method)


def _evaluate_group(
    configurations: Sequence[HeuristicConfig], start: int,
    evaluate: Callable[[HeuristicConfig], float],
    evaluate_many: Callable[[Sequence[HeuristicConfig]], Sequence[float]] | None,
) -> list[SearchTrial]:
    if evaluate_many is None:
        return [_evaluate(config, start + index, evaluate) for index, config in enumerate(configurations)]
    try:
        scores = list(evaluate_many(configurations))
    except Exception as error:
        error.add_note(f"Controller search candidate batch {start}-{start + len(configurations) - 1}.")
        raise
    if len(scores) != len(configurations):
        raise ValueError("Batch evaluation must return one score per candidate in input order.")
    return [
        _evaluate(config, start + index, lambda _, value=score: value)
        for index, (config, score) in enumerate(zip(configurations, scores, strict=True))
    ]


def _cma_trials(
    base: HeuristicConfig, search: SearchConfig, seed: int,
    bounds: tuple[tuple[str, tuple[float, float]], ...],
    parameters: dict[str, float | int | str], evaluate: Callable[[HeuristicConfig], float],
    trials: list[SearchTrial],
    evaluate_many: Callable[[Sequence[HeuristicConfig]], Sequence[float]] | None = None,
) -> None:
    try:
        import cma
    except ModuleNotFoundError as error:
        if error.name != "cma":
            raise
        raise ModuleNotFoundError(
            "CMA controller search requires cma; install the training requirements "
            "(python -m pip install cma==4.4.4).",
            name="cma",
        ) from error
    import numpy as np

    remaining = search.candidates - len(trials) - int(not trials)
    population = min(search.candidates, max(3, min(remaining, 4 + int(3 * math.log(len(bounds))))))
    initial = [
        max(0.0, min(1.0, (getattr(base, name) - lower) / (upper - lower)))
        for name, (lower, upper) in bounds
    ]
    rng = np.random.RandomState(seed % 2**32)
    options = {
        "bounds": [0.0, 1.0],
        # The supplied sampler is explicitly seeded; disable CMA's global NumPy seeding.
        "seed": np.nan,
        "randn": rng.randn,
        "popsize": population,
        "CMA_mirrors": 0,
        "verbose": -9,
        "verb_disp": 0,
        "verb_log": 0,
        "verb_plot": 0,
    }
    if len(bounds) == 1:
        # CMA's scalar diagonal scaling cannot clip a single coordinate's std.
        # The boundary transform still enforces [0, 1] on every proposed value.
        options["maxstd"] = math.inf
    strategy = cma.CMAEvolutionStrategy(initial, search.sigma, options)
    while len(trials) < search.candidates:
        include_baseline = not trials
        count = min(population, search.candidates - len(trials) - int(include_baseline))
        solutions = strategy.ask(number=count)
        proposals = [base] if include_baseline else []
        for solution in solutions:
            proposal = parameters.copy()
            for coordinate, (name, (lower, upper)) in zip(solution, bounds, strict=True):
                if not math.isfinite(coordinate) or not 0.0 <= coordinate <= 1.0:
                    raise ValueError(f"CMA candidate {len(trials)} has an invalid normalized {name}.")
                proposal[name] = max(lower, min(upper, lower + float(coordinate) * (upper - lower)))
            config = HeuristicConfig.model_validate(proposal)
            proposals.append(config)
        evaluated = _evaluate_group(proposals, len(trials), evaluate, evaluate_many)
        trials.extend(evaluated)
        scores = [-trial.score for trial in evaluated[int(include_baseline):]]
        if count == population and count >= 3:
            strategy.tell(solutions, scores)
