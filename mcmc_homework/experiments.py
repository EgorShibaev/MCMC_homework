"""Experiment configuration, repeated-seed evaluation, and benchmarking."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .metrics import (
    compute_metrics,
    compute_ensemble_metrics,
    get_reference_sample,
    standardized_sliced_wasserstein,
)
from .samplers import (
    SamplerResult,
    run_sampler,
    states_for_target_evals,
)
from .targets import Array, TARGETS


METHODS = ("RWMH", "ULA", "MALA", "HMC")
CORE_TARGET_KEYS = ("gaussian", "banana", "mixture")
BENCHMARK_SEEDS = (11, 23, 47)
BENCHMARK_TARGET_EVAL_BUDGET = 40_000
MAX_STUDENT_CHAINS = 8

# A row passes only when its *worst* benchmark-ensemble SWD is at or below the
# target-method-specific threshold and none of the ensembles diverged. Limits
# use a broad search plus local refinement at 40,000 evaluations per ensemble.
# Each admits at least five tested trajectory configurations. Additional seeds
# check sensitivity but do not inflate the grading limits. The 10:90 case is optional.
# They are attainable by every method, but
# the deliberately untuned defaults do not all pass. Passing the numerical gate
# does not replace the notebook's controlled investigations and evidence.
SWD_PASS_THRESHOLDS: dict[tuple[str, str], float] = {
    ("gaussian", "RWMH"): 0.035,
    ("gaussian", "ULA"): 0.06,
    ("gaussian", "MALA"): 0.055,
    ("gaussian", "HMC"): 0.030,
    ("banana", "RWMH"): 0.045,
    ("banana", "ULA"): 0.135,
    ("banana", "MALA"): 0.090,
    ("banana", "HMC"): 0.045,
    ("mixture", "RWMH"): 0.030,
    ("mixture", "ULA"): 0.065,
    ("mixture", "MALA"): 0.055,
    ("mixture", "HMC"): 0.060,
    ("imbalanced_mixture", "RWMH"): 0.025,
    ("imbalanced_mixture", "ULA"): 0.030,
    ("imbalanced_mixture", "MALA"): 0.035,
    ("imbalanced_mixture", "HMC"): 0.035,
}


# Deliberately reasonable starting guesses, not homework answers.
DEFAULT_SETTINGS: dict[tuple[str, str], dict[str, float | int]] = {
    ("gaussian", "RWMH"): {"scale": 0.12},
    ("gaussian", "ULA"): {"scale": 0.008},
    ("gaussian", "MALA"): {"scale": 0.008},
    ("gaussian", "HMC"): {"scale": 0.08, "n_leapfrog": 15},
    ("banana", "RWMH"): {"scale": 0.18},
    ("banana", "ULA"): {"scale": 0.002},
    ("banana", "MALA"): {"scale": 0.003},
    ("banana", "HMC"): {"scale": 0.05, "n_leapfrog": 20},
    ("mixture", "RWMH"): {"scale": 0.90},
    ("mixture", "ULA"): {"scale": 0.060},
    ("mixture", "MALA"): {"scale": 0.080},
    ("mixture", "HMC"): {"scale": 0.18, "n_leapfrog": 20},
    ("imbalanced_mixture", "RWMH"): {"scale": 1.0},
    ("imbalanced_mixture", "ULA"): {"scale": 0.002},
    ("imbalanced_mixture", "MALA"): {"scale": 0.080},
    ("imbalanced_mixture", "HMC"): {"scale": 0.12, "n_leapfrog": 20},
}
for _setting in DEFAULT_SETTINGS.values():
    _setting.update({"n_chains": 2, "burn_fraction": 0.25})


@dataclass
class Experiment:
    target_key: str
    result: SamplerResult
    burn_in: int
    seed: int
    metrics: dict[str, Any]


@dataclass
class EnsembleExperiment:
    """Independent chains sharing one fixed target-evaluation budget."""

    target_key: str
    method: str
    scale: float
    n_leapfrog: int | None
    n_chains: int
    burn_fraction: float
    base_seed: int
    target_eval_budget: int
    chains: list[Experiment]
    metrics: dict[str, Any]


@dataclass
class BenchmarkResult:
    """Repeated-seed result for one target, method, and hyperparameter setting."""

    target_key: str
    method: str
    scale: float
    n_leapfrog: int | None
    n_chains: int
    burn_fraction: float
    target_eval_budget: int
    experiments: list[EnsembleExperiment]
    summary: dict[str, float]

    @property
    def acceptance_rate(self) -> float | None:
        values = [
            experiment.metrics["acceptance_rate"]
            for experiment in self.experiments
            if experiment.metrics["acceptance_rate"] is not None
        ]
        return float(np.mean(values)) if values else None

    @property
    def threshold(self) -> float:
        return SWD_PASS_THRESHOLDS[self.target_key, self.method]

    @property
    def passed(self) -> bool:
        return benchmark_passed(self.target_key, self.method, self.summary)


def run_experiment(
    target_key: str,
    method: str,
    scale: float,
    n_steps: int = 3000,
    burn_fraction: float = 0.25,
    seed: int = 11,
    initial: Array | None = None,
    n_leapfrog: int = 10,
) -> Experiment:
    """Run one chain and score its retained states."""

    if target_key not in TARGETS:
        raise KeyError(f"Unknown target {target_key!r}.")
    if not 0.0 <= burn_fraction < 0.9:
        raise ValueError("burn_fraction must lie in [0, 0.9).")
    target = TARGETS[target_key]
    burn_in = int(np.floor(n_steps * burn_fraction))
    result = run_sampler(
        method,
        target,
        n_steps,
        scale,
        seed,
        initial=initial,
        n_leapfrog=n_leapfrog,
    )
    return Experiment(
        target_key=target_key,
        result=result,
        burn_in=burn_in,
        seed=int(seed),
        metrics=compute_metrics(target, result, burn_in),
    )


def run_ensemble_experiment(
    target_key: str,
    method: str,
    scale: float,
    target_eval_budget: int = BENCHMARK_TARGET_EVAL_BUDGET,
    n_chains: int = 1,
    burn_fraction: float = 0.25,
    base_seed: int = 11,
    initial: Array | None = None,
    n_leapfrog: int = 10,
) -> EnsembleExperiment:
    """Run independent chains that split one fixed target-evaluation budget."""

    if target_key not in TARGETS:
        raise KeyError(f"Unknown target {target_key!r}.")
    if int(n_chains) != n_chains or not 1 <= n_chains <= MAX_STUDENT_CHAINS:
        raise ValueError(f"n_chains must be an integer from 1 to {MAX_STUDENT_CHAINS}.")
    if int(target_eval_budget) != target_eval_budget or target_eval_budget < 1:
        raise ValueError("target_eval_budget must be a positive integer.")
    if not 0.0 <= burn_fraction < 0.9:
        raise ValueError("burn_fraction must lie in [0, 0.9).")

    chain_count = int(n_chains)
    total_budget = int(target_eval_budget)
    base_allocation, remainder = divmod(total_budget, chain_count)
    allocations = [
        base_allocation + int(index < remainder) for index in range(chain_count)
    ]
    state_counts = [
        states_for_target_evals(method, allocation, n_leapfrog)
        for allocation in allocations
    ]
    if any(
        state_count - int(np.floor(state_count * burn_fraction)) < 3
        for state_count in state_counts
    ):
        raise ValueError(
            "This budget, chain count, burn-in, and HMC trajectory leave fewer "
            "than three retained states in at least one chain."
        )

    chains = [
        run_experiment(
            target_key=target_key,
            method=method,
            scale=scale,
            n_steps=state_count,
            burn_fraction=burn_fraction,
            seed=int(base_seed) + 101 * index,
            initial=initial,
            n_leapfrog=n_leapfrog,
        )
        for index, state_count in enumerate(state_counts)
    ]
    target = TARGETS[target_key]
    metrics = compute_ensemble_metrics(
        target,
        [chain.result for chain in chains],
        [chain.burn_in for chain in chains],
    )
    return EnsembleExperiment(
        target_key=target_key,
        method=chains[0].result.method,
        scale=float(scale),
        n_leapfrog=int(n_leapfrog) if chains[0].result.method == "HMC" else None,
        n_chains=chain_count,
        burn_fraction=float(burn_fraction),
        base_seed=int(base_seed),
        target_eval_budget=total_budget,
        chains=chains,
        metrics=metrics,
    )


def summarize_experiments(
    experiments: Sequence[Experiment | EnsembleExperiment],
) -> dict[str, float]:
    """Aggregate repeated-seed metrics while retaining a worst-run check."""

    if not experiments:
        raise ValueError("At least one experiment is required.")
    swd = np.array([experiment.metrics["swd"] for experiment in experiments])
    ess = np.array(
        [experiment.metrics["ess_per_1000_evals"] for experiment in experiments]
    )
    return {
        "mean_swd": float(np.mean(swd)) if np.all(np.isfinite(swd)) else float("inf"),
        "sd_swd": float(np.std(swd, ddof=1))
        if swd.size > 1 and np.all(np.isfinite(swd))
        else 0.0,
        "worst_swd": float(np.max(swd)) if np.all(np.isfinite(swd)) else float("inf"),
        "mean_ess_per_1000": float(np.mean(ess)),
        "divergent_runs": float(
            sum(experiment.metrics["diverged"] for experiment in experiments)
        ),
    }


def benchmark_setting(
    target_key: str,
    method: str,
    scale: float,
    seeds: Sequence[int] = BENCHMARK_SEEDS,
    target_eval_budget: int = BENCHMARK_TARGET_EVAL_BUDGET,
    n_chains: int = 1,
    burn_fraction: float = 0.25,
    n_leapfrog: int = 10,
) -> tuple[list[EnsembleExperiment], dict[str, float]]:
    """Evaluate one ensemble setting over fixed independent benchmark repeats."""

    experiments = [
        run_ensemble_experiment(
            target_key=target_key,
            method=method,
            scale=scale,
            target_eval_budget=target_eval_budget,
            n_chains=n_chains,
            burn_fraction=burn_fraction,
            base_seed=int(seed),
            n_leapfrog=n_leapfrog,
        )
        for seed in seeds
    ]
    return experiments, summarize_experiments(experiments)


def benchmark_passed(target_key: str, method: str, summary: Mapping[str, float]) -> bool:
    """Apply the public homework pass rule to one repeated-seed summary."""

    pair = (target_key, method)
    if pair not in SWD_PASS_THRESHOLDS:
        raise KeyError(f"No SWD pass threshold is defined for {pair!r}.")
    return bool(
        summary["worst_swd"] <= SWD_PASS_THRESHOLDS[pair]
        and summary["divergent_runs"] == 0
    )


def normalized_setting(
    settings: Mapping[tuple[str, str], Mapping[str, float | int]],
    target_key: str,
    method: str,
) -> tuple[float, int, int, float]:
    """Read a student setting with a consistent default for non-HMC methods."""

    setting = settings[(target_key, method)]
    return (
        float(setting["scale"]),
        int(setting.get("n_leapfrog", 10)),
        int(setting.get("n_chains", 1)),
        float(setting.get("burn_fraction", 0.25)),
    )


def benchmark_all_settings(
    settings: Mapping[tuple[str, str], Mapping[str, float | int]],
    target_eval_budget: int = BENCHMARK_TARGET_EVAL_BUDGET,
    seeds: Sequence[int] = BENCHMARK_SEEDS,
) -> list[BenchmarkResult]:
    """Benchmark every core target-method ensemble at one fixed work budget."""

    results: list[BenchmarkResult] = []
    for target_key in CORE_TARGET_KEYS:
        for method in METHODS:
            scale, n_leapfrog, n_chains, burn_fraction = normalized_setting(
                settings, target_key, method
            )
            experiments, summary = benchmark_setting(
                target_key,
                method,
                scale,
                seeds=seeds,
                target_eval_budget=target_eval_budget,
                n_chains=n_chains,
                burn_fraction=burn_fraction,
                n_leapfrog=n_leapfrog,
            )
            results.append(
                BenchmarkResult(
                    target_key=target_key,
                    method=method,
                    scale=scale,
                    n_leapfrog=n_leapfrog if method == "HMC" else None,
                    n_chains=n_chains,
                    burn_fraction=burn_fraction,
                    target_eval_budget=int(target_eval_budget),
                    experiments=experiments,
                    summary=summary,
                )
            )
    return results


def swd_convergence(
    experiments: Sequence[EnsembleExperiment],
    checkpoints: Sequence[int] | None = None,
) -> dict[str, Array]:
    """Track ensemble SWD as cumulative target-evaluation work increases.

    At every checkpoint the available work is split across the chosen number of
    chains and the chosen burn fraction is applied to each resulting prefix.
    The final point therefore equals the official full-budget score.
    """

    if not experiments:
        raise ValueError("At least one experiment is required.")
    target_key = experiments[0].target_key
    if any(experiment.target_key != target_key for experiment in experiments):
        raise ValueError("All experiments must use the same target.")
    first_experiment = experiments[0]
    for experiment in experiments[1:]:
        if (
            experiment.method != first_experiment.method
            or experiment.n_chains != first_experiment.n_chains
            or experiment.n_leapfrog != first_experiment.n_leapfrog
            or experiment.burn_fraction != first_experiment.burn_fraction
            or experiment.target_eval_budget != first_experiment.target_eval_budget
        ):
            raise ValueError("All experiments must use the same ensemble setting.")

    total_budget = first_experiment.target_eval_budget
    if checkpoints is None:
        checkpoints_array = np.unique(
            np.linspace(max(100, total_budget // 10), total_budget, num=11).astype(int)
        )
    else:
        checkpoints_array = np.unique(np.asarray(checkpoints, dtype=int))
        if (
            checkpoints_array.ndim != 1
            or checkpoints_array.size == 0
            or checkpoints_array[0] < 1
            or checkpoints_array[-1] > total_budget
        ):
            raise ValueError(
                f"checkpoints must lie between 1 and the {total_budget} work-unit budget."
            )

    reference = get_reference_sample(target_key)
    score_rows = []
    for experiment in experiments:
        experiment_scores = []
        for checkpoint in checkpoints_array:
            allocation, remainder = divmod(int(checkpoint), experiment.n_chains)
            chain_budgets = [
                allocation + int(index < remainder)
                for index in range(experiment.n_chains)
            ]
            retained_prefixes = []
            try:
                for chain, chain_budget in zip(experiment.chains, chain_budgets):
                    n_states = states_for_target_evals(
                        experiment.method,
                        chain_budget,
                        experiment.n_leapfrog or 10,
                    )
                    n_states = min(n_states, chain.result.samples.shape[0])
                    burn_in = int(np.floor(n_states * experiment.burn_fraction))
                    if n_states - burn_in < 2:
                        raise ValueError
                    retained_prefixes.append(
                        chain.result.samples[burn_in:n_states]
                    )
                combined = np.concatenate(retained_prefixes, axis=0)
                score = standardized_sliced_wasserstein(combined, reference)
            except ValueError:
                score = float("inf")
            experiment_scores.append(score)
        score_rows.append(experiment_scores)
    scores = np.asarray(score_rows, dtype=float)
    finite_columns = np.all(np.isfinite(scores), axis=0)
    means = np.full(checkpoints_array.shape, np.inf, dtype=float)
    standard_deviations = np.full(checkpoints_array.shape, np.inf, dtype=float)
    worst = np.full(checkpoints_array.shape, np.inf, dtype=float)
    means[finite_columns] = np.mean(scores[:, finite_columns], axis=0)
    if scores.shape[0] > 1:
        standard_deviations[finite_columns] = np.std(
            scores[:, finite_columns], axis=0, ddof=1
        )
    else:
        standard_deviations[finite_columns] = 0.0
    worst[finite_columns] = np.max(scores[:, finite_columns], axis=0)
    return {
        "base_seeds": np.asarray([experiment.base_seed for experiment in experiments]),
        "target_evals": checkpoints_array,
        "mean_swd": means,
        "sd_swd": standard_deviations,
        "worst_swd": worst,
        "all_swd": scores,
    }
