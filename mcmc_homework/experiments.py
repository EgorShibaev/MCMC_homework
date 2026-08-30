"""Experiment configuration, repeated-seed evaluation, and benchmarking."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .metrics import (
    compute_metrics,
    get_reference_sample,
    standardized_sliced_wasserstein,
)
from .samplers import SamplerResult, run_sampler
from .targets import Array, TARGETS


METHODS = ("RWMH", "ULA", "MALA", "HMC")
CORE_TARGET_KEYS = ("gaussian", "banana", "mixture")
BENCHMARK_SEEDS = (11, 23, 47)

# A row passes only when its *worst* fixed-seed SWD is at or below the
# target-specific threshold and none of the chains diverged.  These thresholds
# were calibrated with the correct reference implementations at the fixed
# 4,000-state homework budget.  They are attainable by every method, but
# the deliberately untuned defaults do not all pass.
SWD_PASS_THRESHOLDS: dict[str, float] = {
    "gaussian": 0.50,
    "banana": 0.50,
    "mixture": 0.75,
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


@dataclass
class Experiment:
    target_key: str
    result: SamplerResult
    burn_in: int
    seed: int
    metrics: dict[str, Any]


@dataclass
class BenchmarkResult:
    """Repeated-seed result for one target, method, and hyperparameter setting."""

    target_key: str
    method: str
    scale: float
    n_leapfrog: int | None
    experiments: list[Experiment]
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
        return SWD_PASS_THRESHOLDS[self.target_key]

    @property
    def passed(self) -> bool:
        return benchmark_passed(self.target_key, self.summary)


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


def summarize_experiments(experiments: Sequence[Experiment]) -> dict[str, float]:
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
    n_steps: int = 4000,
    burn_fraction: float = 0.25,
    n_leapfrog: int = 10,
) -> tuple[list[Experiment], dict[str, float]]:
    """Evaluate one setting at a fixed budget over multiple seeds."""

    experiments = [
        run_experiment(
            target_key,
            method,
            scale,
            n_steps=n_steps,
            burn_fraction=burn_fraction,
            seed=int(seed),
            n_leapfrog=n_leapfrog,
        )
        for seed in seeds
    ]
    return experiments, summarize_experiments(experiments)


def benchmark_passed(target_key: str, summary: Mapping[str, float]) -> bool:
    """Apply the public homework pass rule to one repeated-seed summary."""

    if target_key not in SWD_PASS_THRESHOLDS:
        raise KeyError(f"No SWD pass threshold is defined for {target_key!r}.")
    return bool(
        summary["worst_swd"] <= SWD_PASS_THRESHOLDS[target_key]
        and summary["divergent_runs"] == 0
    )


def normalized_setting(
    settings: Mapping[tuple[str, str], Mapping[str, float | int]],
    target_key: str,
    method: str,
) -> tuple[float, int]:
    """Read a student setting with a consistent default for non-HMC methods."""

    setting = settings[(target_key, method)]
    return float(setting["scale"]), int(setting.get("n_leapfrog", 10))


def benchmark_all_settings(
    settings: Mapping[tuple[str, str], Mapping[str, float | int]],
    n_steps: int = 4000,
    burn_fraction: float = 0.25,
    seeds: Sequence[int] = BENCHMARK_SEEDS,
) -> list[BenchmarkResult]:
    """Benchmark every core target-method pair at one fixed recorded-state budget."""

    results: list[BenchmarkResult] = []
    for target_key in CORE_TARGET_KEYS:
        for method in METHODS:
            scale, n_leapfrog = normalized_setting(settings, target_key, method)
            experiments, summary = benchmark_setting(
                target_key,
                method,
                scale,
                seeds=seeds,
                n_steps=n_steps,
                burn_fraction=burn_fraction,
                n_leapfrog=n_leapfrog,
            )
            results.append(
                BenchmarkResult(
                    target_key=target_key,
                    method=method,
                    scale=scale,
                    n_leapfrog=n_leapfrog if method == "HMC" else None,
                    experiments=experiments,
                    summary=summary,
                )
            )
    return results


def swd_convergence(
    experiments: Sequence[Experiment],
    checkpoints: Sequence[int] | None = None,
) -> dict[str, Array]:
    """Track mean and worst-seed SWD over nested post-burn-in prefixes.

    The x-axis is the number of retained transitions, not wall-clock time.
    Prefixes reuse the exact reference sample and fixed projection directions
    used by the final benchmark score.
    """

    if not experiments:
        raise ValueError("At least one experiment is required.")
    target_key = experiments[0].target_key
    if any(experiment.target_key != target_key for experiment in experiments):
        raise ValueError("All experiments must use the same target.")

    max_retained = min(
        experiment.result.samples.shape[0] - experiment.burn_in
        for experiment in experiments
    )
    if max_retained < 3:
        raise ValueError("Each experiment must have at least three retained states.")
    if checkpoints is None:
        first = min(50, max_retained)
        checkpoints_array = np.unique(
            np.geomspace(first, max_retained, num=11).astype(int)
        )
    else:
        checkpoints_array = np.unique(np.asarray(checkpoints, dtype=int))
        if (
            checkpoints_array.ndim != 1
            or checkpoints_array.size == 0
            or checkpoints_array[0] < 2
            or checkpoints_array[-1] > max_retained
        ):
            raise ValueError(
                f"checkpoints must lie between 2 and {max_retained} retained states."
            )

    reference = get_reference_sample(target_key)
    score_rows = []
    for experiment in experiments:
        retained = experiment.result.samples[experiment.burn_in :]
        score_rows.append(
            [
                standardized_sliced_wasserstein(
                    retained[: int(checkpoint)], reference
                )
                for checkpoint in checkpoints_array
            ]
        )
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
        "retained": checkpoints_array,
        "mean_swd": means,
        "sd_swd": standard_deviations,
        "worst_swd": worst,
        "all_swd": scores,
    }
