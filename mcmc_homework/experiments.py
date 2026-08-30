"""Experiment configuration, repeated-seed evaluation, and benchmarking."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .metrics import compute_metrics
from .samplers import SamplerResult, run_sampler
from .targets import Array, TARGETS


METHODS = ("RWMH", "ULA", "MALA", "HMC")
CORE_TARGET_KEYS = ("gaussian", "banana", "mixture")
BENCHMARK_SEEDS = (11, 23, 47)


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


def normalized_setting(
    settings: Mapping[tuple[str, str], Mapping[str, float | int]],
    target_key: str,
    method: str,
) -> tuple[float, int]:
    """Read a student setting with a consistent default for non-HMC methods."""

    setting = settings[(target_key, method)]
    return float(setting["scale"]), int(setting.get("n_leapfrog", 10))
