"""Accuracy and mixing diagnostics for comparing sampler output."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import Any

import numpy as np

from .samplers import SamplerResult
from .targets import Array, TARGETS, Target2D


def autocorrelation_1d(values: Array, max_lag: int | None = None) -> Array:
    """Return the FFT autocorrelation estimate, normalized at lag zero."""

    x = np.asarray(values, dtype=float)
    if x.ndim != 1:
        raise ValueError("autocorrelation_1d expects a one-dimensional array.")
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return np.array([])
    centered = x - np.mean(x)
    variance = float(np.dot(centered, centered))
    if variance <= np.finfo(float).eps:
        length = max(0, min(n - 1, max_lag if max_lag is not None else n - 1))
        return np.concatenate(([1.0], np.zeros(length)))
    fft_size = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(centered, n=fft_size)
    autocovariance = np.fft.irfft(
        spectrum * np.conjugate(spectrum), n=fft_size
    )[:n]
    autocorrelation = np.real(autocovariance / autocovariance[0])
    if max_lag is not None:
        autocorrelation = autocorrelation[: min(n, max_lag + 1)]
    return autocorrelation


def effective_sample_size_1d(values: Array) -> float:
    """Estimate ESS with Geyer's initial-positive-pair truncation.

    A constant feature receives ESS zero.  This is useful for detecting a chain
    that never changes mixture component.
    """

    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 3 or float(np.var(x)) <= np.finfo(float).eps:
        return 0.0
    autocorrelation = autocorrelation_1d(x)
    positive_sum = 0.0
    for first_lag in range(1, len(autocorrelation) - 1, 2):
        pair_sum = float(
            autocorrelation[first_lag] + autocorrelation[first_lag + 1]
        )
        if pair_sum <= 0.0:
            break
        positive_sum += pair_sum
    integrated_time = max(1.0, 1.0 + 2.0 * positive_sum)
    return float(np.clip(n / integrated_time, 0.0, n))


@lru_cache(maxsize=None)
def projection_directions(n_directions: int = 32, seed: int = 20240613) -> Array:
    """Return fixed unit projections so settings face identical scoring noise."""

    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(n_directions, 2))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    directions.setflags(write=False)
    return directions


def standardized_sliced_wasserstein(
    samples: Array,
    reference: Array,
    directions: Array | None = None,
    n_quantiles: int = 512,
) -> float:
    """Average standardized Wasserstein-1 error over fixed projections."""

    samples = np.asarray(samples, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if (
        samples.ndim != 2
        or reference.ndim != 2
        or samples.shape[1] != 2
        or reference.shape[1] != 2
    ):
        raise ValueError("samples and reference must both have shape (n, 2).")
    if samples.shape[0] < 2 or reference.shape[0] < 2:
        return float("inf")
    if not np.all(np.isfinite(samples)) or not np.all(np.isfinite(reference)):
        return float("inf")

    directions = projection_directions() if directions is None else np.asarray(directions)
    q_count = int(min(n_quantiles, samples.shape[0], reference.shape[0]))
    probabilities = (np.arange(q_count, dtype=float) + 0.5) / q_count
    projected_samples = samples @ directions.T
    projected_reference = reference @ directions.T
    sample_quantiles = np.quantile(projected_samples, probabilities, axis=0)
    reference_quantiles = np.quantile(projected_reference, probabilities, axis=0)
    projected_scale = np.maximum(
        np.std(projected_reference, axis=0, ddof=1), 1e-12
    )
    distances = np.mean(np.abs(sample_quantiles - reference_quantiles), axis=0)
    return float(np.mean(distances / projected_scale))


@lru_cache(maxsize=None)
def get_reference_sample(target_key: str, n: int = 6000, seed: int = 314159) -> Array:
    """Return a cached exact iid sample for a target."""

    if target_key not in TARGETS:
        raise KeyError(f"Unknown target {target_key!r}.")
    reference = TARGETS[target_key].sample_reference(n, np.random.default_rng(seed))
    reference.setflags(write=False)
    return reference


@lru_cache(maxsize=None)
def reference_noise_floor(target_key: str) -> float:
    """SWD between two independent reference halves."""

    reference = get_reference_sample(target_key)
    midpoint = reference.shape[0] // 2
    return standardized_sliced_wasserstein(
        reference[:midpoint], reference[midpoint:]
    )


def compute_metrics(
    target: Target2D,
    result: SamplerResult,
    burn_in: int,
    reference: Array | None = None,
) -> dict[str, Any]:
    """Compute accuracy, ESS, acceptance, and target-specific diagnostics."""

    if burn_in < 0 or burn_in >= result.samples.shape[0] - 2:
        raise ValueError("burn_in must leave at least three retained states.")
    retained = result.samples[burn_in:]
    reference = (
        get_reference_sample(target.key) if reference is None else np.asarray(reference)
    )
    valid = not result.diverged and np.all(np.isfinite(retained))
    if not valid:
        return {
            "swd": float("inf"),
            "noise_floor": reference_noise_floor(target.key),
            "ess_min": 0.0,
            "ess_per_1000_evals": 0.0,
            "ess_by_feature": {},
            "acceptance_rate": result.acceptance_rate,
            "target_evals": result.target_evals,
            "retained": int(retained.shape[0]),
            "diverged": True,
            "task": {},
        }

    swd = standardized_sliced_wasserstein(retained, reference)
    features, labels = target.diagnostic_features(retained)
    ess_values = [
        effective_sample_size_1d(features[:, index])
        for index in range(features.shape[1])
    ]
    ess_by_feature = dict(zip(labels, ess_values))
    ess_min = float(min(ess_values)) if ess_values else 0.0
    return {
        "swd": swd,
        "noise_floor": reference_noise_floor(target.key),
        "ess_min": ess_min,
        "ess_per_1000_evals": 1000.0 * ess_min / max(1, result.target_evals),
        "ess_by_feature": ess_by_feature,
        "acceptance_rate": result.acceptance_rate,
        "target_evals": result.target_evals,
        "retained": int(retained.shape[0]),
        "diverged": False,
        "task": target.task_diagnostics(retained),
    }


def compute_ensemble_metrics(
    target: Target2D,
    results: Sequence[SamplerResult],
    burn_ins: Sequence[int],
    reference: Array | None = None,
) -> dict[str, Any]:
    """Score independent chains as one fixed-budget sampling ensemble.

    Accuracy and target diagnostics use the concatenated retained draws. ESS is
    computed within each chain and then summed, because concatenating chains
    would create artificial transitions at their boundaries.
    """

    if not results or len(results) != len(burn_ins):
        raise ValueError("results and burn_ins must have the same nonzero length.")
    retained_chains: list[Array] = []
    for result, burn_in in zip(results, burn_ins):
        if burn_in < 0 or burn_in >= result.samples.shape[0] - 2:
            raise ValueError("Every burn_in must leave at least three retained states.")
        retained_chains.append(result.samples[int(burn_in) :])

    reference = (
        get_reference_sample(target.key) if reference is None else np.asarray(reference)
    )
    total_target_evals = int(sum(result.target_evals for result in results))
    total_retained = int(sum(len(retained) for retained in retained_chains))
    valid = all(
        not result.diverged and np.all(np.isfinite(retained))
        for result, retained in zip(results, retained_chains)
    )
    acceptance_arrays = [
        result.accepted for result in results if result.accepted is not None
    ]
    acceptance_rate = (
        float(np.mean(np.concatenate(acceptance_arrays)))
        if acceptance_arrays
        else None
    )
    common = {
        "noise_floor": reference_noise_floor(target.key),
        "acceptance_rate": acceptance_rate,
        "target_evals": total_target_evals,
        "retained": total_retained,
        "n_chains": len(results),
        "states_by_chain": [int(result.samples.shape[0]) for result in results],
        "retained_by_chain": [int(len(retained)) for retained in retained_chains],
    }
    if not valid:
        return {
            "swd": float("inf"),
            "ess_min": 0.0,
            "ess_per_1000_evals": 0.0,
            "ess_by_feature": {},
            "diverged": True,
            "task": {},
            **common,
        }

    combined = np.concatenate(retained_chains, axis=0)
    swd = standardized_sliced_wasserstein(combined, reference)
    feature_ess: dict[str, float] = {}
    for retained in retained_chains:
        features, labels = target.diagnostic_features(retained)
        for index, label in enumerate(labels):
            feature_ess[label] = feature_ess.get(label, 0.0) + effective_sample_size_1d(
                features[:, index]
            )
    ess_min = float(min(feature_ess.values())) if feature_ess else 0.0
    task = target.task_diagnostics(combined)
    if "mode switches" in task:
        task["mode switches"] = float(
            sum(
                target.task_diagnostics(retained).get("mode switches", 0.0)
                for retained in retained_chains
            )
        )
    return {
        "swd": swd,
        "ess_min": ess_min,
        "ess_per_1000_evals": 1000.0 * ess_min / max(1, total_target_evals),
        "ess_by_feature": feature_ess,
        "diverged": False,
        "task": task,
        **common,
    }
