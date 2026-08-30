"""Matplotlib figures and compact notebook metric tables."""

from __future__ import annotations

from html import escape
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .experiments import Experiment
from .metrics import autocorrelation_1d, get_reference_sample
from .targets import TARGETS, Target2D


def _density_contours(ax: Any, target: Target2D, grid_size: int = 150) -> None:
    x = np.linspace(*target.bounds[0], grid_size)
    y = np.linspace(*target.bounds[1], grid_size)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack([xx.ravel(), yy.ravel()])
    with np.errstate(over="ignore", invalid="ignore"):
        log_density = np.asarray(target.log_prob(points)).reshape(xx.shape)
    finite = np.isfinite(log_density)
    if not np.any(finite):
        return
    relative = log_density - np.max(log_density[finite])
    relative[~finite] = np.nan
    ax.contour(
        xx,
        yy,
        relative,
        levels=[-8.0, -5.0, -3.0, -2.0, -1.0, -0.4],
        colors="0.62",
        linewidths=0.8,
    )


def _finite_prefix(samples: np.ndarray) -> np.ndarray:
    valid = np.all(np.isfinite(samples), axis=1)
    invalid_indices = np.flatnonzero(~valid)
    end = int(invalid_indices[0]) if invalid_indices.size else samples.shape[0]
    return samples[:end]


def plot_chain_panel(
    ax: Any,
    target: Target2D,
    experiment: Experiment,
    max_points: int = 700,
) -> None:
    """Plot target contours, orange burn-in, and blue retained states."""

    _density_contours(ax, target, grid_size=120)
    chain = _finite_prefix(experiment.result.samples)
    burn_end = min(experiment.burn_in, len(chain))
    if burn_end > 1:
        stride = max(1, burn_end // 250)
        ax.plot(
            chain[:burn_end:stride, 0],
            chain[:burn_end:stride, 1],
            color="#e67e22",
            linewidth=0.8,
            alpha=0.55,
        )
    retained = chain[burn_end:]
    if len(retained):
        stride = max(1, len(retained) // max_points)
        shown = retained[::stride]
        ax.scatter(
            shown[:, 0],
            shown[:, 1],
            s=7,
            alpha=0.28,
            color="#2468b4",
            edgecolors="none",
        )
    start = chain[0] if len(chain) else target.start
    ax.scatter(
        *start,
        marker="*",
        s=75,
        color="#e67e22",
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set(xlim=target.bounds[0], ylim=target.bounds[1], xlabel="x₁", ylabel="x₂")


def plot_sampling_run(
    experiment: Experiment,
    figure: Figure | None = None,
) -> Figure:
    """Show path, traces, running statistic, and post-burn-in autocorrelation."""

    target = TARGETS[experiment.target_key]
    result = experiment.result
    metrics = experiment.metrics
    figure = plt.figure(figsize=(12.0, 8.2)) if figure is None else figure
    figure.clear()
    ax_path, ax_trace, ax_running, ax_acf = figure.subplots(2, 2).ravel()

    plot_chain_panel(ax_path, target, experiment, max_points=1000)
    ax_path.set_title("Target contours and chain path")

    chain = _finite_prefix(result.samples)
    iterations = np.arange(len(chain))
    ax_trace.plot(iterations, chain[:, 0], linewidth=0.65, alpha=0.85, label="x₁")
    ax_trace.plot(iterations, chain[:, 1], linewidth=0.65, alpha=0.75, label="x₂")
    ax_trace.axvline(
        experiment.burn_in,
        color="#e67e22",
        linestyle="--",
        linewidth=1.2,
        label="burn-in",
    )
    ax_trace.set(xlabel="iteration", ylabel="coordinate", title="Trace: movement and sticking")
    ax_trace.legend(loc="upper right", fontsize=8, ncols=3)

    retained = chain[min(experiment.burn_in, len(chain)) :]
    running_x = np.arange(1, len(retained) + 1)
    if len(retained):
        if "mixture" in target.key:
            indicator = (retained[:, 0] < 0.0).astype(float)
            running = np.cumsum(indicator) / running_x
            target_mass = float(target.weights[0])
            ax_running.plot(running_x, running, color="#2468b4", label="running left-mode mass")
            ax_running.axhline(
                target_mass,
                color="black",
                linestyle="--",
                label=f"target: {target_mass:.2f}",
            )
            ax_running.set_ylim(-0.03, 1.03)
            ylabel = "probability"
        else:
            reference_mean = np.mean(get_reference_sample(target.key), axis=0)
            running_mean = np.cumsum(retained, axis=0) / running_x[:, None]
            ax_running.plot(running_x, running_mean[:, 0], label="running mean x₁")
            ax_running.plot(running_x, running_mean[:, 1], label="running mean x₂")
            ax_running.axhline(reference_mean[0], color="C0", linestyle="--", linewidth=1.0)
            ax_running.axhline(reference_mean[1], color="C1", linestyle="--", linewidth=1.0)
            ylabel = "running mean"
        ax_running.legend(loc="best", fontsize=8)
    else:
        ylabel = "running statistic"
    ax_running.set(
        xlabel="retained iteration",
        ylabel=ylabel,
        title="Convergence of a global statistic",
    )

    if len(retained) >= 3 and np.all(np.isfinite(retained)):
        features, labels = target.diagnostic_features(retained)
        max_lag = min(60, len(retained) - 1)
        for index, label in enumerate(labels):
            autocorrelation = autocorrelation_1d(features[:, index], max_lag=max_lag)
            ax_acf.plot(
                np.arange(len(autocorrelation)),
                autocorrelation,
                linewidth=1.3,
                label=label,
            )
        ax_acf.axhline(0.0, color="black", linewidth=0.8)
        ax_acf.legend(loc="upper right", fontsize=8)
    ax_acf.set(xlabel="lag", ylabel="autocorrelation", title="Autocorrelation after burn-in")

    score_text = "∞" if not np.isfinite(metrics["swd"]) else f"{metrics['swd']:.3f}"
    acceptance = metrics["acceptance_rate"]
    acceptance_text = "N/A" if acceptance is None else f"{100.0 * acceptance:.1f}%"
    integrator = (
        f", L={result.n_leapfrog}" if result.method == "HMC" else ""
    )
    figure.suptitle(
        f"{target.name} — {result.method} | scale={result.scale:.4g}{integrator} "
        f"SWD={score_text} acceptance={acceptance_text}",
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    return figure


def plot_mode_ratio_comparison(
    langevin_experiment: Experiment,
    hmc_experiment: Experiment,
) -> Figure:
    """Compare paths and empirical mode ratios on the 10:90 mixture."""

    if (
        langevin_experiment.target_key != "imbalanced_mixture"
        or hmc_experiment.target_key != "imbalanced_mixture"
    ):
        raise ValueError("Both experiments must use the imbalanced_mixture target.")
    target = TARGETS["imbalanced_mixture"]
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.2))
    ax_langevin, ax_hmc, ax_running, ax_ratio = axes.ravel()
    plot_chain_panel(ax_langevin, target, langevin_experiment, max_points=900)
    plot_chain_panel(ax_hmc, target, hmc_experiment, max_points=900)
    ax_langevin.set_title(f"{langevin_experiment.result.method}: retained states")
    ax_hmc.set_title(
        f"HMC: retained states (L={hmc_experiment.result.n_leapfrog})"
    )

    probabilities: dict[str, list[float]] = {"target": list(target.weights)}
    for experiment, label, color in (
        (langevin_experiment, langevin_experiment.result.method, "#e67e22"),
        (hmc_experiment, "HMC", "#2468b4"),
    ):
        retained = experiment.result.samples[experiment.burn_in :]
        left = (retained[:, 0] < 0.0).astype(float)
        running = np.cumsum(left) / np.arange(1, len(left) + 1)
        ax_running.plot(running, label=label, color=color)
        left_probability = float(np.mean(left))
        probabilities[label] = [left_probability, 1.0 - left_probability]
    ax_running.axhline(
        target.weights[0], color="black", linestyle="--", label="target left mass = 0.10"
    )
    ax_running.set(
        xlabel="retained iteration",
        ylabel="running left-mode fraction",
        ylim=(-0.02, 0.35),
        title="Does the rare-mode fraction approach 0.10?",
    )
    ax_running.legend(fontsize=8)

    x = np.arange(2)
    width = 0.24
    for index, (label, values) in enumerate(probabilities.items()):
        offset = (index - 1) * width
        ax_ratio.bar(x + offset, values, width, label=label)
        for position, value in zip(x + offset, values):
            ax_ratio.text(position, value + 0.015, f"{value:.2f}", ha="center", fontsize=8)
    ax_ratio.set(
        xticks=x,
        xticklabels=["left / rare mode", "right / dominant mode"],
        ylabel="fraction of retained states",
        ylim=(0.0, 1.05),
        title="Final empirical mode proportions",
    )
    ax_ratio.legend(fontsize=8)
    figure.suptitle("10:90 mixture: local Langevin exploration versus HMC", fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    return figure


def _format_number(value: Any, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    number = float(value)
    if not np.isfinite(number):
        return "∞"
    if abs(number) >= 1000:
        return f"{number:,.0f}"
    return f"{number:.{digits}f}"


def metrics_html(
    experiment: Experiment,
    summary: Mapping[str, float] | None = None,
) -> str:
    """Return a compact metric table suitable for notebook display."""

    metrics = experiment.metrics
    rows: list[tuple[str, str, str]] = [
        (
            "Standardized sliced-Wasserstein (primary; lower is better)",
            _format_number(metrics["swd"]),
            f"reference noise floor ≈ {_format_number(metrics['noise_floor'])}",
        ),
        (
            "Minimum ESS after burn-in",
            _format_number(metrics["ess_min"], 1),
            f"{_format_number(metrics['ess_per_1000_evals'], 1)} per 1,000 target evaluations",
        ),
        (
            "Acceptance rate",
            "N/A (ULA has no correction)"
            if metrics["acceptance_rate"] is None
            else f"{100.0 * metrics['acceptance_rate']:.1f}%",
            "diagnostic, not an optimization target",
        ),
    ]
    for label, value in metrics["task"].items():
        rows.append((label, _format_number(value), "target-specific diagnostic"))
    if summary is not None:
        rows.insert(
            0,
            (
                "Repeated-seed mean ± SD SWD",
                f"{_format_number(summary['mean_swd'])} ± {_format_number(summary['sd_swd'])}",
                f"worst run {_format_number(summary['worst_swd'])}; "
                f"{int(summary['divergent_runs'])} divergent run(s)",
            ),
        )
    body = "".join(
        f"<tr><td>{escape(label)}</td><td><b>{escape(value)}</b></td><td>{escape(note)}</td></tr>"
        for label, value, note in rows
    )
    return (
        "<table style='border-collapse:collapse;width:100%;font-size:0.92em'>"
        "<thead><tr><th style='text-align:left'>Metric</th><th style='text-align:left'>Value</th>"
        "<th style='text-align:left'>How to read it</th></tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )
