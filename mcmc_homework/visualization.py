"""Matplotlib figures and compact notebook metric tables."""

from __future__ import annotations

from html import escape
from collections.abc import Sequence
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .experiments import (
    BENCHMARK_SEEDS,
    METHODS,
    SWD_PASS_THRESHOLDS,
    BenchmarkResult,
    EnsembleExperiment,
    Experiment,
    swd_convergence,
)
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


def plot_ensemble_sampling_run(
    ensemble: EnsembleExperiment,
    figure: Figure | None = None,
    benchmark_curve: Mapping[str, np.ndarray] | None = None,
) -> Figure:
    """Show one repeat's paths and highlight its SWD among all repeat curves."""

    target = TARGETS[ensemble.target_key]
    figure = plt.figure(figsize=(12.0, 8.2)) if figure is None else figure
    figure.clear()
    ax_path, ax_trace, ax_swd, ax_acf = figure.subplots(2, 2).ravel()
    colors = plt.get_cmap("tab10")

    _density_contours(ax_path, target, grid_size=120)
    for index, chain_experiment in enumerate(ensemble.chains):
        chain = _finite_prefix(chain_experiment.result.samples)
        burn_end = min(chain_experiment.burn_in, len(chain))
        color = colors(index % 10)
        if burn_end > 1:
            stride = max(1, burn_end // 120)
            ax_path.plot(
                chain[:burn_end:stride, 0],
                chain[:burn_end:stride, 1],
                color=color,
                linewidth=0.65,
                alpha=0.35,
            )
        retained = chain[burn_end:]
        if len(retained):
            stride = max(1, len(retained) // max(80, 700 // ensemble.n_chains))
            ax_path.scatter(
                retained[::stride, 0],
                retained[::stride, 1],
                s=7,
                alpha=0.25,
                color=color,
                edgecolors="none",
                label=f"chain {index + 1}",
            )
        ax_trace.plot(
            np.arange(len(chain)),
            chain[:, 0],
            color=color,
            linewidth=0.65,
            alpha=0.8,
            label=f"chain {index + 1}",
        )
    ax_path.set(
        xlim=target.bounds[0],
        ylim=target.bounds[1],
        xlabel="x₁",
        ylabel="x₂",
        title="All chain paths and retained states",
    )
    ax_path.legend(loc="best", fontsize=7, ncols=2)
    representative_burn = ensemble.chains[0].burn_in
    ax_trace.axvline(
        representative_burn,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="burn-in",
    )
    ax_trace.set(
        xlabel="state index within each chain",
        ylabel="x₁",
        title="Per-chain traces (chain boundaries are not joined)",
    )
    ax_trace.legend(loc="best", fontsize=7, ncols=2)

    curve = swd_convergence([ensemble]) if benchmark_curve is None else benchmark_curve
    if benchmark_curve is None:
        ax_swd.plot(
            curve["target_evals"],
            np.where(np.isfinite(curve["mean_swd"]), curve["mean_swd"], np.nan),
            marker="o", markersize=3.0, color="#2468b4", label="combined-chain SWD",
        )
    else:
        many_seeds = len(curve["base_seeds"]) > 6
        others_labelled = False
        for index, (seed, scores) in enumerate(zip(curve["base_seeds"], curve["all_swd"], strict=True)):
            selected = int(seed) == ensemble.base_seed
            label = f"Seed {seed}" + (" (selected)" if selected else "")
            finite = np.isfinite(scores)
            if not np.all(finite):
                label += " — non-finite"
            if many_seeds and not selected:
                label = f"Other {len(curve['base_seeds']) - 1} seeds" if not others_labelled else "_nolegend_"
                others_labelled = True
            ax_swd.plot(
                curve["target_evals"], np.where(finite, scores, np.nan),
                color="#888888" if many_seeds and not selected else colors(index % 10),
                linewidth=2.6 if selected else 1.3,
                alpha=1.0 if selected else (0.35 if many_seeds else 0.65),
                linestyle="-" if selected else "--",
                marker="o", markersize=5.0 if selected else 3.0,
                zorder=4 if selected else 2,
                label=label,
            )
    threshold = SWD_PASS_THRESHOLDS.get((ensemble.target_key, ensemble.method))
    if threshold is not None:
        ax_swd.axhline(
            threshold,
            color="black",
            linestyle="--",
            linewidth=1.1,
            label=f"official target = {threshold:.3f}",
        )
    ax_swd.set(
        xlabel="target-evaluation budget per repeat",
        ylabel="SWD (lower is better)",
        title="Combined accuracy as the budget is spent" if benchmark_curve is None
        else "SWD by seed",
    )
    ax_swd.legend(loc="best", fontsize=8)

    retained_chains = [
        chain.result.samples[chain.burn_in :] for chain in ensemble.chains
    ]
    max_lag = min(60, *(len(retained) - 1 for retained in retained_chains))
    if max_lag >= 1 and all(np.all(np.isfinite(x)) for x in retained_chains):
        first_features, labels = target.diagnostic_features(retained_chains[0])
        del first_features
        for feature_index, label in enumerate(labels):
            chain_acf = []
            for retained in retained_chains:
                features, _ = target.diagnostic_features(retained)
                chain_acf.append(
                    autocorrelation_1d(features[:, feature_index], max_lag=max_lag)
                )
            ax_acf.plot(
                np.arange(max_lag + 1),
                np.mean(chain_acf, axis=0),
                linewidth=1.3,
                label=label,
            )
        ax_acf.axhline(0.0, color="black", linewidth=0.8)
        ax_acf.legend(loc="upper right", fontsize=8)
    ax_acf.set(
        xlabel="lag within a chain",
        ylabel="mean autocorrelation",
        title="Within-chain autocorrelation (averaged across chains)",
    )

    score = ensemble.metrics["swd"]
    score_text = "∞" if not np.isfinite(score) else f"{score:.3f}"
    subtitle = f"Displayed paths: base seed {ensemble.base_seed}; this repeat's SWD={score_text}"
    integrator = f", L={ensemble.n_leapfrog}" if ensemble.method == "HMC" else ""
    title = (
        f"{target.name} — {ensemble.method} | scale={ensemble.scale:.4g}{integrator}, "
        f"C={ensemble.n_chains}, burn={100.0 * ensemble.burn_fraction:.0f}%, "
        f"budget/repeat={ensemble.target_eval_budget:,}\n"
        f"{subtitle}"
    )
    if benchmark_curve is not None:
        # In the lab, settings are already visible in the controls and scores in
        # the result banner/table. The figure only needs to identify its scope.
        title = (
            f"{target.name} — {ensemble.method}\n"
            f"Paths and ACF: seed {ensemble.base_seed} | "
            f"SWD curves: {len(curve['base_seeds'])} evaluated seeds"
        )
    figure.suptitle(title, fontsize=11)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
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


def plot_swd_convergence(results: Sequence[BenchmarkResult]) -> Figure:
    """Plot the grading metric over target-evaluation budgets for every row."""

    if not results:
        raise ValueError("At least one benchmark result is required.")
    target_keys = list(
        dict.fromkeys(result.target_key for result in results)
    )
    figure, axes = plt.subplots(
        1, len(target_keys), figsize=(5.2 * len(target_keys), 4.2), squeeze=False
    )
    method_colors = dict(zip(METHODS, ("C0", "C1", "C2", "C3")))
    for axis, target_key in zip(axes.ravel(), target_keys):
        target_results = [
            result for result in results if result.target_key == target_key
        ]
        for result in target_results:
            curve = swd_convergence(result.experiments)
            values = curve["worst_swd"]
            finite = np.isfinite(values)
            axis.plot(
                curve["target_evals"][finite],
                values[finite],
                marker="o",
                markersize=3.0,
                linewidth=1.5,
                color=method_colors.get(result.method),
                label=f"{result.method} (≤ {result.threshold:.3f})",
            )
            axis.axhline(
                result.threshold,
                color=method_colors.get(result.method),
                linestyle="--", linewidth=1.0, alpha=0.6,
            )
        axis.set(
            xlabel="cumulative target-evaluation budget",
            ylabel="worst-repeat SWD (lower is better)",
            title=TARGETS[target_key].name,
        )
        axis.set_ylim(bottom=0.0)
        axis.legend(fontsize=8)
    figure.suptitle(
        "Accuracy as the shared target-evaluation budget is spent",
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
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


def _metric_table_html(rows: Sequence[tuple[str, str]], caption: str = "") -> str:
    """Bound the table width and override notebook themes' cell alignment."""
    cell_style = "padding:6px 10px;border-bottom:1px solid #ddd;"
    body = "".join(
        f"<tr><td style='{cell_style}text-align:left'>{escape(label)}</td>"
        f"<td style='{cell_style}text-align:right;font-variant-numeric:tabular-nums'>"
        f"<b>{escape(value)}</b></td></tr>"
        for label, value in rows
    )
    caption_html = (
        f"<caption style='text-align:left;font-weight:600;padding:8px 10px'>"
        f"{escape(caption)}</caption>" if caption else ""
    )
    return (
        "<table style='border-collapse:collapse;width:100%;max-width:620px;"
        "margin:0;font-size:0.95em;table-layout:fixed'>"
        f"{caption_html}"
        f"<thead><tr><th style='{cell_style}text-align:left;width:70%'>Metric</th>"
        f"<th style='{cell_style}text-align:right'>Value</th></tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )


def seed_diagnostics_html(ensemble: EnsembleExperiment) -> str:
    """Selected-repeat results only; settings and benchmark summary live above."""
    metrics = ensemble.metrics
    acceptance = metrics["acceptance_rate"]
    rows = [
        ("SWD", _format_number(metrics["swd"], 4)),
        ("Effective sample size (minimum)", _format_number(metrics["ess_min"], 1)),
        ("ESS / 1,000 evaluations", _format_number(metrics["ess_per_1000_evals"], 1)),
        ("Acceptance", "N/A" if acceptance is None else f"{100.0 * acceptance:.1f}%"),
    ]
    rows.extend((label.capitalize(), _format_number(value)) for label, value in metrics["task"].items())
    return _metric_table_html(rows, caption=f"Seed {ensemble.base_seed} diagnostics")


def metrics_html(
    experiment: Experiment | EnsembleExperiment,
    summary: Mapping[str, float] | None = None,
) -> str:
    """Return a compact metric table suitable for notebook display."""

    metrics = experiment.metrics
    method = experiment.method if isinstance(experiment, EnsembleExperiment) else experiment.result.method
    threshold = SWD_PASS_THRESHOLDS.get((experiment.target_key, method))
    threshold_text = (
        f"≤ {threshold:.3f}" if threshold is not None else "N/A — guided case study"
    )
    swd_label = (
        "Combined retained chains: SWD (lower is better)"
        if isinstance(experiment, EnsembleExperiment)
        else "Single plotted chain: SWD (lower is better)"
    )
    rows: list[tuple[str, str]] = [
        (
            swd_label,
            _format_number(metrics["swd"]),
        ),
        (
            f"Official worst-repeat SWD target ({len(BENCHMARK_SEEDS)} benchmark seeds)",
            threshold_text,
        ),
        (
            "Reference-sample SWD noise floor",
            f"≈ {_format_number(metrics['noise_floor'])}",
        ),
        (
            "Minimum ESS after burn-in",
            _format_number(metrics["ess_min"], 1),
        ),
        (
            "Minimum ESS per 1,000 target evaluations",
            _format_number(metrics["ess_per_1000_evals"], 1),
        ),
        (
            "Acceptance rate",
            "N/A (ULA has no correction)"
            if metrics["acceptance_rate"] is None
            else f"{100.0 * metrics['acceptance_rate']:.1f}%",
        ),
    ]
    if isinstance(experiment, EnsembleExperiment):
        states = metrics["states_by_chain"]
        retained = metrics["retained_by_chain"]
        state_text = (
            str(states[0])
            if len(set(states)) == 1
            else f"{min(states)}–{max(states)}"
        )
        retained_text = (
            str(retained[0])
            if len(set(retained)) == 1
            else f"{min(retained)}–{max(retained)}"
        )
        rows[2:2] = [
            ("Student-selected chains", str(experiment.n_chains)),
            ("Student-selected burn-in", f"{100.0 * experiment.burn_fraction:.0f}% per chain"),
            ("Fixed total target-evaluation budget", f"{experiment.target_eval_budget:,}"),
            ("Actual target evaluations used", f"{metrics['target_evals']:,}"),
            ("Recorded states per chain", state_text),
            ("Retained states per chain", retained_text),
            ("Combined retained states", f"{metrics['retained']:,}"),
        ]
    for label, value in metrics["task"].items():
        rows.append((label, _format_number(value)))
    if summary is not None:
        rows[0:0] = [
            (
                "All repeats: mean ± SD SWD",
                f"{_format_number(summary['mean_swd'])} ± {_format_number(summary['sd_swd'])}",
            ),
            ("All repeats: worst SWD", _format_number(summary["worst_swd"])),
            ("All repeats: divergent runs", str(int(summary["divergent_runs"]))),
        ]
    return _metric_table_html(rows)


def benchmark_results_html(results: Sequence[BenchmarkResult]) -> str:
    """Render the fixed-budget pass rule and all target-method statuses."""

    if not results:
        raise ValueError("At least one benchmark result is required.")
    rendered_rows = []
    passed_count = sum(result.passed for result in results)
    for result in results:
        summary = result.summary
        acceptance = result.acceptance_rate
        acceptance_text = (
            "N/A" if acceptance is None else f"{100.0 * acceptance:.1f}%"
        )
        parameter = "σ" if result.method == "RWMH" else "η"
        setting_text = f"{parameter}={result.scale:.4g}"
        if result.n_leapfrog is not None:
            setting_text = f"ε={result.scale:.4g}, L={result.n_leapfrog}"
        setting_text += (
            f"; C={result.n_chains}, burn={100.0 * result.burn_fraction:.0f}%"
        )
        status = "PASS" if result.passed else "TUNE MORE"
        status_color = "#18864b" if result.passed else "#b33a3a"
        rendered_rows.append(
            "<tr>"
            f"<td>{escape(result.target_key)}</td>"
            f"<td>{escape(result.method)}</td>"
            f"<td>{setting_text}</td>"
            f"<td>{summary['mean_swd']:.3f} ± {summary['sd_swd']:.3f}</td>"
            f"<td><b>{summary['worst_swd']:.3f}</b> / ≤ {result.threshold:.3f}</td>"
            f"<td>{summary['mean_ess_per_1000']:.1f}</td>"
            f"<td>{acceptance_text}</td>"
            f"<td>{int(summary['divergent_runs'])}</td>"
            f"<td><b style='color:{status_color}'>{status}</b></td>"
            "</tr>"
        )
    overall_status = (
        "PASS: all target–method pairs meet the requirement."
        if passed_count == len(results)
        else f"NOT YET: {passed_count}/{len(results)} rows pass. Tune every remaining row."
    )
    header = (
        "<tr><th>target</th><th>method</th><th>setting</th>"
        "<th>mean ± SD SWD</th><th>worst / required</th>"
        "<th>ESS/1k</th><th>accept</th><th>diverged</th><th>status</th></tr>"
    )
    return (
        "<div style='margin:0.4em 0 0.8em 0'>"
        "<b>Pass rule:</b> worst-repeat SWD must meet the target threshold and "
        "all evaluated ensembles must remain finite. Every ensemble receives "
        f"the same {results[0].target_eval_budget:,}-evaluation budget.<br>"
        f"<b>Homework status:</b> {escape(overall_status)}"
        "</div><table>"
        f"{header}{''.join(rendered_rows)}</table>"
    )
