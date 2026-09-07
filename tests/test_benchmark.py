from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import matplotlib.pyplot as plt

from mcmc_homework import (
    SWD_PASS_THRESHOLDS,
    METHODS,
    TARGETS,
    BenchmarkResult,
    benchmark_passed,
    benchmark_results_html,
    benchmark_setting,
    metrics_html,
    plot_ensemble_sampling_run,
    plot_swd_convergence,
    run_ensemble_experiment,
    swd_convergence,
)


class BenchmarkTests(unittest.TestCase):
    def test_lab_plots_each_seed_and_moves_only_the_highlight(self) -> None:
        seeds = (47, 11, 23)  # labels must follow data, not a hard-coded order
        ensembles, _ = benchmark_setting(
            "gaussian", "RWMH", scale=0.7, seeds=seeds, target_eval_budget=800,
        )
        curve = swd_convergence(ensembles, checkpoints=(200, 500, 800))
        np.testing.assert_array_equal(curve["base_seeds"], seeds)
        colors = {}
        for selected in (ensembles[0], ensembles[2]):
            figure = plot_ensemble_sampling_run(selected, benchmark_curve=curve)
            try:
                lines = figure.axes[2].lines
                self.assertEqual(len(lines), 4)  # three seeds plus the threshold
                for index, seed in enumerate(seeds):
                    line = lines[index]
                    active = seed == selected.base_seed
                    self.assertEqual(line.get_label(), f"Seed {seed}" + (" (selected)" if active else ""))
                    np.testing.assert_array_equal(line.get_xdata(), curve["target_evals"])
                    np.testing.assert_allclose(line.get_ydata(), curve["all_swd"][index])
                    self.assertEqual(line.get_linestyle(), "-" if active else "--")
                    self.assertEqual(line.get_linewidth(), 2.6 if active else 1.3)
                    if seed in colors:
                        self.assertEqual(line.get_color(), colors[seed])
                    colors[seed] = line.get_color()
            finally:
                plt.close(figure)
        # A divergent/non-finite prefix must create a gap, not a connecting line.
        curve["all_swd"][1, 1] = np.inf
        figure = plot_ensemble_sampling_run(ensembles[1], benchmark_curve=curve)
        try:
            line = figure.axes[2].lines[1]
            self.assertTrue(np.isnan(line.get_ydata()[1]))
            self.assertIn("non-finite", line.get_label())
        finally:
            plt.close(figure)

    def test_threshold_precision_in_lab_benchmark_and_plots(self) -> None:
        ensembles, summary = benchmark_setting(
            "banana", "RWMH", scale=0.7, target_eval_budget=800, seeds=(11,)
        )
        result = BenchmarkResult(
            target_key="banana", method="RWMH", scale=0.7,
            n_leapfrog=None, n_chains=1, burn_fraction=0.25,
            target_eval_budget=800, experiments=ensembles, summary=summary,
        )
        threshold = f"{result.threshold:.3f}"
        self.assertIn(f"≤ {threshold}", benchmark_results_html([result]))
        self.assertIn(f"≤ {threshold}", metrics_html(ensembles[0]))
        for figure, label in (
            (plot_ensemble_sampling_run(ensembles[0]), f"official target = {threshold}"),
            (plot_swd_convergence([result]), f"RWMH (≤ {threshold})"),
        ):
            try:
                labels = [
                    text for axis in figure.axes
                    for text in axis.get_legend_handles_labels()[1]
                ]
                self.assertIn(label, labels)
            finally:
                plt.close(figure)

    def test_notebook_pass_table_uses_active_thresholds(self) -> None:
        from scripts.build_notebook import build_notebook

        # The prose must follow the grading constants, including after a change.
        with patch.dict(SWD_PASS_THRESHOLDS, {("gaussian", "RWMH"): 0.0123}):
            notebook = build_notebook()
            theory = "\n".join(
                cell.source for cell in notebook.cells if cell.cell_type == "markdown"
            )
            for key, label in (
                ("gaussian", "tilted Gaussian"),
                ("banana", "banana"),
                ("mixture", "65:35 mixture"),
                ("imbalanced_mixture", "10:90 mixture (optional)"),
            ):
                self.assertIn(
                    f"| {label} | " + " | ".join(
                        f"$\\leq {SWD_PASS_THRESHOLDS[key, method]:g}$" for method in METHODS
                    ) + " |", theory
                )

    def test_pass_rule_uses_worst_seed_and_divergence(self) -> None:
        threshold = SWD_PASS_THRESHOLDS["gaussian", "RWMH"]
        passing = {"worst_swd": threshold, "divergent_runs": 0.0}
        too_inaccurate = {"worst_swd": threshold + 1e-6, "divergent_runs": 0.0}
        divergent = {"worst_swd": threshold - 0.1, "divergent_runs": 1.0}
        self.assertTrue(benchmark_passed("gaussian", "RWMH", passing))
        self.assertFalse(benchmark_passed("gaussian", "RWMH", too_inaccurate))
        self.assertFalse(benchmark_passed("gaussian", "RWMH", divergent))

    def test_thresholds_cover_every_pair_and_grade_methods_separately(self) -> None:
        self.assertEqual(set(SWD_PASS_THRESHOLDS), {(target, method) for target in TARGETS for method in METHODS})
        self.assertTrue(all(np.isfinite(value) and value > 0 for value in SWD_PASS_THRESHOLDS.values()))
        summary = {"worst_swd": 0.15, "divergent_runs": 0.0}
        with patch.dict(SWD_PASS_THRESHOLDS, {("gaussian", "RWMH"): .1, ("gaussian", "ULA"): .2}):
            self.assertFalse(benchmark_passed("gaussian", "RWMH", summary))
            self.assertTrue(benchmark_passed("gaussian", "ULA", summary))
            for method, expected in (("RWMH", False), ("ULA", True)):
                result = BenchmarkResult("gaussian", method, .01, None, 1, .25, 40_000, [], summary)
                self.assertEqual(result.threshold, SWD_PASS_THRESHOLDS["gaussian", method])
                self.assertEqual(result.passed, expected)
        with self.assertRaises(KeyError):
            benchmark_passed("gaussian", "unknown", summary)

    def test_swd_convergence_tracks_requested_prefixes(self) -> None:
        experiments = [
            run_ensemble_experiment(
                "gaussian",
                "RWMH",
                scale=0.7,
                target_eval_budget=800,
                n_chains=2,
                burn_fraction=0.25,
                base_seed=seed,
            )
            for seed in (11, 23)
        ]
        curve = swd_convergence(experiments, checkpoints=(200, 500, 800))
        np.testing.assert_array_equal(curve["target_evals"], (200, 500, 800))
        self.assertEqual(curve["all_swd"].shape, (2, 3))
        self.assertTrue(np.all(np.isfinite(curve["worst_swd"])))
        np.testing.assert_allclose(
            curve["worst_swd"], np.max(curve["all_swd"], axis=0)
        )
        self.assertAlmostEqual(
            curve["worst_swd"][-1],
            max(experiment.metrics["swd"] for experiment in experiments),
        )

    def test_metric_table_has_two_columns_and_official_target(self) -> None:
        experiment = run_ensemble_experiment(
            "gaussian",
            "RWMH",
            scale=0.7,
            target_eval_budget=800,
            n_chains=2,
            base_seed=11,
        )
        html = metrics_html(experiment)
        self.assertNotIn("How to read it", html)
        self.assertIn("Official worst-repeat SWD target", html)
        self.assertIn(f"≤ {SWD_PASS_THRESHOLDS['gaussian', 'RWMH']:.3f}", html)
        self.assertEqual(html.count("<th style"), 2)

    def test_budget_is_split_across_chains_and_combined(self) -> None:
        ensemble = run_ensemble_experiment(
            "gaussian",
            "MALA",
            scale=0.04,
            target_eval_budget=4_001,
            n_chains=3,
            burn_fraction=0.20,
            base_seed=11,
        )
        self.assertEqual(len(ensemble.chains), 3)
        self.assertLessEqual(ensemble.metrics["target_evals"], 4_001)
        self.assertEqual(ensemble.metrics["n_chains"], 3)
        self.assertEqual(
            ensemble.metrics["retained"],
            sum(ensemble.metrics["retained_by_chain"]),
        )
        self.assertEqual([chain.seed for chain in ensemble.chains], [11, 112, 213])

    def test_mixture_switches_do_not_count_chain_boundaries(self) -> None:
        ensemble = run_ensemble_experiment(
            "mixture",
            "RWMH",
            scale=2.5,
            target_eval_budget=3_000,
            n_chains=3,
            burn_fraction=0.10,
            base_seed=11,
        )
        within_chain_switches = sum(
            chain.metrics["task"]["mode switches"] for chain in ensemble.chains
        )
        self.assertEqual(
            ensemble.metrics["task"]["mode switches"], within_chain_switches
        )


if __name__ == "__main__":
    unittest.main()
