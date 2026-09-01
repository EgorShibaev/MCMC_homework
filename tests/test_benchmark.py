from __future__ import annotations

import unittest

import numpy as np

from mcmc_homework import (
    SWD_PASS_THRESHOLDS,
    benchmark_passed,
    metrics_html,
    run_ensemble_experiment,
    swd_convergence,
)


class BenchmarkTests(unittest.TestCase):
    def test_pass_rule_uses_worst_seed_and_divergence(self) -> None:
        threshold = SWD_PASS_THRESHOLDS["gaussian"]
        passing = {"worst_swd": threshold, "divergent_runs": 0.0}
        too_inaccurate = {"worst_swd": threshold + 1e-6, "divergent_runs": 0.0}
        divergent = {"worst_swd": threshold - 0.1, "divergent_runs": 1.0}
        self.assertTrue(benchmark_passed("gaussian", passing))
        self.assertFalse(benchmark_passed("gaussian", too_inaccurate))
        self.assertFalse(benchmark_passed("gaussian", divergent))

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
        self.assertIn("≤ 0.080", html)
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
