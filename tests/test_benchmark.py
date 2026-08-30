from __future__ import annotations

import unittest

import numpy as np

from mcmc_homework import (
    SWD_PASS_THRESHOLDS,
    benchmark_passed,
    run_experiment,
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
            run_experiment(
                "gaussian",
                "RWMH",
                scale=0.7,
                n_steps=240,
                burn_fraction=0.25,
                seed=seed,
            )
            for seed in (11, 23)
        ]
        curve = swd_convergence(experiments, checkpoints=(20, 60, 180))
        np.testing.assert_array_equal(curve["retained"], (20, 60, 180))
        self.assertEqual(curve["all_swd"].shape, (2, 3))
        self.assertTrue(np.all(np.isfinite(curve["worst_swd"])))
        np.testing.assert_allclose(
            curve["worst_swd"], np.max(curve["all_swd"], axis=0)
        )


if __name__ == "__main__":
    unittest.main()
