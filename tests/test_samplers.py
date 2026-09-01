from __future__ import annotations

import unittest

import numpy as np

from mcmc_homework import (
    DEFAULT_SETTINGS,
    TARGETS,
    Target2D,
    run_experiment,
    run_sampler,
    states_for_target_evals,
    target_evals_for_states,
)


class FlatTarget(Target2D):
    key = "flat"
    name = "flat"
    challenge = "test"
    bounds = ((-1.0, 1.0), (-1.0, 1.0))
    start = np.zeros(2)

    def log_prob(self, x):
        points = np.asarray(x, dtype=float)
        return 0.0 if points.ndim == 1 else np.zeros(points.shape[0])

    def grad_log_prob(self, x):
        return np.zeros_like(np.asarray(x, dtype=float))

    def sample_reference(self, n, rng):
        raise NotImplementedError


class SamplerTests(unittest.TestCase):
    def test_shapes_finiteness_and_reproducibility(self) -> None:
        for method in ("RWMH", "ULA", "MALA", "HMC"):
            setting = DEFAULT_SETTINGS[("banana", method)]
            first = run_sampler(
                method,
                TARGETS["banana"],
                300,
                float(setting["scale"]),
                seed=17,
                n_leapfrog=int(setting.get("n_leapfrog", 10)),
            )
            second = run_sampler(
                method,
                TARGETS["banana"],
                300,
                float(setting["scale"]),
                seed=17,
                n_leapfrog=int(setting.get("n_leapfrog", 10)),
            )
            self.assertEqual(first.samples.shape, (300, 2))
            self.assertTrue(np.all(np.isfinite(first.samples)))
            np.testing.assert_array_equal(first.samples, second.samples)

    def test_rwmh_rejections_duplicate_the_current_state(self) -> None:
        result = run_sampler(
            "RWMH", TARGETS["gaussian"], n_steps=300, scale=20.0, seed=8
        )
        rejected = np.flatnonzero(~result.accepted)
        self.assertGreater(len(rejected), 0)
        for transition in rejected:
            np.testing.assert_array_equal(
                result.samples[transition + 1], result.samples[transition]
            )

    def test_ula_noise_variance_is_two_eta(self) -> None:
        eta = 0.04
        result = run_sampler("ULA", FlatTarget(), 20_000, eta, seed=123)
        increments = np.diff(result.samples, axis=0)
        observed = np.var(increments, axis=0, ddof=1)
        np.testing.assert_allclose(observed, 2.0 * eta, rtol=0.04)

    def test_large_gaussian_ula_step_reports_divergence(self) -> None:
        result = run_sampler("ULA", TARGETS["gaussian"], 200, 1.0, seed=4)
        self.assertTrue(result.diverged)
        self.assertIn("reduce", result.message.lower())

    def test_hmc_leapfrog_accounting_and_acceptance(self) -> None:
        n_steps = 500
        n_leapfrog = 12
        result = run_sampler(
            "HMC",
            TARGETS["gaussian"],
            n_steps=n_steps,
            scale=0.08,
            seed=19,
            n_leapfrog=n_leapfrog,
        )
        self.assertEqual(result.grad_evals, 1 + (n_steps - 1) * n_leapfrog)
        self.assertGreater(result.acceptance_rate, 0.5)
        self.assertLessEqual(result.acceptance_rate, 1.0)

    def test_exact_samplers_recover_gaussian_moments(self) -> None:
        configurations = (
            ("RWMH", 0.8, 10),
            ("MALA", 0.015, 10),
            ("HMC", 0.08, 15),
        )
        for method, scale, n_leapfrog in configurations:
            experiment = run_experiment(
                "gaussian",
                method,
                scale,
                n_steps=8000,
                burn_fraction=0.25,
                seed=29,
                n_leapfrog=n_leapfrog,
            )
            diagnostics = experiment.metrics["task"]
            self.assertLess(diagnostics["standardized mean error"], 0.35)
            self.assertLess(diagnostics["relative covariance error"], 0.35)

    def test_hmc_requires_positive_leapfrog_count(self) -> None:
        with self.assertRaises(ValueError):
            run_sampler(
                "HMC", TARGETS["gaussian"], 20, 0.08, seed=1, n_leapfrog=0
            )

    def test_budget_to_states_never_exceeds_target_evaluations(self) -> None:
        budget = 40_000
        for method, n_leapfrog in (
            ("RWMH", 10),
            ("ULA", 10),
            ("MALA", 10),
            ("HMC", 20),
        ):
            n_states = states_for_target_evals(method, budget, n_leapfrog)
            used = target_evals_for_states(method, n_states, n_leapfrog)
            self.assertLessEqual(used, budget)
            self.assertGreater(
                target_evals_for_states(method, n_states + 1, n_leapfrog), budget
            )

    def test_imbalanced_case_study_shows_expected_contrast(self) -> None:
        ula = run_experiment(
            "imbalanced_mixture",
            "ULA",
            0.002,
            n_steps=6000,
            burn_fraction=0.20,
            seed=47,
        )
        hmc = run_experiment(
            "imbalanced_mixture",
            "HMC",
            0.12,
            n_steps=6000,
            burn_fraction=0.20,
            seed=47,
            n_leapfrog=20,
        )
        self.assertGreater(ula.metrics["task"]["left-mode mass error"], 0.08)
        self.assertLess(hmc.metrics["task"]["left-mode mass error"], 0.03)
        self.assertGreater(hmc.metrics["task"]["mode switches"], 40)


if __name__ == "__main__":
    unittest.main()
