from __future__ import annotations

import unittest

import numpy as np

from mcmc_homework import TARGETS, gradient_check_report


class TargetTests(unittest.TestCase):
    def test_analytic_gradients(self) -> None:
        report = gradient_check_report()
        self.assertTrue(all(row["passed"] for row in report), report)

    def test_reference_samples_are_finite(self) -> None:
        for index, target in enumerate(TARGETS.values()):
            samples = target.sample_reference(1000, np.random.default_rng(index))
            self.assertEqual(samples.shape, (1000, 2))
            self.assertTrue(np.all(np.isfinite(samples)))

    def test_imbalanced_reference_has_ten_percent_left_mass(self) -> None:
        target = TARGETS["imbalanced_mixture"]
        sample = target.sample_reference(100_000, np.random.default_rng(3))
        self.assertAlmostEqual(float(np.mean(sample[:, 0] < 0)), 0.10, delta=0.01)

    def test_mixture_extreme_points_are_numerically_stable(self) -> None:
        for key in ("mixture", "imbalanced_mixture"):
            target = TARGETS[key]
            points = np.array([[1e3, -1e3], [-1e3, 1e3]])
            self.assertTrue(np.all(np.isfinite(target.log_prob(points))))
            self.assertTrue(np.all(np.isfinite(target.grad_log_prob(points))))


if __name__ == "__main__":
    unittest.main()
