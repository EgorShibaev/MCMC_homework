from __future__ import annotations

import unittest

import numpy as np

from mcmc_homework import (
    autocorrelation_1d,
    effective_sample_size_1d,
    standardized_sliced_wasserstein,
)


class MetricTests(unittest.TestCase):
    def test_wasserstein_identity_and_shift(self) -> None:
        rng = np.random.default_rng(1)
        sample = rng.normal(size=(500, 2))
        self.assertAlmostEqual(
            standardized_sliced_wasserstein(sample, sample), 0.0, places=12
        )
        self.assertGreater(
            standardized_sliced_wasserstein(sample + 1.0, sample), 0.5
        )

    def test_ess_iid_ar_and_constant(self) -> None:
        rng = np.random.default_rng(9)
        iid = rng.normal(size=4000)
        autoregressive = np.empty(4000)
        autoregressive[0] = rng.normal()
        for index in range(1, len(autoregressive)):
            autoregressive[index] = (
                0.95 * autoregressive[index - 1]
                + rng.normal(scale=np.sqrt(1.0 - 0.95**2))
            )
        iid_ess = effective_sample_size_1d(iid)
        ar_ess = effective_sample_size_1d(autoregressive)
        self.assertGreater(iid_ess, ar_ess * 5.0)
        self.assertGreater(iid_ess, 1000.0)
        self.assertEqual(effective_sample_size_1d(np.ones(200)), 0.0)

    def test_acf_shape(self) -> None:
        autocorrelation = autocorrelation_1d(np.arange(20.0), max_lag=7)
        self.assertEqual(autocorrelation.shape, (8,))
        self.assertAlmostEqual(autocorrelation[0], 1.0)


if __name__ == "__main__":
    unittest.main()
