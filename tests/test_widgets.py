from __future__ import annotations

import unittest

from mcmc_homework import build_sampling_lab


class WidgetTests(unittest.TestCase):
    def test_hmc_controls_and_run(self) -> None:
        lab = build_sampling_lab()
        self.assertEqual(len(lab.run_button._click_handlers.callbacks), 1)
        lab.target.value = "imbalanced_mixture"
        lab.method.value = "HMC"
        self.assertEqual(lab.n_leapfrog.layout.display, "flex")
        lab.n_chains.value = 3
        lab.burn_fraction.value = 0.20
        self.assertIn("states per chain", lab.allocation_preview.value)
        self.assertIn("11, 112, 213", lab.allocation_preview.value)
        ensemble = lab.run()
        self.assertEqual(ensemble.method, "HMC")
        self.assertEqual(ensemble.n_chains, 3)
        self.assertEqual(len(ensemble.chains), 3)
        self.assertEqual(ensemble.target_eval_budget, 40_000)
        self.assertIn("swd", ensemble.metrics)


if __name__ == "__main__":
    unittest.main()
