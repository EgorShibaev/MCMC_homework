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
        lab.n_steps.value = 500
        experiments, summary = lab.run()
        self.assertEqual(experiments[0].result.method, "HMC")
        self.assertIn("worst_swd", summary)


if __name__ == "__main__":
    unittest.main()
