from __future__ import annotations

import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
from IPython.utils.capture import capture_output

from mcmc_homework import BENCHMARK_SEEDS, DEFAULT_SETTINGS, benchmark_passed, build_sampling_lab


class WidgetTests(unittest.TestCase):
    def test_switching_pairs_preserves_tuning_without_starting_sampling(self) -> None:
        lab = build_sampling_lab()
        self.addCleanup(lab.close)
        with patch.object(lab, "run", side_effect=AssertionError("Unexpected sampling")):
            lab.scale.value, lab.n_chains.value, lab.burn_fraction.value = .27, 5, .35
            lab.target.value = "banana"  # first visit inherits the controls
            self.assertEqual((lab.scale.value, lab.n_chains.value, lab.burn_fraction.value), (.27, 5, .35))
            lab.scale.value, lab.n_chains.value = .41, 3
            lab.method.value = "HMC"
            lab.scale.value, lab.n_leapfrog.value = .07, 31
            lab.target.value = "mixture"
            self.assertEqual((lab.scale.value, lab.n_leapfrog.value), (.07, 31))
            lab.scale.value, lab.n_leapfrog.value = .9, 7
            lab.target.value = "banana"
            self.assertEqual((lab.scale.value, lab.n_leapfrog.value), (.07, 31))
            lab.method.value = "RWMH"
            self.assertEqual((lab.scale.value, lab.n_chains.value, lab.burn_fraction.value), (.41, 3, .35))
            lab.target.value = "gaussian"
            self.assertEqual((lab.scale.value, lab.n_chains.value, lab.burn_fraction.value), (.27, 5, .35))
            lab.scale.value = 4.0
            lab.method.value = "ULA"  # only out-of-range values are clipped
            self.assertEqual(lab.scale.value, 1.0)
            self.assertIn("clipped", lab.status.value)
            lab.method.value = "RWMH"
            self.assertEqual(lab.scale.value, 4.0)  # clipping did not overwrite it
        self.assertEqual(DEFAULT_SETTINGS["gaussian", "RWMH"]["scale"], .12)
        other_lab = build_sampling_lab()
        self.addCleanup(other_lab.close)
        self.assertEqual(other_lab.scale.value, .12)  # caches are per lab instance

    def test_reported_c1_setting_is_not_a_benchmark_pass_or_duplicate_output(self) -> None:
        lab = build_sampling_lab()
        self.addCleanup(lab.close)
        lab.n_chains.value = 1
        before_figures = plt.get_fignums()
        with capture_output(display=True) as published:
            lab.run_button.click()
        self.assertEqual(published.outputs, [])  # no frontend display broadcasts
        self.assertEqual(plt.get_fignums(), before_figures)
        self.assertEqual([e.base_seed for e in lab.last_ensembles], list(BENCHMARK_SEEDS))
        self.assertAlmostEqual(lab.last_ensembles[0].metrics["swd"], 0.0395868323, places=7)
        self.assertAlmostEqual(lab.last_summary["worst_swd"], 0.0725764047, places=7)
        self.assertAlmostEqual(lab._last_curve["worst_swd"][-1], lab.last_summary["worst_swd"])
        self.assertFalse(benchmark_passed("gaussian", "RWMH", lab.last_summary))
        self.assertIn("TUNE MORE", lab.result_status.value)
        self.assertIn("0.0726", lab.result_status.value)
        self.assertEqual(len(lab.output.outputs), 2)
        self.assertIn("image/png", lab.output.outputs[0]["data"])
        self.assertIn("text/html", lab.output.outputs[1]["data"])
        # Viewing another repeat must not sample, change the score, or append.
        with patch.object(lab, "run", side_effect=AssertionError("Unexpected resampling")):
            lab.view_seed.value = 23
            html = lab.output.outputs[1]["data"]["text/html"]
            self.assertIn("Seed 23 diagnostics", html)
            self.assertIn(f"{lab.last_ensemble.metrics['swd']:.4f}", html)
            for repeated in (
                "All repeats", "mean ± SD", "worst SWD", "Official", "Student-selected",
                "states", "budget", "burn-in", "noise floor",
            ):
                self.assertNotIn(repeated, html)
            self.assertEqual(html.count("<tr>"), 1 + 4 + len(lab.last_ensemble.metrics["task"]))
            self.assertIn("max-width:620px", html)
            self.assertIn("text-align:left", html)
            self.assertIn("mean ± SD", lab.result_status.value)
            lab.view_seed.value = 47
        self.assertEqual(lab.last_ensemble.base_seed, 47)
        self.assertEqual(len(lab.output.outputs), 2)
        self.assertIn("TUNE MORE", lab.result_status.value)
        # Repeated Start replaces output; use the cached numerical result here.
        with patch.object(lab, "run", return_value=lab.last_ensemble) as run:
            lab.run_button.click()
            run.assert_called_once()
        self.assertEqual(len(lab.output.outputs), 2)
        lab.scale.value = 0.2
        self.assertIn("Settings changed", lab.status.value)

    def test_close_detaches_callback_and_failed_run_clears_result(self) -> None:
        lab = build_sampling_lab()
        with patch.object(lab, "run", side_effect=ValueError("test failure")):
            lab.run_button.click()
        self.assertIn("No result", lab.result_status.value)
        self.assertEqual(len(lab.output.outputs), 1)
        self.assertFalse(lab.run_button.disabled)
        self.assertEqual(lab.last_ensembles, [])
        lab.close()
        self.assertEqual(len(lab.run_button._click_handlers.callbacks), 0)
        with patch.object(lab, "run") as run:
            lab.run_button.click()
            run.assert_not_called()

    def test_hmc_controls_and_run(self) -> None:
        lab = build_sampling_lab()
        self.addCleanup(lab.close)
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
        self.assertEqual(len(lab.last_ensembles), 3)


if __name__ == "__main__":
    unittest.main()
