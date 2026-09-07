from __future__ import annotations

import re
import unittest

from mcmc_homework import BENCHMARK_SEEDS
from scripts.build_notebook import build_notebook


class NotebookTests(unittest.TestCase):
    def test_section_five_is_five_questions_without_worked_code(self) -> None:
        notebook = build_notebook()
        start = next(i for i, cell in enumerate(notebook.cells) if cell.source.startswith("## 5."))
        end = next(i for i, cell in enumerate(notebook.cells) if cell.source.startswith("## 6."))
        section = notebook.cells[start:end]
        self.assertEqual(len(section), 1)
        self.assertEqual(section[0].cell_type, "markdown")
        self.assertEqual(re.findall(r"^([1-5])\. ", section[0].source, re.MULTILINE), list("12345"))
        self.assertLess(len(section[0].source.split()), 120)
        self.assertNotIn("rare_mode_ula =", "\n".join(c.source for c in notebook.cells))

    def test_investigations_keep_questions_and_screenshots_concise(self) -> None:
        investigations = [c.source for c in build_notebook().cells if c.source.startswith("### Investigation")]
        self.assertEqual(len(investigations), 6)
        for source in investigations:
            self.assertLess(len(source.split()), 120)
            self.assertEqual(len(re.findall(r"^[1-3]\. ", source, re.MULTILINE)), 3)
            self.assertIn("**Screenshots:**", source)
        self.assertIn(r"\eta<2\lambda_{\min}(\Sigma)", investigations[1])
        self.assertIn("$C=1,2,4,8$", investigations[4])

    def test_fresh_seed_calibration_does_not_expand_student_grading(self) -> None:
        source = "\n".join(c.source for c in build_notebook().cells)
        self.assertEqual(len(BENCHMARK_SEEDS), 20)
        self.assertEqual(BENCHMARK_SEEDS[:3], (11, 23, 47))
        self.assertIn("100 fresh seeds", source)
        self.assertIn("50 seeds were held out", source)
        self.assertIn("1,597/1,600", source)
        self.assertIn("grading now uses 20 fixed seeds", source)
        self.assertIn("seeds=BENCHMARK_SEEDS", source)
        self.assertNotIn("seeds=(11, 23, 47)", source)


if __name__ == "__main__":
    unittest.main()
