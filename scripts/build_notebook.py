"""Generate the student notebook that imports the project's separate modules."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "MCMC_Homework.ipynb"


def markdown(source: str, *tags: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(
        dedent(source).strip(), metadata={"tags": list(tags)} if tags else {}
    )


def code(source: str, *tags: str, hidden: bool = False) -> nbf.NotebookNode:
    metadata: dict = {"tags": list(tags)} if tags else {}
    if hidden:
        metadata.update({"collapsed": True, "jupyter": {"source_hidden": True}})
    return nbf.v4.new_code_cell(dedent(source).strip(), metadata=metadata)


def build_notebook() -> nbf.NotebookNode:
    cells: list[nbf.NotebookNode] = []
    cells.append(
        markdown(
            r"""
            # Homework: Langevin, Metropolis, MALA, and Hamiltonian Monte Carlo

            **Student:** _replace with your name_  
            **Date:** _replace with the submission date_

            In this project you will tune four sampling algorithms on targets with anisotropy, curvature, and multiple modes. You will use distributional error and mixing diagnostics—not visual plausibility or acceptance alone—to decide whether sampling is successful.

            Restart the kernel and run all cells once before using the interactive controls.
            """,
            "assignment",
        )
    )
    cells.append(
        markdown(
            r"""
            ## Project structure

            The implementations intentionally live in normal Python modules:

            | File | Responsibility |
            |---|---|
            | `mcmc_homework/targets.py` | target distributions and exact reference samplers |
            | `mcmc_homework/samplers.py` | RWMH, ULA, MALA, and HMC implementations |
            | `mcmc_homework/metrics.py` | sliced-Wasserstein error, ESS, and diagnostics |
            | `mcmc_homework/visualization.py` | trajectory, trace, ACF, and mode-ratio plots |
            | `mcmc_homework/experiments.py` | reproducible runs and repeated-seed benchmarks |
            | `mcmc_homework/widgets.py` | sliders and the Start button |
            | `tests/` | numerical correctness tests |

            Read the implementation files alongside this notebook. The notebook is an experiment and reporting surface, not the source of every function.
            """,
            "assignment",
        )
    )
    cells.append(
        markdown(
            r"""
            ## 1. Four samplers

            Let $\ell(x)=\log\pi(x)$ and $Z\sim\mathcal N(0,I)$.

            | Method | Transition | Hyperparameters | Correction? |
            |---|---|---|---|
            | **RWMH** | $Y=X_k+\sigma Z$ | proposal scale $\sigma$ | symmetric MH ratio |
            | **ULA** | $X_{k+1}=X_k+\eta\nabla\ell(X_k)+\sqrt{2\eta}Z$ | step $\eta$ | none; finite-step bias |
            | **MALA** | propose the ULA transition | step $\eta$ | asymmetric MH ratio |
            | **HMC** | simulate position and momentum by leapfrog | integrator step $\epsilon$, number of steps $L$ | Hamiltonian MH ratio |

            HMC augments the state with momentum $p\sim\mathcal N(0,I)$ and uses

            $$H(x,p)=-\ell(x)+\tfrac12p^\top p.$$

            A leapfrog trajectory approximately conserves $H$. The proposal is accepted with

            $$\alpha=\min\{1,\exp(H(x,p)-H(x',p'))\}.$$

            Small steps can waste computation; large steps can cause ULA bias/divergence or MH/HMC rejection. Large $L$ lets HMC travel farther but costs more gradients and can create resonances. No hyperparameter or acceptance target is universally best.
            """,
            "theory",
        )
    )
    cells.append(
        markdown(
            r"""
            ## 2. Targets and metrics

            The core tuning tasks use:

            1. a tilted ill-conditioned Gaussian;
            2. a thin banana-shaped target;
            3. a 65:35 bimodal mixture.

            A fourth **10:90 mixture** is a guided case study about rare-mode discovery.

            The primary error is standardized sliced Wasserstein-1 distance (SWD) to an exact iid reference sample. **Lower is better.** Report mean, standard deviation, and worst-seed SWD over fixed seeds. Secondary diagnostics are minimum ESS, ESS per 1,000 target evaluations, acceptance, and target-specific quantities such as mode mass and switches.

            **Formal definitions.** For a retained scalar chain $Z_1,\ldots,Z_N$ with lag-$k$ autocorrelation $\rho_k$, its effective sample size is approximately

            $$\operatorname{ESS}=\frac{N}{1+2\sum_{k\geq 1}\rho_k}.$$

            In practice the sum is truncated using a stable positive-sequence rule. For a multivariate chain, this notebook computes ESS for several diagnostic features and reports the minimum. The cost-normalized value is $1000\,\operatorname{ESS}/(\text{target evaluations})$.

            For fixed unit directions $\theta_1,\ldots,\theta_M$, the standardized sliced Wasserstein score used here is

            $$\operatorname{SWD}(P,Q)=\frac{1}{M}\sum_{m=1}^{M}
            \frac{W_1\!\left((\theta_m^\top)_\#P,(\theta_m^\top)_\#Q\right)}
            {\operatorname{SD}_{Y\sim Q}(\theta_m^\top Y)},$$

            where $P$ is the retained-chain distribution, $Q$ is the exact target reference, $(\theta^\top)_\#P$ is the one-dimensional projected distribution, and $W_1(F,G)=\int_0^1|F^{-1}(u)-G^{-1}(u)|\,du$. Thus **ESS measures efficiency (higher is better), while SWD measures distributional error (lower is better)**. High ESS does not guarantee low SWD when a chain is trapped in one mode.

            ### Explicit pass criterion

            A target–method row passes only when **the worst SWD over seeds `(11, 23, 47)` is at or below the target threshold** and no run diverges:

            | target | required worst-seed SWD |
            |---|---:|
            | tilted Gaussian | $\leq 0.50$ |
            | banana | $\leq 0.50$ |
            | 65:35 mixture | $\leq 0.75$ |

            To pass the homework, all **12** core target–method rows must say **PASS** at the fixed budget of 4,000 recorded chain states and 25% burn-in. The thresholds differ by target difficulty, but within a target every method faces the same requirement. The convergence figure later in the notebook plots this exact worst-seed metric against retained iterations.

            Burn-in removes an initial transient. It cannot fix discretization bias, mode trapping, or a poorly tuned integrator.
            """,
            "theory",
        )
    )
    cells.append(code("# %pip install -r requirements.txt\n%matplotlib inline", "setup"))
    cells.append(
        code(
            """
            import numpy as np
            import matplotlib.pyplot as plt
            from IPython.display import HTML, display

            from mcmc_homework import (
                BENCHMARK_SEEDS,
                CORE_TARGET_KEYS,
                DEFAULT_SETTINGS,
                METHODS,
                SWD_PASS_THRESHOLDS,
                TARGETS,
                benchmark_all_settings,
                benchmark_results_html,
                build_sampling_lab,
                gradient_check_report,
                metrics_html,
                plot_mode_ratio_comparison,
                plot_sampling_run,
                plot_swd_convergence,
                run_experiment,
            )

            plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.18})
            """,
            "setup",
        )
    )
    cells.append(markdown("### Numerical implementation check", "validation"))
    cells.append(
        code(
            """
            gradient_checks = gradient_check_report()
            assert all(row["passed"] for row in gradient_checks), gradient_checks
            display(HTML(
                "<table><tr><th>target</th><th>maximum gradient error</th><th>status</th></tr>"
                + "".join(
                    f"<tr><td>{row['target']}</td><td>{row['max_abs_error']:.2e}</td><td>pass</td></tr>"
                    for row in gradient_checks
                )
                + "</table>"
            ))
            """,
            "validation",
        )
    )
    cells.append(
        markdown(
            r"""
            ## 3. Read an HMC diagnostic plot

            Orange is the burn-in path; blue points are retained states. The trace panel shows $x_1$ and $x_2$ over transitions. Running statistics expose slow convergence, and the ACF panel shows correlation among retained states.

            This is a demonstration setting, not an answer key for the final fixed-budget benchmark.
            """,
            "demo",
        )
    )
    cells.append(
        code(
            """
            hmc_demo = run_experiment(
                "banana", "HMC", scale=0.05,
                n_steps=2500, burn_fraction=0.25, seed=11, n_leapfrog=20,
            )
            plot_sampling_run(hmc_demo)
            plt.show()
            display(HTML(metrics_html(hmc_demo)))
            """,
            "demo",
        )
    )
    cells.append(
        markdown(
            r"""
            ## 4. Interactive tuning lab

            Select a target and method, adjust the logarithmic scale slider, and press **Start sampling**. HMC adds a second slider for leapfrog count $L$. Use one repeat while exploring and at least three repeats when comparing serious candidates.

            Inspect paths and traces before looking at SWD. A chain that never discovers a mode can have attractive within-mode scatter and high acceptance.
            """,
            "interactive",
        )
    )
    cells.append(code("lab = build_sampling_lab()\ndisplay(lab.ui)", "interactive"))

    cells.append(
        markdown(
            r"""
            ## 5. Guided case study: a rare 10% mode

            Both Gaussian components have unit covariance, but the left component has probability 0.10 and the right component 0.90. Both chains start in the dominant right mode and receive the same number of transitions.

            The ULA setting below is deliberately local: in this finite run it never represents the rare mode. HMC refreshes momentum, integrates longer trajectories, crosses the density barrier repeatedly, and approaches the target ratio. This illustrates a specific geometry/budget/tuning combination—not a theorem that all Langevin samplers always fail on mixtures.
            """,
            "case-study",
        )
    )
    cells.append(
        code(
            """
            rare_mode_ula = run_experiment(
                "imbalanced_mixture", "ULA", scale=0.002,
                n_steps=6000, burn_fraction=0.20, seed=47,
            )
            rare_mode_hmc = run_experiment(
                "imbalanced_mixture", "HMC", scale=0.12,
                n_steps=6000, burn_fraction=0.20, seed=47, n_leapfrog=20,
            )
            plot_mode_ratio_comparison(rare_mode_ula, rare_mode_hmc)
            plt.show()
            """,
            "case-study",
        )
    )
    cells.append(
        markdown(
            r"""
            **Case-study questions**

            1. Report the final left:right proportions for ULA and HMC.
            2. Why can a sampler produce a visually plausible cloud in the right mode while still having a large distributional error?
            3. Compare the number of mode switches. Why is one transition weaker evidence than many transitions?
            4. Increase ULA's step. Can it cross? If so, what happens to SWD and finite-step bias?
            5. Reduce HMC's $L$ while keeping $\epsilon$ fixed. When does rare-mode discovery degrade?

            **Your answer:** _replace this text._
            """,
            "student-answer",
        )
    )

    cells.append(
        markdown(
            r"""
            ## 6. Record and benchmark your settings

            Replace every value below. RWMH uses `scale = sigma`; ULA/MALA use `scale = eta`; HMC uses `scale = epsilon` plus `n_leapfrog = L`. The supplied values are starting guesses, not optimized answers.

            Final comparisons use 4,000 states, 25% burn-in, and seeds `(11, 23, 47)`. You may change hyperparameters, but do not change this grading budget, burn-in, seeds, or thresholds. The table gives each row a direct **PASS / TUNE MORE** status. The plot then shows whether the worst-seed SWD approaches and crosses the dashed requirement as retained iterations accumulate. Curves need not decrease monotonically.
            """,
            "assignment",
        )
    )
    cells.append(
        code(
            """
            # TODO: replace all 12 core settings after exploration.
            student_settings = {
                key: value.copy()
                for key, value in DEFAULT_SETTINGS.items()
                if key[0] in CORE_TARGET_KEYS
            }
            student_settings
            """,
            "student-edit",
        )
    )
    cells.append(
        code(
            """
            RUN_FULL_BENCHMARK = True  # set False only while doing quick exploration

            if RUN_FULL_BENCHMARK:
                benchmark_results = benchmark_all_settings(student_settings)
                display(HTML(benchmark_results_html(benchmark_results)))
                plot_swd_convergence(benchmark_results)
                plt.show()
            else:
                print("Set RUN_FULL_BENCHMARK = True to see the official PASS / TUNE MORE result.")
            """,
            "student-edit",
        )
    )

    cells.append(
        markdown(
            r"""
            ## Task A — tilted Gaussian

            1. Tune all four methods. For HMC, explore both $\epsilon$ and $L$.
            2. Derive the Gaussian ULA stability condition $\eta<2\lambda_{\min}(\Sigma)$ and compare it with empirical instability.
            3. Explain the isotropic RWMH compromise between the long and narrow axes.
            4. Compare MALA and HMC gradient costs as well as raw ESS.

            **Your answer:** _replace this text._
            """,
            "student-answer",
        )
    )
    cells.append(
        markdown(
            r"""
            ## Task B — banana geometry

            1. Tune all four methods and report the fixed-budget rows.
            2. Find a high-acceptance setting with poor exploration and explain its trace/ACF.
            3. Find a ULA step with visible bias. Which task-specific diagnostics expose it?
            4. Explain how longer HMC trajectories follow the curved typical set and when they cease to help.

            **Your answer:** _replace this text._
            """,
            "student-answer",
        )
    )
    cells.append(
        markdown(
            r"""
            ## Task C — 65:35 mixture

            1. Tune all four methods; report mode mass and switches as well as SWD.
            2. Construct a run with convincing within-mode scatter that misses a component.
            3. Explain why coordinate ESS can be misleading and why the mode indicator is included.
            4. Compare two initial components. Which methods remain sensitive to initialization at this budget?

            **Your answer:** _replace this text._
            """,
            "student-answer",
        )
    )
    cells.append(
        markdown(
            r"""
            ## Task D — bias, cost, and synthesis

            1. Increase the chain length at fixed scale. Contrast exact MH-corrected methods with finite-step ULA.
            2. Halve ULA's step and increase transitions. Explain the bias–mixing–cost trade-off.
            3. Rank the four methods for each core target using mean/worst SWD first and efficiency diagnostics second.
            4. Use the SWD convergence figure: identify when each curve first crosses its threshold and whether it remains below it. Explain any non-monotonicity.
            5. Explain why a low SWD from one lucky mixture trajectory is insufficient evidence.

            **Your answer:** _replace this text._
            """,
            "student-answer",
        )
    )
    cells.append(
        markdown(
            r"""
            ## Submission checklist

            - [ ] I inspected the separate implementation modules.
            - [ ] I replaced all 12 core settings and ran the unchanged fixed benchmark.
            - [ ] Every row in the benchmark table says **PASS** (12/12).
            - [ ] I reported mean $\pm$ SD and worst-seed SWD.
            - [ ] I interpreted the SWD-over-retained-iterations figure.
            - [ ] I interpreted acceptance, ESS/target-evaluation cost, and task diagnostics.
            - [ ] I completed the 10:90 rare-mode case-study questions.
            - [ ] I answered Tasks A–D and restarted the kernel before submission.
            """,
            "assignment",
        )
    )

    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.update(
        {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.12",
                "mimetype": "text/x-python",
                "codemirror_mode": {"name": "ipython", "version": 3},
                "pygments_lexer": "ipython3",
                "nbconvert_exporter": "python",
                "file_extension": ".py",
            },
        }
    )
    return notebook


def main() -> None:
    notebook = build_notebook()
    nbf.validate(notebook)
    nbf.write(notebook, OUTPUT)
    print(f"Wrote {OUTPUT} with {len(notebook.cells)} cells.")


if __name__ == "__main__":
    main()
