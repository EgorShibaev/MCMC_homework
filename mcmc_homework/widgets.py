"""Button-driven ipywidgets interface for exploring sampler hyperparameters."""

from __future__ import annotations

from html import escape
from typing import Any

import matplotlib.pyplot as plt

from .experiments import (
    BENCHMARK_TARGET_EVAL_BUDGET,
    DEFAULT_SETTINGS,
    MAX_STUDENT_CHAINS,
    METHODS,
    EnsembleExperiment,
    run_ensemble_experiment,
)
from .targets import TARGETS
from .samplers import states_for_target_evals
from .visualization import metrics_html, plot_ensemble_sampling_run


class SamplingLab:
    """Interactive target/method selector with explicit Start button."""

    def __init__(self) -> None:
        try:
            import ipywidgets as widgets
            from IPython.display import HTML, clear_output, display
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "SamplingLab requires ipywidgets; install the project requirements."
            ) from exc

        self.widgets = widgets
        self._display = display
        self._HTML = HTML
        self._clear_output = clear_output
        self.last_ensemble: EnsembleExperiment | None = None

        common_layout = widgets.Layout(width="470px")
        common_style = {"description_width": "90px"}
        self.target = widgets.Dropdown(
            options=[(target.name, key) for key, target in TARGETS.items()],
            value="gaussian",
            description="Target",
            layout=common_layout,
            style=common_style,
        )
        self.method = widgets.ToggleButtons(
            options=list(METHODS),
            value="RWMH",
            description="Method",
            style=common_style,
        )
        self.scale = widgets.FloatLogSlider(
            value=float(DEFAULT_SETTINGS[("gaussian", "RWMH")]["scale"]),
            base=10,
            min=-2.0,
            max=0.8,
            step=0.05,
            description="proposal σ",
            readout_format=".4f",
            continuous_update=False,
            layout=common_layout,
            style=common_style,
        )
        self.n_leapfrog = widgets.IntSlider(
            value=15,
            min=1,
            max=40,
            step=1,
            description="leapfrog L",
            continuous_update=False,
            layout=widgets.Layout(width="470px", display="none"),
            style=common_style,
        )
        self.burn_fraction = widgets.FloatSlider(
            value=0.25,
            min=0.05,
            max=0.40,
            step=0.05,
            description="burn-in",
            readout_format=".0%",
            continuous_update=False,
            layout=common_layout,
            style=common_style,
        )
        self.n_chains = widgets.IntSlider(
            value=2,
            min=1,
            max=MAX_STUDENT_CHAINS,
            step=1,
            description="chains C",
            continuous_update=False,
            layout=common_layout,
            style=common_style,
        )
        self.budget = widgets.HTML(
            value=(
                "<b>Fixed total budget:</b> "
                f"{BENCHMARK_TARGET_EVAL_BUDGET:,} target evaluations"
            )
        )
        self.allocation_preview = widgets.HTML()
        self.run_button = widgets.Button(
            description="Start sampling",
            button_style="primary",
            tooltip="Run only when clicked; slider changes do not launch work.",
        )
        self.status = widgets.HTML()
        self.output = widgets.Output(
            layout=widgets.Layout(border="1px solid #ddd", padding="6px")
        )

        controls = widgets.VBox(
            [
                self.target,
                self.method,
                self.scale,
                self.n_leapfrog,
                self.burn_fraction,
                self.n_chains,
                self.budget,
                self.allocation_preview,
                widgets.HBox([self.run_button, self.status]),
            ]
        )
        self.ui = widgets.VBox([controls, self.output])
        self.target.observe(self._on_configuration_change, names="value")
        self.method.observe(self._on_configuration_change, names="value")
        self.n_chains.observe(self._update_allocation_preview, names="value")
        self.n_leapfrog.observe(self._update_allocation_preview, names="value")
        self.burn_fraction.observe(self._update_allocation_preview, names="value")
        self.run_button.on_click(self._on_click)
        self._on_configuration_change(None)

    def _on_configuration_change(self, _change: Any) -> None:
        method = self.method.value
        if method == "RWMH":
            self.scale.description = "proposal σ"
            lower, upper = -2.0, 0.8
        elif method == "ULA":
            self.scale.description = "step η"
            lower, upper = -4.0, 0.0
        elif method == "MALA":
            self.scale.description = "step η"
            lower, upper = -4.0, 0.8
        else:
            self.scale.description = "HMC step ε"
            lower, upper = -3.0, 0.3
        self.scale.max = upper
        self.scale.min = lower
        setting = DEFAULT_SETTINGS[(self.target.value, method)]
        self.scale.value = float(setting["scale"])
        self.n_chains.value = int(setting["n_chains"])
        self.burn_fraction.value = float(setting["burn_fraction"])
        if method == "HMC":
            self.n_leapfrog.layout.display = "flex"
            self.n_leapfrog.value = int(setting["n_leapfrog"])
        else:
            self.n_leapfrog.layout.display = "none"
        target = TARGETS[self.target.value]
        self.status.value = (
            f"<span style='margin-left:8px;color:#555'>{escape(target.challenge)}</span>"
        )
        self._update_allocation_preview(None)

    def _update_allocation_preview(self, _change: Any) -> None:
        chain_count = int(self.n_chains.value)
        allocation, remainder = divmod(
            BENCHMARK_TARGET_EVAL_BUDGET, chain_count
        )
        budgets = [
            allocation + int(index < remainder) for index in range(chain_count)
        ]
        state_counts = [
            states_for_target_evals(
                self.method.value, budget, int(self.n_leapfrog.value)
            )
            for budget in budgets
        ]
        retained_counts = [
            count - int(count * float(self.burn_fraction.value))
            for count in state_counts
        ]
        state_range = (
            f"{state_counts[0]:,}"
            if len(set(state_counts)) == 1
            else f"{min(state_counts):,}–{max(state_counts):,}"
        )
        seeds = ", ".join(str(11 + 101 * index) for index in range(chain_count))
        self.allocation_preview.value = (
            f"<b>Derived allocation:</b> {state_range} states per chain; "
            f"{sum(retained_counts):,} retained states combined; seeds {seeds}."
        )

    def run(self) -> EnsembleExperiment:
        ensemble = run_ensemble_experiment(
            target_key=self.target.value,
            method=self.method.value,
            scale=float(self.scale.value),
            target_eval_budget=BENCHMARK_TARGET_EVAL_BUDGET,
            n_chains=int(self.n_chains.value),
            burn_fraction=float(self.burn_fraction.value),
            base_seed=11,
            n_leapfrog=int(self.n_leapfrog.value),
        )
        self.last_ensemble = ensemble
        return ensemble

    def _on_click(self, _button: Any) -> None:
        self.run_button.disabled = True
        self.run_button.description = "Running…"
        self.status.value = "<span style='margin-left:8px'>Sampling and scoring…</span>"
        try:
            ensemble = self.run()
            with self.output:
                self._clear_output(wait=True)
                figure = plot_ensemble_sampling_run(ensemble)
                self._display(figure)
                plt.close(figure)
                self._display(self._HTML(metrics_html(ensemble)))
                messages = [
                    chain.result.message
                    for chain in ensemble.chains
                    if chain.result.message
                ]
                if messages:
                    self._display(
                        self._HTML(
                            f"<p style='color:#b00020'><b>Warning:</b> "
                            f"{escape(' | '.join(messages))}</p>"
                        )
                    )
            self.status.value = "<span style='margin-left:8px;color:#176b2c'>Finished.</span>"
        except Exception as exc:
            with self.output:
                self._clear_output(wait=True)
                self._display(
                    self._HTML(
                        f"<p style='color:#b00020'><b>Run failed:</b> {escape(str(exc))}</p>"
                    )
                )
            self.status.value = "<span style='margin-left:8px;color:#b00020'>Failed.</span>"
        finally:
            self.run_button.disabled = False
            self.run_button.description = "Start sampling"


def build_sampling_lab() -> SamplingLab:
    return SamplingLab()
