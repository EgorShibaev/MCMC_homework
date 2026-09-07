"""Button-driven ipywidgets interface for exploring sampler hyperparameters."""

from __future__ import annotations

from base64 import b64encode
from html import escape
from io import BytesIO
from typing import Any

import matplotlib.pyplot as plt

from .experiments import (
    BENCHMARK_TARGET_EVAL_BUDGET,
    BENCHMARK_SEEDS,
    DEFAULT_SETTINGS,
    MAX_STUDENT_CHAINS,
    METHODS,
    EnsembleExperiment,
    SWD_PASS_THRESHOLDS,
    benchmark_passed,
    benchmark_setting,
    swd_convergence,
)
from .targets import TARGETS
from .samplers import states_for_target_evals
from .visualization import seed_diagnostics_html, plot_ensemble_sampling_run


class SamplingLab:
    """Interactive target/method selector with explicit Start button."""

    def __init__(self) -> None:
        try:
            import ipywidgets as widgets
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "SamplingLab requires ipywidgets; install the project requirements."
            ) from exc

        self.widgets = widgets
        self.last_ensemble: EnsembleExperiment | None = None
        self.last_ensembles: list[EnsembleExperiment] = []
        self.last_summary: dict[str, float] | None = None
        self._last_curve = None
        self._closed = False
        self._active_pair: tuple[str, str] | None = None
        self._settings_by_pair: dict[tuple[str, str], dict[str, float | int]] = {}

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
                f"{BENCHMARK_TARGET_EVAL_BUDGET:,} target evaluations per repeat; "
                f"{len(BENCHMARK_SEEDS)} fixed repeats "
                f"({len(BENCHMARK_SEEDS) * BENCHMARK_TARGET_EVAL_BUDGET:,} "
                "calls maximum per click)."
            )
        )
        self.allocation_preview = widgets.HTML()
        self.run_button = widgets.Button(
            description="Start sampling",
            button_style="primary",
            tooltip="Run only when clicked; slider changes do not launch work.",
        )
        self.status = widgets.HTML()
        self.result_status = widgets.HTML()
        self.view_seed = widgets.Dropdown(
            options=[(f"Base seed {seed}", seed) for seed in BENCHMARK_SEEDS],
            value=BENCHMARK_SEEDS[0], description="View traces",
            style=common_style, layout=common_layout,
        )
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
        self.ui = widgets.VBox([controls, self.result_status, self.view_seed, self.output])
        self.target.observe(self._on_configuration_change, names="value")
        self.method.observe(self._on_configuration_change, names="value")
        self.n_chains.observe(self._update_allocation_preview, names="value")
        self.n_leapfrog.observe(self._update_allocation_preview, names="value")
        self.burn_fraction.observe(self._update_allocation_preview, names="value")
        self.scale.observe(self._mark_settings_changed, names="value")
        self.view_seed.observe(self._on_view_seed, names="value")
        self.run_button.on_click(self._on_click)
        self._on_configuration_change(None)

    def _on_configuration_change(self, _change: Any) -> None:
        current = {
            "scale": float(self.scale.value),
            "n_chains": int(self.n_chains.value),
            "burn_fraction": float(self.burn_fraction.value),
            "n_leapfrog": int(self.n_leapfrog.value),
        }
        if self._active_pair is not None:
            self._settings_by_pair[self._active_pair] = current.copy()
        pair = (self.target.value, self.method.value)
        # Returning to a pair restores its tuning. A new pair inherits the
        # current controls instead of resetting them to starting guesses.
        setting = self._settings_by_pair.get(pair, current)
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
        requested_scale = float(setting["scale"])
        self.scale.value = min(max(requested_scale, 10**lower), 10**upper)
        self.n_chains.value = int(setting["n_chains"])
        self.burn_fraction.value = float(setting["burn_fraction"])
        self.n_leapfrog.value = int(setting["n_leapfrog"])
        if method == "HMC":
            self.n_leapfrog.layout.display = "flex"
        else:
            self.n_leapfrog.layout.display = "none"
        target = TARGETS[self.target.value]
        self.status.value = (
            f"<span style='margin-left:8px;color:#555'>{escape(target.challenge)}</span>"
        )
        self._update_allocation_preview(None)
        self._active_pair = pair
        if self.scale.value != requested_scale:
            self.status.value += " Scale clipped to this method's slider range."

    def _update_allocation_preview(self, _change: Any) -> None:
        self._mark_settings_changed(None)
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
        seeds = ", ".join(str(BENCHMARK_SEEDS[0] + 101 * index) for index in range(chain_count))
        self.allocation_preview.value = (
            f"<b>Derived allocation:</b> {state_range} states per chain; "
            f"{sum(retained_counts):,} retained states combined per repeat; "
            f"first-repeat chain seeds {seeds}."
        )

    def _mark_settings_changed(self, _change: Any) -> None:
        if self.last_ensembles:
            self.status.value = "Settings changed — press Start to evaluate."

    def run(self) -> EnsembleExperiment:
        ensembles, summary = benchmark_setting(
            target_key=self.target.value,
            method=self.method.value,
            scale=float(self.scale.value),
            target_eval_budget=BENCHMARK_TARGET_EVAL_BUDGET,
            n_chains=int(self.n_chains.value),
            burn_fraction=float(self.burn_fraction.value),
            seeds=BENCHMARK_SEEDS,
            n_leapfrog=int(self.n_leapfrog.value),
        )
        self.last_ensembles = ensembles
        self.last_summary = summary
        self._last_curve = swd_convergence(ensembles)
        self.last_ensemble = next(e for e in ensembles if e.base_seed == self.view_seed.value)
        return self.last_ensemble

    def _on_view_seed(self, _change: Any) -> None:
        if self.last_ensembles and not self.run_button.disabled:
            self.last_ensemble = next(
                e for e in self.last_ensembles if e.base_seed == self.view_seed.value
            )
            self._render_result()

    def _render_result(self) -> None:
        ensemble = self.last_ensemble
        summary = self.last_summary
        assert ensemble is not None and summary is not None
        threshold = SWD_PASS_THRESHOLDS.get((ensemble.target_key, ensemble.method))
        status = "GUIDED CASE" if threshold is None else (
            "PASS" if benchmark_passed(ensemble.target_key, ensemble.method, summary) else "TUNE MORE"
        )
        limit = "" if threshold is None else f"; required ≤ {threshold:.3f}"
        self.result_status.value = (
            f"<b>Last evaluated setting — {status}</b>: "
            f"worst-repeat SWD = {summary['worst_swd']:.4f}{limit}.<br>"
            f"3 seeds · mean ± SD: {summary['mean_swd']:.4f} ± {summary['sd_swd']:.4f}"
        )
        if summary["divergent_runs"]:
            self.result_status.value += f" · divergent repeats: {int(summary['divergent_runs'])}"
        # Set serialized output data directly. Output context managers capture
        # kernel display messages, which can be duplicated by multiple frontend
        # views/hooks. No display() or inline-backend auto-display is needed here.
        with plt.ioff():
            figure = plot_ensemble_sampling_run(ensemble, benchmark_curve=self._last_curve)
            try:
                buffer = BytesIO()
                figure.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
                png = b64encode(buffer.getvalue()).decode("ascii")
            finally:
                plt.close(figure)
        messages = [
            f"seed {e.base_seed}: {chain.result.message}"
            for e in self.last_ensembles for chain in e.chains if chain.result.message
        ]
        table = seed_diagnostics_html(ensemble)
        if messages:
            table += f"<p style='color:#b00020'><b>Warning:</b> {escape(' | '.join(messages))}</p>"
        self.output.outputs = (
            {"output_type": "display_data", "data": {"image/png": png}, "metadata": {}},
            {"output_type": "display_data", "data": {"text/html": table}, "metadata": {}},
        )

    def close(self) -> None:
        """Dispose callbacks and views before replacing a notebook lab instance."""
        self._closed = True
        self.run_button.on_click(self._on_click, remove=True)

        def close_widget(widget):
            for child in getattr(widget, "children", ()):
                close_widget(child)
            widget.close()

        close_widget(self.ui)

    def _on_click(self, _button: Any) -> None:
        if self._closed or self.run_button.disabled:
            return
        self.run_button.disabled = True
        self.run_button.description = "Running…"
        self.status.value = "<span style='margin-left:8px'>Sampling and scoring…</span>"
        try:
            self.run()
            self._render_result()
            self.status.value = "<span style='margin-left:8px;color:#176b2c'>Finished.</span>"
        except Exception as exc:
            self.last_ensembles = []
            self.last_ensemble = None
            self.last_summary = None
            self._last_curve = None
            self.result_status.value = "<b>No result: evaluation failed.</b>"
            self.output.outputs = ({
                "output_type": "display_data", "metadata": {},
                "data": {"text/html": f"<p><b>Run failed:</b> {escape(str(exc))}</p>"},
            },)
            self.status.value = "<span style='margin-left:8px;color:#b00020'>Failed.</span>"
        finally:
            self.run_button.disabled = False
            self.run_button.description = "Start sampling"


def build_sampling_lab() -> SamplingLab:
    return SamplingLab()
