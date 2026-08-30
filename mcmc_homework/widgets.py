"""Button-driven ipywidgets interface for exploring sampler hyperparameters."""

from __future__ import annotations

from html import escape
from typing import Any

import matplotlib.pyplot as plt

from .experiments import DEFAULT_SETTINGS, METHODS, Experiment, run_experiment, summarize_experiments
from .targets import TARGETS
from .visualization import metrics_html, plot_sampling_run


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
        self.last_experiments: list[Experiment] = []
        self.last_summary: dict[str, float] | None = None

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
        self.n_steps = widgets.IntSlider(
            value=3000,
            min=500,
            max=10000,
            step=500,
            description="iterations",
            continuous_update=False,
            layout=common_layout,
            style=common_style,
        )
        self.burn_fraction = widgets.FloatSlider(
            value=0.25,
            min=0.05,
            max=0.50,
            step=0.05,
            description="burn-in",
            readout_format=".0%",
            continuous_update=False,
            layout=common_layout,
            style=common_style,
        )
        self.repetitions = widgets.IntSlider(
            value=1,
            min=1,
            max=5,
            step=1,
            description="repeats",
            continuous_update=False,
            layout=common_layout,
            style=common_style,
        )
        self.seed = widgets.IntSlider(
            value=11,
            min=0,
            max=999,
            step=1,
            description="base seed",
            continuous_update=False,
            layout=common_layout,
            style=common_style,
        )
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
                self.n_steps,
                self.burn_fraction,
                self.repetitions,
                self.seed,
                widgets.HBox([self.run_button, self.status]),
            ]
        )
        self.ui = widgets.VBox([controls, self.output])
        self.target.observe(self._on_configuration_change, names="value")
        self.method.observe(self._on_configuration_change, names="value")
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
        if method == "HMC":
            self.n_leapfrog.layout.display = "flex"
            self.n_leapfrog.value = int(setting["n_leapfrog"])
        else:
            self.n_leapfrog.layout.display = "none"
        target = TARGETS[self.target.value]
        self.status.value = (
            f"<span style='margin-left:8px;color:#555'>{escape(target.challenge)}</span>"
        )

    def run(self) -> tuple[list[Experiment], dict[str, float]]:
        seeds = [
            int(self.seed.value) + 101 * repeat
            for repeat in range(self.repetitions.value)
        ]
        experiments = [
            run_experiment(
                target_key=self.target.value,
                method=self.method.value,
                scale=float(self.scale.value),
                n_steps=int(self.n_steps.value),
                burn_fraction=float(self.burn_fraction.value),
                seed=seed,
                n_leapfrog=int(self.n_leapfrog.value),
            )
            for seed in seeds
        ]
        summary = summarize_experiments(experiments)
        self.last_experiments = experiments
        self.last_summary = summary
        return experiments, summary

    def _on_click(self, _button: Any) -> None:
        self.run_button.disabled = True
        self.run_button.description = "Running…"
        self.status.value = "<span style='margin-left:8px'>Sampling and scoring…</span>"
        try:
            experiments, summary = self.run()
            with self.output:
                self._clear_output(wait=True)
                figure = plot_sampling_run(experiments[0])
                self._display(figure)
                plt.close(figure)
                self._display(self._HTML(metrics_html(experiments[0], summary)))
                if experiments[0].result.message:
                    self._display(
                        self._HTML(
                            f"<p style='color:#b00020'><b>Warning:</b> "
                            f"{escape(experiments[0].result.message)}</p>"
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
