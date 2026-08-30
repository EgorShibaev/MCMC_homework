"""Render Python-level visual previews before notebook generation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from mcmc_homework import (
    CORE_TARGET_KEYS,
    DEFAULT_SETTINGS,
    TARGETS,
    gradient_check_report,
    plot_chain_panel,
    plot_mode_ratio_comparison,
    plot_sampling_run,
    run_experiment,
)
from mcmc_homework.experiments import METHODS


def main() -> None:
    output_dir = Path("artifacts")
    output_dir.mkdir(exist_ok=True)
    report = gradient_check_report()
    if not all(row["passed"] for row in report):
        raise RuntimeError(report)

    experiments = []
    for target_key in CORE_TARGET_KEYS:
        for method in METHODS:
            setting = DEFAULT_SETTINGS[(target_key, method)]
            experiment = run_experiment(
                target_key,
                method,
                float(setting["scale"]),
                n_steps=1800,
                burn_fraction=0.25,
                seed=11,
                n_leapfrog=int(setting.get("n_leapfrog", 10)),
            )
            experiments.append(experiment)
            print(
                f"{target_key:9s} {method:5s} scale={setting['scale']:<6} "
                f"SWD={experiment.metrics['swd']:.3f}"
            )

    figure, axes = plt.subplots(3, 4, figsize=(15.5, 10.5), constrained_layout=True)
    for row, target_key in enumerate(CORE_TARGET_KEYS):
        for column, method in enumerate(METHODS):
            experiment = next(
                item
                for item in experiments
                if item.target_key == target_key and item.result.method == method
            )
            plot_chain_panel(axes[row, column], TARGETS[target_key], experiment, 400)
            axes[row, column].set_title(
                f"{method}: SWD={experiment.metrics['swd']:.2f}", fontsize=10
            )
    figure.suptitle("Correct sampler implementations at untuned starting settings")
    figure.savefig(output_dir / "sampler_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(figure)

    showcase = next(
        item
        for item in experiments
        if item.target_key == "banana" and item.result.method == "HMC"
    )
    dashboard = plot_sampling_run(showcase)
    dashboard.savefig(output_dir / "hmc_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close(dashboard)

    ula = run_experiment(
        "imbalanced_mixture", "ULA", 0.002, 6000, 0.20, 47
    )
    hmc = run_experiment(
        "imbalanced_mixture", "HMC", 0.12, 6000, 0.20, 47, n_leapfrog=20
    )
    comparison = plot_mode_ratio_comparison(ula, hmc)
    comparison.savefig(
        output_dir / "imbalanced_mode_ratio.png", dpi=150, bbox_inches="tight"
    )
    plt.close(comparison)
    print("Saved previews in artifacts/.")


if __name__ == "__main__":
    main()
