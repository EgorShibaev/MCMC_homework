"""Public API for the MCMC sampling homework project."""

from .experiments import (
    BENCHMARK_SEEDS,
    CORE_TARGET_KEYS,
    DEFAULT_SETTINGS,
    METHODS,
    Experiment,
    benchmark_setting,
    run_experiment,
    summarize_experiments,
)
from .metrics import (
    autocorrelation_1d,
    compute_metrics,
    effective_sample_size_1d,
    get_reference_sample,
    reference_noise_floor,
    standardized_sliced_wasserstein,
)
from .samplers import (
    SamplerResult,
    hamiltonian_monte_carlo,
    metropolis_adjusted_langevin,
    random_walk_metropolis,
    run_sampler,
    unadjusted_langevin,
)
from .targets import (
    TARGETS,
    BananaTarget,
    ImbalancedGaussianMixture,
    Target2D,
    TiltedGaussian,
    UnequalGaussianMixture,
)
from .validation import finite_difference_gradient, gradient_check_report
from .visualization import (
    metrics_html,
    plot_chain_panel,
    plot_mode_ratio_comparison,
    plot_sampling_run,
)
from .widgets import SamplingLab, build_sampling_lab

__all__ = [
    "BENCHMARK_SEEDS",
    "CORE_TARGET_KEYS",
    "DEFAULT_SETTINGS",
    "METHODS",
    "TARGETS",
    "BananaTarget",
    "Experiment",
    "ImbalancedGaussianMixture",
    "SamplerResult",
    "SamplingLab",
    "Target2D",
    "TiltedGaussian",
    "UnequalGaussianMixture",
    "autocorrelation_1d",
    "benchmark_setting",
    "build_sampling_lab",
    "compute_metrics",
    "effective_sample_size_1d",
    "finite_difference_gradient",
    "get_reference_sample",
    "gradient_check_report",
    "hamiltonian_monte_carlo",
    "metrics_html",
    "metropolis_adjusted_langevin",
    "plot_chain_panel",
    "plot_mode_ratio_comparison",
    "plot_sampling_run",
    "random_walk_metropolis",
    "reference_noise_floor",
    "run_experiment",
    "run_sampler",
    "standardized_sliced_wasserstein",
    "summarize_experiments",
    "unadjusted_langevin",
]
