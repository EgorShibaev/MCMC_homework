"""Two-dimensional target distributions used by the homework.

Each target provides a log density, an analytic gradient, and an exact iid
reference sampler.  Exact references let students measure sampling error rather
than judge a chain only by whether its scatter plot looks plausible.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


Array = np.ndarray


def _as_points(x: Array | Sequence[float]) -> tuple[Array, bool]:
    points = np.asarray(x, dtype=float)
    was_single = points.ndim == 1
    if points.shape[-1:] != (2,):
        raise ValueError(f"Expected points with final dimension 2, got {points.shape}.")
    return points, was_single


def _scalar_if_single(value: Array, was_single: bool) -> Array | float:
    return float(np.asarray(value)) if was_single else np.asarray(value)


class Target2D:
    """Interface for a normalized-or-unnormalized two-dimensional target."""

    key: str
    name: str
    challenge: str
    bounds: tuple[tuple[float, float], tuple[float, float]]
    start: Array

    def log_prob(self, x: Array | Sequence[float]) -> Array | float:
        raise NotImplementedError

    def grad_log_prob(self, x: Array | Sequence[float]) -> Array:
        raise NotImplementedError

    def sample_reference(self, n: int, rng: np.random.Generator) -> Array:
        raise NotImplementedError

    def diagnostic_features(self, samples: Array) -> tuple[Array, list[str]]:
        return np.asarray(samples, dtype=float), ["x₁", "x₂"]

    def task_diagnostics(self, samples: Array) -> dict[str, float]:
        return {}


class TiltedGaussian(Target2D):
    """A narrow rotated Gaussian with covariance condition number 100."""

    key = "gaussian"
    name = "Tilted ill-conditioned Gaussian"
    challenge = "Can the chain move along the long axis without crossing the narrow axis?"
    bounds = ((-4.2, 4.2), (-3.4, 3.4))

    def __init__(self) -> None:
        angle = np.deg2rad(35.0)
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        self.cov = rotation @ np.diag([1.5**2, 0.15**2]) @ rotation.T
        self.precision = np.linalg.inv(self.cov)
        self.start = np.array([-3.2, 2.4])

    def log_prob(self, x: Array | Sequence[float]) -> Array | float:
        points, was_single = _as_points(x)
        values = -0.5 * np.einsum("...i,ij,...j->...", points, self.precision, points)
        return _scalar_if_single(values, was_single)

    def grad_log_prob(self, x: Array | Sequence[float]) -> Array:
        points, _ = _as_points(x)
        return -(points @ self.precision)

    def sample_reference(self, n: int, rng: np.random.Generator) -> Array:
        return rng.multivariate_normal(np.zeros(2), self.cov, size=n)

    def task_diagnostics(self, samples: Array) -> dict[str, float]:
        mean = np.mean(samples, axis=0)
        covariance = np.cov(samples, rowvar=False)
        return {
            "standardized mean error": float(np.sqrt(mean @ self.precision @ mean)),
            "relative covariance error": float(
                np.linalg.norm(covariance - self.cov, ord="fro")
                / np.linalg.norm(self.cov, ord="fro")
            ),
        }


class BananaTarget(Target2D):
    """A curved target obtained by a unit-Jacobian Gaussian transform."""

    key = "banana"
    name = "Banana-shaped density"
    challenge = "Can a local isotropic proposal follow a thin, curved typical set?"
    bounds = ((-4.5, 4.5), (-2.0, 7.0))

    def __init__(self, sigma_x: float = 1.3, sigma_r: float = 0.30, bend: float = 0.45):
        self.sigma_x = float(sigma_x)
        self.sigma_r = float(sigma_r)
        self.bend = float(bend)
        self.start = np.array([-3.0, self.bend * (3.0**2 - self.sigma_x**2)])

    def residual(self, x: Array) -> Array:
        return x[..., 1] - self.bend * (x[..., 0] ** 2 - self.sigma_x**2)

    def log_prob(self, x: Array | Sequence[float]) -> Array | float:
        points, was_single = _as_points(x)
        with np.errstate(over="ignore", invalid="ignore"):
            residual = self.residual(points)
            values = -0.5 * (points[..., 0] / self.sigma_x) ** 2
            values -= 0.5 * (residual / self.sigma_r) ** 2
        return _scalar_if_single(values, was_single)

    def grad_log_prob(self, x: Array | Sequence[float]) -> Array:
        points, _ = _as_points(x)
        with np.errstate(over="ignore", invalid="ignore"):
            residual = self.residual(points)
            grad_x = -(points[..., 0] / self.sigma_x**2)
            grad_x += 2.0 * self.bend * points[..., 0] * residual / self.sigma_r**2
            grad_y = -residual / self.sigma_r**2
        return np.stack([grad_x, grad_y], axis=-1)

    def sample_reference(self, n: int, rng: np.random.Generator) -> Array:
        x = rng.normal(scale=self.sigma_x, size=n)
        residual = rng.normal(scale=self.sigma_r, size=n)
        y = residual + self.bend * (x**2 - self.sigma_x**2)
        return np.column_stack([x, y])

    def diagnostic_features(self, samples: Array) -> tuple[Array, list[str]]:
        features = np.column_stack([samples[:, 0], samples[:, 1], samples[:, 0] ** 2])
        return features, ["x₁", "x₂", "x₁² (curvature)"]

    def task_diagnostics(self, samples: Array) -> dict[str, float]:
        residual = self.residual(samples)
        return {
            "residual-scale error": abs(float(np.std(residual, ddof=1) / self.sigma_r) - 1.0),
            "x₁ variance error": abs(
                float(np.var(samples[:, 0], ddof=1) / self.sigma_x**2) - 1.0
            ),
        }


class UnequalGaussianMixture(Target2D):
    """Two separated Gaussian modes with unequal probability mass."""

    key = "mixture"
    name = "Unequal bimodal Gaussian mixture"
    challenge = "Can the chain discover both modes with the correct 65:35 mass split?"
    bounds = ((-5.5, 5.5), (-3.5, 3.5))

    def __init__(self) -> None:
        self.weights = np.array([0.65, 0.35])
        self.means = np.array([[-2.5, 0.0], [2.5, 0.0]])
        self.variances = np.array([0.65**2, 0.80**2])
        self.start = self.means[1].copy()

    def _component_log_prob(self, points: Array) -> Array:
        difference = points[..., None, :] - self.means
        quadratic = np.sum(difference**2 / self.variances, axis=-1)
        log_normalizer = np.log(2.0 * np.pi) + 0.5 * np.sum(np.log(self.variances))
        return np.log(self.weights) - 0.5 * quadratic - log_normalizer

    def _responsibilities(self, points: Array) -> tuple[Array, Array]:
        component_log_prob = self._component_log_prob(points)
        maximum = np.max(component_log_prob, axis=-1, keepdims=True)
        shifted = np.exp(component_log_prob - maximum)
        total = np.sum(shifted, axis=-1, keepdims=True)
        responsibilities = shifted / total
        log_prob = np.squeeze(maximum + np.log(total), axis=-1)
        return responsibilities, log_prob

    def log_prob(self, x: Array | Sequence[float]) -> Array | float:
        points, was_single = _as_points(x)
        _, values = self._responsibilities(points)
        return _scalar_if_single(values, was_single)

    def grad_log_prob(self, x: Array | Sequence[float]) -> Array:
        points, _ = _as_points(x)
        responsibilities, _ = self._responsibilities(points)
        component_scores = -(points[..., None, :] - self.means) / self.variances
        return np.sum(responsibilities[..., None] * component_scores, axis=-2)

    def sample_reference(self, n: int, rng: np.random.Generator) -> Array:
        components = rng.choice(2, size=n, p=self.weights)
        noise = rng.normal(size=(n, 2)) * np.sqrt(self.variances)
        return self.means[components] + noise

    def diagnostic_features(self, samples: Array) -> tuple[Array, list[str]]:
        left_mode = (samples[:, 0] < 0.0).astype(float)
        features = np.column_stack([samples[:, 0], samples[:, 1], left_mode])
        return features, ["x₁", "x₂", "left-mode indicator"]

    def task_diagnostics(self, samples: Array) -> dict[str, float]:
        left_mode = samples[:, 0] < 0.0
        switches = int(np.count_nonzero(left_mode[1:] != left_mode[:-1]))
        return {
            "left-mode mass error": abs(float(np.mean(left_mode)) - self.weights[0]),
            "mode switches": float(switches),
        }


class ImbalancedGaussianMixture(UnequalGaussianMixture):
    """A 10:90 mixture used to compare rare-mode discovery by ULA and HMC."""

    key = "imbalanced_mixture"
    name = "Imbalanced Gaussian mixture (10:90)"
    challenge = "Can the sampler find the rare 10% left mode often enough?"
    bounds = ((-6.0, 6.0), (-4.0, 4.0))

    def __init__(self) -> None:
        self.weights = np.array([0.10, 0.90])
        self.means = np.array([[-2.5, 0.0], [2.5, 0.0]])
        self.variances = np.array([1.0, 1.0])
        self.start = self.means[1].copy()


TARGETS: dict[str, Target2D] = {
    target.key: target
    for target in (
        TiltedGaussian(),
        BananaTarget(),
        UnequalGaussianMixture(),
        ImbalancedGaussianMixture(),
    )
}
