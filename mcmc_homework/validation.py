"""Small numerical checks that can also be shown inside the notebook."""

from __future__ import annotations

from typing import Any

import numpy as np

from .targets import Array, TARGETS, Target2D


def finite_difference_gradient(
    target: Target2D, point: Array, epsilon: float = 1e-6
) -> Array:
    point = np.asarray(point, dtype=float)
    estimate = np.empty(2, dtype=float)
    for dimension in range(2):
        offset = np.zeros(2)
        offset[dimension] = epsilon
        estimate[dimension] = (
            float(target.log_prob(point + offset))
            - float(target.log_prob(point - offset))
        ) / (2.0 * epsilon)
    return estimate


def gradient_check_report() -> list[dict[str, Any]]:
    points = {
        "gaussian": [np.array([0.2, -0.1]), np.array([1.0, 0.7])],
        "banana": [np.array([0.2, -0.4]), np.array([1.4, 0.3])],
        "mixture": [np.array([-2.0, 0.3]), np.array([0.2, -0.4])],
        "imbalanced_mixture": [np.array([-2.0, 0.3]), np.array([0.2, -0.4])],
    }
    report: list[dict[str, Any]] = []
    for key, target_points in points.items():
        target = TARGETS[key]
        errors = [
            float(
                np.max(
                    np.abs(
                        np.asarray(target.grad_log_prob(point))
                        - finite_difference_gradient(target, point)
                    )
                )
            )
            for point in target_points
        ]
        report.append(
            {
                "target": key,
                "max_abs_error": max(errors),
                "passed": max(errors) < 1e-5,
            }
        )
    return report
