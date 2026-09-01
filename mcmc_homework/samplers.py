"""Correct implementations of the four samplers used in the homework."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .targets import Array, Target2D


@dataclass
class SamplerResult:
    """Raw chain output and target-evaluation accounting."""

    samples: Array
    log_probs: Array
    method: str
    scale: float
    accepted: Array | None
    log_prob_evals: int
    grad_evals: int
    n_leapfrog: int | None = None
    diverged: bool = False
    message: str = ""

    @property
    def acceptance_rate(self) -> float | None:
        if self.accepted is None:
            return None
        return float(np.mean(self.accepted)) if self.accepted.size else float("nan")

    @property
    def target_evals(self) -> int:
        return int(self.log_prob_evals + self.grad_evals)


def target_evals_for_states(
    method: str,
    n_states: int,
    n_leapfrog: int = 10,
) -> int:
    """Return the sampler's target-oracle cost for a finite chain.

    One call to ``log_prob`` and one call to ``grad_log_prob`` each count as one
    work unit. Input validation, diagnostic scoring, and reference sampling are
    excluded. HMC's value is the maximum cost when every trajectory remains
    finite; an invalid trajectory can stop early and use less work.
    """

    if int(n_states) != n_states or n_states < 2:
        raise ValueError("n_states must be an integer of at least 2.")
    normalized = method.strip().upper()
    if normalized in {"RWM", "RWMH", "MH", "RANDOM-WALK METROPOLIS"}:
        return int(n_states)
    if normalized == "ULA":
        return int(n_states - 1)
    if normalized == "MALA":
        return int(2 * n_states)
    if normalized in {"HMC", "HAMILTONIAN MONTE CARLO"}:
        if int(n_leapfrog) != n_leapfrog or n_leapfrog < 1:
            raise ValueError("n_leapfrog must be a positive integer.")
        return int(2 + (n_states - 1) * (int(n_leapfrog) + 1))
    raise ValueError(f"Unknown method {method!r}; choose RWMH, ULA, MALA, or HMC.")


def states_for_target_evals(
    method: str,
    target_eval_budget: int,
    n_leapfrog: int = 10,
) -> int:
    """Return the longest chain whose sampler cost cannot exceed the budget."""

    if int(target_eval_budget) != target_eval_budget or target_eval_budget < 2:
        raise ValueError("target_eval_budget must be an integer of at least 2.")
    normalized = method.strip().upper()
    budget = int(target_eval_budget)
    if normalized in {"RWM", "RWMH", "MH", "RANDOM-WALK METROPOLIS"}:
        n_states = budget
    elif normalized == "ULA":
        n_states = budget + 1
    elif normalized == "MALA":
        n_states = budget // 2
    elif normalized in {"HMC", "HAMILTONIAN MONTE CARLO"}:
        if int(n_leapfrog) != n_leapfrog or n_leapfrog < 1:
            raise ValueError("n_leapfrog must be a positive integer.")
        n_states = 1 + (budget - 2) // (int(n_leapfrog) + 1)
    else:
        raise ValueError(
            f"Unknown method {method!r}; choose RWMH, ULA, MALA, or HMC."
        )
    if n_states < 2:
        raise ValueError(
            "The target-evaluation budget is too small for this method setting."
        )
    return int(n_states)


def _validate_sampler_inputs(
    target: Target2D, n_steps: int, scale: float, initial: Array | None
) -> Array:
    if int(n_steps) != n_steps or n_steps < 2:
        raise ValueError("n_steps must be an integer of at least 2.")
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("The proposal scale / integrator step size must be positive.")
    current = target.start.copy() if initial is None else np.asarray(initial, dtype=float).copy()
    if current.shape != (2,):
        raise ValueError("initial must be a length-2 position.")
    if not np.all(np.isfinite(current)) or not np.isfinite(target.log_prob(current)):
        raise ValueError("initial must have a finite target log density.")
    return current


def random_walk_metropolis(
    target: Target2D,
    n_steps: int,
    proposal_scale: float,
    rng: np.random.Generator,
    initial: Array | None = None,
) -> SamplerResult:
    """Isotropic Gaussian random-walk Metropolis-Hastings."""

    current = _validate_sampler_inputs(target, n_steps, proposal_scale, initial)
    samples = np.empty((n_steps, 2), dtype=float)
    accepted = np.zeros(n_steps - 1, dtype=bool)
    samples[0] = current
    current_log_prob = float(target.log_prob(current))
    log_prob_evals = 1

    for step in range(1, n_steps):
        proposal = current + proposal_scale * rng.normal(size=2)
        proposal_log_prob = float(target.log_prob(proposal))
        log_prob_evals += 1
        if np.isfinite(proposal_log_prob):
            log_acceptance_ratio = proposal_log_prob - current_log_prob
            if np.log(rng.random()) < min(0.0, log_acceptance_ratio):
                current = proposal
                current_log_prob = proposal_log_prob
                accepted[step - 1] = True
        # Rejections must duplicate the current state in the Markov chain.
        samples[step] = current

    return SamplerResult(
        samples=samples,
        log_probs=np.asarray(target.log_prob(samples)),
        method="RWMH",
        scale=float(proposal_scale),
        accepted=accepted,
        log_prob_evals=log_prob_evals,
        grad_evals=0,
    )


def unadjusted_langevin(
    target: Target2D,
    n_steps: int,
    step_size: float,
    rng: np.random.Generator,
    initial: Array | None = None,
    divergence_limit: float = 1e8,
) -> SamplerResult:
    r"""Unadjusted Langevin algorithm.

    Uses ``X[k+1] = X[k] + eta*grad_log_prob(X[k]) + sqrt(2*eta)*Z[k]``.
    At finite ``eta`` this is generally biased, and large values may be unstable.
    """

    current = _validate_sampler_inputs(target, n_steps, step_size, initial)
    samples = np.full((n_steps, 2), np.nan, dtype=float)
    samples[0] = current
    grad_evals = 0
    diverged = False
    message = ""
    noise_scale = np.sqrt(2.0 * step_size)

    for step in range(1, n_steps):
        gradient = np.asarray(target.grad_log_prob(current), dtype=float)
        grad_evals += 1
        proposal = current + step_size * gradient + noise_scale * rng.normal(size=2)
        if (
            not np.all(np.isfinite(gradient))
            or not np.all(np.isfinite(proposal))
            or np.max(np.abs(proposal)) > divergence_limit
        ):
            diverged = True
            message = f"ULA diverged at transition {step}; reduce the step size η."
            break
        current = proposal
        samples[step] = current

    with np.errstate(over="ignore", invalid="ignore"):
        log_probs = np.asarray(target.log_prob(samples))
    return SamplerResult(
        samples=samples,
        log_probs=log_probs,
        method="ULA",
        scale=float(step_size),
        accepted=None,
        log_prob_evals=0,
        grad_evals=grad_evals,
        diverged=diverged,
        message=message,
    )


def metropolis_adjusted_langevin(
    target: Target2D,
    n_steps: int,
    step_size: float,
    rng: np.random.Generator,
    initial: Array | None = None,
) -> SamplerResult:
    r"""Metropolis-adjusted Langevin algorithm (MALA).

    The proposal is ``N(x + eta*grad_log_prob(x), 2*eta*I)``.  The asymmetric
    forward/reverse proposal correction is included in the MH ratio.
    """

    current = _validate_sampler_inputs(target, n_steps, step_size, initial)
    samples = np.empty((n_steps, 2), dtype=float)
    accepted = np.zeros(n_steps - 1, dtype=bool)
    samples[0] = current
    current_log_prob = float(target.log_prob(current))
    current_gradient = np.asarray(target.grad_log_prob(current), dtype=float)
    log_prob_evals = 1
    grad_evals = 1
    noise_scale = np.sqrt(2.0 * step_size)

    for step in range(1, n_steps):
        forward_mean = current + step_size * current_gradient
        proposal = forward_mean + noise_scale * rng.normal(size=2)
        proposal_log_prob = float(target.log_prob(proposal))
        proposal_gradient = np.asarray(target.grad_log_prob(proposal), dtype=float)
        log_prob_evals += 1
        grad_evals += 1

        if np.isfinite(proposal_log_prob) and np.all(np.isfinite(proposal_gradient)):
            reverse_mean = proposal + step_size * proposal_gradient
            forward_residual = proposal - forward_mean
            reverse_residual = current - reverse_mean
            log_q_forward = -np.dot(forward_residual, forward_residual) / (4.0 * step_size)
            log_q_reverse = -np.dot(reverse_residual, reverse_residual) / (4.0 * step_size)
            log_acceptance_ratio = (
                proposal_log_prob - current_log_prob + log_q_reverse - log_q_forward
            )
            if np.log(rng.random()) < min(0.0, log_acceptance_ratio):
                current = proposal
                current_log_prob = proposal_log_prob
                current_gradient = proposal_gradient
                accepted[step - 1] = True
        samples[step] = current

    return SamplerResult(
        samples=samples,
        log_probs=np.asarray(target.log_prob(samples)),
        method="MALA",
        scale=float(step_size),
        accepted=accepted,
        log_prob_evals=log_prob_evals,
        grad_evals=grad_evals,
    )


def hamiltonian_monte_carlo(
    target: Target2D,
    n_steps: int,
    step_size: float,
    n_leapfrog: int,
    rng: np.random.Generator,
    initial: Array | None = None,
) -> SamplerResult:
    r"""Hamiltonian Monte Carlo with identity mass and leapfrog integration.

    A fresh momentum ``p ~ N(0, I)`` is sampled for each transition.  The
    leapfrog proposal is accepted using the exact Hamiltonian difference, which
    corrects finite integrator error.
    """

    current = _validate_sampler_inputs(target, n_steps, step_size, initial)
    if int(n_leapfrog) != n_leapfrog or n_leapfrog < 1:
        raise ValueError("n_leapfrog must be a positive integer.")
    n_leapfrog = int(n_leapfrog)

    samples = np.empty((n_steps, 2), dtype=float)
    accepted = np.zeros(n_steps - 1, dtype=bool)
    samples[0] = current
    current_log_prob = float(target.log_prob(current))
    current_gradient = np.asarray(target.grad_log_prob(current), dtype=float)
    log_prob_evals = 1
    grad_evals = 1

    for step in range(1, n_steps):
        initial_momentum = rng.normal(size=2)
        position = current.copy()
        momentum = initial_momentum.copy()

        momentum += 0.5 * step_size * current_gradient
        proposal_gradient = current_gradient
        valid_proposal = True
        for leapfrog_step in range(n_leapfrog):
            position += step_size * momentum
            proposal_gradient = np.asarray(target.grad_log_prob(position), dtype=float)
            grad_evals += 1
            if not np.all(np.isfinite(position)) or not np.all(np.isfinite(proposal_gradient)):
                valid_proposal = False
                break
            if leapfrog_step < n_leapfrog - 1:
                momentum += step_size * proposal_gradient
            else:
                momentum += 0.5 * step_size * proposal_gradient

        if valid_proposal:
            proposal_log_prob = float(target.log_prob(position))
            log_prob_evals += 1
            if np.isfinite(proposal_log_prob) and np.all(np.isfinite(momentum)):
                current_hamiltonian = -current_log_prob + 0.5 * np.dot(
                    initial_momentum, initial_momentum
                )
                proposal_hamiltonian = -proposal_log_prob + 0.5 * np.dot(momentum, momentum)
                log_acceptance_ratio = current_hamiltonian - proposal_hamiltonian
                if np.log(rng.random()) < min(0.0, log_acceptance_ratio):
                    current = position
                    current_log_prob = proposal_log_prob
                    current_gradient = proposal_gradient
                    accepted[step - 1] = True

        samples[step] = current

    return SamplerResult(
        samples=samples,
        log_probs=np.asarray(target.log_prob(samples)),
        method="HMC",
        scale=float(step_size),
        accepted=accepted,
        log_prob_evals=log_prob_evals,
        grad_evals=grad_evals,
        n_leapfrog=n_leapfrog,
    )


def run_sampler(
    method: str,
    target: Target2D,
    n_steps: int,
    scale: float,
    seed: int,
    initial: Array | None = None,
    n_leapfrog: int = 10,
) -> SamplerResult:
    """Run a sampler using a fresh, reproducible random-number generator."""

    rng = np.random.default_rng(seed)
    normalized = method.strip().upper()
    if normalized in {"RWM", "RWMH", "MH", "RANDOM-WALK METROPOLIS"}:
        return random_walk_metropolis(target, n_steps, scale, rng, initial)
    if normalized == "ULA":
        return unadjusted_langevin(target, n_steps, scale, rng, initial)
    if normalized == "MALA":
        return metropolis_adjusted_langevin(target, n_steps, scale, rng, initial)
    if normalized in {"HMC", "HAMILTONIAN MONTE CARLO"}:
        return hamiltonian_monte_carlo(
            target, n_steps, scale, n_leapfrog, rng, initial
        )
    raise ValueError(f"Unknown method {method!r}; choose RWMH, ULA, MALA, or HMC.")
