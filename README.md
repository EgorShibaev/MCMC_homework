# MCMC sampling homework

Student project for comparing random-walk Metropolis-Hastings (RWMH), the
unadjusted Langevin algorithm (ULA), Metropolis-adjusted Langevin (MALA), and
Hamiltonian Monte Carlo (HMC).

All implementations in this version are intentional, correct reference
implementations. The notebook asks students to tune hyperparameters and explain
sampling failures; it does not hide deliberate implementation bugs.

## What is included

- `MCMC_Homework.ipynb` — interactive student assignment.
- `mcmc_homework/targets.py` — four two-dimensional target distributions and
  exact iid reference samplers.
- `mcmc_homework/samplers.py` — RWMH, ULA, MALA, identity-mass HMC, and exact
  target-evaluation accounting for a shared work budget.
- `mcmc_homework/metrics.py` — standardized sliced-Wasserstein error, ESS, and
  target-specific diagnostics.
- `mcmc_homework/visualization.py` — trajectory, trace, autocorrelation, and
  mode-proportion plots, plus the fixed-budget SWD convergence figure.
- `mcmc_homework/experiments.py` — fixed-budget multi-chain ensembles,
  repeated-seed benchmarks, target-specific pass thresholds, and convergence
  calculations.
- `mcmc_homework/widgets.py` — sliders and button-driven notebook interface.
- `tests/` — numerical correctness and regression tests.

The fourth target is a 10:90 Gaussian mixture. Its guided notebook experiment
shows a finite-budget ULA chain missing the rare mode while tuned HMC repeatedly
crosses the barrier and approaches the correct mode proportions.

## Homework pass rule

Every benchmark ensemble receives 40,000 target-evaluation work units, including
burn-in and shared across all selected chains. One log-density or gradient call
counts as one unit. Students tune a separate proposal scale `sigma` for RWMH,
step size `eta` for ULA and MALA, and HMC step size `epsilon` plus leapfrog count
`L`. Every row also tunes burn-in and a chain count from 1 through 8.

The entire selected multi-chain ensemble is repeated with base seeds
`(11, 23, 47)`. A row passes when its **worst-repeat standardized
sliced-Wasserstein distance** is at most 0.06 for the tilted Gaussian, 0.115 for
the banana, or 0.15 for the 65:35 mixture, with no divergent ensemble. All 12
core rows must pass. The notebook table reports `PASS` or `TUNE MORE`, and the
adjacent plot tracks worst-repeat SWD against the cumulative target-evaluation
budget.

The total work budget, benchmark base seeds, initial states, accounting rule,
and pass thresholds are fixed. Choosing more chains makes each chain shorter;
increasing burn-in leaves fewer scored states; increasing HMC's `L` reduces its
number of transitions. These allocation trade-offs are part of the assignment.

## Required experimental analysis

A passing benchmark table alone is not a complete submission. Students keep a
search log with at least three candidate configurations per target–method row
(at least 36 entries), then evaluate the two strongest candidates per row on all
three benchmark repeats. Failed runs remain in the log.

The notebook replaces Tasks A–D with six controlled investigations:

1. RWMH proposal scale: acceptance versus movement on Gaussian and mixture targets.
2. ULA step size: slow exploration, finite-step bias, and instability.
3. ULA versus MALA at matched step sizes: the effect and cost of MH correction.
4. HMC: short versus long trajectories at fixed step size, then an integrator-step comparison.
5. One versus many chains: independent restarts versus shorter chains at fixed total cost.
6. Burn-in: initialization bias versus retained information, followed by repeated-seed verification.

For each investigation, students submit a results table, a short explanation,
and their own labelled screenshots of contrasting lab runs and their metrics.
The notebook specifies which comparisons to capture. Screenshots must be
embedded as notebook attachments or included in a portable `screenshots/`
directory. The final submission also includes the 12/12 PASS benchmark table
and SWD-versus-budget plot. The numerical thresholds are calibrated finite-run
requirements, not proofs of convergence.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
jupyter lab MCMC_Homework.ipynb
```

Restart the kernel and choose **Run All** before using **Start sampling**. A
static HTML export can display saved controls but cannot execute their Python
callbacks.

## Validate or rebuild

```bash
MPLBACKEND=Agg python -m unittest discover -v
MPLBACKEND=Agg python -m scripts.preview
python -m scripts.build_notebook
```

The preview script writes visual QA artifacts to the ignored `artifacts/`
directory. The notebook builder writes a clean `MCMC_Homework.ipynb` from the
project modules and assignment text.
