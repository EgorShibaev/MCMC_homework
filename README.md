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
- `mcmc_homework/samplers.py` — RWMH, ULA, MALA, and identity-mass HMC with
  leapfrog integration.
- `mcmc_homework/metrics.py` — standardized sliced-Wasserstein error, ESS, and
  target-specific diagnostics.
- `mcmc_homework/visualization.py` — trajectory, trace, autocorrelation, and
  mode-proportion plots, plus the fixed-budget SWD convergence figure.
- `mcmc_homework/experiments.py` — reproducible runs and repeated-seed
  benchmarks, target-specific pass thresholds, and convergence calculations.
- `mcmc_homework/widgets.py` — sliders and button-driven notebook interface.
- `tests/` — numerical correctness and regression tests.

The fourth target is a 10:90 Gaussian mixture. Its guided notebook experiment
shows a finite-budget ULA chain missing the rare mode while tuned HMC repeatedly
crosses the barrier and approaches the correct mode proportions.

## Homework pass rule

The official benchmark uses 4,000 recorded chain states, 25% burn-in, and seeds
`(11, 23, 47)`. A target-method row passes when its **worst-seed standardized
sliced-Wasserstein distance** is at most 0.50 for the tilted Gaussian, 0.50 for
the banana, or 0.75 for the 65:35 mixture, with no divergent run. All 12 core
rows must pass. The notebook table reports `PASS` or `TUNE MORE`, and an
adjacent plot tracks the same worst-seed SWD as retained iterations accumulate.

Students tune a separate proposal scale `sigma` for each RWMH row, step size
`eta` for each ULA and MALA row, and HMC step size `epsilon` plus leapfrog count
`L` for each HMC row. Iterations, burn-in, number of chains, benchmark seeds,
initial states, and pass thresholds are fixed for submission; the corresponding
interactive controls are for exploration only.

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
