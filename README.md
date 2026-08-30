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
  mode-proportion plots.
- `mcmc_homework/experiments.py` — reproducible runs and repeated-seed
  benchmarks.
- `mcmc_homework/widgets.py` — sliders and button-driven notebook interface.
- `tests/` — numerical correctness and regression tests.

The fourth target is a 10:90 Gaussian mixture. Its guided notebook experiment
shows a finite-budget ULA chain missing the rare mode while tuned HMC repeatedly
crosses the barrier and approaches the correct mode proportions.

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
