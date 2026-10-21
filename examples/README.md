# Examples

Standalone, **signac-free** examples of running emulator GPBO on the paper's case studies
(Carlozo, Wang & Dowling, *"Bayesian Optimization Methods for Nonlinear Model Calibration"*,
Ind. Eng. Chem. Res. 2025). Each script builds synthetic data, runs the algorithm via
`GPBODriver.run(job=None)`, and prints the recovered parameters — no signac and
no cluster required.

You can also calibrate **your own model** (not just the built-in case studies) by defining a
`CalibrationProblem` around a `model(theta, x)` function — see `run_user_defined_problem.py`.

## Setup

```bash
pip install -e "..[gpflow]"          # GP backend (from the package root)
# For the Müller examples (CS2/CS3) you also need Pyomo + an ipopt solver:
pip install -e "..[gpflow,muller]"   # and: conda install -c conda-forge ipopt
```

Run from *this* `examples/` directory (the per-case-study scripts import the shared
`common.py` helper):

```bash
cd examples
python run_simple_linear.py
```

Each example uses small budgets (≈10 BO iterations, reduced GP re-trainings) so it finishes
in ~1–2 minutes — these demonstrate the API, they are not the paper's full study. Raise
`iters` / `retrain_gp` / `reopt_obj` (see `common.run_case_study`) for better convergence.

## Scripts → paper case studies (Table 2)

| Script | CS | Paper name | Notes |
|--------|----|-----------|-------|
| `run_simple_linear.py` | 1 | Simple Linear | **Annotated** — full recipe written out inline (start here) |
| `run_muller_x0.py` | 2 | Müller x0 | needs `muller` extra + ipopt |
| `run_muller_y0.py` | 3 | Müller y0 | needs `muller` extra + ipopt |
| `run_large_linear.py` | 10 | Large Linear | 5 parameters (highest-dimensional) |
| `run_bod_curve.py` | 11 | BOD Curve | |
| `run_yield_loss.py` | 12 | Yield-Loss | large-magnitude objective |
| `run_log_logistic.py` | 13 | Log Logistic | |
| `run_2d_log_logistic.py` | 14 | 2D Log Logistic | |
| `run_simple_multimodal.py` | 15 | Simple Multimodal | multiple local minima |
| `run_water_glycerol_vle.py` | 16 | Water-Glycerol VLE | fixed experimental x-grid |
| `run_acn_water_vle.py` | 17 | ACN-Water VLE | fixed experimental x-grid |
| `run_user_defined_problem.py` | — | — | **Bring your own model** via `CalibrationProblem` (synthetic mode) |
| `run_real_calibration.py` | — | — | **Real calibration**: fit a model to measured `(x, y)`, no ground truth |
| `run_external_toy.py` | — | — | **External simulator**: calibrate a separate program (subprocess) via `emcal.external`; runs are stored and reused |
| `inspect_gp_fit.py` | 1 | — | **GP diagnostics**: accuracy (R²/RMSE/MAE/MAPE) + uncertainty calibration (coverage, NLPD, reduced χ²) on held-out + cross-validation, with parity/calibration/residual plots — before running BO |
| `analyze_and_plot.py` | 1 | — | run → save → analyze via `JobContext` → plot (no signac) |

## Methods

The examples default to **method 7 (E[SSE])**, an emulator acquisition recommended in the
paper (analytic; no Tasmanian needed). All seven methods are available — see
`common.METHODS`. Method 5 (Sparse Grid) additionally needs the `sparsegrid` extra
(Tasmanian). Pass a different `method_val` to `run_case_study` to try others.

## Note

The paper's full study (parameter sweeps, cluster submission, cross-method and
derivative-free benchmarking, and all figures) used signac on an HPC cluster; that
reproduction code is **not** part of the installed package (see `refactor_notes.md`).
These examples cover what a typical user needs: running the algorithm and inspecting results.
