# emcal

Gaussian-process Bayesian optimization (GPBO) methods for nonlinear model calibration —
the library accompanying Carlozo, Wang & Dowling, *"Bayesian Optimization Methods for
Nonlinear Model Calibration"* (Ind. Eng. Chem. Res. 2025).

## Install (development)

```bash
pip install -e ".[gpflow,sparsegrid,muller,dev]"
```

The core algorithm requires the `gpflow` and `sparsegrid` extras; the Müller case
studies (CS2/CS3) also need `muller` (plus an `ipopt` solver on PATH). `import
emcal` itself pulls in no heavy dependencies — gpflow/TensorFlow load only when a
GP model is built. See `devtools/conda-envs/gpbo-dev-SETUP.md` for the validated
environment.

### GP backends

- **`gpflow`** (default): gpflow on TensorFlow.
- **`gpytorch`**: the same model on torch + GPyTorch. It needs no TensorFlow, and numpy 2 works.

```bash
pip install -e ".[gpytorch,dev]"
```

```python
driver = GPBODriver(config, method, sim, exp, sim_data, sse_data,
                    None, None, None, ep, GenMethod.LHS, backend="gpytorch")
emulator = diagnostics.fit_gp(method, sim, exp, sim_data, sse_data, config, backend="gpytorch")
```

At equal hyperparameters, the two backends give the same predictions. Trained runs
differ slightly, so the golden baselines are gpflow-only. gpytorch keeps
hyperparameters within emcal's bounds, while gpflow only enforces positivity.
`GPyTorchBackend(bounded=False)` behaves like gpflow.

## Quick start — a built-in case study

```python
from emcal import (
    MethodName, EpSchedule, Kernel, GenMethod,
    GPBOMethod, ExplorationBias, BOConfig, GPBODriver,
    get_case_study, make_case_study_simulator,
)

problem = get_case_study(1)                       # CS1 "Simple Linear" (a CalibrationProblem)
method = GPBOMethod(MethodName.EXPECTED_SSE)      # emulator acquisition (analytic; method 7)

sim = make_case_study_simulator(problem, 0, None, 1)
exp = sim.generate_experimental_data(5, GenMethod.MESHGRID, None, 0.01)
n = len(sim.indices_to_consider)
sim_data = sim.generate_simulation_data(
    10 * n, 5, GenMethod.LHS, GenMethod.MESHGRID, 1.0, 1, False, None, with_noise=False
)
sse_data = sim.to_sse_data(method, sim_data, exp, 1.0, False)

ep = ExplorationBias(1, None, EpSchedule.CONSTANT, None, None, None, None, None, None, None)
config = BOConfig(problem.name, kernel=Kernel.MAT_52, retrain_gp=10, reoptimize_obj=10,
                  bo_iter_tot=10, bo_run_tot=1)

driver = GPBODriver(config, method, sim, exp, sim_data, sse_data,
                    None, None, None, ep, GenMethod.LHS)
results_simple, results_gp = driver.run(job=None)   # job=None => signac-free, in-memory
best = results_simple[0].results_df.tail(1)["Min Obj Act Cum"].iloc[0]
print("best SSE:", best)
```

See `examples/` for all 11 case studies as runnable scripts (start with the annotated
`examples/run_simple_linear.py`).

## Calibrate your own model

Define a `CalibrationProblem` around a `model(theta, x)` function instead of using a
built-in (see `examples/run_user_defined_problem.py`):

```python
from emcal import CalibrationProblem, make_case_study_simulator
import numpy as np

def my_model(theta, x):
    return theta[0] * np.sin(x) + theta[1] * x

problem = CalibrationProblem(
    model=my_model,
    param_names=["amp", "slope"],
    param_bounds=[(-5.0, 5.0), (-5.0, 5.0)],
    x_bounds=[(-3.0, 3.0)],
    true_params=np.array([2.0, -0.5]),   # synthetic-benchmark mode
)
sim = make_case_study_simulator(problem, 0, None, 1)
# ... same recipe as above.
```

`CalibrationProblem` validates its inputs eagerly (parameter/bounds lengths, bound
ordering, `true_params` in-bounds, and a trial `model(theta, x)` call). **Real calibration**
against measured data — pass `experimental_data=(x, y)` and omit `true_params` — is
expressed the same way and runs end-to-end (see `examples/run_real_calibration.py`, which
recovers parameters from noisy measurements with no ground truth).

## Methods (paper Table 1)

`MethodName`: `CONVENTIONAL`, `LOG_CONVENTIONAL` (standard/objective GP); `INDEPENDENCE`,
`LOG_INDEPENDENCE`, `SPARSE_GRID`, `MONTE_CARLO`, `EXPECTED_SSE` (emulator GP). The
sparse-grid method needs the `sparsegrid` extra (Tasmanian).

## GP diagnostics — check the emulator before running BO

Emulator GPBO only works well when the GP is good — accurate **and** with trustworthy
uncertainty. `emcal.diagnostics` reports both, using GP best practices, and is
designed to be easy to run before committing to a full BO study:

```python
from emcal import diagnostics

emulator = diagnostics.fit_gp(method, sim, exp, sim_data, sse_data, config, sep_fact=0.8)
report = diagnostics.evaluate_gp(emulator)      # held-out split
print(report.summary())                          # accuracy + calibration verdict
cv = diagnostics.cross_validate_gp(method, sim, exp, sim_data, sse_data, config,
                                    holdout_frac=0.2, n_repeats=5)
report.plot_all()                                # parity | calibration | residual-QQ
```

`GPDiagnostics` covers accuracy (R²/RMSE/MAE/MAPE) and uncertainty calibration —
standardized residuals, reduced χ², interval coverage (50/90/95%), NLPD, CRPS, sharpness,
and miscalibration area — on a held-out split and via cross-validation. See
`examples/inspect_gp_fit.py`.

## Analysis & plotting (signac-free)

`emcal.analysis` and `emcal.plotting` provide per-run analysis and
diagnostic plots that work without signac, via a `JobContext` handle over a results
directory (see `examples/analyze_and_plot.py`).

## Scope

This package is the **algorithm plus a general user's needs**. The paper's full
reproduction workflow (signac parameter sweeps, HPC cluster submission, cross-method and
derivative-free benchmarking, and all figures) is **not** part of the installed package;
it lives with the archived research repository.

## Citation

> Carlozo, M., Wang, ..., Dowling, A. W. "Bayesian Optimization Methods for Nonlinear
> Model Calibration." *Industrial & Engineering Chemistry Research*, 2025.

## License

BSD-3-Clause
