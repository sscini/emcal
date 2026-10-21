#!/usr/bin/env python3
"""Example: calibrate an EXTERNAL simulator, i.e. a separate program run as a subprocess.

The "simulator" is emcal's toy script (y = a*x + k*x**2 + x**3), but the workflow is the
same for any program that reads input files and writes output files:

1. Wrap it as an ExternalModel (write_inputs / command / parse_outputs / x_grid).
2. Describe the calibrated parameters with a ParameterSpace (k is searched in log10).
3. Pass an ExternalModelAdapter as the CalibrationProblem's model.

Every run gets its own folder under ./toy_store. Re-running this script reuses the
stored simulations instead of launching them again.

Run:  python run_external_toy.py
Needs: a GP backend extra ('gpytorch' or 'gpflow').
"""
import numpy as np

from emcal import (BOConfig, CalibrationProblem, EpSchedule, ExplorationBias, GenMethod,
                   GPBODriver, GPBOMethod, Kernel, MethodName, make_case_study_simulator)
from emcal.external import ExternalModelAdapter, LocalExecutor
from emcal.external.toy import TOY_TRUTH, ToyModel, toy_space


def main(store="toy_store", backend=None):
    # --- 1. Wrap the simulator; each run is a separate Python process.
    space = toy_space()
    adapter = ExternalModelAdapter(ToyModel(), space, store, executor=LocalExecutor(timeout=60))

    # --- 2. Synthetic benchmark: "measured" data come from the simulator at TOY_TRUTH.
    problem = CalibrationProblem(
        model=adapter,
        param_names=space.names,
        param_bounds=space.bounds,
        x_values=adapter.x_grid,
        true_params=space.to_calibration(TOY_TRUTH),
        name="Toy external",
        validate_model=False,  # skip the trial call: each call can be a full simulation
    )
    sim = make_case_study_simulator(problem, 0, None, 1)
    exp = sim.generate_experimental_data(adapter.n_x, GenMethod.MESHGRID, adapter.x_grid, 0.01)

    # --- 3. Initial design: 10 parameter sets -> 10 simulator runs (one per set, not per x).
    method = GPBOMethod(MethodName.EXPECTED_SSE)
    sim_data = sim.generate_simulation_data(10, adapter.n_x, GenMethod.LHS, GenMethod.MESHGRID,
                                            1.0, 1, False, exp.x_vals, with_noise=False)
    sse_data = sim.to_sse_data(method, sim_data, exp, 1.0, False)

    # --- 4. BO. Each iteration runs the simulator at the proposed point(s).
    ep = ExplorationBias(1, None, EpSchedule.CONSTANT, None, None, None, None, None, None, None)
    cfg = BOConfig(problem.name, kernel=Kernel.MAT_52, retrain_gp=5, reoptimize_obj=5,
                   bo_iter_tot=5, bo_run_tot=1, compute_y_sse=True)
    driver = GPBODriver(cfg, method, sim, exp, sim_data, sse_data, None, None, None, ep,
                        GenMethod.LHS, backend=backend)
    results, _ = driver.run(job=None)

    final = results[0].results_df.tail(1)
    best = space.to_physical(np.asarray(final["theta_best_actual"].iloc[0], dtype=float))
    print(f"simulator launches this session: {adapter.n_runs} (stored runs in {store}/)")
    print(f"best SSE = {final['best_sse_actual'].iloc[0]:.4g}")
    print(f"best a = {best['a']:.4f} (true {TOY_TRUTH['a']}), k = {best['k']:.4f} (true {TOY_TRUTH['k']})")
    return results


if __name__ == "__main__":
    try:
        import gpytorch  # noqa: F401
        chosen = "gpytorch"
    except ImportError:
        chosen = None  # gpflow
    main(backend=chosen)
