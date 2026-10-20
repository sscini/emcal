"""Per-run analysis.py + plotting.py coverage, via a pytest port of
devtools/verify_analysis.py's build_fixture()/JobContext pattern (signac-free).

Runs BO via FakeGPBackend (no real GP training), so this is fast enough for the
"not slow" tier -- unlike the real-GP version this replaced, which took ~1 minute.
devtools/verify_analysis.py (the real-GP numerical net) is unchanged and still the
regression gate for exact values; these tests assert structure/shape/no-crash instead.
"""
import gzip
import json
import os
import pathlib
import pickle
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from emcal import (
    MethodName, EpSchedule, Kernel, GenMethod, GPBOMethod, ExplorationBias,
    BOConfig, GPBODriver, get_case_study, make_case_study_simulator,
)
from emcal.analysis import JobContext, RunAnalysis
from emcal.data import Data
from emcal.plotting import Plotter

from _fakes import FakeGPBackend, sum_features


def _build_job_ctx(tmp_path_factory, method_val, backend=None):
    # Mirrors devtools/verify_analysis.py's build_fixture(): a small deterministic CS1 run,
    # saved as a JobContext workspace so the per-run analysis/plotting methods (which read a
    # job's result files + statepoint) can be exercised without signac.
    method = GPBOMethod(MethodName(method_val))
    problem = get_case_study(1)
    sim = make_case_study_simulator(problem, 0, None, 1)
    exp = sim.generate_experimental_data(5, GenMethod(2), None, 0.01)
    ep = ExplorationBias(1, None, EpSchedule(1), None, None, None, None, None, None, None)
    simd = sim.generate_simulation_data(
        20, 5, GenMethod(1), GenMethod(2), 1.0, 1, False, None, with_noise=False
    )
    ssed = sim.to_sse_data(method, simd, exp, 1.0, False)
    cfg = BOConfig(problem.name, 1, 1.0, True, Kernel(1), None, None,
                    3, 3, False, 3, 1, False, None, 1, 1e-7, 1e-7, True, False)
    if backend is None:
        # A theta/x-dependent mean (not a flat constant): plot_gp_fit's heat-map contourf
        # used to degenerate (vmin == vmax) on a perfectly flat surface -- fixed in
        # plotting.py, and test_plot_gp_fit_constant_surface_does_not_crash below regression-
        # tests that fix directly with a genuinely constant mean_fn=None backend.
        backend = FakeGPBackend(mean_value=1.0, variance_value=0.25, mean_fn=sum_features)
    drv = GPBODriver(cfg, method, sim, exp, simd, ssed, None, None, None, ep, GenMethod(1),
                      backend=backend)
    res_simple, res_gp = drv.run(job=None)

    ws = tmp_path_factory.mktemp(f"gpbo_analysis_m{method_val}")
    with gzip.open(ws / "BO_Results.gz", "wb") as f:
        pickle.dump(res_simple, f)
    with gzip.open(ws / "BO_Results_GPs.gz", "wb") as f:
        pickle.dump(res_gp, f)
    # Statepoint keys the per-run analysis methods read; bo_runs_in_job/bo_iter_tot mirror the
    # fixture's bo_run_tot=1/bo_iter_tot=3 above, and the ep0.../ei_tol block mirrors the
    # BOConfig exactly so __rebuild_cs (gp_heat_map_data) reproduces the saved run.
    sp = {"cs_name_val": 1, "meth_name_val": method_val, "ep_enum_val": 1, "bo_run_num": 1,
          "bo_run_tot": 1, "kernel_enum_val": 1, "gen_meth_theta": 1,
          "bo_runs_in_job": 1, "bo_iter_tot": 3, "with_noise": False,
          "ep0": 1, "sep_fact": 1.0, "normalize": True, "lengthscale": None, "outputscale": None,
          "retrain_gp": 3, "reoptimize_obj": 3, "gen_heat_map_data": False,
          "seed": 1, "obj_tol": 1e-7, "ei_tol": 1e-7}
    with open(ws / "signac_statepoint.json", "w") as f:
        json.dump(sp, f)
    return JobContext(str(ws), sp, job_id=f"fix_m{method_val}")


@pytest.fixture(scope="module")
def job_ctx(tmp_path_factory):
    """Method 7 (E[SSE]): the main fixture used by most per-run/plotting checks."""
    return _build_job_ctx(tmp_path_factory, method_val=7)


@pytest.fixture(scope="module")
def job_ctx_method1(tmp_path_factory):
    """Method 1 (ObjectiveGP, is_emulator=False): guards gp_heat_map_data's non-emulator
    EI branch (the 7C-a guard template)."""
    return _build_job_ctx(tmp_path_factory, method_val=1)


@pytest.fixture(scope="module")
def job_ctx_method6(tmp_path_factory):
    """Method 6 (Monte Carlo): guards gp_heat_map_data's per-theta sparse-grid/MC EI loop."""
    return _build_job_ctx(tmp_path_factory, method_val=6)


@pytest.fixture(scope="module")
def job_ctx_flat_surface(tmp_path_factory):
    """A genuinely constant-mean backend (no mean_fn): regression fixture for the
    plot_gp_fit near-flat-surface crash -- every heat-map z value is bit-identical."""
    return _build_job_ctx(
        tmp_path_factory, method_val=7,
        backend=FakeGPBackend(mean_value=1.0, variance_value=0.25),
    )


@pytest.fixture(scope="module")
def job_ctx_plot_params(tmp_path_factory, job_ctx):
    # plot_parameters -> parameter_trajectories -> __preprocess_analyze reads
    # sp_data["num_theta_multiplier"] from the workspace FILE, not job.sp -- so the augmented
    # key needs its own workspace dir (a copy of the same result files) rather than mutating
    # job_ctx's statepoint, which other tests depend on staying as build_fixture wrote it.
    # Copy, not symlink: symlinks need admin rights on Windows.
    ws = tmp_path_factory.mktemp("gpbo_analysis_plotparams")
    for fname in ("BO_Results.gz", "BO_Results_GPs.gz"):
        shutil.copyfile(os.path.join(job_ctx.workspace_dir, fname), str(ws / fname))
    sp = dict(job_ctx.sp)
    sp["num_theta_multiplier"] = 10
    with open(ws / "signac_statepoint.json", "w") as f:
        json.dump(sp, f)
    return JobContext(str(ws), sp, job_id="fix_plotparams")


@pytest.fixture(scope="module")
def ga():
    return RunAnalysis({"cs_name_val": 1, "meth_name_val": 7}, project=None,
                             mode="act", save_csv=False)


@pytest.fixture(scope="module")
def plotter(ga):
    return Plotter(ga, save_figs=False)


def _capture_shown_figure(plot_fn):
    """Call a Plotter method that ends in plt.show()/plt.close() (save_figs=False path) and
    return the Figure it produced, captured just before it would be closed."""
    captured = {}
    orig_close = plt.close

    def _capture(*a, **kw):
        captured["fig"] = plt.gcf()

    plt.close = _capture
    try:
        plot_fn()
    finally:
        plt.close = orig_close
    fig = captured.get("fig")
    if fig is not None:
        orig_close(fig)
    return fig


# --- per-run analysis methods (RunAnalysis) -----------------------------------------

def test_get_run_dataframe_has_run_metadata_and_true_params(ga, job_ctx):
    df, (theta_true, theta_true_bnds) = ga.get_run_dataframe(job_ctx)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3  # bo_iter_tot=3, bo_run_tot=1
    for col in ("best_error", "run_number", "job_id", "cs_name_val", "cs_name"):
        assert col in df.columns
    assert theta_true == {"theta_1": 1.0, "theta_2": -1.0}


def test_best_error_returns_finite_error_and_theta(ga, job_ctx):
    be, be_theta = ga.best_error(job_ctx)

    assert be.shape == (1,)
    assert np.isfinite(be[0]) and be[0] >= 0
    assert be_theta.shape == (1, 2)
    assert np.all(np.isfinite(be_theta))


def test_hyperparameter_trajectories_shape(ga, job_ctx):
    data, data_names, data_true, sp_data = ga.hyperparameter_trajectories(job_ctx)

    assert data.shape == (1, 3, len(data_names))  # 1 run x 3 iters x n_hyperparams
    assert data_true is None
    assert sp_data["meth_name_val"] == 7


def test_objective_trajectories_all_z_choices(ga, job_ctx):
    data, data_names, data_true, sp_data, data_true_med = ga.objective_trajectories(
        job_ctx, ["min_sse", "sse", "acq"]
    )

    assert data.shape == (1, 3, 3)
    assert len(data_names) == 3
    assert set(data_true.keys()) == {"min_sse", "sse", "acq"}


def test_parameter_trajectories_matches_true_theta(ga, job_ctx):
    data, data_names, data_true, sp_data = ga.parameter_trajectories(job_ctx, "min_sse")

    assert data.shape == (1, 3, 2)
    assert data_true == {"theta_1": 1.0, "theta_2": -1.0}


def test_gp_parity_data_attaches_gp_predictions(ga, job_ctx):
    test_data = ga.gp_parity_data(job_ctx, 1, 1)

    assert isinstance(test_data, Data)
    assert test_data.gp_mean is not None
    assert test_data.gp_var is not None
    assert test_data.gp_covar is not None
    assert np.all(np.isfinite(test_data.gp_mean))


def test_gp_heat_map_data_without_ei(ga, job_ctx):
    sim_sse_var_ei, test_mesh, param_info_dict, sp_data = ga.gp_heat_map_data(
        job_ctx, 1, 1, 0
    )

    assert len(sim_sse_var_ei) == 4
    assert sim_sse_var_ei[3] is None  # ei not requested
    xx, yy = test_mesh
    assert xx.shape == yy.shape
    for key in ("true", "min_sse", "opt_acq", "train", "names", "idcs"):
        assert key in param_info_dict


def test_gp_heat_map_data_with_ei(ga, job_ctx):
    sim_sse_var_ei, test_mesh, param_info_dict, sp_data = ga.gp_heat_map_data(
        job_ctx, 1, 1, 0, get_ei=True
    )
    assert sim_sse_var_ei[3] is not None
    assert np.all(np.isfinite(sim_sse_var_ei[3]))


def test_gp_heat_map_data_ei_method1_objective_gp(ga, job_ctx_method1):
    # Method 1 (ObjectiveGP, is_emulator=False) exercises gp_heat_map_data's
    # `elif method.is_emulator == False` acquisition branch.
    sim_sse_var_ei, _, _, sp_data = ga.gp_heat_map_data(job_ctx_method1, 1, 1, 0, get_ei=True)
    assert sp_data["meth_name_val"] == 1
    assert np.all(np.isfinite(sim_sse_var_ei[3]))


def test_gp_heat_map_data_ei_method6_monte_carlo(ga, job_ctx_method6):
    # Method 6 (Monte Carlo) exercises the per-theta sparse-grid/MC EI loop.
    sim_sse_var_ei, _, _, sp_data = ga.gp_heat_map_data(job_ctx_method6, 1, 1, 0, get_ei=True)
    assert sp_data["meth_name_val"] == 6
    assert np.all(np.isfinite(sim_sse_var_ei[3]))


# --- plotting methods (Plotter), Agg backend, show/close path ---------------------------

def test_plot_hyperparameters(plotter, job_ctx):
    fig = _capture_shown_figure(lambda: plotter.plot_hyperparameters(job_ctx))
    assert fig is not None
    assert len(fig.axes) > 0


def test_plot_parameters(plotter, job_ctx_plot_params):
    fig = _capture_shown_figure(
        lambda: plotter.plot_parameters(job_ctx_plot_params, "min_sse")
    )
    assert fig is not None
    assert len(fig.axes) > 0


def test_plot_gp_fit_sse_choices(plotter, job_ctx):
    fig = _capture_shown_figure(
        lambda: plotter.plot_gp_fit(job_ctx, 1, 1, 0, ["sse_sim", "sse_mean"])
    )
    assert fig is not None
    assert len(fig.axes) > 0


def test_plot_gp_fit_with_acquisition(plotter, job_ctx):
    # "acq" alongside another z_choice exercises gp_heat_map_data's get_ei=True dispatch
    # from inside plot_gp_fit.
    fig = _capture_shown_figure(
        lambda: plotter.plot_gp_fit(job_ctx, 1, 1, 0, ["sse_mean", "acq"])
    )
    assert fig is not None
    assert len(fig.axes) > 0


@pytest.mark.parametrize("z_choice", ["sse_sim", "sse_mean", "sse_var", "acq"])
def test_plot_gp_fit_single_z_choice(plotter, job_ctx, z_choice):
    # Regression test: __get_z_plot_names_hms used to unwrap its list return values to bare
    # scalars/strings when given a single z_choice, so plot_gp_fit's `all_z_data[i]` indexed
    # into the ndarray itself (a 1D row) instead of getting the whole 2D array, and
    # matplotlib's contourf raised "Input z must be 2D, not 1D". Fixed by always returning
    # lists from __get_z_plot_names_hms, regardless of how many z_choices were requested.
    fig = _capture_shown_figure(lambda: plotter.plot_gp_fit(job_ctx, 1, 1, 0, [z_choice]))
    assert fig is not None
    assert len(fig.axes) > 0


@pytest.mark.parametrize("z_choices", [
    ["sse_sim"], ["sse_mean"], ["sse_var"], ["acq"], ["sse_sim", "sse_mean"],
])
def test_plot_gp_fit_constant_surface_does_not_crash(plotter, job_ctx_flat_surface, z_choices):
    # Regression test: a genuinely flat GP predictive surface (vmin == vmax exactly, bit-
    # identical everywhere) used to crash contourf with "Contour levels must be increasing".
    # The existing vmin/vmax-nudge guard used a fixed 1e-14 absolute epsilon, which is a
    # no-op at any magnitude where 1e-14 is below the value's own float64 precision (e.g.
    # vmin=vmax=337: 337 - 1e-14 rounds right back to 337) -- fixed by scaling the nudge to
    # the data's own magnitude.
    fig = _capture_shown_figure(
        lambda: plotter.plot_gp_fit(job_ctx_flat_surface, 1, 1, 0, z_choices)
    )
    assert fig is not None
    assert len(fig.axes) > 0


def test_plot_gp_fit_log_data(plotter, job_ctx):
    fig = _capture_shown_figure(
        lambda: plotter.plot_gp_fit(job_ctx, 1, 1, 0, ["sse_var", "sse_mean"], log_data=True)
    )
    assert fig is not None
    assert len(fig.axes) > 0


# --- save_figs=True path (__save_fig) -----------------------------------------------------

def test_plot_hyperparameters_save_figs_writes_under_job_workspace(ga, job_ctx):
    plotter_save = Plotter(ga, save_figs=True)
    plotter_save.plot_hyperparameters(job_ctx)

    # Scoped to plot_hyperparameters' own subdirectory: job_ctx is module-scoped and shared
    # with the plot_gp_fit save_figs test below, which writes into a sibling "heat_maps" dir.
    saved = list(pathlib.Path(job_ctx.workspace_dir, "line_plots").rglob("*.png"))
    assert len(saved) == 1


def test_plot_gp_fit_save_figs_writes_under_job_workspace(ga, job_ctx, monkeypatch, tmp_path):
    # Regression test: plot_gp_fit's save path used to be built from
    # make_dir_name_from_criteria (cwd-relative), unlike plot_hyperparameters/plot_parameters
    # which both correctly use job.fn("") -- so save_figs=True wrote a "results/" directory
    # into whatever the cwd happened to be (e.g. the repo root) instead of the job's own
    # workspace. Fixed to use job.fn("") like its siblings. chdir into a throwaway tmp_path
    # anyway, purely as a belt-and-suspenders guard against a future regression polluting cwd.
    monkeypatch.chdir(tmp_path)
    plotter_save = Plotter(ga, save_figs=True)
    plotter_save.plot_gp_fit(job_ctx, 1, 1, 0, ["sse_sim", "sse_mean"])

    assert list(tmp_path.rglob("*.png")) == []  # nothing written relative to cwd
    # Scoped to plot_gp_fit's own subdirectory: job_ctx is module-scoped and shared with the
    # plot_hyperparameters save_figs test above, which writes into a sibling "line_plots" dir.
    saved = list(pathlib.Path(job_ctx.workspace_dir, "heat_maps").rglob("*.png"))
    assert len(saved) == 1
