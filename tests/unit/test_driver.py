"""Fast orchestration tests for driver.py (GPBODriver), driven by FakeGPBackend -- a
full run(job=None) completes in ~1s (no real GP training), so the BO loop's structure
(results_df shape/columns, training-data growth, acquisition optimization, the
__min_obj_val/__min_obj_prediction companion tracking) can be exercised without the
`slow` marker.
"""
import numpy as np
import pytest

from emcal import (
    BOConfig, EpSchedule, ExplorationBias, GenMethod, GPBODriver, GPBOMethod, Kernel,
    MethodName, get_case_study, make_case_study_simulator,
)

from _fakes import FakeGPBackend

# The 23 columns GPBODriver.run() produces in results_df (get_run_dataframe in analysis.py
# adds 5 more -- ep_method_val, run_number, job_id, cs_name_val, cs_name -- for 28 total).
EXPECTED_COLUMNS = [
    "bo_iter", "best_error", "alpha", "theta_at_acq", "acq_value", "sse_at_acq",
    "mse_at_acq", "theta_at_min", "sse_gp", "sse_actual", "mse_gp", "mse_actual",
    "time_per_iter", "method", "max_evals", "theta_best_at_acq", "theta_best_gp",
    "theta_best_actual", "termination_reason", "total_time", "best_sse_gp",
    "best_sse_actual", "best_sse_at_acq",
]


def _build_driver(method_val=7, mean_value=1.0, variance_value=0.25, bo_iter_tot=3,
                   retrain_gp=1, reoptimize_obj=1, gen_heat_map_data=False, seed=1,
                   bo_run_tot=1, compute_y_sse=True):
    # compute_y_sse=True also simulates the min-SSE theta, so sse_actual is filled in.
    method = GPBOMethod(MethodName(method_val))
    problem = get_case_study(1)
    sim = make_case_study_simulator(problem, 0, None, 1)
    exp = sim.generate_experimental_data(5, GenMethod.MESHGRID, None, 0.01)
    n = len(sim.indices_to_consider)
    simd = sim.generate_simulation_data(
        10 * n, 5, GenMethod.LHS, GenMethod.MESHGRID, 1.0, 1, False, None, with_noise=False
    )
    ssed = sim.to_sse_data(method, simd, exp, 1.0, False)
    ep = ExplorationBias(1, None, EpSchedule.CONSTANT, None, None, None, None, None, None, None)
    cfg = BOConfig(problem.name, kernel=Kernel.MAT_52, retrain_gp=retrain_gp,
                    reoptimize_obj=reoptimize_obj, bo_iter_tot=bo_iter_tot,
                    bo_run_tot=bo_run_tot, gen_heat_map_data=gen_heat_map_data, seed=seed,
                    compute_y_sse=compute_y_sse)
    backend = FakeGPBackend(mean_value=mean_value, variance_value=variance_value)
    driver = GPBODriver(cfg, method, sim, exp, simd, ssed, None, None, None, ep,
                         GenMethod.LHS, backend=backend)
    return driver, backend


def _cs1_sse(driver, theta):
    """SSE of the CS1 (Simple Linear) model at theta against the driver's experimental data."""
    x = driver.exp_data.x_vals.reshape(-1)
    y = theta[0] * x + theta[1] * x**2 + x**3
    return float(np.sum((y - driver.exp_data.y_vals) ** 2))


# --- run(job=None): end-to-end structure, via FakeGPBackend (no real GP training) ----

# Regression: the driver used to report the SSE of the min-SSE theta as sse_at_acq for
# emulator methods, and crashed when compute_y_sse=False (that theta has no y values).
@pytest.mark.parametrize("compute_y_sse", [True, False])
@pytest.mark.parametrize("method_val", [3, 7])  # Independence EI and E[SSE] (both emulators)
def test_sse_at_acq_is_the_sse_at_the_acquisition_theta(method_val, compute_y_sse):
    driver, _ = _build_driver(method_val=method_val, bo_iter_tot=2, compute_y_sse=compute_y_sse)
    res_simple, _ = driver.run(job=None)
    df = res_simple[0].results_df

    assert len(df) == 2
    for i in range(len(df)):
        theta_acq = np.asarray(df["theta_at_acq"].iloc[i], dtype=float)
        assert df["sse_at_acq"].iloc[i] == pytest.approx(_cs1_sse(driver, theta_acq))
    if not compute_y_sse:
        assert df["sse_actual"].isna().all()


def test_run_produces_expected_shape_and_columns():
    driver, _ = _build_driver(method_val=7, bo_iter_tot=3)
    res_simple, res_gp = driver.run(job=None)
    df = res_simple[0].results_df

    assert df.shape == (3, len(EXPECTED_COLUMNS))
    assert list(df.columns) == EXPECTED_COLUMNS


def test_run_terminates_at_max_budget_with_a_flat_objective():
    # bo_iter_tot < 4 means the acq/obj-improvement termination flags (which need i>=4)
    # can never fire first, so this always terminates via the budget check -- deterministic
    # regardless of the (flat, constant-mean) fake GP's acquisition behavior.
    driver, _ = _build_driver(method_val=7, bo_iter_tot=3)
    res_simple, _ = driver.run(job=None)

    assert res_simple[0].why_term == "max_budget"
    assert (res_simple[0].results_df["termination_reason"] == "max_budget").all()


@pytest.mark.parametrize("method_val", [1, 7])  # ObjectiveGP (Type 1) and EmulatorGP (Type 2)
def test_run_grows_training_data_every_iteration(method_val):
    driver, _ = _build_driver(method_val=method_val, bo_iter_tot=3)
    _, res_gp = driver.run(job=None)

    sizes = [res_gp[0].list_gp_emulator_class[i].train_data.n_theta for i in range(3)]
    assert sizes[0] < sizes[1] < sizes[2]
    # Growth per iteration is n_x (EmulatorGP repeats theta over x) or 1 (ObjectiveGP).
    step = sizes[1] - sizes[0]
    assert sizes[2] - sizes[1] == step
    assert step == (5 if method_val == 7 else 1)  # n_x=5 for the CS1 experimental grid


def test_run_trains_a_fresh_emulator_each_restart():
    """Seam #1 guard (PHASE8_AUDIT.md §3.8, risk #1): each restart in run() must build and
    split a genuinely fresh gp_emulator, never reuse a reference held over from the previous
    restart. A single-restart run/golden config can't exercise this at all -- the
    decomposition's riskiest failure mode (a collaborator caching gp_emulator at
    construction instead of re-syncing it every restart) only manifests starting on the
    *second* restart. Pin this now, on the pre-decomposition driver, so Phase 8.3-c's
    AcquisitionOptimizer extraction has a net that would catch a stale-reference
    regression that golden (single-restart configs) would silently miss.
    """
    driver, _ = _build_driver(method_val=7, bo_iter_tot=3, bo_run_tot=2)
    _, res_gp = driver.run(job=None)

    assert len(res_gp) == 2
    run0_first = res_gp[0].list_gp_emulator_class[0]
    run0_last = res_gp[0].list_gp_emulator_class[-1]
    run1_first = res_gp[1].list_gp_emulator_class[0]

    # Restart 2 built a genuinely new emulator object...
    assert run1_first is not run0_last
    assert run1_first is not run0_first
    # ...freshly split, so its training set starts back at the initial split size instead
    # of continuing to grow from wherever restart 1 left off (which is what silently
    # reusing a stale reference would produce).
    assert run1_first.train_data.n_theta == run0_first.train_data.n_theta
    assert run1_first.train_data.n_theta < run0_last.train_data.n_theta


def test_run_calls_backend_train_exactly_bo_iter_tot_times_retrain_gp():
    driver, backend = _build_driver(bo_iter_tot=3, retrain_gp=2)
    driver.run(job=None)
    assert len(backend.train_calls) == 3 * 2


def test_run_columns_are_identical_for_objective_and_emulator_gp():
    driver1, _ = _build_driver(method_val=1, bo_iter_tot=3)
    driver7, _ = _build_driver(method_val=7, bo_iter_tot=3)
    res1, _ = driver1.run(job=None)
    res7, _ = driver7.run(job=None)
    assert list(res1[0].results_df.columns) == list(res7[0].results_df.columns)


def test_run_best_sse_columns_are_cumulative_minimums():
    driver, _ = _build_driver(method_val=7, bo_iter_tot=3)
    res_simple, _ = driver.run(job=None)
    df = res_simple[0].results_df

    assert np.array_equal(df["best_sse_gp"].to_numpy(), df["sse_gp"].cummin().to_numpy())
    assert np.array_equal(
        df["best_sse_actual"].to_numpy(), df["sse_actual"].cummin().to_numpy()
    )


# --- __opt_with_scipy / __scipy_fxn: direct orchestration checks ---------------------

def _prep_for_opt(driver):
    """Replicates the __run_bo_iter setup __opt_with_scipy needs: a fitted emulator and
    a set of acquisition-optimizer starting points."""
    driver.reset_rng()
    driver.gp_emulator = driver._GPBODriver__gen_emulator()
    driver.gp_emulator.split_train_test(driver.cs_params.sep_fact, driver.cs_params.seed)
    driver.gp_emulator.fit()
    be_data, best_error_metrics = driver._GPBODriver__get_best_error()
    driver.ep_bias.update()  # sets ep_curr; expected_improvement needs it non-None
    driver.opt_start_pts = driver._GPBODriver__make_starting_opt_pts(best_error_metrics, None)
    return best_error_metrics


def test_opt_with_scipy_sse_sets_min_obj_companions_and_augments_train_data():
    driver, _ = _build_driver(method_val=7, bo_iter_tot=3, reoptimize_obj=2)
    _prep_for_opt(driver)
    n_before = driver.gp_emulator.train_data.n_theta

    best_val, best_class, best_prediction = driver._GPBODriver__opt_with_scipy(
        "sse", get_y=True
    )

    assert best_val is driver.acq_optimizer.min_obj_val
    assert best_class is driver.acq_optimizer.min_obj_class
    assert best_prediction is driver.acq_optimizer.min_obj_prediction
    assert np.all(np.isfinite(best_prediction.mean))

    driver._GPBODriver__augment_train_data(best_class)
    assert driver.gp_emulator.train_data.n_theta == n_before + 5  # n_x=5, EmulatorGP


def test_opt_with_scipy_neg_ei_returns_the_raw_unnegated_ei():
    # Method 7 (E[SSE]) never optimizes "neg_ei" (only "E_sse"); use method 3
    # (INDEPENDENCE), a proper expected-improvement emulator method, instead.
    driver, _ = _build_driver(method_val=3, bo_iter_tot=3)
    _prep_for_opt(driver)

    best_val, best_class, best_prediction = driver._GPBODriver__opt_with_scipy(
        "neg_ei", get_y=True
    )

    # __scipy_fxn negates ei internally for the minimizer but stores the raw ei value as
    # the companion __min_obj_val -- so this must be >= 0 (an EI), not the negated value
    # scipy actually minimized.
    assert best_val >= 0
    assert best_class.y_vals is not None  # get_y=True populated the simulated y


def test_scipy_fxn_nan_theta_returns_the_penalty_without_touching_the_gp():
    driver, _ = _build_driver(method_val=7, bo_iter_tot=3)
    best_error_metrics = _prep_for_opt(driver)
    nan_theta = np.array([np.nan, np.nan])

    assert driver._GPBODriver__scipy_fxn(nan_theta, "sse", best_error_metrics) == driver.sse_penalty
    assert driver._GPBODriver__scipy_fxn(nan_theta, "E_sse", best_error_metrics) == driver.sse_penalty
    assert driver._GPBODriver__scipy_fxn(nan_theta, "neg_ei", best_error_metrics) == 1


def test_augment_train_data_grows_by_one_for_objective_gp():
    driver, _ = _build_driver(method_val=1, bo_iter_tot=3, reoptimize_obj=2)
    _prep_for_opt(driver)
    n_before = driver.gp_emulator.train_data.n_theta

    _, best_class, _ = driver._GPBODriver__opt_with_scipy("sse", get_y=True)
    driver._GPBODriver__augment_train_data(best_class)

    assert driver.gp_emulator.train_data.n_theta == n_before + 1


def test_get_best_error_matches_gp_emulator_calc_best_error():
    driver, _ = _build_driver(method_val=1, bo_iter_tot=3)
    driver.reset_rng()
    driver.gp_emulator = driver._GPBODriver__gen_emulator()
    driver.gp_emulator.split_train_test(driver.cs_params.sep_fact, driver.cs_params.seed)
    driver.gp_emulator.fit()

    be_data, best_error_metrics = driver._GPBODriver__get_best_error()
    direct_best_error, direct_be_theta, _ = driver.gp_emulator.calc_best_error()

    assert best_error_metrics[0] == direct_best_error
    assert np.array_equal(best_error_metrics[1], direct_be_theta)
    assert best_error_metrics[2] is None  # Type 1: no per-x squared-error breakdown
