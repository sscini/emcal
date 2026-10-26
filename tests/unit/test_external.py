"""External-model layer: parameters, store, executor, adapter, and the BO loop driving a
real subprocess (the toy simulator). Toy runs take ~1 s each, so designs here are small."""
import copy
import json
import pickle
import sys

import numpy as np
import pytest

from emcal import (BOConfig, CalibrationProblem, EpSchedule, ExplorationBias, GenMethod,
                   GPBODriver, GPBOMethod, Kernel, MethodName, make_case_study_simulator)
from emcal.external import (EvaluationStore, ExternalModelAdapter, LocalExecutor, Parameter,
                            ParameterSpace, SimulationFailed)
from emcal.external.toy import TOY_TRUTH, ToyModel, toy_space

from _fakes import FakeGPBackend


def _toy_y(a, k, x):
    return a * x + k * x**2 + x**3


@pytest.fixture
def adapter(tmp_path):
    return ExternalModelAdapter(ToyModel(), toy_space(), tmp_path / "store")


# ------------------------------------------------------------------ ParameterSpace
def test_parameter_rejects_bad_bounds():
    with pytest.raises(AssertionError):
        Parameter("a", (1.0, 1.0))


def test_space_log10_round_trip_and_fixed_values():
    space = ParameterSpace([Parameter("a", (-2, 2)), Parameter("k", (-3, 1), log10=True)],
                           fixed={"c": 7.0})
    phys = space.to_physical([0.5, -1.0])
    assert phys == {"a": 0.5, "k": pytest.approx(0.1), "c": 7.0}
    np.testing.assert_allclose(space.to_calibration(phys), [0.5, -1.0])
    assert space.names == ["a", "k"] and space.bounds == [(-2, 2), (-3, 1)]


def test_space_rejects_duplicate_or_overlapping_names():
    with pytest.raises(AssertionError):
        ParameterSpace([Parameter("a", (0, 1)), Parameter("a", (0, 1))])
    with pytest.raises(AssertionError):
        ParameterSpace([Parameter("a", (0, 1))], fixed={"a": 0.5})


# ----------------------------------------------------------------- EvaluationStore
def test_store_key_is_stable():
    k1 = EvaluationStore.key("m", "1", {"a": 1.0, "b": 2.0})
    assert k1 == EvaluationStore.key("m", "1", {"b": 2.0, "a": 1.0 + 1e-15})  # order, round-off
    assert k1 != EvaluationStore.key("m", "2", {"a": 1.0, "b": 2.0})          # version
    assert k1 != EvaluationStore.key("m", "1", {"a": 1.1, "b": 2.0})          # values


def test_store_round_trip_and_pickle(tmp_path):
    store = EvaluationStore(tmp_path)
    key = store.key("m", "1", {"a": 1.0})
    assert store.load(key) is None
    run_dir = store.prepare(key, {"a": 1.0})
    assert json.loads((run_dir / "params.json").read_text()) == {"a": 1.0}
    store.save(key, {"a": 1.0}, np.array([1.0, 2.0]), "ok", 0.5)
    record = store.load(key)
    assert record["status"] == "ok"
    np.testing.assert_array_equal(record["y"], [1.0, 2.0])
    assert (tmp_path / "index.csv").read_text().count("\n") == 2  # header + 1 row
    assert pickle.loads(pickle.dumps(store)).load(key)["status"] == "ok"


def test_store_prepare_clears_an_unfinished_attempt(tmp_path):
    store = EvaluationStore(tmp_path)
    key = store.key("m", "1", {"a": 1.0})
    (store.prepare(key, {"a": 1.0}) / "stale.txt").write_text("x")
    assert not (store.prepare(key, {"a": 1.0}) / "stale.txt").exists()


# ------------------------------------------------------------------- LocalExecutor
def test_local_executor_captures_output_and_return_code(tmp_path):
    res = LocalExecutor().run([sys.executable, "-c", "print('hi'); raise SystemExit(4)"], tmp_path)
    assert res.returncode == 4 and not res.timed_out
    assert (tmp_path / "stdout.txt").read_text().strip() == "hi"


def test_local_executor_timeout(tmp_path):
    res = LocalExecutor(timeout=0.5).run([sys.executable, "-c", "import time; time.sleep(30)"], tmp_path)
    assert res.timed_out and res.returncode is None


def test_local_executor_closes_stdin(tmp_path):
    # A program that reads input must get end-of-input, not wait on the parent's stdin.
    # pytest's capture already closes fd 0; run with -s to test against a live stdin.
    code = "import sys; print(len(sys.stdin.read()))"
    res = LocalExecutor(timeout=10).run([sys.executable, "-c", code], tmp_path)
    assert not res.timed_out and res.returncode == 0
    assert (tmp_path / "stdout.txt").read_text().strip() == "0"


def test_local_executor_rejects_shell_strings_and_missing_programs(tmp_path):
    with pytest.raises(AssertionError):
        LocalExecutor().run("echo hi", tmp_path)
    with pytest.raises(FileNotFoundError):
        LocalExecutor().run(["definitely-not-a-program-xyz"], tmp_path)


# ------------------------------------------------------------------------- adapter
def test_adapter_runs_simulator_once_and_caches(adapter, tmp_path):
    theta = adapter.space.to_calibration(TOY_TRUTH)
    first = adapter.evaluate(theta)
    assert first.status == "ok" and not first.cached
    np.testing.assert_allclose(first.y, _toy_y(1.0, 0.5, adapter.x_grid[:, 0]))
    assert adapter.evaluate(theta).cached
    # every x served from the one run
    for x, y in zip(adapter.x_grid, first.y):
        assert adapter(theta, x) == pytest.approx(y)
    np.testing.assert_allclose(adapter.evaluate_all(theta, adapter.x_grid[::-1]), first.y[::-1])
    assert adapter.n_runs == 1
    # a new adapter on the same store reuses the stored run
    fresh = ExternalModelAdapter(ToyModel(), toy_space(), tmp_path / "store")
    assert fresh.evaluate(theta).cached and fresh.n_runs == 0


def test_adapter_x_not_on_grid(adapter):
    with pytest.raises(KeyError):
        adapter(adapter.space.to_calibration(TOY_TRUTH), np.array([0.123]))


def test_adapter_raise_policy_and_failed_runs_are_not_rerun(tmp_path):
    space = toy_space(crash_above_a=0.0)
    ad = ExternalModelAdapter(ToyModel(), space, tmp_path / "store")
    theta = space.to_calibration({"a": 1.0, "k": 0.5})
    with pytest.raises(SimulationFailed) as err:
        ad(theta, ad.x_grid[0])
    assert err.value.status == "failed" and "return code 3" in str(err.value)
    assert "crashing" in (err.value.run_dir / "stderr.txt").read_text()
    with pytest.raises(SimulationFailed):
        ExternalModelAdapter(ToyModel(), space, tmp_path / "store")(theta, ad.x_grid[0])
    assert ad.n_runs == 1  # the stored failure was reused
    retry = ExternalModelAdapter(ToyModel(), space, tmp_path / "store", rerun_failed=True)
    assert retry.evaluate(theta).status == "failed" and retry.n_runs == 1


def test_adapter_penalty_policy(tmp_path):
    space = toy_space(crash_above_a=0.0)
    ad = ExternalModelAdapter(ToyModel(), space, tmp_path / "store", failure="penalty", penalty=1e3)
    theta = space.to_calibration({"a": 1.0, "k": 0.5})
    np.testing.assert_array_equal(ad.evaluate_all(theta, ad.x_grid), np.full(ad.n_x, 1e3))
    with pytest.raises(AssertionError):
        ExternalModelAdapter(ToyModel(), space, tmp_path / "s2", failure="penalty")


def test_adapter_wrong_output_length_fails(tmp_path):
    class ShortToy(ToyModel):
        name = "short_toy"

        def parse_outputs(self, run_dir):
            return super().parse_outputs(run_dir)[:-1]

    ad = ExternalModelAdapter(ShortToy(), toy_space(), tmp_path / "store")
    result = ad.evaluate(ad.space.to_calibration(TOY_TRUTH))
    assert result.status == "failed" and "expected 8 outputs" in result.message


def test_evaluate_many_runs_in_parallel_and_dedupes(tmp_path):
    ad = ExternalModelAdapter(ToyModel(), toy_space(), tmp_path / "store",
                              executor=LocalExecutor(max_workers=3))
    thetas = np.array([[0.0, -1.0], [0.5, -0.5], [0.0, -1.0], [1.0, 0.0]])
    results = ad.evaluate_many(thetas)
    assert [r.status for r in results] == ["ok"] * 4
    assert ad.n_runs == 3 and results[2].cached
    np.testing.assert_allclose(results[2].y, results[0].y)


def _toy_outputs(adapter, theta):
    """Toy outputs at theta, computed here instead of by the simulator."""
    p = adapter.space.to_physical(theta)
    return _toy_y(p["a"], p["k"], adapter.x_grid[:, 0])


def test_imported_run_is_reused_without_running(adapter):
    theta = np.array([0.3, -1.2])
    y = _toy_outputs(adapter, theta)
    result = adapter.import_evaluation(theta, y, source="earlier project")
    assert result.status == "ok" and result.source == "earlier project"
    np.testing.assert_array_equal(adapter.evaluate_all(theta, adapter.x_grid), y)
    assert adapter.n_runs == 0

    # A fresh adapter on the same store sees it too
    other = ExternalModelAdapter(ToyModel(), toy_space(), adapter.store)
    assert other.evaluate(theta).source == "earlier project" and other.n_runs == 0

    # Importing the same run again is allowed; a different one needs overwrite
    adapter.import_evaluation(theta, y, source="earlier project")
    with pytest.raises(FileExistsError):
        adapter.import_evaluation(theta, y + 1.0, source="other")
    adapter.import_evaluation(theta, y + 1.0, source="other", overwrite=True)
    np.testing.assert_array_equal(other.store.load(other._key(other.space.to_physical(theta)))["y"], y + 1.0)


def test_import_rejects_wrong_outputs(adapter):
    with pytest.raises(AssertionError, match="expected 8 outputs"):
        adapter.import_evaluation([0.0, -1.0], np.zeros(7), source="x")
    with pytest.raises(AssertionError, match="finite"):
        adapter.import_evaluation([0.0, -1.0], np.full(8, np.nan), source="x")


def test_stored_design_collects_runs_for_this_model_only(tmp_path):
    store = tmp_path / "store"
    ad = ExternalModelAdapter(ToyModel(), toy_space(), store)
    ad.evaluate([0.0, -1.0])  # simulated
    ad.import_evaluation([0.5, -0.5], _toy_outputs(ad, [0.5, -0.5]), source="import")

    # Runs with other fixed values (a different space) share the store but not the design
    crashing = ExternalModelAdapter(ToyModel(), toy_space(crash_above_a=0.0), store)
    assert crashing.evaluate([1.0, -0.3]).status == "failed"

    fresh = ExternalModelAdapter(ToyModel(), toy_space(), store)
    thetas, outputs = fresh.stored_design()
    assert thetas.shape == (2, 2) and outputs.shape == (2, 8)
    for theta, y in zip(thetas, outputs):
        np.testing.assert_allclose(y, _toy_outputs(fresh, theta), rtol=1e-6)
    assert len(crashing.stored_evaluations()) == 0
    assert len(crashing.stored_evaluations(include_failed=True)) == 1
    assert fresh.n_runs == 0


def test_adapter_survives_pickle_and_deepcopy(adapter):
    theta = adapter.space.to_calibration(TOY_TRUTH)
    y = adapter.evaluate(theta).y
    for clone in (pickle.loads(pickle.dumps(adapter)), copy.deepcopy(adapter)):
        np.testing.assert_allclose(clone.evaluate(theta).y, y)


# ------------------------------------------------------------- emcal integration
def _toy_problem(adapter, **kw):
    truth = adapter.space.to_calibration(TOY_TRUTH)
    return CalibrationProblem(model=adapter, param_names=adapter.space.names,
                              param_bounds=adapter.space.bounds, x_values=adapter.x_grid,
                              true_params=truth, name="toy", **kw)


def test_validate_model_false_skips_the_trial_run(adapter):
    _toy_problem(adapter, validate_model=False)
    assert adapter.n_runs == 0
    _toy_problem(adapter)
    assert adapter.n_runs == 1


def test_simulator_runs_once_per_parameter_set(adapter):
    sim = make_case_study_simulator(_toy_problem(adapter, validate_model=False), 0, None, 1)
    exp = sim.generate_experimental_data(adapter.n_x, GenMethod.MESHGRID, adapter.x_grid, 0.01)
    assert adapter.n_runs == 1
    simd = sim.generate_simulation_data(4, adapter.n_x, GenMethod.LHS, GenMethod.MESHGRID, 1.0, 1,
                                        False, exp.x_vals, with_noise=False)
    assert simd.theta_vals.shape[0] == 4 * adapter.n_x
    assert adapter.n_runs == 1 + 4  # one launch per sampled theta, not per (theta, x) row
    phys = [adapter.space.to_physical(t) for t in simd.theta_vals]
    expected = [_toy_y(p["a"], p["k"], x) for p, x in zip(phys, simd.x_vals[:, 0])]
    np.testing.assert_allclose(simd.y_vals, expected, rtol=1e-6)


def test_simulation_data_from_evaluations_matches_generated_data(adapter):
    sim = make_case_study_simulator(_toy_problem(adapter, validate_model=False), 0, None, 1)
    exp = sim.generate_experimental_data(adapter.n_x, GenMethod.MESHGRID, adapter.x_grid, 0.01)
    simd = sim.generate_simulation_data(4, adapter.n_x, GenMethod.LHS, GenMethod.MESHGRID, 1.0, 1,
                                        False, exp.x_vals, with_noise=False)
    runs_before = adapter.n_runs

    # The same runs, given as (n_theta, dim_theta) and (n_theta, n_x)
    thetas = simd.theta_vals[::adapter.n_x]
    outputs = simd.y_vals.reshape(4, adapter.n_x)
    rebuilt = sim.simulation_data_from_evaluations(thetas, outputs, exp.x_vals)

    np.testing.assert_array_equal(rebuilt.theta_vals, simd.theta_vals)
    np.testing.assert_array_equal(rebuilt.x_vals, simd.x_vals)
    np.testing.assert_array_equal(rebuilt.y_vals, simd.y_vals)
    np.testing.assert_array_equal(rebuilt.bounds_theta, simd.bounds_theta)
    assert adapter.n_runs == runs_before  # nothing is simulated

    with pytest.raises(AssertionError, match="y_vals must have shape"):
        sim.simulation_data_from_evaluations(thetas, outputs[:, :-1], exp.x_vals)
    with pytest.raises(AssertionError, match="inside the parameter bounds"):
        sim.simulation_data_from_evaluations(thetas + 100.0, outputs, exp.x_vals)


def _toy_driver(adapter, backend, n_theta=6, bo_iter_tot=2, method_val=7):
    method = GPBOMethod(MethodName(method_val))
    sim = make_case_study_simulator(_toy_problem(adapter, validate_model=False), 0, None, 1)
    exp = sim.generate_experimental_data(adapter.n_x, GenMethod.MESHGRID, adapter.x_grid, 0.01)
    simd = sim.generate_simulation_data(n_theta, adapter.n_x, GenMethod.LHS, GenMethod.MESHGRID,
                                        1.0, 1, False, exp.x_vals, with_noise=False)
    ssed = sim.to_sse_data(method, simd, exp, 1.0, False)
    ep = ExplorationBias(1, None, EpSchedule.CONSTANT, None, None, None, None, None, None, None)
    # compute_y_sse=False: one simulation per BO iteration (the acquisition point).
    cfg = BOConfig("toy", kernel=Kernel.MAT_52, retrain_gp=1, reoptimize_obj=1,
                   bo_iter_tot=bo_iter_tot, bo_run_tot=1, seed=1, compute_y_sse=False)
    return GPBODriver(cfg, method, sim, exp, simd, ssed, None, None, None, ep, GenMethod.LHS,
                      backend=backend)


def test_bo_loop_drives_the_subprocess_and_resumes_from_the_store(tmp_path):
    ad = ExternalModelAdapter(ToyModel(), toy_space(), tmp_path / "store")
    res, _ = _toy_driver(ad, FakeGPBackend(mean_value=1.0, variance_value=0.25)).run(job=None)
    assert len(res[0].results_df) == 2
    first_runs = ad.n_runs
    # experimental data + design + one run per BO iteration (fewer if a point repeats)
    assert 1 + 6 + 1 <= first_runs <= 1 + 6 + 2
    # Same study again with a fresh adapter: every simulation comes from the store.
    ad2 = ExternalModelAdapter(ToyModel(), toy_space(), tmp_path / "store")
    _toy_driver(ad2, FakeGPBackend(mean_value=1.0, variance_value=0.25)).run(job=None)
    assert ad2.n_runs == 0


def test_bo_starts_from_imported_design(tmp_path):
    ad = ExternalModelAdapter(ToyModel(), toy_space(), tmp_path / "store")
    sim = make_case_study_simulator(_toy_problem(ad, validate_model=False), 0, None, 1)
    exp = sim.generate_experimental_data(ad.n_x, GenMethod.MESHGRID, ad.x_grid, 0.01)

    # Six runs "made elsewhere", imported instead of simulated
    design = np.array([[-1.5, -1.8], [-0.8, -1.0], [0.0, -0.2], [0.6, 0.3], [1.2, -1.5], [1.8, 0.0]])
    for theta in design:
        ad.import_evaluation(theta, _toy_outputs(ad, theta), source="test design")
    thetas, outputs = ad.stored_design()
    simd = sim.simulation_data_from_evaluations(thetas, outputs, exp.x_vals)

    method = GPBOMethod(MethodName(7))
    ssed = sim.to_sse_data(method, simd, exp, 1.0, False)
    ep = ExplorationBias(1, None, EpSchedule.CONSTANT, None, None, None, None, None, None, None)
    cfg = BOConfig("toy", kernel=Kernel.MAT_52, retrain_gp=1, reoptimize_obj=1,
                   bo_iter_tot=2, bo_run_tot=1, seed=1, compute_y_sse=False)
    driver = GPBODriver(cfg, method, sim, exp, simd, ssed, None, None, None, ep, GenMethod.LHS,
                        backend=FakeGPBackend(mean_value=1.0, variance_value=0.25))
    res, _ = driver.run(job=None)
    assert len(res[0].results_df) == 2
    # experimental data + at most one run per BO iteration; the design is never simulated
    assert ad.n_runs <= 1 + 2


@pytest.mark.slow
@pytest.mark.parametrize("backend", ["gpytorch", "gpflow"])
def test_bo_recovers_toy_parameters(tmp_path, backend):
    pytest.importorskip("gpytorch" if backend == "gpytorch" else "gpflow")
    ad = ExternalModelAdapter(ToyModel(), toy_space(), tmp_path / "store",
                              executor=LocalExecutor(timeout=120))
    res, _ = _toy_driver(ad, backend, n_theta=10, bo_iter_tot=5).run(job=None)
    df = res[0].results_df
    best = np.asarray(df["theta_best_actual"].iloc[-1], dtype=float)
    phys = ad.space.to_physical(best)
    assert phys["a"] == pytest.approx(TOY_TRUTH["a"], abs=0.05)
    assert phys["k"] == pytest.approx(TOY_TRUTH["k"], abs=0.05)
