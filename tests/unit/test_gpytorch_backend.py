"""GPyTorch backend: contract, closed-form numpy reference, gpflow parity, end-to-end.
Skipped when torch/gpytorch are not installed."""
import copy
import pickle

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch not installed")
gpytorch = pytest.importorskip("gpytorch", reason="gpytorch not installed")

from emcal.gp_backend import get_backend
from emcal.gp_backend.gpytorch_backend import BoundedValue, GPyTorchBackend

# emcal's hyperparameter bounds (GPEmulator.__init_hyper_parameters)
LENSCL_BNDS = (1e-5, 1000.0)
VAR_BNDS = (1e-5, 100.0)
WHITE_BNDS = (1e-5, 10.0)
NOISE = 1e-5

KERNELS = {3: "rbf", 2: "matern32", 1: "matern52"}


# ----------------------------------------------------------------- numpy reference GP
def _base_kernel(kernel_value, xa, xb, lengthscale, outputscale):
    r = np.sqrt(np.sum(((xa[:, None, :] - xb[None, :, :]) / lengthscale) ** 2, axis=-1))
    if kernel_value == 3:
        k = np.exp(-0.5 * r**2)
    elif kernel_value == 2:
        k = (1.0 + np.sqrt(3.0) * r) * np.exp(-np.sqrt(3.0) * r)
    else:
        k = (1.0 + np.sqrt(5.0) * r + 5.0 / 3.0 * r**2) * np.exp(-np.sqrt(5.0) * r)
    return outputscale * k


def _reference_gp(kernel_value, x, y, x_star, lengthscale, outputscale, white):
    """Closed-form GP posterior for f; White adds to train and test diagonals only."""
    n = len(x)
    k_xx = _base_kernel(kernel_value, x, x, lengthscale, outputscale) + (white + NOISE) * np.eye(n)
    k_sx = _base_kernel(kernel_value, x_star, x, lengthscale, outputscale)
    k_ss = _base_kernel(kernel_value, x_star, x_star, lengthscale, outputscale) + white * np.eye(len(x_star))
    chol = np.linalg.cholesky(k_xx)
    alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, y))
    v = np.linalg.solve(chol, k_sx.T)
    mean = k_sx @ alpha
    cov = k_ss - v.T @ v
    nlml = 0.5 * (y.T @ alpha).item() + np.sum(np.log(np.diag(chol))) + 0.5 * n * np.log(2 * np.pi)
    return mean, cov, nlml


# ------------------------------------------------------------------------- fixtures
@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    x = rng.uniform(-2.0, 2.0, size=(10, 2))
    y = (np.sin(x[:, 0]) + 0.5 * x[:, 1] ** 2).reshape(-1, 1)
    x_star = rng.uniform(-2.0, 2.0, size=(5, 2))
    return x, y, x_star


HYPER = dict(lengthscale=np.array([0.7, 1.3]), outputscale=1.5, white=0.05)


def _build(backend, data, kernel_value=1, fix_lengthscale=False, fix_outputscale=False, hyper=HYPER):
    x, y, _ = data
    return backend.build_model(
        (x, y), kernel_value,
        backend.make_bounded_parameter(*LENSCL_BNDS, hyper["lengthscale"]),
        backend.make_bounded_parameter(*VAR_BNDS, hyper["outputscale"]),
        backend.make_bounded_parameter(*WHITE_BNDS, hyper["white"]),
        fix_lengthscale, fix_outputscale, noise_variance=NOISE,
    )


# -------------------------------------------------------------------------- contract
def test_registry_returns_gpytorch_backend():
    assert isinstance(get_backend("gpytorch"), GPyTorchBackend)
    assert isinstance(get_backend("GPyTorch"), GPyTorchBackend)  # case-insensitive


def test_unknown_backend_lists_both_choices():
    with pytest.raises(ValueError, match="gpytorch"):
        get_backend("not-a-backend")


def test_implements_full_abc_interface():
    backend = GPyTorchBackend()
    for method in ("configure", "make_bounded_parameter", "build_model", "train",
                   "get_hyperparameters", "make_posterior", "predict_f"):
        assert callable(getattr(backend, method))
    backend.configure()
    assert torch.get_default_dtype() == torch.float32  # no global dtype change


def test_bounded_value_scales_with_numpy_like_gpemulator_does():
    # GPEmulator builds isotropic lengthscales as np.ones(d) * bounded_parameter(...)
    spec = np.ones(3) * GPyTorchBackend().make_bounded_parameter(1e-5, 1000.0, 2.0)
    assert isinstance(spec, BoundedValue)
    np.testing.assert_array_equal(spec.value, [2.0, 2.0, 2.0])
    assert (spec.low, spec.high) == (1e-5, 1000.0)
    assert isinstance(spec * 0.5, BoundedValue)
    np.testing.assert_array_equal((spec * 0.5).value, [1.0, 1.0, 1.0])


def test_build_model_sets_initial_hyperparameters(data):
    backend = GPyTorchBackend()
    model = _build(backend, data)
    lengthscale, white, outputscale = backend.get_hyperparameters(model)
    np.testing.assert_allclose(lengthscale, HYPER["lengthscale"], rtol=1e-12)
    assert outputscale == pytest.approx(HYPER["outputscale"], rel=1e-12)
    assert white == pytest.approx(HYPER["white"], rel=1e-12)
    assert float(model.likelihood.noise) == pytest.approx(NOISE, rel=1e-9)
    assert model.train_inputs[0].dtype == torch.float64


def test_isotropic_lengthscale_from_np_ones_times_parameter(data):
    x, y, _ = data
    backend = GPyTorchBackend()
    ls = np.ones(x.shape[1]) * backend.make_bounded_parameter(*LENSCL_BNDS, 1.0)
    model = backend.build_model((x, y), 1, ls, backend.make_bounded_parameter(*VAR_BNDS, 1.0),
                                backend.make_bounded_parameter(*WHITE_BNDS, 1.0), False, False)
    np.testing.assert_allclose(backend.get_hyperparameters(model)[0], [1.0, 1.0])


def test_trainability_flags(data):
    model = _build(GPyTorchBackend(), data, fix_lengthscale=True, fix_outputscale=True)
    scale, white = model.covar_module.kernels
    assert not scale.base_kernel.raw_lengthscale.requires_grad
    assert not scale.raw_outputscale.requires_grad
    assert not model.likelihood.raw_noise.requires_grad  # noise is never trained
    assert white.raw_variance.requires_grad              # white variance always trained


def test_predict_f_shapes_and_psd(data):
    backend = GPyTorchBackend()
    posterior = backend.make_posterior(_build(backend, data))
    _, _, x_star = data
    mean, cov = backend.predict_f(posterior, x_star, full_cov=True)
    assert mean.shape == (5, 1)
    assert cov.shape == (5, 5)
    np.testing.assert_allclose(cov, cov.T, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(cov) > 0)
    _, var = backend.predict_f(posterior, x_star, full_cov=False)
    assert var.shape == (5, 1)
    np.testing.assert_allclose(var.ravel(), np.diag(cov), rtol=1e-10)


# ------------------------------------------------- correctness vs closed-form numpy GP
@pytest.mark.parametrize("kernel_value", sorted(KERNELS), ids=lambda k: KERNELS[k])
def test_prediction_matches_closed_form_gp(data, kernel_value):
    x, y, x_star = data
    backend = GPyTorchBackend()
    posterior = backend.make_posterior(_build(backend, data, kernel_value))
    mean, cov = backend.predict_f(posterior, x_star)
    ref_mean, ref_cov, _ = _reference_gp(kernel_value, x, y, x_star, **HYPER)
    np.testing.assert_allclose(mean, ref_mean, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(cov, ref_cov, rtol=1e-8, atol=1e-10)


def test_white_term_is_zero_across_train_and_test_even_at_training_points(data):
    # Like gpflow: no white term in the cross covariance, even at training inputs.
    x, y, _ = data
    backend = GPyTorchBackend()
    posterior = backend.make_posterior(_build(backend, data))
    mean, cov = backend.predict_f(posterior, x)
    ref_mean, ref_cov, _ = _reference_gp(1, x, y, x, **HYPER)
    np.testing.assert_allclose(mean, ref_mean, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(cov, ref_cov, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("kernel_value", sorted(KERNELS), ids=lambda k: KERNELS[k])
def test_training_loss_is_closed_form_negative_log_marginal_likelihood(data, kernel_value):
    x, y, x_star = data
    backend = GPyTorchBackend()
    loss = backend.training_loss(_build(backend, data, kernel_value))
    _, _, ref_nlml = _reference_gp(kernel_value, x, y, x_star, **HYPER)
    assert loss == pytest.approx(ref_nlml, rel=1e-10)


# ------------------------------------------------------------------------- training
def test_train_lowers_loss_and_reports_success(data):
    backend = GPyTorchBackend()
    model = _build(backend, data)
    before = backend.training_loss(model)
    success, after = backend.train(model)
    assert success
    assert after < before
    assert after == pytest.approx(backend.training_loss(model), rel=1e-12)


def test_train_respects_bounds_when_bounded(data):
    backend = GPyTorchBackend(bounded=True)
    model = _build(backend, data)
    backend.train(model)
    lengthscale, white, outputscale = backend.get_hyperparameters(model)
    assert np.all((LENSCL_BNDS[0] < lengthscale) & (lengthscale < LENSCL_BNDS[1]))
    assert VAR_BNDS[0] < outputscale < VAR_BNDS[1]
    assert WHITE_BNDS[0] < white < WHITE_BNDS[1]


def test_unbounded_mode_uses_positive_constraints(data):
    model = _build(GPyTorchBackend(bounded=False), data)
    scale, white = model.covar_module.kernels
    for constraint in (scale.base_kernel.raw_lengthscale_constraint,
                       scale.raw_outputscale_constraint, white.raw_variance_constraint):
        assert isinstance(constraint, gpytorch.constraints.Positive)


def test_fixed_hyperparameters_do_not_move_in_training(data):
    backend = GPyTorchBackend()
    model = _build(backend, data, fix_lengthscale=True, fix_outputscale=True)
    backend.train(model)
    lengthscale, _, outputscale = backend.get_hyperparameters(model)
    np.testing.assert_allclose(lengthscale, HYPER["lengthscale"], rtol=1e-12)
    assert outputscale == pytest.approx(HYPER["outputscale"], rel=1e-12)


def test_training_is_deterministic(data):
    backend = GPyTorchBackend()
    results = []
    for _ in range(2):
        model = _build(backend, data)
        backend.train(model)
        results.append(np.concatenate([np.ravel(h) for h in backend.get_hyperparameters(model)]))
    np.testing.assert_array_equal(results[0], results[1])


def test_posterior_survives_deepcopy_and_pickle(data):
    # The driver deep-copies emulators and pickles results.
    _, _, x_star = data
    backend = GPyTorchBackend()
    model = _build(backend, data)
    backend.train(model)
    posterior = backend.make_posterior(model)
    mean, cov = backend.predict_f(posterior, x_star)  # populate the prediction cache
    for clone in (copy.deepcopy(posterior), pickle.loads(pickle.dumps(posterior))):
        m2, c2 = backend.predict_f(clone, x_star)
        np.testing.assert_allclose(m2, mean, rtol=1e-12)
        np.testing.assert_allclose(c2, cov, rtol=1e-12)


# ---------------------------------------------------------------- emcal integration
def test_gpemulator_accepts_backend_name():
    from emcal.emulators import GPEmulator
    from emcal.enums import Kernel

    emu = GPEmulator(None, None, None, Kernel.MAT_52, None, 1.0, None, 0, None, False,
                     backend="gpytorch")
    assert isinstance(emu._backend, GPyTorchBackend)


@pytest.mark.parametrize("kernel_value", sorted(KERNELS), ids=lambda k: KERNELS[k])
def test_parity_with_gpflow_at_fixed_hyperparameters(data, kernel_value):
    pytest.importorskip("gpflow", reason="gpflow not installed")
    from emcal.gp_backend.gpflow_backend import GpflowBackend

    x, y, x_star = data
    gpf = GpflowBackend()
    gpf.configure()
    gpf_model = gpf.build_model((x, y), kernel_value, HYPER["lengthscale"], HYPER["outputscale"],
                                HYPER["white"], False, False, noise_variance=NOISE)
    gpf_mean, gpf_cov = gpf.predict_f(gpf.make_posterior(gpf_model), x_star)
    gpf_loss = float(gpf_model.training_loss().numpy())

    gpt = GPyTorchBackend()
    gpt_model = _build(gpt, data, kernel_value)
    gpt_loss = gpt.training_loss(gpt_model)
    gpt_mean, gpt_cov = gpt.predict_f(gpt.make_posterior(gpt_model), x_star)

    np.testing.assert_allclose(gpt_mean, gpf_mean, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(gpt_cov, gpf_cov, rtol=1e-8, atol=1e-10)
    assert gpt_loss == pytest.approx(gpf_loss, rel=1e-9)


def _cs1_setup(method_value):
    from emcal import (
        MethodName, EpSchedule, Kernel, GenMethod, GPBOMethod, ExplorationBias, BOConfig,
        get_case_study, make_case_study_simulator,
    )

    method = GPBOMethod(MethodName(method_value))
    problem = get_case_study(1)
    sim = make_case_study_simulator(problem, 0, None, 1)
    exp = sim.generate_experimental_data(5, GenMethod(2), None, 0.01)
    n = len(sim.indices_to_consider)
    simd = sim.generate_simulation_data(10 * n, 5, GenMethod(1), GenMethod(2), 1.0, 1, False, None,
                                        with_noise=False)
    ssed = sim.to_sse_data(method, simd, exp, 1.0, False)
    ep = ExplorationBias(1, None, EpSchedule(1), None, None, None, None, None, None, None)
    cfg = BOConfig(problem.name, kernel=Kernel.MAT_52, retrain_gp=3, reoptimize_obj=3,
                   bo_iter_tot=3, bo_run_tot=1, compute_y_sse=True)
    return method, sim, exp, simd, ssed, ep, cfg


@pytest.mark.slow
@pytest.mark.parametrize("method_value", [1, 7], ids=["conventional", "expected_sse"])
def test_bo_run_with_gpytorch_backend(method_value):
    from emcal import GPBODriver, GenMethod

    method, sim, exp, simd, ssed, ep, cfg = _cs1_setup(method_value)
    initial_best = float(np.min(ssed.y_vals))
    driver = GPBODriver(cfg, method, sim, exp, simd, ssed, None, None, None, ep, GenMethod(1),
                        backend="gpytorch")
    res_simple, res_gp = driver.run(job=None)
    df = res_simple[0].results_df
    assert len(df) == cfg.bo_iter_tot
    assert np.isfinite(float(df["best_sse_actual"].iloc[-1]))
    # Incumbent (training data + new points) never gets worse.
    best_error = df["best_error"].to_numpy(dtype=float)
    assert best_error[0] <= initial_best + 1e-9
    assert np.all(np.diff(best_error) <= 1e-9)
    emulators = res_gp[0].list_gp_emulator_class
    assert all(isinstance(e._backend, GPyTorchBackend) for e in emulators)


@pytest.mark.slow
def test_diagnostics_fit_gp_with_gpytorch_backend():
    from emcal import diagnostics

    method, sim, exp, simd, ssed, _, cfg = _cs1_setup(7)
    emulator = diagnostics.fit_gp(method, sim, exp, simd, ssed, cfg, sep_fact=0.8,
                                  shuffle_seed=0, backend="gpytorch")
    assert isinstance(emulator._backend, GPyTorchBackend)
    report = diagnostics.evaluate_gp(emulator)
    assert np.isfinite(report.r2)
