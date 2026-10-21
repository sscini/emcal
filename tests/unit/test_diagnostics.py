"""Tests for the GP-diagnostics API (accuracy + uncertainty calibration + CV)."""
import matplotlib
matplotlib.use("Agg")  # headless: must be set before any pyplot import (incl. via diagnostics plots)
import numpy as np
import pytest

from emcal import (
    MethodName, Kernel, GenMethod, GPBOMethod, BOConfig,
    get_case_study, make_case_study_simulator, diagnostics,
)
from emcal.diagnostics import GPDiagnostics, _diagnostics_from_arrays


# ---- fast: metric math (no GP fitting) ----------------------------------------------
def test_metrics_on_perfect_predictions():
    actual = np.linspace(0, 10, 50)
    diag = _diagnostics_from_arrays(actual, actual.copy(), np.ones_like(actual),
                                    n_train=100, label="perfect", hyper=[])
    assert diag.r2 == pytest.approx(1.0)
    assert diag.rmse == pytest.approx(0.0, abs=1e-12)
    assert diag.mae == pytest.approx(0.0, abs=1e-12)


def test_calibration_metrics_on_well_calibrated_samples():
    # actual = predicted + sigma * N(0,1)  ->  standardized residuals ~ N(0,1).
    rng = np.random.default_rng(0)
    n = 4000
    predicted = rng.uniform(-5, 5, n)
    sigma = rng.uniform(0.5, 2.0, n)
    actual = predicted + sigma * rng.standard_normal(n)
    diag = _diagnostics_from_arrays(actual, predicted, sigma, n_train=n, label="calib", hyper=[])
    assert diag.mean_z == pytest.approx(0.0, abs=0.05)
    assert diag.std_z == pytest.approx(1.0, abs=0.05)
    assert diag.reduced_chi2 == pytest.approx(1.0, abs=0.1)
    assert diag.coverage[0.95] == pytest.approx(0.95, abs=0.03)
    assert diag.coverage[0.50] == pytest.approx(0.50, abs=0.04)
    assert diag.miscalibration_area < 0.03
    assert diag.sharpness == pytest.approx(float(np.mean(sigma)), rel=1e-6)
    assert np.isfinite(diag.crps) and diag.crps > 0


def test_crps_closed_form():
    # For a perfectly-centered Gaussian (y = mu) with sigma = 1, the Gaussian CRPS is
    # 2*phi(0) - 1/sqrt(pi) = 1/sqrt(pi) * (sqrt(2/pi) ... ) ~= 0.23370.
    actual = np.zeros(200)
    diag = _diagnostics_from_arrays(actual, actual.copy(), np.ones_like(actual),
                                    n_train=200, label="crps", hyper=[])
    assert diag.crps == pytest.approx(0.233696, abs=1e-4)
    assert diag.sharpness == pytest.approx(1.0)


def test_overconfident_detected():
    # predictive sigma too SMALL -> std_z > 1 (overconfident).
    rng = np.random.default_rng(1)
    n = 2000
    predicted = np.zeros(n)
    actual = predicted + rng.standard_normal(n)   # true spread 1.0 ...
    diag = _diagnostics_from_arrays(actual, predicted, np.full(n, 0.5), n_train=n,
                                    label="oc", hyper=[])   # ... but claimed sigma 0.5
    assert diag.std_z > 1.5
    assert "overconfident" in diag._calibration_verdict()


def test_summary_is_string():
    actual = np.arange(10.0)
    diag = _diagnostics_from_arrays(actual, actual + 0.1, np.ones(10), 20, "x", [])
    assert isinstance(diag.summary(), str) and "calibration" in diag.summary()


def _diag_with_std_z(std_z):
    # Direct dataclass construction: _calibration_verdict only reads self.std_z, so this
    # pins its three branches exactly rather than relying on a statistical sample landing
    # in the right bucket.
    return GPDiagnostics(label="x", n_train=1, n_eval=1, r2=1.0, rmse=0.0, mae=0.0, mape=0.0,
                         mean_z=0.0, std_z=std_z, reduced_chi2=1.0, coverage={}, nlpd=0.0,
                         crps=0.0, sharpness=0.0, miscalibration_area=0.0)


@pytest.mark.parametrize("std_z,expected", [
    (0.5, "underconfident"), (1.5, "overconfident"), (1.0, "reasonably calibrated"),
])
def test_calibration_verdict_boundaries(std_z, expected):
    assert expected in _diag_with_std_z(std_z)._calibration_verdict()


def test_str_matches_summary():
    actual = np.arange(10.0)
    diag = _diagnostics_from_arrays(actual, actual + 0.1, np.ones(10), 20, "x", [])
    assert str(diag) == diag.summary()


# ---- fast: plots on synthetic-array diagnostics (no GP fitting) --------------------

def _synthetic_diag():
    rng = np.random.default_rng(0)
    n = 200
    predicted = rng.uniform(-5, 5, n)
    sigma = rng.uniform(0.5, 2.0, n)
    actual = predicted + sigma * rng.standard_normal(n)
    return _diagnostics_from_arrays(actual, predicted, sigma, n_train=n, label="plots",
                                    hyper=["ell", 1.0])


def test_parity_plot_returns_axes():
    diag = _synthetic_diag()
    ax = diag.parity_plot()
    assert ax.get_xlabel() == "actual"
    assert ax.get_ylabel() == "GP predicted"


def test_calibration_plot_returns_axes():
    diag = _synthetic_diag()
    ax = diag.calibration_plot()
    assert ax.get_xlabel() == "nominal coverage"
    assert ax.get_ylabel() == "empirical coverage"


def test_residual_plot_returns_axes():
    diag = _synthetic_diag()
    ax = diag.residual_plot()
    assert ax.get_xlabel() == "theoretical quantile N(0,1)"


def test_plot_all_returns_figure_with_three_axes():
    diag = _synthetic_diag()
    fig = diag.plot_all()
    assert len(fig.axes) == 3


def test_plots_accept_an_existing_axes():
    import matplotlib.pyplot as plt

    diag = _synthetic_diag()
    _, ax = plt.subplots()
    assert diag.parity_plot(ax=ax) is ax
    assert diag.calibration_plot(ax=ax) is ax
    assert diag.residual_plot(ax=ax) is ax


# ---- slow: end-to-end fit + evaluate + CV -------------------------------------------
def _build(method_val, retrain=20):
    method = GPBOMethod(MethodName(method_val))
    problem = get_case_study(1)
    sim = make_case_study_simulator(problem, 0, None, 1)
    exp = sim.generate_experimental_data(5, GenMethod.MESHGRID, None, 0.01)
    simd = sim.generate_simulation_data(
        20, 5, GenMethod.LHS, GenMethod.MESHGRID, 1.0, 1, False, None, with_noise=False
    )
    ssed = sim.to_sse_data(method, simd, exp, 1.0, False)
    cfg = BOConfig(problem.name, kernel=Kernel.MAT_52, retrain_gp=retrain, reoptimize_obj=3)
    return method, sim, exp, simd, ssed, cfg


@pytest.mark.slow
@pytest.mark.parametrize("method_val", [1, 3])  # ObjectiveGP and EmulatorGP
def test_evaluate_gp_well_trained(method_val):
    method, sim, exp, simd, ssed, cfg = _build(method_val, retrain=20)
    # Fixed split: some random splits give degenerate gpflow fits (see notes 06).
    emu = diagnostics.fit_gp(method, sim, exp, simd, ssed, cfg, sep_fact=0.75, shuffle_seed=1)
    diag = diagnostics.evaluate_gp(emu)
    assert isinstance(diag, GPDiagnostics)
    assert diag.n_train > 0 and diag.n_eval > 0
    assert set(diag.coverage) == {0.50, 0.90, 0.95}
    assert np.isfinite(diag.nlpd) and np.isfinite(diag.std_z)
    assert diag.r2 > 0.9        # CS1 is smooth; a well-trained GP fits held-out well


@pytest.mark.slow
def test_cross_validate_gp():
    method, sim, exp, simd, ssed, cfg = _build(3, retrain=10)
    cv = diagnostics.cross_validate_gp(method, sim, exp, simd, ssed, cfg,
                                       holdout_frac=0.25, n_repeats=2)
    assert isinstance(cv, GPDiagnostics)
    assert cv.n_eval > 0 and "CV" in cv.label
    assert np.isfinite(cv.rmse) and np.isfinite(cv.std_z)


@pytest.mark.slow
def test_plots_return_axes_and_figure():
    import matplotlib
    matplotlib.use("Agg")
    method, sim, exp, simd, ssed, cfg = _build(1, retrain=10)
    diag = diagnostics.evaluate_gp(diagnostics.fit_gp(method, sim, exp, simd, ssed, cfg, sep_fact=0.75))
    assert diag.parity_plot().get_xlabel() == "actual"
    assert diag.calibration_plot().get_xlabel() == "nominal coverage"
    assert diag.residual_plot() is not None
    assert len(diag.plot_all().axes) == 3
