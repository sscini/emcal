"""GP-diagnostics API: examine a GP emulator's quality BEFORE running Bayesian optimization.

Emulator GPBO only works well when the GP is good -- both its MEAN (accuracy) and its
predictive UNCERTAINTY (calibration). This module bundles the standard GP "best practices"
into an easy API:

- accuracy on held-out data  : R^2, RMSE, MAE, MAPE
- uncertainty calibration    : standardized residuals (mean~0, std~1), reduced chi^2 (~1),
                               interval coverage @ 50/90/95%, NLPD, miscalibration area
- validation schemes         : a single train/test split, or leave-N-out / repeated
                               cross-validation
- plots                      : parity (with +/-1.96 sigma), calibration curve, residual QQ

Quick start (after generating experimental + simulation data, as in the examples)::

    from emcal import diagnostics
    diag = diagnostics.evaluate_gp(
        diagnostics.fit_gp(method, simulator, exp_data, sim_data, sim_sse_data, config)
    )
    print(diag.summary())
    diag.plot_all()                                  # parity | calibration | residuals

    # cross-validated (leave-N-out style) performance + calibration:
    cv = diagnostics.cross_validate_gp(
        method, simulator, exp_data, sim_data, sim_sse_data, config, holdout_frac=0.2, n_repeats=5
    )
    print(cv.summary())

The calibration metrics are computed on the GP's NATIVE target (the SSE for the standard
"objective" GP; the model output for the emulator GP) -- i.e. exactly the quantity whose
uncertainty the acquisition function relies on.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm

from .emulators import build_gp_emulator

# central-interval nominal levels reported by default
_COVERAGE_LEVELS = (0.50, 0.90, 0.95)


@dataclass
class GPDiagnostics:
    """Accuracy + uncertainty-calibration diagnostics for a GP, on out-of-sample data.

    Produced by :func:`evaluate_gp` (single train/test split) or
    :func:`cross_validate_gp` (leave-N-out / repeated CV). Holds both scalar metrics and
    the raw (actual, predicted, pred_std) arrays so the plots can be regenerated.
    """

    label: str
    n_train: int
    n_eval: int
    # --- accuracy of the predictive MEAN ---
    r2: float
    rmse: float
    mae: float
    mape: float
    # --- calibration of the predictive UNCERTAINTY (standardized residual z = (y-mu)/sigma) ---
    mean_z: float          # ~0 if unbiased
    std_z: float           # ~1 if well-calibrated; <1 underconfident, >1 overconfident
    reduced_chi2: float    # mean(z^2), ~1 if the predictive variance is right on average
    coverage: dict         # {nominal level -> empirical coverage}, e.g. {0.95: 0.93}
    nlpd: float            # mean negative log predictive density (Gaussian); lower is better
    crps: float            # mean continuous ranked probability score (Gaussian); lower is better
    sharpness: float       # mean predictive std (tightness of the error bars; lower = sharper)
    miscalibration_area: float  # mean|empirical - nominal| over the calibration curve; 0 is perfect
    # --- raw data for plots ---
    actual: np.ndarray = field(repr=False, default=None)
    predicted: np.ndarray = field(repr=False, default=None)
    pred_std: np.ndarray = field(repr=False, default=None)
    hyperparameters: list = field(default_factory=list)

    # ---- reporting -------------------------------------------------------------------
    def _calibration_verdict(self):
        if self.std_z < 0.8:
            return "underconfident (predictive std too LARGE)"
        if self.std_z > 1.25:
            return "overconfident (predictive std too SMALL)"
        return "reasonably calibrated spread"

    def summary(self):
        cov = "  ".join(f"{int(p*100)}%:{self.coverage[p]:.2f}" for p in sorted(self.coverage))
        return (
            f"GP diagnostics [{self.label}]  (train={self.n_train}, eval={self.n_eval})\n"
            f"  accuracy      R^2={self.r2:.4f}  RMSE={self.rmse:.4g}  "
            f"MAE={self.mae:.4g}  MAPE={self.mape:.2f}%\n"
            f"  calibration   mean(z)={self.mean_z:+.3f} (~0)  std(z)={self.std_z:.3f} (~1)  "
            f"reduced_chi2={self.reduced_chi2:.3f} (~1)\n"
            f"                coverage[{cov}]  (target = nominal)\n"
            f"                NLPD={self.nlpd:.4g}  CRPS={self.crps:.4g} (lower better)  "
            f"sharpness={self.sharpness:.4g}\n"
            f"                miscalibration_area={self.miscalibration_area:.3f} (0 best)\n"
            f"  verdict       {self._calibration_verdict()}\n"
            f"  hyperparams   {self.hyperparameters}"
        )

    def __str__(self):
        return self.summary()

    # ---- plots -----------------------------------------------------------------------
    def parity_plot(self, ax=None):
        """Predicted vs. actual with +/-1.96 sigma error bars (points on y=x = good mean)."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(5, 5))
        ax.errorbar(self.actual, self.predicted, yerr=1.96 * self.pred_std, fmt="o",
                    ms=4, alpha=0.6, elinewidth=0.8, capsize=2, label="pred +/-1.96 sigma")
        lo = float(min(self.actual.min(), self.predicted.min()))
        hi = float(max(self.actual.max(), self.predicted.max()))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
        ax.set_xlabel("actual"); ax.set_ylabel("GP predicted")
        ax.set_title(f"Parity [{self.label}]"); ax.legend(fontsize=8)
        return ax

    def calibration_plot(self, ax=None):
        """Reliability curve: nominal vs. empirical central-interval coverage (diagonal = perfect)."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(5, 5))
        nominal = np.linspace(0.0, 1.0, 21)
        z = _safe_z(self.actual, self.predicted, self.pred_std)
        empirical = [float(np.mean(np.abs(z) <= norm.ppf(0.5 + p / 2))) if p > 0 else 0.0
                     for p in nominal]
        ax.plot(nominal, empirical, "o-", ms=3, label="GP")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
        ax.set_xlabel("nominal coverage"); ax.set_ylabel("empirical coverage")
        ax.set_title(f"Calibration [{self.label}]"); ax.legend(fontsize=8)
        return ax

    def residual_plot(self, ax=None):
        """QQ plot of standardized residuals vs. N(0,1) (on the line = calibrated)."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(5, 5))
        z = np.sort(_safe_z(self.actual, self.predicted, self.pred_std))
        theo = norm.ppf((np.arange(1, len(z) + 1) - 0.5) / len(z))
        ax.plot(theo, z, "o", ms=4, alpha=0.6)
        lim = float(max(np.abs(theo).max(), np.abs(z).max())) if len(z) else 1.0
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=1)
        ax.set_xlabel("theoretical quantile N(0,1)"); ax.set_ylabel("standardized residual")
        ax.set_title(f"Residual QQ [{self.label}]")
        return ax

    def plot_all(self):
        """A 1x3 figure: parity | calibration | residual QQ. Returns the Figure."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        self.parity_plot(axes[0]); self.calibration_plot(axes[1]); self.residual_plot(axes[2])
        fig.tight_layout()
        return fig


def _safe_z(actual, predicted, pred_std):
    std = np.where(pred_std > 0, pred_std, np.nan)
    return (actual - predicted) / std


def _diagnostics_from_arrays(actual, predicted, pred_std, n_train, label, hyper):
    actual = np.asarray(actual, dtype=float).ravel()
    predicted = np.asarray(predicted, dtype=float).ravel()
    pred_std = np.asarray(pred_std, dtype=float).ravel()
    assert len(actual) > 0, "no evaluation points"

    resid = predicted - actual
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(np.mean(resid**2)))
    mae = float(np.mean(np.abs(resid)))
    nz = actual != 0
    mape = float(np.mean(np.abs(resid[nz] / actual[nz])) * 100.0) if np.any(nz) else float("nan")

    z = _safe_z(actual, predicted, pred_std)
    zf = z[np.isfinite(z)]
    mean_z = float(np.mean(zf)) if len(zf) else float("nan")
    std_z = float(np.std(zf)) if len(zf) else float("nan")
    reduced_chi2 = float(np.mean(zf**2)) if len(zf) else float("nan")
    coverage = {
        p: (float(np.mean(np.abs(zf) <= norm.ppf(0.5 + p / 2))) if len(zf) else float("nan"))
        for p in _COVERAGE_LEVELS
    }
    var = pred_std**2
    good = np.isfinite(z) & (var > 0)
    nlpd = (
        float(np.mean(0.5 * np.log(2 * np.pi * var[good]) + 0.5 * z[good] ** 2))
        if np.any(good)
        else float("nan")
    )
    # CRPS for a Gaussian predictive N(mu, sigma^2) (closed form; omega = z = (y-mu)/sigma).
    w = z[good]
    crps = (
        float(np.mean(pred_std[good] * (w * (2 * norm.cdf(w) - 1)
                                        + 2 * norm.pdf(w) - 1.0 / np.sqrt(np.pi))))
        if np.any(good)
        else float("nan")
    )
    sharpness = float(np.mean(pred_std[pred_std > 0])) if np.any(pred_std > 0) else float("nan")
    # miscalibration area: |empirical - nominal| averaged over a nominal grid
    grid = np.linspace(0.0, 1.0, 21)
    emp = np.array([np.mean(np.abs(zf) <= norm.ppf(0.5 + p / 2)) if (p > 0 and len(zf)) else 0.0
                    for p in grid])
    miscal = float(np.mean(np.abs(emp - grid)))

    return GPDiagnostics(
        label=label, n_train=int(n_train), n_eval=len(actual),
        r2=r2, rmse=rmse, mae=mae, mape=mape,
        mean_z=mean_z, std_z=std_z, reduced_chi2=reduced_chi2, coverage=coverage,
        nlpd=nlpd, crps=crps, sharpness=sharpness, miscalibration_area=miscal,
        actual=actual, predicted=predicted, pred_std=pred_std,
        hyperparameters=list(hyper) if hyper is not None else [],
    )


# ---- building / evaluating a single fit ---------------------------------------------
def fit_gp(
    method,
    simulator,
    exp_data,
    sim_data,
    sim_sse_data,
    config,
    val_data=None,
    val_sse_data=None,
    sep_fact=0.8,
    shuffle_seed=None,
    backend=None,
):
    """
    Build and train the GP emulator for `method`, holding out a test set for evaluation.

    `sep_fact` is the fraction used for TRAINING (default 0.8 -> 20% held out), so this is
    deliberately different from a full-data BO run (config.sep_fact).

    Parameters
    ----------
    method: GPBOMethod
        Class containing GPBO method information
    simulator: Simulator
        Class containing values of simulation parameters
    exp_data: Data
        Experimental data containing at least exp_data.x_vals, exp_data.y_vals
    sim_data: Data
        Simulation data used to build the emulator's training/test sets
    sim_sse_data: Data
        SSE-transformed simulation data (used for Type-1/emulator-of-SSE methods)
    config: BOConfig
        Class containing the values associated with BOConfig
    val_data: Data or None, default None
        Validation data, if any
    val_sse_data: Data or None, default None
        SSE-transformed validation data, if any
    sep_fact: float, default 0.8
        Fraction of data used for training (1 - sep_fact is held out as the test set)
    shuffle_seed: int or None, default None
        Seed for the train/test split
    backend: GPBackend, str, or None, default None
        GP backend ("gpflow", "gpytorch", or an instance). None means gpflow.

    Returns
    -------
    emulator: ObjectiveGP or EmulatorGP
        A fitted emulator with train/test data populated (ready for evaluate_gp / plots /
        predict)
    """
    emulator = build_gp_emulator(
        method, sim_data, sim_sse_data, val_data, val_sse_data,
        config.kernel, config.lengthscale, config.outputscale, config.retrain_gp,
        config.seed, config.normalize, simulator.noise_std, exp_data.n_x,
        backend=backend,
    )
    # The split is driven by the source data's sep_fact; set it so a test set exists even
    # if the data was built for a full-data BO run (sep_fact = 1.0).
    emulator.gp_sim_data.sep_fact = sep_fact
    emulator.split_train_test(sep_fact, shuffle_seed)
    emulator.fit()
    return emulator


def evaluate_gp(emulator, label="held-out test"):
    """
    Accuracy + uncertainty-calibration diagnostics on the emulator's held-out test set.

    `emulator` must be fitted with a non-trivial test split (see fit_gp with sep_fact < 1).

    Parameters
    ----------
    emulator: ObjectiveGP or EmulatorGP
        A fitted emulator with a non-trivial test split (see fit_gp)
    label: str, default "held-out test"
        Label attached to the returned GPDiagnostics (for reports/plots)

    Returns
    -------
    diagnostics: GPDiagnostics
        Accuracy + calibration diagnostics computed on the emulator's test set
    """
    gp_mean, gp_var = emulator.predict(target="test")
    actual = np.asarray(emulator.test_data.y_vals, dtype=float).ravel()
    pred = np.asarray(gp_mean, dtype=float).ravel()
    std = np.sqrt(np.clip(np.asarray(gp_var, dtype=float).ravel(), 0.0, None))
    n_train = len(np.asarray(emulator.train_data.y_vals).ravel())
    hyper = getattr(emulator, "trained_hyperparams", None)
    return _diagnostics_from_arrays(actual, pred, std, n_train, label, hyper)


def gp_fit_report(emulator):
    """Backwards-compatible alias for :func:`evaluate_gp` (returns a GPDiagnostics)."""
    return evaluate_gp(emulator)


# ---- cross-validation (leave-N-out style) -------------------------------------------
def cross_validate_gp(
    method,
    simulator,
    exp_data,
    sim_data,
    sim_sse_data,
    config,
    holdout_frac=0.2,
    n_repeats=5,
    base_seed=0,
    label=None,
    backend=None,
):
    """
    Cross-validated GP accuracy + calibration via repeated random hold-out (Monte-Carlo CV).

    Each repeat holds out `holdout_frac` of the sampled points (a "leave-N-out" split with
    N = holdout_frac * n_points), refits the GP on the rest, and predicts the held-out
    points; the out-of-sample (actual, predicted, sigma) from all repeats are pooled into a
    single GPDiagnostics. Repeated random hold-out is used (rather than exact k-fold) because
    it reuses the proven fit path and is robust; raise n_repeats for a tighter estimate.

    Parameters
    ----------
    holdout_frac : float in (0, 1), default 0.2
        Fraction held out each repeat (train fraction = 1 - holdout_frac).
    n_repeats : int, default 5
        Number of random hold-out repeats to pool.
    base_seed : int, default 0
        Shuffle seeds are base_seed + repeat index (reproducible).
    label : str or None, default None
        Label attached to the returned GPDiagnostics. If None, a label is generated from
        n_repeats/holdout_frac
    backend : GPBackend, str, or None, default None
        GP backend ("gpflow", "gpytorch", or an instance). None means gpflow.

    Returns
    -------
    diagnostics: GPDiagnostics
        Accuracy + calibration diagnostics pooled across all n_repeats hold-out repeats

    Raises
    ------
    AssertionError
        If holdout_frac is not in (0, 1) or n_repeats < 1
    """
    assert 0 < holdout_frac < 1, "holdout_frac must be in (0, 1)"
    assert n_repeats >= 1, "n_repeats must be >= 1"
    actuals, preds, stds = [], [], []
    n_train_last = 0
    for r in range(n_repeats):
        emu = fit_gp(
            method, simulator, exp_data, sim_data, sim_sse_data, config,
            sep_fact=1.0 - holdout_frac, shuffle_seed=base_seed + r, backend=backend,
        )
        gp_mean, gp_var = emu.predict(target="test")
        actuals.append(np.asarray(emu.test_data.y_vals, dtype=float).ravel())
        preds.append(np.asarray(gp_mean, dtype=float).ravel())
        stds.append(np.sqrt(np.clip(np.asarray(gp_var, dtype=float).ravel(), 0.0, None)))
        n_train_last = len(np.asarray(emu.train_data.y_vals).ravel())
        hyper = getattr(emu, "trained_hyperparams", None)
    if label is None:
        label = f"{n_repeats}x {int(holdout_frac*100)}%-holdout CV"
    return _diagnostics_from_arrays(
        np.concatenate(actuals), np.concatenate(preds), np.concatenate(stds),
        n_train_last, label, hyper,
    )
