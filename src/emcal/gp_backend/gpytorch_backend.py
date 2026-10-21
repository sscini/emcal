"""GPyTorch implementation of the GPBackend interface.

Same model as GpflowBackend: zero mean, ARD base kernel + White kernel, fixed noise,
L-BFGS-B training, exact float64 Cholesky. Hyperparameters keep emcal's (low, high)
bounds. gpflow drops them, and ``bounded=False`` matches that.
"""
import contextlib
import math
import warnings

import numpy as np
import torch
import gpytorch
from linear_operator.utils.warnings import NumericalWarning
from scipy import optimize

from .base import GPBackend

_DTYPE = torch.float64


class BoundedValue:
    """(low, high, value) spec from make_bounded_parameter. GPEmulator builds
    lengthscales as ``np.ones(d) * spec``, so multiplying scales the value."""

    # Makes ``ndarray * spec`` call spec.__rmul__ instead of building an object array.
    __array_ufunc__ = None

    def __init__(self, low, high, value):
        self.low = float(low)
        self.high = float(high)
        self.value = np.asarray(value, dtype=np.float64)

    def __mul__(self, other):
        scaled_value = self.value * np.asarray(other, dtype=np.float64)
        return BoundedValue(self.low, self.high, scaled_value)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __repr__(self):
        return f"BoundedValue(low={self.low}, high={self.high}, value={self.value})"


class WhiteKernel(gpytorch.kernels.Kernel):
    """gpflow.kernels.White: variance * I on one input set, zero across sets.
    Needs eager kernels (see _exact_settings) so the train/test block stays zero."""

    is_stationary = True

    def __init__(self, variance_constraint=None, **kwargs):
        super().__init__(**kwargs)
        self.register_parameter(
            name="raw_variance", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1))
        )
        if variance_constraint is None:
            variance_constraint = gpytorch.constraints.Positive()
        self.register_constraint("raw_variance", variance_constraint)

    @property
    def variance(self):
        return self.raw_variance_constraint.transform(self.raw_variance)

    @variance.setter
    def variance(self, value):
        value = torch.as_tensor(value, dtype=self.raw_variance.dtype).reshape(self.raw_variance.shape)
        self.initialize(raw_variance=self.raw_variance_constraint.inverse_transform(value))

    def forward(self, x1, x2, diag=False, **params):
        n1 = x1.shape[-2]
        n2 = x2.shape[-2]
        var = self.variance.reshape(())
        same_inputs = n1 == n2 and torch.equal(x1, x2)

        if diag:
            if same_inputs:
                return var.expand(n1)
            return torch.zeros(n1, dtype=x1.dtype)

        if same_inputs:
            return var * torch.eye(n1, dtype=x1.dtype)
        return torch.zeros(n1, n2, dtype=x1.dtype)


class _ExactGPModel(gpytorch.models.ExactGP):
    """Zero-mean exact GP (counterpart of gpflow.models.GPR)."""

    def __init__(self, train_x, train_y, likelihood, covar_module):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = covar_module

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))


@contextlib.contextmanager
def _exact_settings():
    """Exact Cholesky, eager kernels, no debug warnings."""
    with contextlib.ExitStack() as stack:
        stack.enter_context(gpytorch.settings.lazily_evaluate_kernels(False))
        stack.enter_context(gpytorch.settings.max_cholesky_size(10**9))
        stack.enter_context(gpytorch.settings.fast_computations(
            covar_root_decomposition=False, log_prob=False, solves=False
        ))
        stack.enter_context(gpytorch.settings.debug(False))
        yield


# ---- flat-vector helpers for scipy, which optimizes one 1D array of all parameters ----
def _get_flat_values(params):
    """Values of the torch parameters, concatenated into one 1D numpy array."""
    pieces = []
    for p in params:
        pieces.append(p.detach().cpu().numpy().ravel())
    return np.concatenate(pieces)


def _get_flat_grads(params):
    """Gradients of the torch parameters, concatenated into one 1D numpy array."""
    pieces = []
    for p in params:
        pieces.append(p.grad.detach().cpu().numpy().ravel())
    return np.concatenate(pieces)


def _set_flat_values(params, flat):
    """Copy a 1D array (as from _get_flat_values) back into the torch parameters."""
    offset = 0
    with torch.no_grad():
        for p in params:
            size = p.numel()
            chunk = torch.as_tensor(flat[offset:offset + size], dtype=p.dtype)
            p.copy_(chunk.reshape(p.shape))
            offset += size


def _loss_and_grad(flat, model, mll, params):
    """Negative log marginal likelihood and its gradient at ``flat``, for scipy."""
    _set_flat_values(params, flat)
    for p in params:
        p.grad = None

    n_data = model.train_targets.shape[0]
    # Trial steps can hit non-PD matrices; linear_operator adds jitter and warns.
    with _exact_settings(), warnings.catch_warnings():
        warnings.simplefilter("ignore", NumericalWarning)
        out = model(model.train_inputs[0])
        loss = -mll(out, model.train_targets) * n_data
        loss.backward()

    grad = _get_flat_grads(params)
    return float(loss.item()), grad


class GPyTorchBackend(GPBackend):
    """
    GPyTorch implementation of GPBackend.

    Parameters
    ----------
    bounded: bool, default True
        Keep hyperparameters in (low, high) via a sigmoid. False: positive only, like gpflow.
    """

    name = "gpytorch"

    def __init__(self, bounded=True):
        self.bounded = bounded

    def configure(self):
        """See GPBackend.configure. No-op: tensors are created in float64 explicitly."""

    def make_bounded_parameter(self, low, high, initial_value):
        """See GPBackend.make_bounded_parameter."""
        return BoundedValue(low, high, initial_value)

    # ------------------------------------------------------------------ helpers
    def _is_bounded(self, spec):
        """True if this spec's (low, high) bounds are enforced."""
        return self.bounded and spec.low is not None

    def _constraint(self, spec):
        if self._is_bounded(spec):
            low = torch.tensor(spec.low, dtype=_DTYPE)
            high = torch.tensor(spec.high, dtype=_DTYPE)
            return gpytorch.constraints.Interval(low, high)
        return gpytorch.constraints.Positive()

    def _initial(self, spec):
        """Start value, clipped just inside (low, high) when bounded."""
        value = np.asarray(spec.value, dtype=np.float64)
        if self._is_bounded(spec):
            eps = 1e-9 * (spec.high - spec.low)
            value = np.clip(value, spec.low + eps, spec.high - eps)
        return value

    @staticmethod
    def _as_spec(obj):
        """BoundedValue as-is, or wrap a plain value (positive only)."""
        if isinstance(obj, BoundedValue):
            return obj
        # low=None marks a spec with no bounds (see _is_bounded)
        spec = BoundedValue(0.0, math.inf, obj)
        spec.low = None
        return spec

    @staticmethod
    def _parts(model):
        """(ScaleKernel, base kernel, WhiteKernel) of a model from build_model."""
        scale, white = model.covar_module.kernels
        return scale, scale.base_kernel, white

    # --------------------------------------------------------------- interface
    def build_model(self, data, kernel_value, lengthscales, outputscale, white_var,
                    fix_lengthscale, fix_outputscale, noise_variance=1e-5):
        """See GPBackend.build_model. kernel_value: 3=RBF, 2=Matern32, else Matern52."""
        x_np, y_np = data
        train_x = torch.as_tensor(np.asarray(x_np, dtype=np.float64))
        train_y = torch.as_tensor(np.asarray(y_np, dtype=np.float64).reshape(-1))
        dim = train_x.shape[1]

        ls_spec = self._as_spec(lengthscales)
        os_spec = self._as_spec(outputscale)
        wv_spec = self._as_spec(white_var)

        ls_constraint = self._constraint(ls_spec)
        if kernel_value == 3:
            base = gpytorch.kernels.RBFKernel(ard_num_dims=dim, lengthscale_constraint=ls_constraint)
        elif kernel_value == 2:
            base = gpytorch.kernels.MaternKernel(
                nu=1.5, ard_num_dims=dim, lengthscale_constraint=ls_constraint
            )
        else:
            base = gpytorch.kernels.MaternKernel(
                nu=2.5, ard_num_dims=dim, lengthscale_constraint=ls_constraint
            )
        scale = gpytorch.kernels.ScaleKernel(base, outputscale_constraint=self._constraint(os_spec))
        white = WhiteKernel(variance_constraint=self._constraint(wv_spec))

        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(torch.tensor(0.0, dtype=_DTYPE))
        )
        model = _ExactGPModel(train_x, train_y, likelihood, scale + white).to(_DTYPE)

        # Set after .to(float64) so inverse transforms run in float64.
        # A scalar lengthscale is repeated for every input dimension: shape (1, dim).
        ls_initial = np.broadcast_to(self._initial(ls_spec), (dim,)).reshape(1, dim)
        base.lengthscale = torch.as_tensor(ls_initial.copy(), dtype=_DTYPE)  # copy: broadcast is read-only
        scale.outputscale = torch.as_tensor(float(self._initial(os_spec)), dtype=_DTYPE)
        white.variance = torch.as_tensor(float(self._initial(wv_spec)), dtype=_DTYPE)
        likelihood.noise = torch.as_tensor([float(noise_variance)], dtype=_DTYPE)

        # Noise is fixed, as in GpflowBackend.
        likelihood.raw_noise.requires_grad_(False)
        if fix_lengthscale:
            base.raw_lengthscale.requires_grad_(False)
        if fix_outputscale:
            scale.raw_outputscale.requires_grad_(False)
        return model

    def training_loss(self, model):
        """Negative log marginal likelihood, same as gpflow's GPR.training_loss()."""
        model.train()
        model.likelihood.train()
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
        n_data = model.train_targets.shape[0]
        with torch.no_grad(), _exact_settings():
            out = model(model.train_inputs[0])
            loss = -mll(out, model.train_targets) * n_data
            return float(loss)

    def train(self, model):
        """See GPBackend.train. scipy L-BFGS-B with torch autograd gradients."""
        model.train()
        model.likelihood.train()
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

        params = []
        for p in model.parameters():
            if p.requires_grad:
                params.append(p)
        if len(params) == 0:
            return True, self.training_loss(model)

        x0 = _get_flat_values(params)
        try:
            result = optimize.minimize(
                _loss_and_grad, x0, args=(model, mll, params), jac=True, method="L-BFGS-B",
                options={"maxiter": 10**9},
            )
        except (RuntimeError, ValueError):
            # Non-PSD matrix (NotPSDError is a RuntimeError) or NaN: GPEmulator.fit skips it.
            _set_flat_values(params, x0)
            return False, math.inf

        _set_flat_values(params, result.x)
        training_loss = self.training_loss(model)
        success = bool(result.success) and math.isfinite(training_loss)
        return success, training_loss

    def get_hyperparameters(self, model):
        """See GPBackend.get_hyperparameters."""
        scale, base, white = self._parts(model)
        lengthscale_final = base.lengthscale.detach().cpu().numpy().reshape(-1)
        noise_final = float(white.variance.detach().reshape(()))
        outputscale_final = float(scale.outputscale.detach().reshape(()))
        return [lengthscale_final, noise_final, outputscale_final]

    def make_posterior(self, model):
        """See GPBackend.make_posterior. The eval-mode model is the posterior."""
        model.eval()
        model.likelihood.eval()
        return model

    def predict_f(self, posterior, eval_points, full_cov=True):
        """See GPBackend.predict_f. Latent f: mean (n, 1), covariance (n, n) or (n, 1)."""
        x_eval = torch.as_tensor(np.asarray(eval_points, dtype=np.float64))
        with torch.no_grad(), _exact_settings():
            pred = posterior(x_eval)
            mean = pred.mean.cpu().numpy().reshape(-1, 1)
            if full_cov:
                covar = pred.covariance_matrix.cpu().numpy()
            else:
                covar = pred.variance.cpu().numpy().reshape(-1, 1)
        return mean, covar
