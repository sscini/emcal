"""Abstract GP-backend interface.

The GPBO algorithm (``GP_Emulator`` in ``GPBO_Classes_New``) owns all orchestration —
data scaling, the multi-restart training loop, hyperparameter initialization — and
delegates only the library-specific Gaussian-process primitives to a ``GPBackend``.
Model / parameter / posterior objects returned by a backend are opaque to the caller
and are only ever passed back to the same backend instance.

Implementations: GpflowBackend (default) and GPyTorchBackend. Select one with
``get_backend(name)`` or ``backend=`` on GPBODriver, build_gp_emulator, or diagnostics.fit_gp.
"""
from abc import ABC, abstractmethod


class GPBackend(ABC):
    """
    Abstract interface a GP library backend must implement (see module docstring).
    Model/parameter/posterior objects are opaque -- only ever passed back to the same
    backend instance, never introspected by the caller.

    Attributes
    ----------
    name : str
        Identifier used by gp_backend.get_backend(name) to select this backend.
    """

    name = "abstract"

    @abstractmethod
    def configure(self):
        """One-time global configuration (e.g. set default float precision)."""

    @abstractmethod
    def make_bounded_parameter(self, low, high, initial_value):
        """
        Return a trainable parameter constrained to (low, high) via a sigmoid transform.

        Parameters
        ----------
        low: float
            Lower bound of the parameter
        high: float
            Upper bound of the parameter
        initial_value: float
            Initial value of the parameter, must be within (low, high)

        Returns
        -------
        parameter: object
            Backend-specific trainable, bounded parameter object
        """

    @abstractmethod
    def build_model(self, data, kernel_value, lengthscales, outputscale, white_var,
                    fix_lengthscale, fix_outputscale, noise_variance=1e-5):
        """
        Build an (untrained) GP regression model.

        Parameters
        ----------
        data: tuple(np.ndarray, np.ndarray)
            (X, y) training data
        kernel_value: int
            3=RBF, 2=Matern32, else Matern52
        lengthscales: object
            Lengthscale parameter(s), from make_bounded_parameter
        outputscale: object
            Outputscale parameter, from make_bounded_parameter
        white_var: object
            Noise/white-kernel variance parameter, from make_bounded_parameter
        fix_lengthscale: bool
            If True, mark the lengthscale hyperparameter non-trainable
        fix_outputscale: bool
            If True, mark the outputscale hyperparameter non-trainable
        noise_variance: float, default 1e-5
            Initial noise variance

        Returns
        -------
        model: object
            Backend-specific (untrained) GP regression model
        """

    @abstractmethod
    def train(self, model):
        """
        Optimize the model hyperparameters.

        Parameters
        ----------
        model: object
            A model returned by build_model

        Returns
        -------
        success: bool
            Whether the optimizer converged
        training_loss: float
            The final training loss (negative log marginal likelihood)
        """

    @abstractmethod
    def get_hyperparameters(self, model):
        """
        Return the model's trained hyperparameters.

        Parameters
        ----------
        model: object
            A (trained) model returned by build_model

        Returns
        -------
        hyperparameters: list
            [lengthscale (np.ndarray), noise (float), outputscale (float)]
        """

    @abstractmethod
    def make_posterior(self, model):
        """
        Return a posterior object for fast repeated prediction.

        Parameters
        ----------
        model: object
            A (trained) model returned by build_model

        Returns
        -------
        posterior: object
            Backend-specific posterior object, passed to predict_f
        """

    @abstractmethod
    def predict_f(self, posterior, eval_points, full_cov=True):
        """
        Predict latent mean and (co)variance at eval_points.

        Parameters
        ----------
        posterior: object
            A posterior object returned by make_posterior
        eval_points: np.ndarray
            Points at which to evaluate the GP
        full_cov: bool, default True
            Whether to return the full covariance matrix

        Returns
        -------
        mean: np.ndarray
            Predicted latent mean at eval_points
        covar: np.ndarray
            Predicted latent (co)variance at eval_points, with the leading singleton
            dimension already squeezed out
        """
