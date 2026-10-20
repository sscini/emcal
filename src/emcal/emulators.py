"""GP emulators: the GPEmulator base class plus ObjectiveGP (GP fits the SSE objective;
Type-1) and EmulatorGP (GP emulates the model output; Type-2).
"""
import numpy as np
import pandas as pd
import math
from enum import Enum
from sklearn.preprocessing import RobustScaler
from .enums import Kernel
from .data import Data, SimulationData, GPPrediction
from .methods import GPBOMethod
from .exploration import ExplorationBias
from .acquisition import ExpectedImprovement
from .gp_backend import get_backend
from ._utils import blockwise_sse


class GPEmulator:
    """
    The base class for Gaussian Processes used in this workflow

    Methods
    --------------
    __init__(*) : Constructor method
    get_num_gp_data(): Defines the total number of all simulated accessible to the GP
    __set_lenscl_guess(lb, ub): Sets the lengthscale guess
    __set_white_kern(lb, ub): Sets the white kernel guess
    __set_outputscl(lb, ub): Sets the outputscale guess
    set_gp_model_data(): Sets training data for the GP model data
    __init_hyper_parameters(retrain_count): Initializes hyperparameters for the GP model
    set_gp_model(retrain_count): Builds the GP model
    fit(): Trains the GP model
    __eval_gp_mean_var(data): Evaluates the mean and variance of the GP model
    predict(target=None, data=None, featurized_data=None, covar=False): Evaluates the GP mean and (co)variance for a data set
    """

    # Class variables and attributes

    def __init__(
        self,
        gp_sim_data,
        gp_val_data,
        cand_data,
        kernel,
        lengthscale,
        noise_std,
        outputscale,
        retrain_gp,
        set_seed,
        normalize,
        backend=None,
    ):
        """
        Parameters
        ----------
        gp_sim_data: Data
            All simulation data for the GP
        gp_val_data: Data
            The validation data for the GP. None if not saving validation data
        cand_data: Data
            Candidate theta value for evaluation with GPBODriver.opt_with_scipy()
        kernel: Kernel
            Determines which GP Kerenel to use
        lengthscale: float or None
            Value of the lengthscale hyperparameter - None if hyperparameters will be updated during training
        noise_std: float, int
            The standard deviation of the noise
        outputscale: float or None
            Determines value of outputscale
        retrain_gp: int
            Number of times to (re)do GP training. If 0, no training is done and default/initial values are used
        set_seed: int or None
            Random seed
        normalize: bool
            Determines whether data is standardized (using the sklearn RobustScaler)
        backend: GPBackend, str, or None, default None
            GP backend instance or name ("gpflow", "gpytorch"). None means gpflow.
            Tests inject a fake backend here.

        Raises
        ------
        AssertionError
            If any of the inputs are not of the correct type or value
        """
        # Assert statements
        # Check for int/float
        assert (
            isinstance(outputscale, (float, int)) or outputscale is None
        ), "outputscale must be float, int, or None"
        # Check that set values for outputscale are in range
        if outputscale is not None:
            assert (
                100 > outputscale > 1e-5
            ), "outputscale must be in range [1e-5,1e3] if it is not None"

        # Check lengthscale, float, int, array, or None
        if isinstance(lengthscale, list):
            lengthscale = np.array(lengthscale)
        assert (
            isinstance(lengthscale, (float, int, np.ndarray)) or lengthscale is None
        ), "lengthscale must be float, int, np.ndarray, or None"
        # Check that set values for lengthscale are in range
        if lengthscale is not None:
            if isinstance(lengthscale, (float, int)):
                assert (
                    1000 > lengthscale > 1e-5
                ), "lengthscale must be in range [1e-5,1e3] if lengthscale is not None"
            else:
                assert all(
                    isinstance(var, (np.int64, np.float64, float, int))
                    for var in lengthscale
                ), "All lengthscale elements must float or int"
                assert all(
                    1000 > item > 1e-5 for item in lengthscale
                ), "lengthscale elements must be in range [1e-5,1e3] if lengthscale is not None"
                lengthscale = lengthscale.astype(np.float64)  # Convert all guesses to float64

        assert isinstance(normalize, bool), "normalize must be bool"
        assert (
            isinstance(retrain_gp, int) == True and retrain_gp >= 0
        ), "retrain_gp must be int greater than or equal to 0"
        # Check for Enum
        assert isinstance(kernel, Enum) == True, "kernel must be type Enum"
        # Check for instance of Data class or None
        assert (
            isinstance(gp_sim_data, (Data)) == True or gp_sim_data == None
        ), "gp_sim_data must be an instance of the Data class or None"
        assert (
            isinstance(gp_val_data, (Data)) == True or gp_val_data == None
        ), "gp_sim_data must be an instance of the Data class or None"

        # Constructor method
        self.gp_sim_data = gp_sim_data
        self.gp_val_data = gp_val_data
        self.cand_data = cand_data
        self.kernel = kernel
        self.lengthscale = lengthscale
        self.noise_std = noise_std
        self.outputscale = outputscale
        self.retrain_gp = retrain_gp
        self.seed = set_seed
        self.normalize = normalize
        # If normalize, create the scalers
        if normalize == True:
            self.scaler_x = RobustScaler(unit_variance=True)
            self.scaler_y = RobustScaler(unit_variance=True)

        # Resolve the GP library backend (gpflow by default). Selecting it here keeps the
        # heavy gpflow/TF import lazy: it only happens when a GPEmulator is constructed.
        # Override by setting the `gp_package` class/instance attribute before construction,
        # or (e.g. in tests) by passing `backend=` directly to bypass the registry entirely.
        # Also accept a backend name, e.g. backend="gpytorch".
        if isinstance(backend, str):
            backend = get_backend(backend)
        self._backend = (
            backend if backend is not None
            else get_backend(getattr(self, "gp_package", "gpflow"))
        )

        self.rng_rand = np.random.default_rng()
        if self.seed != None:
            self.rng_set = np.random.default_rng(self.seed)
        else:
            self.rng_set = self.rng_rand

    def get_num_gp_data(self):
        """
        Defines the total number of data the GP will have access to

        Returns
        -------
        num_data: int
            The number of data the GP will have access to

        Raises
        ------
        AssertionError
            If self.gp_sim_data is not an instance of the Data class
        """
        assert isinstance(
            self.gp_sim_data, Data
        ), "self.gp_sim_data must be instance of Data class"
        # Number of available gp data determined by number of sim data
        num_gp_data = int(self.gp_sim_data.n_theta)

        return num_gp_data

    def get_dim_gp_data(self):
        """
        Defines the total dimension of the input data used by the GP

        Returns
        -------
        dim_gp_data: int
            The dimensions of the input data that the GP will use

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value

        Notes
        -----
        The dimension count itself is subclass-specific (Type 1 uses only theta;
        Type 2 also carries x), so this delegates to the _gp_input_dim() hook.
        """
        return self._gp_input_dim()

    def bounded_parameter(self, low, high, initial_value):
        """
        Creates a bounded parameter for the GP model

        Parameters
        ----------
        low: float, int
            Lower bound of the parameter
        high: float, int
            Upper bound of the parameter
        initial_value: float, int, np.ndarray
            Initial value of the parameter

        Returns
        -------
        parameter: gpflow.Parameter
            The bounded parameter

        Raises
        ------
        AssertionError
            If the lower bound is higher than the upper bound
        """
        assert isinstance(low, (float, int)), "low must be float or int"
        assert isinstance(high, (float, int)), "low must be float or int"
        assert isinstance(
            initial_value, (float, int, np.ndarray)
        ), "initial_value must be float, int, or np.ndarray of shape (n,)"
        if isinstance(initial_value, np.ndarray):
            assert len(initial_value.shape) <= 1, "initial_value must be a scalar or 1D array"
        assert low < high, "low must be less than high"
        return self._backend.make_bounded_parameter(low, high, initial_value)

    def __set_lenscl_guess(self, lb, ub):
        """
        Sets the lengthscale guess for the GP model

        Parameters
        ----------
        lb: float, int
            Lower bound of the lengthscale
        ub: float, int
            Upper bound of the lengthscale

        Returns
        --------
        lenscl_guess: np.ndarray
            The intial lengthscale of the GP model

        Raises
        ------
        AssertionError
            If self.train_data_init is not an array
        """
        rng = self.rng_set
        # Set lengthscale bounds using the original training data to ensure distance
        # Between min and max lengthscales does not collapse as iterations progress
        assert isinstance(
            self.train_data_init, np.ndarray
        ), "self.train_data_init must be an array"
        if self.normalize:
            org_scalerX = RobustScaler(unit_variance=True)
            points = org_scalerX.fit_transform(self.train_data_init)
        else:
            points = self.train_data_init

        # Compute pairwise differences for each column
        pairwise_diffs = np.abs(
            points[:, :, None] - points[:, :, None].transpose(0, 2, 1)
        )
        # Compute Euclidean distances
        euclidean_distances = np.sqrt(np.sum(pairwise_diffs**2, axis=1))
        # Set diagonal elements (distance between the same point) to infinity
        np.fill_diagonal(euclidean_distances, np.inf)
        euclidean_distances = np.ma.masked_invalid(euclidean_distances)
        # Find the smallest/largest distance for each column and ensure it is within the bounds
        min_distance = np.min(euclidean_distances, axis=0)
        max_distance = np.max(euclidean_distances, axis=0)

        lb_array = np.ones(len(min_distance)) * lb
        ub_array = np.ones(len(max_distance)) * ub
        lower = np.maximum(min_distance, lb_array)
        upper = np.minimum(max_distance, ub_array)

        lenscl_guess = rng.uniform(lower, upper, size=len(max_distance))
        return lenscl_guess

    def __set_white_kern(self, lb, ub):
        """
        Sets the white kernel value guess for the GP model

        Parameters
        ----------
        lb: float, int
            Lower bound of the white kernel
        ub: float, int
            Upper bound of the white kernel

        Returns
        --------
        noise_guess: float
            The initial white noise variance for the GP model
        """
        # Set the noise guess or allow gp to tune the noise parameter
        if self.normalize:
            self.scaler_y.fit(self.train_data.y_vals.reshape(-1, 1))
            sclr = np.float64(self.scaler_y.scale_.item())
        else:
            sclr = 1.0

        if self.noise_std is not None:
            # If we know the noise, use it
            noise_guess = float((self.noise_std / sclr) ** 2)

        else:
            # Otherwise, set the guess as 1% the taining data median
            if not math.isclose(np.median(self.gp_sim_data.y_vals),0):
                data_mean = np.abs(np.median(self.gp_sim_data.y_vals))
            elif not math.isclose(np.mean(self.gp_sim_data.y_vals),0):
                data_mean = np.abs(np.mean(self.gp_sim_data.y_vals))
            else:
                data_mean = np.max(np.abs(self.gp_sim_data.y_vals))
            noise_guess = np.float64(data_mean * 0.01 / sclr) ** 2

        if not lb < noise_guess < ub:
            noise_guess = 1.0

        return noise_guess

    def __set_outputscl(self, lb, ub):
        """
        Set the initial output scale of the model

        Parameters
        ----------
        lb: float, int
            Lower bound of the output scale
        ub: float, int
            Upper bound of the output scale

        Returns
        -------
        outputscale: float
            Initial output scale guess for the GP model

        Notes
        ------
        Need to have training data before using this function
        """

        # Set outputscale kernel to be optimized based on guess if desired
        if self.outputscale == None:
            train_y = self.train_data.y_vals.reshape(-1, 1)
            if self.normalize:
                scl_y = self.scaler_y.fit_transform(train_y)
            else:
                scl_y = train_y

            c_guess = sum(scl_y.flatten() ** 2) / len(scl_y)
            outputscale = c_guess

        elif isinstance(self.outputscale, (float, int, np.float64)):
            assert self.outputscale > 0, "outputscale must be positive int or float"
            outputscale = self.outputscale
        else:
            outputscale = 1.0

        if not lb < outputscale < ub:
            outputscale = 1.0

        return outputscale

    def set_gp_model_data(self):
        """
        Sets the training data for the GP model

        Returns
        -------
        data: tuple(np.ndarrays, len=2)
            The feature and output training data for the GP model

        Raises
        ------
        AssertionError
            If self.feature_train_data or self.train_data.y_vals are not defined
        """
        assert (
            self.feature_train_data is not None
        ), "self.feature_train_data must be defined"
        assert (
            self.train_data.y_vals is not None
        ), "self.train_data.y_vals must be defined"
        # Set new model data
        # Preprocess Training data
        if self.normalize == True:
            # Update scaler to be the fitted scaler. This scaler will change as the training data is updated
            # Scale training data if necessary
            ft_td_scl = self.scaler_x.fit_transform(self.feature_train_data)
            y_td_scl = self.scaler_y.fit_transform(self.train_data.y_vals.reshape(-1, 1))
        else:
            ft_td_scl = self.feature_train_data
            y_td_scl = self.train_data.y_vals.reshape(-1, 1)
        data = (ft_td_scl, y_td_scl)
        return data

    def __init_hyper_parameters(self, retrain_count):
        """
        Initializes hyperparameters for the GP model

        Parameters
        ----------
        retrain_count: int
            The number of times the GP will be (re)trained

        Returns
        --------
        lengthscales: np.ndarray
            The initial lengthscale of the GP model
        outputscale: float
            The initial output scale guess for the GP model
        white_var: float
            The initial white noise variance for the GP model
        """
        self._backend.configure()
        rng = self.rng_set

        # Set bounds for hyperparameters
        lenscl_bnds = [0.00001, 1000.0]
        var_bnds = [0.00001, 100.0]
        white_var_bnds = [0.00001, 10.0]

        # Get X and Y Data
        data = self.set_gp_model_data()
        x_train, y_train = data

        if isinstance(self.lengthscale, np.ndarray):
            lengthscales = self.bounded_parameter(
                lenscl_bnds[0], lenscl_bnds[1], self.lengthscale
            )
        elif isinstance(self.lengthscale, (int, float)):
            lengthscales = np.ones(x_train.shape[1]) * self.bounded_parameter(
                lenscl_bnds[0], lenscl_bnds[1], self.lengthscale
            )
        if self.outputscale is not None:
            outputscale = self.bounded_parameter(var_bnds[0], var_bnds[1], self.outputscale)

        # On the 1st iteration, use initial guesses initialized to 1
        if retrain_count == 0:
            if self.lengthscale is None:
                lengthscale_1 = self.bounded_parameter(
                    lenscl_bnds[0], lenscl_bnds[1], 1.0
                )
                lengthscales = np.ones(x_train.shape[1]) * lengthscale_1
            if self.outputscale is None:
                outputscale = self.bounded_parameter(var_bnds[0], var_bnds[1], 1.0)
            white_var = self.bounded_parameter(
                white_var_bnds[0], white_var_bnds[1], 1.0
            )
        # On second iteration, base guesses on initial data values
        elif retrain_count == 1:
            if self.lengthscale is None:
                initial_lenscls = np.array(
                    self.__set_lenscl_guess(lenscl_bnds[0], lenscl_bnds[1]),
                    dtype="float64",
                )
                lengthscales = self.bounded_parameter(
                    lenscl_bnds[0], lenscl_bnds[1], initial_lenscls
                )
            if self.outputscale is None:
                initial_tau = np.array(
                    self.__set_outputscl(var_bnds[0], var_bnds[1]), dtype="float64"
                )
                outputscale = self.bounded_parameter(var_bnds[0], var_bnds[1], initial_tau)
            initial_white_var = np.array(
                self.__set_white_kern(white_var_bnds[0], white_var_bnds[1]),
                dtype="float64",
            )
            white_var = self.bounded_parameter(
                white_var_bnds[0], white_var_bnds[1], initial_white_var
            )
        # On all other iterations, use random guesses
        else:
            if self.lengthscale is None:
                initial_lenscls = np.array(
                    rng.uniform(0.1, 100.0, x_train.shape[1]), dtype="float64"
                )
                lengthscales = self.bounded_parameter(
                    lenscl_bnds[0], lenscl_bnds[1], initial_lenscls
                )
            if self.outputscale is None:
                outputscale = self.bounded_parameter(
                    var_bnds[0],
                    var_bnds[1],
                    np.array(rng.lognormal(0.0, 1.0), dtype="float64"),
                )
            white_var = self.bounded_parameter(
                white_var_bnds[0],
                white_var_bnds[1],
                np.array(rng.uniform(0.05, 10), dtype="float64"),
            )
        return lengthscales, outputscale, white_var

    def set_gp_model(self, retrain_count):
        """
        Generates the GP model for the process in sklearn

        Parameters
        ----------
        retrain_count: int
            The number of times the GP will be (re)trained

        Returns
        --------
        gp_model: gpflow.models.GPR
            The untrained GP model with all hyperparameters set

        Raises
        ------
        AssertionError
            If retrains are not an integer greater than or equal to 0
        """

        assert (
            isinstance(retrain_count, int) and retrain_count >= 0
        ), "retrain_count must be an int greater than or equal to 0"
        data = self.set_gp_model_data()
        lengthscales, outputscale, white_var = self.__init_hyper_parameters(retrain_count)

        fix_lengthscale = isinstance(self.lengthscale, (np.ndarray, float, int))
        fix_outputscale = self.outputscale is not None
        gp_model = self._backend.build_model(
            data,
            self.kernel.value,
            lengthscales,
            outputscale,
            white_var,
            fix_lengthscale,
            fix_outputscale,
            noise_variance=10**-5,
        )

        return gp_model

    def fit(self):
        """
        Trains the GP with restarts given training data.

        Raises
        ------
        AssertionError
            If self.feature_train_data is not an np.ndarray or is undefined

        Notes
        ------
        Sets the following parameters of self
        self.trained_hyperparams: list, the trained hyperparameters of the GP model
        self.fit_gp_model:  gpflow.models.GPR, the trained GP model
        self.posterior:  gpflow.mean_field.KFGaussian, the posterior of the GP model
        """
        assert isinstance(
            self.feature_train_data, np.ndarray
        ), "self.feature_train_data must be np.ndarray"
        assert (
            self.feature_train_data is not None
        ), "Must have training data. Run split_train_test() to generate"

        # Train the model multiple times and keep track of the model with the lowest minimum training loss
        best_minimum_loss = float("inf")
        best_model = None

        # If we are not retraining the GP, set the model once with default/set hyperparameters
        if self.retrain_gp == 0:
            best_model = self.set_gp_model(0)
        # Otherwise train the model and keep the best model over all retrains
        else:
            # While you still have retrains left
            for i in range(self.retrain_gp):
                # Create and fit the model
                gp_model = self.set_gp_model(i)
                # Train hyperparameters via the GP backend. compile=False (eager) is used for
                # reliability + determinism — see GpflowBackend.train / refactor_notes Phase 2a.
                success, training_loss = self._backend.train(gp_model)
                if i == 0:
                    first_model = gp_model
                    first_loss = training_loss
                if success:
                    # Check if this model has the best minimum training loss
                    if training_loss < best_minimum_loss:
                        best_minimum_loss = training_loss
                        best_model = gp_model

            # If we have no good models, use the first one
            if best_model is None:
                best_model = first_model
                best_minimum_loss = first_loss

        # Pull out kernel parameters after GP training: [lengthscale, noise, outputscale]
        trained_hyperparams = self._backend.get_hyperparameters(best_model)

        # Assign self parameters
        self.trained_hyperparams = trained_hyperparams
        self.fit_gp_model = best_model
        self.posterior = self._backend.make_posterior(self.fit_gp_model)

        # gpflow.utilities.print_summary(best_model)

    def __eval_gp_mean_var(self, data):
        """
        Calculates the GP mean and variance for a given input set and adds it to the instance of the data class

        Parameters
        -----------
        data: Data
            Data to evaluate GP for containing at least parameter sets (data.theta_vals) and state points (data.x_vals)

        Returns
        -------
        gp_mean: np.ndarray
            GP mean prediction for the data set
        gp_var: np.ndarray
            GP variance prediction for the data set
        gp_covar: np.ndarray
            GP covariance prediction for the data set

        """
        # Get data in vector form into array form
        if len(data.shape) < 2:
            data.reshape(1, -1)
        # scale eval_point if necessary
        if self.normalize == True:
            eval_points = self.scaler_x.transform(data)
        else:
            eval_points = data

        # Evaluate GP given parameter set theta and state point value (mean + full covariance).
        # The backend handles tensor conversion and returns numpy arrays with the leading
        # singleton covariance dimension already squeezed out.
        gp_mean_scl, gp_covar_scl = self._backend.predict_f(
            self.posterior, eval_points, full_cov=True
        )

        # Unscale gp_mean and gp_covariance
        if self.normalize == True:
            gp_mean = self.scaler_y.inverse_transform(
                gp_mean_scl.reshape(-1, 1)
            ).flatten()
            gp_covar = float(self.scaler_y.scale_.item() ** 2) * gp_covar_scl
        else:
            gp_mean = gp_mean_scl
            gp_covar = gp_covar_scl

        gp_var = np.diag(gp_covar)

        return gp_mean, gp_var, gp_covar

    def predict(self, target=None, data=None, featurized_data=None, covar=False):
        """
        Evaluate the GP posterior mean and (co)variance.

        Collapses the former eval_gp_mean_var_{test,val,cand,misc}. Choose a built-in
        `target` ("test", "val", or "cand") to evaluate the emulator's own split, or pass
        an arbitrary `data` (a Data instance) -- e.g. a heat-map / diagnostic set -- with
        an optional pre-featurized `featurized_data` (computed via featurize_data() if omitted).

        Parameters
        ----------
        target : {"test", "val", "cand"} or None
            Which built-in data split to evaluate. None => use `data`.
        data : Data or None
            The data to evaluate when `target` is None.
        featurized_data : np.ndarray or None
            Pre-featurized form of `data`; computed via featurize_data(data) if None.
        covar : bool, default False
            Return the full covariance (True) or the marginal variance (False).

        Returns
        -------
        GPPrediction
            .mean, .variance, and .covariance are always populated (the backend computes
            the full covariance unconditionally); `covar` only selects which one appears
            as the second element of the legacy 2-tuple iteration.
        """
        if target == "test":
            data, featurized_data = self.test_data, self.feature_test_data
        elif target == "val":
            data, featurized_data = self.gp_val_data, self.feature_val_data
        elif target == "cand":
            data, featurized_data = self.cand_data, self.feature_cand_data
        elif target is not None:
            raise ValueError("target must be 'test', 'val', 'cand', or None")

        assert isinstance(data, Data), "data must be type Data"
        if featurized_data is None:
            featurized_data = self.featurize_data(data)
        assert isinstance(
            featurized_data, np.ndarray
        ), "featurized_data must be np.ndarray"
        assert len(featurized_data) > 0, "Must have data"
        assert isinstance(covar, bool), "covar must be bool!"

        gp_mean, gp_var, gp_covar = self.__eval_gp_mean_var(featurized_data)

        return GPPrediction(gp_mean, variance=gp_var, covariance=gp_covar, as_covar=covar)

    def _resolve_target(self, target, data):
        """
        Resolve a built-in `target` to the corresponding stored Data instance.

        Shared by predict_sse/expected_improvement (both subclasses), which each had an
        identical target -> {"test", "val", "cand"} dispatch block.

        Parameters
        ----------
        target: {"test", "val", "cand"} or None
            Which built-in data split to use. None => return `data` unchanged.
        data: Data or None
            Passed through unchanged when target is None.

        Returns
        -------
        Data or None

        Raises
        ------
        ValueError
            If target is not one of "test", "val", "cand", or None.
        """
        if target == "test":
            return self.test_data
        elif target == "val":
            return self.gp_val_data
        elif target == "cand":
            return self.cand_data
        elif target is not None:
            raise ValueError("target must be 'test', 'val', 'cand', or None")
        return data

    def _extend_train_data_arrays(self, new_point_data):
        """
        Hook for growing any train_data fields beyond theta_vals/y_vals when a new
        training point is appended. No-op by default (ObjectiveGP's training data is
        theta_vals/y_vals only); EmulatorGP overrides this to also grow x_vals (Type-2
        carries per-point x alongside theta).
        """
        pass

    def featurize_data(self, data):
        """
        Collects the features of the GP into ndarray form from an instance of the Data class.

        Abstract hook: the feature construction genuinely differs by GP type (Type 1 uses
        theta_vals only; Type 2 concatenates theta_vals and x_vals), so each subclass
        provides its own implementation rather than sharing a base body.

        Parameters
        ----------
        data: Data
            Data to evaluate GP for

        Returns
        -------
        feature_eval_data: np.ndarray
            The feature data for the GP
        """
        raise NotImplementedError

    def _select_train_test_rows(self, train_idx, test_idx):
        """
        Hook for selecting the train/test theta, x, and y rows from self.gp_sim_data given
        the theta-level train_idx/test_idx produced by self.gp_sim_data.train_test_idx_split.

        Genuinely differs by GP type: Type 1's x_vals is shared experimental data (not
        indexed at all), while Type 2 must expand the theta-level indices to full
        (theta, x)-row boolean masks via get_unique_theta() + np.isin before indexing
        theta_vals/x_vals/y_vals.

        Parameters
        ----------
        train_idx: np.ndarray
        test_idx: np.ndarray

        Returns
        -------
        theta_train, x_train, y_train, theta_test, x_test, y_test: np.ndarray
        """
        raise NotImplementedError

    def split_train_test(self, sep_fact, shuffle_seed=None):
        """
        Splits simulated data into training and testing data

        Parameters
        ----------
        sep_fact: float or int
            The separation factor that decides what percentage of data will be training data. Between 0 and 1.
        set_seed: int or None
            Determines seed for randomizations. None if seed is random

        Returns
        -------
        train_data: Data
            Contains all input/output data and bounds for training data
        test_data: Data
            Contains all input/output data and bounds for testing data

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value

        Notes
        -----
        Sets self.train_data, self.test_data, self.feature_train_data, self.feature_test_data, and self.feature_val_data
        """
        assert isinstance(
            sep_fact, (float, int)
        ), "Separation factor must be float or int > 0"
        assert 0 < sep_fact <= 1, "sep_fact must be > 0 and less than or equal to 1!"
        assert isinstance(
            self.gp_sim_data, Data
        ), "self.gp_sim_data must be instance of Data"
        assert np.all(
            self.gp_sim_data.x_vals is not None
        ), "Must have simulation x, theta, and y data to create train/test data"
        assert np.all(
            self.gp_sim_data.theta_vals is not None
        ), "Must have simulation x, theta, and y data to create train/test data"
        assert np.all(
            self.gp_sim_data.y_vals is not None
        ), "Must have simulation x, theta, and y data to create train/test data"
        assert np.all(
            self.gp_sim_data.bounds_x is not None
        ), "Must have simulation x bounds to create train/test data"
        assert np.all(
            self.gp_sim_data.bounds_theta is not None
        ), "Must have simulation theta bounds to create train/test data"

        # Get train test idx
        train_idx, test_idx = self.gp_sim_data.train_test_idx_split(shuffle_seed)

        # Select the train/test theta, x, and y rows (subclass-specific)
        theta_train, x_train, y_train, theta_test, x_test, y_test = (
            self._select_train_test_rows(train_idx, test_idx)
        )

        # Get train data and set it as an instance of the data class
        train_data = SimulationData(
            theta_train,
            x_train,
            y_train,
            bounds_theta=self.gp_sim_data.bounds_theta,
            bounds_x=self.gp_sim_data.bounds_x,
            sep_fact=sep_fact,
        )
        self.train_data = train_data

        # Get test data and set it as an instance of the data class
        test_data = SimulationData(
            theta_test,
            x_test,
            y_test,
            bounds_theta=self.gp_sim_data.bounds_theta,
            bounds_x=self.gp_sim_data.bounds_x,
            sep_fact=sep_fact,
        )
        self.test_data = test_data

        # Set training and validation data features in GPEmulator base class
        feature_train_data = self.featurize_data(train_data)
        feature_test_data = self.featurize_data(test_data)

        self.feature_train_data = feature_train_data
        self.feature_test_data = feature_test_data

        if self.gp_val_data is not None:
            feature_val_data = self.featurize_data(self.gp_val_data)
            self.feature_val_data = feature_val_data

        # Set the initial training data for the GP Emulator upon creation
        if self.train_data_init is None:
            self.train_data_init = feature_train_data

        return train_data, test_data

    def append_training_point(self, new_point_data):
        """
        Adds parameter set which optimizes the acquisition function to the training data set

        Parameters
        ----------
        new_point_data: Data
            The class containing the data relevant to argmin(acq. func.) for the GP

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value

        Notes
        ------
        This function updates self.train_data.theta_vals, self.train_data.y_vals (and, for
        EmulatorGP, self.train_data.x_vals via _extend_train_data_arrays), and
        self.feature_train_data
        """
        assert self.train_data is not None, "self.train_data must be Data"
        assert isinstance(self.train_data, Data), "self.train_data must be Data"
        assert isinstance(new_point_data, Data), "new_point_data must be Data"
        assert all(
            isinstance(var, np.ndarray)
            for var in [self.train_data.theta_vals, self.train_data.y_vals]
        ), "self.train_data.theta_vals and self.train_data.y_vals must be np.ndarray"
        assert all(
            isinstance(var, np.ndarray)
            for var in [new_point_data.theta_vals, new_point_data.y_vals]
        ), "new_point_data.theta_vals and new_point_data.y_vals must be np.ndarray"

        # Update training theta, x, and y separately
        self.train_data.theta_vals = np.vstack(
            (self.train_data.theta_vals, new_point_data.theta_vals)
        )
        self._extend_train_data_arrays(new_point_data)
        self.train_data.y_vals = np.concatenate(
            (self.train_data.y_vals, new_point_data.y_vals)
        )
        feature_train_data = self.featurize_data(self.train_data)

        # Reset training data feature array
        self.feature_train_data = feature_train_data

    def _sse_from_prediction(self, data, covar=False, prediction=None, **kw):
        """
        Hook computing (sse, sse_var, sse_covar) for `data`, given (or deriving) a
        GPPrediction. Genuinely differs by GP type: Type 1's SSE *is* the GP output;
        Type 2 must derive SSE from the GP-predicted model output via a blockwise
        sum-of-squared-errors reduction against experimental data (hence the extra
        `method`/`exp_data` kwargs it requires, threaded through predict_sse's **kw).

        Parameters
        ----------
        data: Data
        covar: bool, default False
        prediction: GPPrediction or None, default None
        **kw: subclass-specific extra arguments (e.g. EmulatorGP's method, exp_data)

        Returns
        -------
        sse, sse_var, sse_covar
        """
        raise NotImplementedError

    def predict_sse(self, target=None, data=None, covar=False, prediction=None, **kw):
        """
        GP-predicted SSE mean and (co)variance.

        Collapses the former eval_gp_sse_var_{test,val,cand,misc}. Choose a built-in
        `target` ("test"/"val"/"cand") or pass an arbitrary `data`. Extra keyword
        arguments (e.g. EmulatorGP's `method`/`exp_data`) are forwarded to the
        _sse_from_prediction hook.

        Parameters
        ----------
        prediction: GPPrediction or None, default None
            The GPPrediction returned by a prior predict() call on the same data, reused
            instead of computing a fresh one. If None, this calls predict(data=data)
            internally.

        Returns
        -------
        GPPrediction

        Raises
        ------
        AssertionError
            If covar is not a boolean, or (subclass-specific) required parameters are
            missing or not of the correct type or value
        """
        data = self._resolve_target(target, data)
        assert isinstance(covar, bool), "covar must be bool!"

        sse, sse_var, sse_covar = self._sse_from_prediction(
            data, covar=covar, prediction=prediction, **kw
        )
        return GPPrediction(sse, variance=sse_var, covariance=sse_covar, as_covar=covar)

    def _compute_ei(self, sim_data, exp_data, ep_bias, best_error_metrics, gp_prediction=None, **kw):
        """
        Hook evaluating the expected-improvement acquisition function for `sim_data`.
        Near-identical between subclasses (predict-if-none -> build
        ExpectedImprovement(...) -> .compute()), but EmulatorGP requires extra `method`/
        `sg_mc_samples` arguments and an upfront method-range validation, so each
        subclass keeps its own body rather than forcing a shared implementation.

        Parameters
        ----------
        sim_data: Data
        exp_data: Data
        ep_bias: ExplorationBias
        best_error_metrics: tuple(float, np.ndarray, np.ndarray)
        gp_prediction: GPPrediction or None, default None
        **kw: subclass-specific extra arguments (e.g. EmulatorGP's method, sg_mc_samples)

        Returns
        -------
        ei, ei_terms_df
        """
        raise NotImplementedError

    def expected_improvement(self, target=None, data=None, exp_data=None, ep_bias=None,
                              best_error_metrics=None, gp_prediction=None, **kw):
        """
        Expected-improvement acquisition function.

        Collapses the former eval_ei_{test,val,cand,misc}. Choose a built-in `target`
        ("test"/"val"/"cand") or pass an arbitrary `data`. `best_error_metrics` is the
        (best_error, best_theta, best_error_x) tuple from calc_best_error(). Extra
        keyword arguments (e.g. EmulatorGP's `method`/`sg_mc_samples`) are forwarded to
        the _compute_ei hook.

        Parameters
        ----------
        gp_prediction: GPPrediction or None, default None
            The GPPrediction returned by a prior predict() call on the same data, reused
            instead of computing a fresh one. If None, this calls predict(data=data)
            internally.

        Returns
        -------
        ei : np.ndarray
        ei_terms_df : pd.DataFrame

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or
            value
        """
        data = self._resolve_target(target, data)

        assert isinstance(data, Data), "data must be type Data"
        assert isinstance(exp_data, Data), "exp_data must be type Data"
        assert isinstance(
            ep_bias, ExplorationBias
        ), "ep_bias must be type Exploration_bias"
        assert (
            isinstance(best_error_metrics, tuple) and len(best_error_metrics) == 3
        ), "Error metric must be a tuple of length 3"

        return self._compute_ei(
            data, exp_data, ep_bias, best_error_metrics, gp_prediction=gp_prediction, **kw
        )


class ObjectiveGP(GPEmulator):
    """
    The base class for Gaussian Processes
    Parameters

    Methods
    --------------
    __init__(*): Constructor method
    get_dim_gp_data (inherited from GPEmulator): Defines the total dimension of data used by the GP
    _gp_input_dim(): Type 1 GP input is theta only, so the dimension is just theta_dim
    featurize_data(data) (overrides abstract GPEmulator.featurize_data): Collects the featues of the GP into ndarray form from an instance of the Data class
    split_train_test(sep_fact, seed) (inherited from GPEmulator): Finds the simulation data to use as training/testing data
    _select_train_test_rows(train_idx, test_idx): Type 1's x_vals is shared experimental data, so it is not indexed at all
    predict_sse(target=None, data=None, covar=False) (inherited from GPEmulator): GP-predicted SSE mean and (co)variance
    _sse_from_prediction(data, covar, prediction): For Type 1, sse is the gp_mean, so this is a direct passthrough of the prediction
    calc_best_error(): Calculates the best error metrics for the GP
    expected_improvement(target=None, data=None, exp_data=None, ep_bias=None, best_error_metrics=None) (inherited from GPEmulator): Expected improvement acquisition
    _compute_ei(sim_data, exp_data, ep_bias, best_error_metrics): Predict-if-none, build ExpectedImprovement(...), .compute()

    append_training_point (inherited from GPEmulator): Adds the next parameter set to the training data
    """

    # Class variables and attributes

    def __init__(
        self,
        gp_sim_data,
        gp_val_data,
        cand_data,
        train_data,
        test_data,
        kernel,
        lengthscale,
        noise_std,
        outputscale,
        retrain_gp,
        set_seed,
        normalize,
        backend=None,
    ):
        """
        Parameters
        ----------
        gp_sim_data: Data
            All simulation data for the GP
        gp_val_data: Data
            The validation data for the GP. None if not saving validation data
        cand_data: Data
            Candidate theta value for evaluation with GPBODriver.opt_with_scipy()
        train_data: Data
            The training data for the GP
        test_data: Data
            The testing data for the GP
        kernel: Kernel
            Determines which GP Kerenel to use
        lengthscale: float or None
            Value of the lengthscale hyperparameter - None if hyperparameters will be updated during training
        noise_std: float, int
            The standard deviation of the noise
        outputscale: float or None
            Determines value of outputscale - None if hyperparameters will be updated during training
        retrain_gp: int
            Number of times to (re)train the GP. If 0, the GP is not trained and default/initial hyperparameters are used
        set_seed: int or None
            Random seed
        normalize: bool
            Determines whether data is standardized (with sklearn RobustScaler) before training
        backend: GPBackend or None, default None
            The GP backend to use. If None, resolved via get_backend (gpflow by default).

        Raises
        ------
        AssertionError
            If any of the inputs are not of the correct type or value
        """
        # Constructor method
        # Inherit objects from GPEmulator Base Class
        super().__init__(
            gp_sim_data,
            gp_val_data,
            cand_data,
            kernel,
            lengthscale,
            noise_std,
            outputscale,
            retrain_gp,
            set_seed,
            normalize,
            backend=backend,
        )
        # Add training and testing data as child features
        self.train_data = train_data
        self.test_data = test_data
        self.train_data_init = (
            None  # Will be populated with the 1st instance of train data
        )

    def _gp_input_dim(self):
        """
        Type 1 GP input is theta only, so the dimension is just theta_dim.
        """
        assert np.all(
            self.gp_sim_data.theta_vals is not None
        ), "self.gp_sim_data.theta_vals must exist!"

        # Just use number of theta dimensions for Type 1
        dim_gp_data = self.gp_sim_data.theta_dim

        return dim_gp_data

    def featurize_data(self, data):
        """
        Collects the features (parameter set values) of the GP into ndarray form from an instance of the Data class

        Parameters
        -----------
        data: Data
            Data to evaluate GP for containing at least parameter sets (data.theta_vals)

        Returns
        -------
        feature_eval_data: np.ndarray
            The feature data for the GP

        Raises
        ------
        AssertionError
            If any of the inputs are not of the correct type or not defined

        """
        assert isinstance(data, Data), "data must be an instance of Data"
        assert np.all(
            data.theta_vals is not None
        ), "Must have validation data theta_vals and x_vals to evaluate the GP"

        # Assign feature evaluation data as theta and x values. Create empty list to store gp approximations
        feature_eval_data = data.theta_vals

        return feature_eval_data

    def _select_train_test_rows(self, train_idx, test_idx):
        """
        Type 1's x_vals is shared experimental data, so it is not indexed at all.
        """
        theta_train = self.gp_sim_data.theta_vals[train_idx]
        x_train = (
            self.gp_sim_data.x_vals
        )  # x_vals for Type 1 is the same as exp_data. No need to index x
        y_train = self.gp_sim_data.y_vals[train_idx]

        theta_test = self.gp_sim_data.theta_vals[test_idx]
        x_test = (
            self.gp_sim_data.x_vals
        )  # x_vals for Type 1 is the same as exp_data. No need to index x
        y_test = self.gp_sim_data.y_vals[test_idx]

        return theta_train, x_train, y_train, theta_test, x_test, y_test

    def _sse_from_prediction(self, data, covar=False, prediction=None):
        """
        For Type 1, sse is the gp_mean, so this is a direct passthrough of the prediction.
        """
        assert isinstance(data, Data), "data must be type Data"
        if prediction is None:
            prediction = self.predict(data=data)
        # For type 1, sse is the gp_mean
        sse = prediction.mean
        sse_var = prediction.variance
        sse_covar = prediction.covariance

        return sse, sse_var, sse_covar

    def calc_best_error(self):
        """
        Calculates the best error of the model

        Returns
        -------
        best_error: float
            The best error of the method
        be_theta: np.ndarray
            The parameter set associated with the best error of the method
        train_idc: int
            The index of the best error in the training data

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value

        """
        assert self.train_data is not None, "self.train_data must exist!"
        assert isinstance(self.train_data, Data), "self.train_data must be type Data"
        assert np.all(
            self.train_data.y_vals is not None
        ), "self.train_data.y_vals and self.train_data.theta_vals must exist!"
        assert np.all(
            self.train_data.theta_vals is not None
        ), "self.train_data.y_vals and self.train_data.theta_vals must exist!"

        # Best error is the minimum sse value of the training data for Type 1
        best_error = np.min(self.train_data.y_vals)
        train_idc = np.argmin(self.train_data.y_vals)
        be_theta = self.train_data.theta_vals[train_idc]

        return best_error, be_theta, train_idc

    def _compute_ei(self, sim_data, exp_data, ep_bias, best_error_metrics, gp_prediction=None):
        """
        Predict-if-none, build ExpectedImprovement(...), .compute().
        """
        if gp_prediction is None:
            gp_prediction = self.predict(data=sim_data)
        # Call instance of expected improvement class
        ei_class = ExpectedImprovement(
            ep_bias,
            gp_prediction.mean,
            gp_prediction.covariance,
            exp_data,
            best_error_metrics,
            self.seed,
            None,
        )
        # Call correct method of ei calculation
        ei, ei_terms_df = ei_class.compute()

        return ei, ei_terms_df


class EmulatorGP(GPEmulator):
    """
    The base class for Gaussian Processes
    Parameters

    Methods
    --------------
    __init__(*) : Constructor method
    get_dim_gp_data (inherited from GPEmulator): Defines the total dimension of input data used by the GP
    _gp_input_dim(): Type 2 GP input is theta and x, so the dimension is x_dim + theta_dim
    featurize_data(data) (overrides abstract GPEmulator.featurize_data): Collects the features of the GP into ndarray form from an instance of the Data class
    split_train_test(sep_fact, seed) (inherited from GPEmulator): Finds the simulation data to use as training/testing data
    _select_train_test_rows(train_idx, test_idx): Expands theta-level indices to full (theta, x)-row boolean masks via get_unique_theta() + np.isin
    predict_sse(target=None, data=None, method=None, exp_data=None, covar=False) (inherited from GPEmulator): GP-predicted SSE mean and (co)variance
    _sse_from_prediction(data, method, exp_data, covar, prediction): Derives SSE from the GP-predicted model output via a blockwise sum-of-squared-errors reduction against experimental data
    calc_best_error(method, exp_data): Calculates the best error metrics for the GP
    expected_improvement(target=None, data=None, exp_data=None, ep_bias=None, best_error_metrics=None, method=None, sg_mc_samples=2000) (inherited from GPEmulator): Expected improvement acquisition
    _compute_ei(sim_data, exp_data, ep_bias, best_error_metrics, method, sg_mc_samples): Evaluates the expected improvement for the GP
    _extend_train_data_arrays(new_point_data): Grows x_vals too when append_training_point (inherited from GPEmulator) adds a point
    """

    # Class variables and attributes
    def __init__(
        self,
        gp_sim_data,
        gp_val_data,
        cand_data,
        train_data,
        test_data,
        kernel,
        lengthscale,
        noise_std,
        outputscale,
        retrain_gp,
        set_seed,
        normalize,
        backend=None,
    ):
        """
        Parameters
        ----------
        gp_sim_data: Data,
            All simulation data for the GP
        gp_val_data: Data
            The validation data for the GP. None if not saving validation data
        cand_data: Data
            Candidate theta value for evaluation with GPBODriver.opt_with_scipy()
        train_data: Data
            The training data for the GP
        testing_data: Data
            The testing data for the GP
        kernel: Kernel
            Determines which GP Kerenel to use
        lengthscale: float or None
            Value of the lengthscale hyperparameter - None if hyperparameters will be updated during training
        noise_std: float, int
            The standard deviation of the noise
        outputscale: float or None
            Determines value of outputscale - None if hyperparameters will be updated during training
        retrain_gp: int
            Number of times to (re)train the GP. If 0, the GP is not trained and default/initial hyperparameters are used
        set_seed: int or None
            Random seed
        normalize: bool
            Determines whether data is standardized (with sklearn RobustScaler) before training
        backend: GPBackend or None, default None
            The GP backend to use. If None, resolved via get_backend (gpflow by default).

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value
        """
        # Constructor method
        # Inherit objects from GPEmulator Base Class
        super().__init__(
            gp_sim_data,
            gp_val_data,
            cand_data,
            kernel,
            lengthscale,
            noise_std,
            outputscale,
            retrain_gp,
            set_seed,
            normalize,
            backend=backend,
        )
        # Set training and testing data as child class specific objects
        assert (
            isinstance(train_data, Data) or train_data is None
        ), "train_data must be instance of Data or None"
        assert (
            isinstance(test_data, Data) or train_data is None
        ), "test_data must be instance of Data or None"

        self.train_data = train_data
        self.test_data = test_data
        self.train_data_init = (
            None  # This will be populated with the first set of training thetas
        )

    def _gp_input_dim(self):
        """
        Type 2 GP input is theta and x, so the dimension is x_dim + theta_dim.
        """
        assert isinstance(
            self.gp_sim_data, Data
        ), "self.gp_sim_data must be instance of Data"
        assert np.all(
            self.gp_sim_data.x_vals is not None
        ), "self.gp_sim_data.x_vals and self.gp_sim_data.theta_vals must exist!"
        assert np.all(
            self.gp_sim_data.theta_vals is not None
        ), "self.gp_sim_data.x_vals and self.gp_sim_data.theta_vals must exist!"

        # Number of theta dimensions + number of x dimensions
        dim_gp_data = int(
            self.gp_sim_data.x_dim + self.gp_sim_data.theta_dim
        )

        return dim_gp_data

    def featurize_data(self, data):
        """
        Collects the features of the GP into ndarray form from an instance of the Data class

        Parameters
        -----------
        data: Data
            Data to evaluate GP for containing at least data.theta_vals and data.x_vals

        Returns
        --------
        feature_eval_data: np.ndarray
            The feature data for the GP

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value

        """
        assert isinstance(data, Data), "data must be instance of Data"
        assert np.all(
            data.x_vals is not None
        ), "data.x_vals and data.theta_vals must exist!"
        assert np.all(
            data.theta_vals is not None
        ), "data.x_vals and data.theta_vals must exist!"

        # Assign feature evaluation data as theta and x values. Create empty list to store gp approximations
        feature_eval_data = np.concatenate((data.theta_vals, data.x_vals), axis=1)

        return feature_eval_data

    def _select_train_test_rows(self, train_idx, test_idx):
        """
        Type 2 must expand the theta-level train_idx/test_idx to full (theta, x)-row
        boolean masks via get_unique_theta() + np.isin before indexing.
        """
        # Find unique theta_values
        unique_theta_vals = self.gp_sim_data.get_unique_theta()

        # Check which rows in theta_vals match the rows in Theta_unique based on theta_idx
        train_mask = np.isin(self.gp_sim_data.theta_vals, unique_theta_vals[train_idx])
        test_mask = np.isin(
            self.gp_sim_data.theta_vals, unique_theta_vals[train_idx], invert=True
        )

        # Get the indices of the matching rows
        train_rows_idx = np.all(train_mask, axis=1)
        test_rows_idx = np.all(test_mask, axis=1)

        # Use the indices to select the specific rows from theta_vals
        theta_train = self.gp_sim_data.theta_vals[train_rows_idx]
        x_train = self.gp_sim_data.x_vals[train_rows_idx]
        y_train = self.gp_sim_data.y_vals[train_rows_idx]

        theta_test = self.gp_sim_data.theta_vals[test_rows_idx]
        x_test = self.gp_sim_data.x_vals[test_rows_idx]
        y_test = self.gp_sim_data.y_vals[test_rows_idx]

        return theta_train, x_train, y_train, theta_test, x_test, y_test

    def _sse_from_prediction(self, data, method=None, exp_data=None, covar=False, prediction=None):
        """
        Type 2 derives SSE from the GP-predicted model output via a blockwise
        sum-of-squared-errors reduction against experimental data. `method` is validated
        but not otherwise used here.
        """
        assert isinstance(
            method, GPBOMethod
        ), "method must be instance of GPBOMethod class"
        assert all(
            isinstance(var, Data) for var in [data, exp_data]
        ), "data and exp_data must be type Data"
        assert np.all(
            data.x_vals is not None
        ), "data.x_vals and data.theta_vals must exist!"
        assert np.all(
            data.theta_vals is not None
        ), "data.x_vals and data.theta_vals must exist!"
        assert np.all(
            exp_data.x_vals is not None
        ), "exp_data.x_vals and exp_data.y_vals must exist!"
        assert np.all(
            exp_data.y_vals is not None
        ), "exp_data.x_vals and exp_data.y_vals must exist!"
        assert isinstance(covar, bool), "covar must be bool!"

        if prediction is None:
            prediction = self.predict(data=data)
        gp_mean = prediction.mean
        gp_covar = prediction.covariance

        # Find length of theta and number of unique x in data arrays
        len_theta = data.n_theta
        len_x = len(data.get_unique_x())
        # Infer number of thetas
        num_uniq_theta = int(len_theta / len_x)

        # Reshape y_sim into n_theta rows x n_x columns
        indices = np.arange(0, len_theta, len_x)
        n_blocks = len(indices)
        # gp_mean is flat, one entry per (theta, x) row (EmulatorGP repeats each theta
        # len_x times); blockwise_sse reshapes it into n_blocks (one per unique theta)
        # rows of len_x columns, so each block's residual-vs-exp_data.y_vals and
        # summed-squared-error can be computed per theta in one vectorized call.
        sse_mean, block_errors = blockwise_sse(gp_mean, exp_data.y_vals, n_blocks, len_x)
        residuals = block_errors.reshape(gp_covar.shape[0], -1)

        # Propagate gp_covar (over the flat (theta,x) rows) into SSE variance via the
        # quadratic-form identity for sums of squared correlated Gaussians:
        # Var(sum r_i^2) = 2*tr(Sigma^2) + 4*r^T*Sigma*r, r = residuals, Sigma = gp_covar.
        # This SSE_variance CAN'T be negative
        sse_var_all = (
            2 * np.trace(gp_covar**2) + 4 * residuals.T @ gp_covar @ residuals
        )

        # Calculate individual variances Var(SSE[t1]), and Var(SSE[t2])
        if num_uniq_theta == 1:
            sse_var = sse_var_all
            sse_covar = sse_var
        else:
            sse_var = np.zeros(n_blocks)
            for i in range(n_blocks):
                # Get section of covariance matrix that corresponds to the covariance of the different thetas
                covar_t_t = gp_covar[
                    i * len_x : (i + 1) * len_x, i * len_x : (i + 1) * len_x
                ]
                # Get row of block error corresponding to this matrix
                res_theta = block_errors[i].reshape(-1, 1)
                # Calculate Variance
                sse_var[i] = (
                    2 * np.trace(covar_t_t**2) + 4 * res_theta.T @ covar_t_t @ res_theta
                ).item()
            if num_uniq_theta == 2 and covar == True:
                sse_covar = sse_var_all
            else:
                sse_covar = None

        return sse_mean, sse_var, sse_covar

    def calc_best_error(self, method, exp_data):
        """
        Calculates the best error of the model (sse) and squared error for each state point x (squared error)

        Parameters
        ----------
        method: GPBOMethod
            Class containing method information
        exp_data: Data
            Experimental data. Must contain exp_data.x_vals, exp_data.theta_vals, and exp_data.y_vals

        Returns
        -------
        best_error: float
            The best error (sse) of the method
        be_theta: np.ndarray
            The parameter set associated with the best error value
        best_sq_error: np.ndarray
            Array of squared errors for each value of x
        org_train_idcs: list(int)
            The original training indices of be_theta

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value
        """
        assert isinstance(
            method, GPBOMethod
        ), "method must be instance of GPBOMethod class"
        assert all(
            isinstance(var, Data) for var in [self.train_data, exp_data]
        ), "self.tain_data and exp_data must be type Data"
        assert np.all(
            self.train_data.x_vals is not None
        ), "self.train_data.x_vals, self.train_data.theta_vals, and self.train_data.y_vals must exist!"
        assert np.all(
            self.train_data.theta_vals is not None
        ), "self.train_data.x_vals, self.train_data.theta_vals, and self.train_data.y_vals must exist!"
        assert np.all(
            self.train_data.y_vals is not None
        ), "self.train_data.x_vals, self.train_data.theta_vals, and self.train_data.y_vals must exist!"
        assert np.all(
            exp_data.x_vals is not None
        ), "exp_data.x_vals and exp_data.y_vals must exist!"
        assert np.all(
            exp_data.y_vals is not None
        ), "exp_data.x_vals and exp_data.y_vals must exist!"

        # Find length of theta and x in data arrays
        len_theta = self.train_data.n_theta
        len_x = len(self.train_data.get_unique_x())

        # #Reshape y_sim into n_theta rows x n_x columns
        indices = np.arange(0, len_theta, len_x)
        n_blocks = len(indices)
        # train_data.y_vals is flat (one entry per (theta, x) training row); reshape into
        # one row per unique training theta so the SSE-vs-exp_data can be computed (and
        # then argmin'd over) per theta rather than per individual (theta, x) row.
        sse_train_vals, block_errors = blockwise_sse(
            self.train_data.y_vals, exp_data.y_vals, n_blocks, len_x
        )
        ind_errors = block_errors**2

        # List to array
        be_theta = self.train_data.theta_vals[int(np.argmin(sse_train_vals) * len_x)]
        org_train_idcs = [
            int(np.argmin(sse_train_vals) * len_x),
            int((np.argmin(sse_train_vals) + 1) * len_x),
        ]

        # Best error is the minimum of these values
        best_error = np.amin(sse_train_vals)
        best_sq_error = ind_errors[np.argmin(sse_train_vals)]

        return best_error, be_theta, best_sq_error, org_train_idcs

    def _compute_ei(self, sim_data, exp_data, ep_bias, best_error_metrics, method=None,
                     sg_mc_samples=2000, gp_prediction=None):
        """
        Emulator EI needs the `method` and experimental `exp_data`; `sg_mc_samples` sets
        the sparse-grid / Monte-Carlo sample count. The sparse-grid and Monte-Carlo
        methods require a single sample covariance matrix, so `sim_data` must contain a
        single unique parameter set for those.
        """
        assert isinstance(
            method, GPBOMethod
        ), "method must be instance of GPBOMethod"
        assert (
            6 >= method.method_name.value > 2
        ), "method must be Type 2. Hint: Must have method.method_name.value > 2"

        if method.method_name.value in [5, 6]:
            if len(sim_data.get_unique_theta()) > 1:
                raise ValueError(
                    "Sparse Grid and Monte Carlo methods require a single sample covariance matrix"
                )

        assert (
            6 >= method.method_name.value >= 3
        ), "Must be using method 2A, 2B, 2C, or 2D"
        # Set sparse grid depth if applicable
        if method.uses_sparse_grid == True or method.uses_monte_carlo == True:
            assert (
                isinstance(sg_mc_samples, int) and sg_mc_samples > 0
            ), "sg_mc_samples must be positive int for sparse grid and Monte Carlo methods"
        if gp_prediction is None:
            gp_prediction = self.predict(data=sim_data)
        # Call instance of expected improvement class
        ei_class = ExpectedImprovement(
            ep_bias,
            gp_prediction.mean,
            gp_prediction.covariance,
            exp_data,
            best_error_metrics,
            self.seed,
            sg_mc_samples,
            method,
        )
        # Call correct method of ei calculation
        ei, ei_terms_df = ei_class.compute()

        return ei, ei_terms_df

    def _extend_train_data_arrays(self, new_point_data):
        """Type-2 also carries per-point x alongside theta, so grow x_vals here too."""
        self.train_data.x_vals = np.vstack(
            (self.train_data.x_vals, new_point_data.x_vals)
        )


##Again, composition instead of inheritance


def build_gp_emulator(
    method,
    sim_data,
    sim_sse_data,
    val_data,
    val_sse_data,
    kernel,
    lengthscale,
    outputscale,
    retrain_gp,
    seed,
    normalize,
    simulator_noise_std,
    n_x,
    backend=None,
):
    """
    Construct the right GP emulator (ObjectiveGP or EmulatorGP) for a method.

    Factory extracted verbatim from GPBODriver.__gen_emulator so it can be reused (e.g. by
    the GP-diagnostics API) without a driver. The returned emulator is UNfitted -- call
    split_train_test(...) then fit().

    Parameters
    ----------
    method : GPBOMethod
    sim_data, sim_sse_data : Data
        Simulation data (model outputs) and its SSE form. EmulatorGP uses sim_data;
        ObjectiveGP uses sim_sse_data.
    val_data, val_sse_data : Data or None
        Optional validation data (same two forms).
    kernel, lengthscale, outputscale, retrain_gp, seed, normalize :
        GP hyperparameter / training configuration (from BOConfig).
    simulator_noise_std : float or None
        The simulator's noise standard deviation.
    n_x : int
        Number of experimental conditions (exp_data.n_x); sets the chi-squared noise scale
        for the standard (objective) GP.
    backend : GPBackend or None, default None
        The GP backend to use. If None, resolved via get_backend (gpflow by default).

    Returns
    -------
    GPEmulator (ObjectiveGP or EmulatorGP)
    """
    if method.is_emulator == False:
        all_gp_data = sim_sse_data
        all_val_data = val_sse_data
        k = np.maximum(n_x - 1, 1)
        # If using objective sse use var of a chi^2 distribution (2k)
        if not method.log_scaled and simulator_noise_std is not None:
            noise_scl_fact = np.sqrt(2 * k)
            noise_std = simulator_noise_std * noise_scl_fact
        # If using objective ln(sse) guess the noise std
        else:
            noise_std = None
        gp_emulator = ObjectiveGP(
            all_gp_data, all_val_data, None, None, None, kernel, lengthscale, noise_std,
            outputscale, retrain_gp, seed, normalize,
            backend=backend,
        )
    else:
        all_gp_data = sim_data
        all_val_data = val_data
        noise_std = simulator_noise_std  # Yexp_std is exactly the noise_std of the GP Kernel
        gp_emulator = EmulatorGP(
            all_gp_data, all_val_data, None, None, None, kernel, lengthscale, noise_std,
            outputscale, retrain_gp, seed, normalize,
            backend=backend,
        )
    return gp_emulator
