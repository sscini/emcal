"""Simulator: the internal engine that generates experimental / simulation data and the
SSE objective data for a calibration problem (built from a CalibrationProblem).
"""
import numpy as np
import pandas as pd
import math
import warnings
import itertools
from scipy.stats import qmc
from .enums import GenMethod
from .data import ExperimentalData, SimulationData, ObjectiveData
from ._utils import vector_to_1D_array, blockwise_sse


class Simulator:
    """
    The base class for differet simulators. Defines a simulation

    Methods
    --------------
    __init__(*): Constructor method
    __set_true_params(): Sets true parameter value array and the corresponding names based on parameter dictionary and indices to consider
    __grid_sampling(num_points, bounds): Generates Grid sampled data
    __lhs_sampling(num_points, bounds, seed): Design LHS Samples
    __create_param_data(num_points, bounds, gen_meth, seed): Generates data based off of bounds, and sampling scheme
    evaluate_model(data, noise_mean, noise_std): Generates y data with noise
    generate_experimental_data(num_x_data, gen_meth_x, set_seed=None, x_vals = None): Generates experimental data
    generate_simulation_data(num_theta_data, num_x_data, gen_meth_theta, gen_meth_x, sep_fact, gen_val_data): Generates simulation data
    simulation_data_from_evaluations(theta_vals, y_vals, x_vals, sep_fact): Simulation data from runs already done
    generate_parameter_samples(num_theta_data, rng): Generates parameter sets
    to_sse_data(method, sim_data, exp_data, sep_fact, gen_val_data)
    """

    def __init__(
        self,
        indices_to_consider,
        theta_ref,
        theta_names,
        bounds_theta_l,
        bounds_x_l,
        bounds_theta_u,
        bounds_x_u,
        noise_mean,
        noise_std,
        set_seed,
        calc_y_fxn,
        calc_y_fxn_args,
    ):
        """
        Parameters
        ----------
        indices_to_consider: list(int)
            The indices corresponding to which parameters are being guessed
        theta_ref: ndarray
            The array containing the true values of problem constants
        theta_names: list
            List of names of each parameter that will be plotted named by index w.r.t Theta_True
        bounds_theta_l: list
            Lower bounds of theta
        bounds_x_l: list
            Lower bounds of x
        bounds_theta_u: list
            Upper bounds of theta
        bounds_x_u: list
            Upper bounds of x
        noise_mean: float, int
            The mean of the noise
        noise_std: float, int, or None
            The standard deviation of the noise. If None, 1% of median of Y-exp will be used
        set_seed: int or None
            Determines seed for randomizations. None if seed is random
        calc_y_fxn: function
            The function to calculate ysim data with
        calc_y_fxn_args: dict
            Dictionary of arguments other than parameters and x to pass to calc_y_fxn

        Raises
        ------
        AssertionError
            If any of the inputs are not of the correct type or value
        """
        # Check for float/int
        assert isinstance(noise_mean, (float, int)), "noise_mean must be int or float"
        assert (
            isinstance(noise_std, (float, int)) or noise_std is None
        ), "noise_std must be int, float, or None"
        assert isinstance(set_seed, int) or set_seed is None, "Seed must be int or None"
        # theta_ref is the reference/true parameter vector; it is None for real calibration
        # (no ground truth). When None, every parameter must be calibrated (indices cover all
        # of theta_names), since evaluate_model then uses the sampled theta as the full vector.
        assert theta_ref is None or isinstance(
            theta_ref, (list, np.ndarray)
        ), "theta_ref must be a list, np.ndarray, or None"
        # Check for list or ndarray
        list_vars = [
            indices_to_consider,
            theta_names,
            bounds_theta_l,
            bounds_x_l,
            bounds_theta_u,
            bounds_x_u,
        ]
        assert (
            all(isinstance(var, (list, np.ndarray)) for var in list_vars) == True
        ), "indices_to_consider, theta_names, bounds_theta_l, bounds_x_l, bounds_theta_u, and bounds_x_u must be list or np.ndarray"
        # Check for list lengths > 0
        assert (
            all(len(var) > 0 for var in list_vars) == True
        ), "indices_to_consider, theta_names, bounds_theta_l, bounds_x_l, bounds_theta_u, and bounds_x_u must have length > 0"
        # Check that bound_x and bounds_theta have same lengths
        assert len(bounds_theta_l) == len(bounds_theta_u) and len(bounds_x_l) == len(
            bounds_x_u
        ), "bounds lists for x and theta must be same length"
        # Check indeces to consider in theta_ref (synthetic) or that all params are calibrated (real)
        if theta_ref is not None:
            assert (
                all(0 <= idx <= len(theta_ref) - 1 for idx in indices_to_consider) == True
            ), "indeces to consider must be in range of theta_ref"
        else:
            assert list(indices_to_consider) == list(
                range(len(theta_names))
            ), "when theta_ref is None (real calibration) every parameter must be calibrated"
        assert (
            isinstance(calc_y_fxn_args, dict) or calc_y_fxn_args is None
        ), "calc_y_fxn_args must be dict or None"
        assert callable(
            calc_y_fxn
        ), "The argument 'calc_y_fxn' must be a callable (function) with 3 arguments."

        # Constructor method
        self.dim_x = len(bounds_x_l)
        self.dim_theta = len(
            indices_to_consider
        )  # Length of theta is equivalent to the number of indeces to consider
        self.indices_to_consider = indices_to_consider
        self.theta_ref = theta_ref
        self.theta_names = theta_names
        self.theta_true, self.theta_true_names = (
            self.__set_true_params()
        )  # Would this be better as a dictionary?
        self.bounds_theta = np.array([bounds_theta_l, bounds_theta_u])
        self.bounds_theta_reg = self.bounds_theta[
            :, self.indices_to_consider
        ]  # This is the theta_bounds for parameters we will regress
        self.bounds_x = np.array([bounds_x_l, bounds_x_u])
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.calc_y_fxn = calc_y_fxn
        self.calc_y_fxn_args = calc_y_fxn_args

        self.rng_rand = np.random.default_rng()

        # If set_seed is given, every RNG below derives deterministically from it (full
        # run reproducibility); rng_exp reuses rng_set's own stream so experimental-data
        # generation advances in lockstep with the rest of the class. If set_seed is None,
        # rng_set/rng_exp are seeded independently (from OS entropy / random_set_seed) so
        # repeated construction is intentionally non-reproducible.
        if set_seed is not None:
            self.rng_set = np.random.default_rng(set_seed)
            self.rng_exp = self.rng_set
        else:
            # Own Generator so rng_set's stream is untouched; int() keeps it JSON-safe.
            random_set_seed = int(np.random.default_rng().integers(1, 10**8, endpoint=True))
            self.rng_set = self.rng_rand
            self.rng_exp = np.random.default_rng(random_set_seed)

        # Ensure LHS for sim, val, and starting pts for EI will all be different: offsetting
        # by +1/+2 keeps those three data-generation seeds distinct even though they all
        # derive from the same set_seed.
        if set_seed is not None:
            self.sim_seed = set_seed
            self.sim_x_seed = set_seed
            self.val_seed = set_seed + 1
            self.start_seed = set_seed + 2
        else:
            self.sim_seed = self.val_seed = self.start_seed = None
            self.sim_x_seed = random_set_seed

    def __set_true_params(self):
        """
        Sets true parameter value array and the corresponding names based on parameter dictionary and indices to consider

        Returns
        -------
        true_params: ndarray
            The true parameter of the model
        true_param_names: list(str)
            The names of the true parameter of the model
        """
        # Define theta_true and theta_true_names from theta_ref, theta_names, and indeces to consider
        true_param_names = [self.theta_names[idx] for idx in self.indices_to_consider]
        # No ground truth for real calibration (theta_ref is None).
        true_params = (
            None if self.theta_ref is None else self.theta_ref[self.indices_to_consider]
        )

        return true_params, true_param_names

    def __grid_sampling(self, num_points, bounds):
        """
        Generates grid sampled data

        Parameters
        ----------
        num_points: int
            Number of points to generate in each dimension, should be greater than # of dimensions
        bounds: ndarray
            Array containing upper and lower bounds of elements in each dimension.

        Returns
        ----------
        grid_data: np.ndarray
            (num_points)**bounds.shape[1] grid sample of data

        """
        # Generate mesh_grid data for theta_set in 2D
        # Define linspace for theta
        params = np.linspace(0, 1, num_points)
        # Define dimensions of parameter
        dimensions = bounds.shape[1]
        # Generate the equivalent of all meshgrid points
        df = pd.DataFrame(list(itertools.product(params, repeat=dimensions)))
        df2 = df.drop_duplicates()
        scaled_data = df2.to_numpy()
        # Normalize to bounds
        lower_bound = bounds[0]
        upper_bound = bounds[1]
        grid_data = scaled_data * (upper_bound - lower_bound) + lower_bound
        return grid_data

    def __lhs_sampling(self, num_points, bounds, rng):
        """
        Design LHS Samples

        Parameters
        ----------
        num_points: int
            Number of points in LHS, should be greater than # of dimensions
        bounds: np.ndarray
            Array containing upper and lower bounds of elements in LHS sample
        set_seed: int
            Seed of random generation

        Returns
        -------
        lhs_data: np.ndarray
            Array of LHS sampling points with length (num_points)
        """
        # Define number of dimensions
        dimensions = bounds.shape[1]
        # Define sampler
        #Note: "seed" in qmc.LatinHypercube will be deprecated after version 1.15.2, use rng if this becomes an issue
        sampler = qmc.LatinHypercube(d=dimensions, seed=rng)
        lhs_data = sampler.random(n=num_points)

        # Generate LHS data given bounds
        lhs_data = qmc.scale(
            lhs_data, bounds[0], bounds[1]
        )  # Using this because I like that bounds can be different shapes

        return lhs_data

    def __create_param_data(self, num_points, bounds, gen_meth, rng):
        """
        Generates data based off of bounds, and and generation scheme

        Parameters
        ----------
        num_points: int
            Number of data to generate
        bounds: np.ndarray
            Array of parameter bounds
        gen_meth: GenMethod
            ("LHS", "Meshgrid"). Determines whether data will be generated with an LHS or meshgrid
        set_seed: int
            Seed of random generation

        Returns
        --------
        data: np.ndarray
            An array of data

        Raises
        ------
        ValueError
            If gen_meth.value is not 1 or 2

        Notes
        ------
        Meshgrid generated data will output num_points in each dimension, LHS generates num_points of data
        """

        # Set dimensions
        dimensions = bounds.shape[
            1
        ]  # Want to do it this way to make it general for either x or theta parameters

        # Decide on a method to use based on gen_meth_value. LHS or Grid
        if gen_meth.value == 2:
            data = self.__grid_sampling(num_points, bounds)

        elif gen_meth.value == 1:
            # Generate LHS sample
            data = self.__lhs_sampling(num_points, bounds, rng)

        else:
            raise ValueError("gen_meth.value must be 1 or 2!")

        return data

    def __model_coefficients(self, theta_row):
        """
        Full parameter vector for one sampled theta

        Parameters
        ----------
        theta_row: np.ndarray
            One row of sampled (regressed) parameter values

        Returns
        -------
        model_coefficients: np.ndarray
            Parameter vector passed to the model
        """
        # Real calibration: the sampled theta is the full parameter vector
        if self.theta_ref is None:
            model_coefficients = np.array(theta_row, dtype=float)
            return model_coefficients

        # Synthetic benchmark: put the sampled values into a copy of the reference vector
        model_coefficients = self.theta_ref.copy()
        model_coefficients[self.indices_to_consider] = theta_row
        return model_coefficients

    def __evaluate_by_theta(self, data, evaluate_all):
        """
        Evaluates the model with one evaluate_all(theta, X) call per unique theta

        Parameters
        ----------
        data: Data
            Parameter sets and x values to generate y data for
        evaluate_all: callable
            evaluate_all(theta, X) returns the model output at every row of X

        Returns
        -------
        y_data: np.ndarray
            The model output for every row of data
        """
        thetas = np.asarray(data.theta_vals, dtype=float)
        y_data = np.empty(len(thetas))

        # Collect the rows that share each theta, in order of first appearance
        rows_by_theta = {}
        for row in range(len(thetas)):
            theta_key = tuple(thetas[row])
            if theta_key not in rows_by_theta:
                rows_by_theta[theta_key] = []
            rows_by_theta[theta_key].append(row)

        # Evaluate each theta once, at all of its x values
        for rows in rows_by_theta.values():
            theta = thetas[rows[0]]
            y = evaluate_all(self.__model_coefficients(theta), data.x_vals[rows])
            y_data[rows] = np.asarray(y, dtype=float).reshape(-1)

        return y_data

    def evaluate_model(self, data, noise_mean, noise_std, rng, noise_std_pct = 0.01):
        """
        Creates simulated data based on the function self.calc_y_fxn

        Parameters
        ----------
        data: Data
            Parameter sets to generate y data for
        noise_mean: float, int
            The mean of the noise
        noise_std: float, int, None
            The standard deviation of the noise
        rng: np.random.Generator
            Random number generator used to sample the noise
        noise_std_pct: float or int, default 0.01
            Percentage of the mean of the y data to use as the standard deviation of the
            noise when noise_std is None

        Returns
        -------
        y_data: np.ndarray The simulated y training data
        """
        if noise_std is None:
            assert isinstance(noise_std_pct, (float, int)) and noise_std_pct >= 0, "noise_std_pct must be positive float or int"

        # Models that run once per parameter set (external simulators) expose evaluate_all.
        evaluate_all = getattr(self.calc_y_fxn, "evaluate_all", None)
        if evaluate_all is not None:
            y_data = self.__evaluate_by_theta(data, evaluate_all)
        else:
            # Define an array to store y values in
            y_data = []
            # Loop over all theta values
            for i in range(data.n_theta):
                # Create y data coefficients
                y_data.append(
                    self.calc_y_fxn(
                        self.__model_coefficients(data.theta_vals[i]), data.x_vals[i],
                        self.calc_y_fxn_args,
                    )
                )

            # Convert list to array and flatten array
            y_data = np.array(y_data).flatten()

        # Creates noise values with a certain stdev and mean from a normal distribution
        # If noise is none
        if noise_std is None:
            # Set the noise as 1% of the median as a default. 
            if not math.isclose(np.median(y_data),0):
                noise_std = np.abs(np.median(y_data)) * noise_std_pct
            #If the median value is 0, use 1% of the mean as the default.
            elif not math.isclose(np.mean(y_data),0):
                noise_std = np.abs(np.mean(y_data)) * noise_std_pct
            #If both values are zero, Use 1% of the abs max value
            else:
                noise_std = np.max(np.abs(y_data)) * noise_std_pct
            #Set temp noise to the noise value that was just generated. This value is only used if generate_experimental_data is called
            self.temp_noise = noise_std
        else:
            noise_std = noise_std

        noise = rng.normal(size=len(y_data), loc=noise_mean, scale=noise_std)
        # print(noise.flatten())

        # Add noise to data
        y_data = y_data + noise

        return y_data

    def generate_experimental_data(self, num_x_data, gen_meth_x, x_vals=None, noise_std_pct = 0.01):
        """
        Generates experimental data in an instance of the Data class

        Parameters
        ----------
        num_x_data: int
            Number of experiments
        gen_meth_x: GenMethod
            Whether to generate X data with LHS or grid method
        x_vals: np.ndarray or None, default None
            X values to use for experimental data. If None, x_vals will be generated based on bounds and num_x_data
        noise_std_pct: float or int, default 0.01
            Percentage of the mean of the y data to use as the standard deviation of the noise

        Returns
        --------
        exp_data: Data
            Experimental x and y data along with parameter bounds

        Raises
        ------
        AssertionError
            If any of the inputs are not of the correct type or value
        ValueError
            If num_x_data is not a positive integer

        Notes:
        ------
        Warning: This function will not generate exactly the same values of y when repeatedly called, even with the same seed. 
        """
        assert x_vals is None or isinstance(x_vals, np.ndarray), "x_vals must be np.ndarray or None"

        assert isinstance(noise_std_pct, (int,float)) and noise_std_pct >= 0, "noise_std_pct must be a positive int/float"
        # check that num_data > 0
        if num_x_data <= 0 or isinstance(num_x_data, int) == False:
            raise ValueError("num_x_data must be a positive integer")

        # Create x vals based on bounds and num_x_data if x_vals are not specified
        #For data generation we always want instances of exp_data to be reproduceable, so sim_x_seed and rng_exp are used
        if x_vals is None:
            x_vals = vector_to_1D_array(
                self.__create_param_data(num_x_data, self.bounds_x, gen_meth_x, self.sim_x_seed)
            )
        else:
            x_vals = vector_to_1D_array(x_vals)
    
        # Reshape theta_true to correct dimensions and stack it once for each xexp value
        theta_true = self.theta_true.reshape(1, -1)
        theta_true_repeated = np.vstack([theta_true] * len(x_vals))
        # Create exp_data class and add values
        # (synthetic: theta_vals carries the repeated true params so evaluate_model
        # can generate y_vals below).
        exp_data = ExperimentalData(
            x_vals,
            None,
            theta_vals=theta_true_repeated,
            bounds_theta=self.bounds_theta_reg,
            bounds_x=self.bounds_x,
        )
        # Generate y data for exp_data calss instance
        #We will always use the set_rng for data generation
        exp_data.y_vals = self.evaluate_model(exp_data, self.noise_mean, self.noise_std, self.rng_exp, noise_std_pct = noise_std_pct)

        #Set simulator noise after exp_data is generated if self.noise_std is None
        if self.noise_std == None:
            self.noise_std = self.temp_noise

        return exp_data

    def set_experimental_data(self, x_vals, y_vals, noise_std_pct=0.01):
        """
        Build an experimental-data Data object from measured (x, y) for REAL calibration.

        Unlike generate_experimental_data (which synthesizes noisy y from theta_ref), this
        uses the user's measured observations directly and requires no ground-truth
        parameters. Use it when the Simulator was built with theta_ref=None
        (i.e. from a CalibrationProblem with experimental_data).

        Parameters
        ----------
        x_vals : np.ndarray
            Experimental conditions (one row per observation).
        y_vals : np.ndarray
            Measured responses (one per condition); len(y_vals) == len(x_vals).
        noise_std_pct : float, default 0.01
            If the simulator's noise_std is unknown (None), it is set to this fraction of
            the median |y| so downstream code has a sensible noise scale.

        Returns
        -------
        exp_data : Data
            Experimental x and y data (theta_vals is None -- there is no true parameter set).
        """
        x_arr = vector_to_1D_array(np.asarray(x_vals, dtype=float))
        y_arr = np.asarray(y_vals, dtype=float).flatten()
        assert len(x_arr) == len(y_arr), "x_vals and y_vals must have the same length"

        exp_data = ExperimentalData(
            x_arr,
            y_arr,
            theta_vals=None,   # no true parameters for real calibration
            bounds_theta=self.bounds_theta_reg,
            bounds_x=self.bounds_x,
        )

        # Give downstream code a noise scale if none was provided (mirrors the synthetic default).
        if self.noise_std is None:
            median_abs = float(np.median(np.abs(y_arr)))
            self.noise_std = median_abs * noise_std_pct if median_abs > 0 else noise_std_pct

        return exp_data

    def generate_simulation_data(
        self,
        num_theta_data,
        num_x_data,
        gen_meth_theta,
        gen_meth_x,
        sep_fact,
        set_seed = None,
        gen_val_data=False,
        x_vals = None,
        with_noise = False,
    ):
        """
        Generates simulated data in an instance of the Data class

        Parameters
        ----------
        num_theta_data: int
            Number of parameter sets
        num_x_data: int
            Number of experiments
        gen_meth_theta: GenMethod
            Whether to generate theta data with LHS or grid method
        gen_meth_x: GenMethod
            Whether to generate X data with LHS or grid method
        sep_fact: float or int
            The separation factor that decides what percentage of data will be training data. Between 0 and 1.
        set_seed: int or None, default None
            Optional seed to generate initial LHS training data with. If None, seed will be the seed of the class
        gen_val_data: bool, default False
            Whether validation data (no y vals) or simulation data (has y vals) will be generated
        x_vals: np.ndarray or None, default None
            X values to use for simulation data. If None, x_vals will be generated based on bounds and num_x_data
        with_noise: bool, default False
            Whether to generate data with noise

        Returns
        --------
        sim_data: Data
            Simulated x and y data along with parameter bounds

        Raises
        ------
        AssertionError
            If any of the inputs are not of the correct type or value
        ValueError
            If num_theta_data or num_x_data are not a positive integer or gen_val is not a boolean
        Warning
            If more than 5000 points are generated

        Notes:
        -------
        Warning: This function will not generate exactly the same values of y when repeatedly called, even with the same seed. 
        """
        assert isinstance(
            sep_fact, (float, int)
        ), "Separation factor must be float or int > 0"
        assert 0 < sep_fact <= 1, "sep_fact must be > 0 and less than or equal to 1!"

        #Random if rng is not set, otherwise set by seed of simulator
        rng = self.rng_set

        # Pick theta/x seeds by data role: simulation data (gen_val_data=False) seeds both
        # theta and x from sim_seed; validation data reuses sim_seed for x (so it shares the
        # same x-grid as the simulation data) but val_seed for theta (so its theta draws are
        # independent of the simulation data's).
        if gen_val_data == False and self.sim_seed is not None:
            seed_theta = self.sim_seed
            seed_x = self.sim_seed
        elif gen_val_data == True and self.sim_seed is not None:
            seed_theta = self.val_seed
            seed_x = self.sim_seed
        else:
            seed_theta = None
            seed_x = self.sim_x_seed #For data generation we always want x to be the same
        
        #Set the theta seed to the given seed if one is provided
        if set_seed is not None:
            seed_theta = set_seed

        if isinstance(gen_val_data, bool) == False:
            raise ValueError("gen_val_data must be bool")

        # Chck that num_data > 0
        if num_theta_data <= 0 or isinstance(num_theta_data, int) == False:
            raise ValueError("num_theta_data must be a positive integer")

        if num_x_data <= 0 or isinstance(num_x_data, int) == False:
            raise ValueError("num_x_data must be a positive integer")

        # Set bounds on theta which we are regressing given bounds_theta and indeces to consider
        # X data we always want the same between simulation and validation data
        if x_vals is None:
            x_data = vector_to_1D_array(
                self.__create_param_data(num_x_data, self.bounds_x, gen_meth_x, seed_x)
            )
        else:
            x_data = vector_to_1D_array(x_vals)

        # Infer how many times to repeat theta and x values given whether they were generated by LHS or a meshgrid
        # X and theta repeated at least once per time the other is generated
        repeat_x = num_theta_data
        repeat_theta = len(x_data)

        # If using a meshgrid this number is exponentiated by the number of dimensions of itself
        if gen_meth_theta.value == 2:
            repeat_x = num_theta_data ** (self.dim_theta)
        if gen_meth_x.value == 2:
            repeat_theta = num_x_data ** (self.dim_x)

        # Warn user if >5000 pts generated
        if repeat_x * repeat_theta > 5000:
            warnings.warn("More than 5000 points will be generated!")

        # Generate all rows of simulation data (empty shell; theta/x/y filled in below)
        sim_data = SimulationData(
            None,
            None,
            None,
            bounds_theta=self.bounds_theta_reg,
            bounds_x=self.bounds_x,
            sep_fact=sep_fact,
        )

        # Generate simulation data theta_vals and create instance of data class
        sim_theta_vals = vector_to_1D_array(
            self.__create_param_data(
                num_theta_data, self.bounds_theta_reg, gen_meth_theta, seed_theta
            )
        )

        # Add repeated theta_vals and x_data to sim_data
        sim_data.theta_vals = np.repeat(sim_theta_vals, repeat_theta, axis=0)
        sim_data.x_vals = np.vstack([x_data] * repeat_x)

        # Add y_vals for sim_data
        if with_noise == False:
            #Default to noiseless training data
            sim_data.y_vals = self.evaluate_model(sim_data, self.noise_mean, 0, rng)
        else:
            # Generate train data with noise if some noise is specified
            sim_data.y_vals = self.evaluate_model(sim_data, self.noise_mean, self.noise_std, rng)

        return sim_data

    def simulation_data_from_evaluations(self, theta_vals, y_vals, x_vals, sep_fact=1.0):
        """
        Simulation data from runs that were already done (e.g. an existing design)

        Parameters
        ----------
        theta_vals: np.ndarray
            (n_theta, dim_theta) parameter sets, in the same space as the parameter bounds
        y_vals: np.ndarray
            (n_theta, n_x) outputs: row i holds the outputs of theta_vals[i] at every x_vals row
        x_vals: np.ndarray
            (n_x, dim_x) conditions, e.g. the experimental x values
        sep_fact: float or int, default 1.0
            Fraction of the data used for training. Between 0 and 1.

        Returns
        --------
        sim_data: SimulationData
            Same layout as generate_simulation_data: each theta repeated once per x
        """
        assert 0 < sep_fact <= 1, "sep_fact must be > 0 and less than or equal to 1!"
        theta_vals = vector_to_1D_array(np.asarray(theta_vals, dtype=float))
        x_vals = vector_to_1D_array(np.asarray(x_vals, dtype=float))
        y_vals = np.asarray(y_vals, dtype=float)
        n_theta = theta_vals.shape[0]
        n_x = x_vals.shape[0]

        assert theta_vals.shape[1] == self.dim_theta, f"theta_vals must have {self.dim_theta} columns"
        assert x_vals.shape[1] == self.dim_x, f"x_vals must have {self.dim_x} columns"
        assert y_vals.shape == (n_theta, n_x), f"y_vals must have shape ({n_theta}, {n_x})"
        assert np.all(np.isfinite(theta_vals)), "theta_vals must be finite"
        assert np.all(np.isfinite(y_vals)), "y_vals must be finite"
        lower = self.bounds_theta_reg[0]
        upper = self.bounds_theta_reg[1]
        assert np.all(theta_vals >= lower) and np.all(theta_vals <= upper), "theta_vals must be inside the parameter bounds"

        sim_data = SimulationData(
            np.repeat(theta_vals, n_x, axis=0),
            np.vstack([x_vals] * n_theta),
            y_vals.reshape(-1),
            bounds_theta=self.bounds_theta_reg,
            bounds_x=self.bounds_x,
            sep_fact=sep_fact,
        )
        return sim_data

    def generate_parameter_samples(self, num_theta_data, rng_seed = None):
        """
        Generates parameter sets for an instance of the Data class

        Parameters
        ----------
        num_theta_data: int
            Number of parameter sets
        rng_seed: int or None, default None
            Offset added to self.start_seed to derive the sampling seed. If None,
            self.start_seed is used directly

        Returns
        --------
        theta_vals: np.ndarray
            Generated parameter sets

        Raises
        ------
        AssertionError
            If num_theta_data is not a positive integer
        Warning
            If more than 5000 points are generated
        """
        #Ensures seed will never be the same as the data generation seeds (always higher)
        if rng_seed == None:
            rng_seed = self.start_seed
        else:
            rng_seed += self.start_seed

        assert (
            isinstance(num_theta_data, int) and num_theta_data > 0
        ), "num_theta_data must be int > 0"
        gen_meth_theta = GenMethod(1)

        # Warn user if >5000 pts generated
        if num_theta_data > 5000:
            warnings.warn("More than 5000 points will be generated!")

        # Generate simulation data theta_vals and create instance of data class
        theta_vals = vector_to_1D_array(
            self.__create_param_data(
                num_theta_data, self.bounds_theta_reg, gen_meth_theta, rng_seed
            )
        )

        return theta_vals

    def to_sse_data(
        self, method, sim_data, exp_data, sep_fact, y_to_sse=False
    ):
        """
        Creates objective function simulation data based on state points, parameter sets, the GPBO method, and self.calc_y_fxn

        Parameters
        ----------
        method: GPBOMethod
            Fully defined methods class which determines which method will be used
        sim_data: Data
            Class containing at least the theta_vals, x_vals, and y_vals for simulation
        exp_data: Data
            Class containing at least the x_data and y_data for the experimental data
        sep_fact: float or int
            The separation factor that decides what percentage of data will be training data. Between 0 and 1.
        y_to_sse: bool, default False
            Whether sim_data.y_vals will be set as y (True) or sse(y) (False)

        Returns
        --------
        sim_sse_data: np.ndarray
            Objective function data generated from y_vals

        Raises
        ------
        AssertionError
            If sep_fact is not between 0 and 1
        ValueError
            If y_to_sse is not a boolean
        """

        assert isinstance(
            sep_fact, (float, int)
        ), "Separation factor must be float or int > 0"
        assert 0 < sep_fact <= 1, "sep_fact must be > 0 and less than or equal to 1!"

        if isinstance(y_to_sse, bool) == False:
            raise ValueError("y_to_sse must be bool")

        # Find length of theta and x in data arrays
        len_theta = sim_data.n_theta
        len_x = exp_data.n_x

        # Q: For this dataset does it make more sense to have all theta and x values or just the unique thetas and x values?
        # A: Just the unique ones. No need to store extra data if we won't use it and it will be saved somewhere else regardless
        # Assign unique theta indeces and create an array of them
        unique_indexes = np.unique(sim_data.theta_vals, axis=0, return_index=True)[1]
        unique_theta_vals = np.array(
            [sim_data.theta_vals[index] for index in sorted(unique_indexes)]
        )
        # Add the unique theta_vals and exp_data x values to the new data class instance
        sim_sse_data = ObjectiveData(
            unique_theta_vals,
            x_vals=exp_data.x_vals,
            sse=sim_data.sse,
            sse_var=sim_data.sse_var,
            acq=sim_data.acq,
            bounds_theta=self.bounds_theta,
            bounds_x=self.bounds_x,
            sep_fact=sep_fact,
        )

        if y_to_sse == False and sim_data.y_vals is not None:
            # Define all y_sims
            y_sim = sim_data.y_vals

            # Reshape y_sim into n_theta rows x n_x columns
            indices = np.arange(0, len_theta, len_x)
            n_blocks = len(indices)
            # Slice y_sim into blocks of size len_x and calculate squared errors for each block
            sum_error_sq, _ = blockwise_sse(y_sim, exp_data.y_vals, n_blocks, len_x)
            # objective function only log if using 1B
            if method.log_scaled:
                #Set a minimum error to avoid log(0)
                sum_error_sq[sum_error_sq < 1e-16] = 1e-16
                sum_error_sq = np.log(sum_error_sq)  # Scaler

            # Add y_values to data class instance
            sim_sse_data.y_vals = sum_error_sq

        return sim_sse_data
