"""GPBODriver: orchestrates the Bayesian-optimization loop (data prep, GP fitting,
acquisition optimization, termination, and results assembly).
"""
import numpy as np
import pandas as pd
import copy
import time
import pickle
import gzip
from itertools import combinations
from .enums import GenMethod
from .methods import GPBOMethod
from .config import BOConfig
from .simulator import Simulator
from .data import Data, CandidateSet
from .exploration import ExplorationBias
from .emulators import GPEmulator, ObjectiveGP, EmulatorGP, build_gp_emulator
from .results import BOResults, build_iteration_row, ITERATION_COLUMNS
from .acquisition_optimizer import AcquisitionOptimizer


class GPBODriver:
    """
    The base class for running the GPBO Workflow. Delegates acquisition/SSE scipy
    optimization to the collaborator `self.acq_optimizer` (AcquisitionOptimizer); the
    thin wrappers below (__get_best_error, __make_starting_opt_pts, __opt_with_scipy,
    __scipy_fxn) exist only to preserve existing call-site signatures.

    Methods
    --------------
    __init__
    gp_emulator (property): re-syncs self.acq_optimizer.gp_emulator on every assignment
    __gen_emulator()
    __get_best_error()
    __make_starting_opt_pts(best_error_metrics, rng_seed)
    __opt_with_scipy(opt_obj, get_y, with_noise)
    __scipy_fxn(theta, opt_obj, best_error_metrics)
    create_heat_map_param_data(n_points_set)
    __augment_train_data(theta_best_data)
    create_data_instance_from_theta(theta_array)
    __run_bo_iter(iteration)
    __run_bo_to_term(run_num, job)
    __run_bo_workflow(run_num, job)
    reset_rng()
    run(job)
    save_results_run(job)
    """

    # Class variables and attributes

    def __init__(
        self,
        cs_params,
        method,
        simulator,
        exp_data,
        sim_data,
        sim_sse_data,
        val_data,
        val_sse_data,
        gp_emulator,
        ep_bias,
        gen_meth_theta,
        backend=None,
    ):
        """
        Parameters
        ----------
        cs_params: BOConfig
            Class containing the values associated with BOConfig
        method: GPBOMethod
            Class containing GPBO method information
        simulator: Simulator
            Class containing values of simulation parameters
        exp_data: Data
            Experimental data containing at least exp_data.theta_vals, exp_data.x_vals, and exp_data.y_vals
        sim_data: Data
            Simulated data containing at least sim_data.theta_vals, sim_data.x_vals, and sim_data.y_vals
        sim_sse_data: Data
            Simulated objective data containing at least sim_sse_data.theta_vals, sim_sse_data.x_vals, and sim_sse_data.y_vals
        val_data: Data or None
            Validation data containing at least val_data.theta_vals, val_data.x_vals, and val_data.y_vals
        val_sse_data: Data or None
            Validation data containing at least val_sse_data.theta_vals, val_sse_data.x_vals, and val_sse_data.y_vals
        gp_emulator: GPEmulator
            Class containing gp_emulator data (set after training)
        ep_bias: ExplorationBias class
            Class containing exploration parameter info
        gen_meth_theta: GenMethod or None
            The method by which simulation data is generated. For heat map making
        backend: GPBackend, str, or None, default None
            GP backend instance or name ("gpflow", "gpytorch"), passed to
            build_gp_emulator. None means gpflow. Tests inject a fake backend here.

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value
        """
        assert isinstance(
            cs_params, BOConfig
        ), "cs_params must be instance of BOConfig"
        assert isinstance(
            method, GPBOMethod
        ), "method must be instance of GPBOMethod"
        assert isinstance(
            simulator, Simulator
        ), "simulator must be instance of Simulator"
        assert isinstance(exp_data, Data), "exp_data must be instance of Data"
        assert isinstance(sim_data, Data), "sim_data must be instance of Data"
        assert isinstance(sim_sse_data, Data), "sim_sse_data must be instance of Data"
        assert (
            isinstance(val_data, Data) or val_data is None
        ), "val_data must be instance of Data or None"
        assert (
            isinstance(val_sse_data, Data) or val_sse_data is None
        ), "val_sse_data must be instance of Data or None"
        assert (
            isinstance(gp_emulator, (ObjectiveGP, EmulatorGP))
            or gp_emulator is None
        ), "gp_emulator must be instance of ObjectiveGP, EmulatorGP, or None"
        assert isinstance(
            ep_bias, ExplorationBias
        ), "ep_bias must be instance of ExplorationBias"
        assert isinstance(
            gen_meth_theta, GenMethod
        ), "gen_meth_theta must be instance of GenMethod"

        # Constructor method
        self.cs_params = cs_params
        self.method = method
        self.simulator = simulator
        self.exp_data = exp_data
        self.sim_data = sim_data
        self.sim_sse_data = sim_sse_data
        self.val_data = val_data
        self.val_sse_data = val_sse_data
        self.ep_bias = ep_bias
        self.gen_meth_theta = gen_meth_theta
        self._backend = backend
        self.bo_iter_term_frac = 0.3  # The fraction of iterations after which to terminate bo if no sse improvement is made
        self.sse_penalty = 1e7  # The penalty the __scipy_opt function gets for choosing nan theta values
        self.sg_mc_samples = 2000  # This can be changed at will
        # AcquisitionOptimizer owns the scipy-optimization scratch state (min_obj_*,
        # opt_start_pts) and a gp_emulator reference that is re-synced (never cached) by
        # the gp_emulator property below. Must be constructed before self.gp_emulator is
        # first assigned, since that assignment goes through the property setter.
        self._gp_emulator = None
        self.acq_optimizer = AcquisitionOptimizer(
            cs_params, method, simulator, exp_data, ep_bias, self.sse_penalty,
            self.sg_mc_samples, self.create_data_instance_from_theta,
        )
        self.gp_emulator = gp_emulator
        # self.reset_rng()

    @property
    def gp_emulator(self):
        return self._gp_emulator

    @gp_emulator.setter
    def gp_emulator(self, value):
        """
        Every assignment to self.gp_emulator re-syncs self.acq_optimizer.gp_emulator to the
        same object -- gp_emulator is *replaced* (not just mutated) every restart, and
        AcquisitionOptimizer must never hold a stale reference (PHASE8_AUDIT.md §3.8,
        risk #1). This covers both replacement sites in __run_bo_workflow, plus any other
        assignment (e.g. tests that set driver.gp_emulator directly).
        """
        self._gp_emulator = value
        self.acq_optimizer.gp_emulator = value
        assert self._gp_emulator is self.acq_optimizer.gp_emulator, (
            "acq_optimizer.gp_emulator must always be identical to GPBODriver.gp_emulator"
        )

    def __make_BO_results_temp(self, results_df, why_term, max_ei_details_df, list_gp_emulator_class):
        "Makes BO results from minimum data"

        # Set results for all compiled iterations for that run
        bo_results_res = BOResults(
            None, None, self.exp_data, None, results_df, None, why_term, None
        )

        bo_results_GPs = BOResults(
            None,
            None,
            None,
            list_gp_emulator_class,
            None,
            max_ei_details_df,
            None,
            None,
        )
        return bo_results_res, bo_results_GPs


    def __gen_emulator(self):
        """
        Sets GP Emulator class with training data and validation data based on the method class instance

        Returns
        --------
        gp_emulator: GPEmulator
            Class for the GP emulator
        """
        # Build the right emulator (ObjectiveGP / EmulatorGP) via the shared factory.
        return build_gp_emulator(
            self.method,
            self.sim_data,
            self.sim_sse_data,
            self.val_data,
            self.val_sse_data,
            self.cs_params.kernel,
            self.cs_params.lengthscale,
            self.cs_params.outputscale,
            self.cs_params.retrain_gp,
            self.cs_params.seed,
            self.cs_params.normalize,
            self.simulator.noise_std,
            self.exp_data.n_x,
            backend=self._backend,
        )

    def __get_best_error(self):
        """
        Helper function to calculate the best error and squared error calculations over x given the method.

        Returns
        -------
        be_data: Data
            Contains best_error as an instance of the data class
        be_metrics: tuple(float, np.ndarray, np.ndarray)
            The min_SSE, param at min_SSE, and squared residuals
        """
        return self.acq_optimizer.get_best_error()

    def __make_starting_opt_pts(self, best_error_metrics, rng_seed):
        """
        Makes starting point for optimization with scipy

        Parameters
        -----------
        best_error_metrics: tuple(float, np.ndarray, np.ndarray)
            The best error, best error parameter set, and best_error_x values of the method. Hint: Use self.__get_best_error()

        Returns
        --------
        starting_pts: np.ndarray
            Array of parameter set initializations for self.__opt_with_scipy
        """
        starting_pts = self.acq_optimizer.make_starting_points(best_error_metrics, rng_seed)
        self.acq_optimizer.opt_start_pts = starting_pts
        return starting_pts

    def __opt_with_scipy(self, opt_obj, get_y = False, with_noise = False):
        """
        Optimizes a function with scipy.optimize

        Parameters
        ----------
        opt_obj: str
            Which objective to calculate. neg_ei, E[SSE], or SSE
        get_y: bool, default False
            Whether to return the y values of the optimized theta
        with_noise: bool, default False
            Whether to return the y values with noise

        Returns
        --------
        best_val: float
            The optimized value of the function
        best_theta: np.ndarray
            The parameter set corresponding to best_val

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value
        """
        # Seam #2: thread the driver's own rng_set Generator explicitly -- the optimizer
        # must never own/cache its own rng reference (PHASE8_AUDIT.md §3.8, risk #2).
        return self.acq_optimizer.optimize(opt_obj, self.rng_set, get_y=get_y, with_noise=with_noise)

    def __scipy_fxn(self, theta, opt_obj, best_error_metrics):
        """
        Calculates either -ei, sse objective, or E[SSE] at a candidate parameter set value

        Parameters
        -----------
        theta: np.ndarray
            Array of theta values to optimize
        opt_obj: str
            Which objective to calculate. 'neg_ei', 'E_sse', or 'sse'
        best_error_metrics: tuple(float, np.ndarray, np.ndarray)
            The best error, best error parameter set, and best_error_x values of the method. Hint: Use self.__get_best_error()

        Returns
        --------
        obj: float, the value of the specified objective function for the given candidate parameter set

        Notes
        -----
        If there are nan values in theta, the objective function is set to 1 for neg_ei and self.sse_penalty for sse and E_sse

        """
        return self.acq_optimizer._objective(theta, opt_obj, best_error_metrics, self.rng_set)

    def create_heat_map_param_data(self, n_points_set=None):
        """
        Creates parameter sets that can be used to generate heat maps of data at any given iteration

        Parameters
        -----------
        n_points_set: int or None, default None
            The number of points to use per axis for creating heat maps. If None, the number of unique simulation points is used

        Returns
        --------
        heat_map_data_dict: dict
            Heat map data for each set of 2 parameters indexed by parameter name tuple ("param_1,param_2")
        """
        assert isinstance(
            self.gp_emulator, (ObjectiveGP, EmulatorGP)
        ), "self.gp_emulator must be instance of ObjectiveGP or EmulatorGP"
        assert isinstance(
            self.gp_emulator.gp_sim_data, Data
        ), "self.gp_emulator.gp_sim_data must be an instance of Data!"
        assert isinstance(
            self.gen_meth_theta, GenMethod
        ), "self.gen_meth_theta must be instance of GenMethod"
        assert isinstance(
            self.exp_data.x_vals, (np.ndarray)
        ), "self.exp_data.x_vals must be np.ndarray"
        assert (
            isinstance(n_points_set, int) or n_points_set is None
        ), "n_points_set must be None or int"

        # Create list of heat map theta data
        heat_map_data_dict = {}

        # Create a linspace for the number of dimensions and define number of points
        dim_list = np.linspace(
            0, self.simulator.dim_theta - 1, self.simulator.dim_theta
        )
        # Create a list of all combinations (without repeats e.g no (1,1), (2,2)) of dimensions of theta
        mesh_combos = np.array(list(combinations(dim_list, 2)), dtype=int)

        # Set x_vals
        norm_x_vals = self.exp_data.x_vals
        num_x = self.exp_data.n_x

        # If no number of points is set, use the length of the unique simulation thetas
        if n_points_set == None:
            # Use number of training theta for number of theta points
            n_thetas_points = len(self.gp_emulator.gp_sim_data.get_unique_theta())
            # Initialze meshgrid-like set of theta values at their true values
            # If points were generated with an LHS, the number of points per parameter is n_thetas_points for the meshgrid
            if self.gen_meth_theta.value == 1:
                n_points = n_thetas_points
            else:
                # For a meshgrid, the number of theta values/ parameter is n_thetas_points for the meshgrid ^(1/theta_dim)
                n_points = int((n_thetas_points) ** (1 / self.simulator.dim_theta))
        else:
            n_points = n_points_set

        # Ensure we will never generate more than 5000 pts per heat map
        # if self.method.is_emulator == True:
        if num_x * n_points**2 >= 5000:
            n_points = int(np.sqrt(5000 / (num_x)))

        # Meshgrid set always defined by n_points**2
        # Set thetas for meshgrid. Never use more than 10000 points
        theta_set = np.tile(np.array(self.simulator.theta_true), (n_points**2, 1))

        # Infer how many times to repeat theta and x values given that heat maps are meshgrid form by definition
        # The meshgrid of parameter values created below is symmetric, therefore, x is repeated by n_points**2 for a 2D meshgrid
        repeat_x = n_points**2  # Square because only 2 values at a time change
        x_vals = np.vstack([norm_x_vals] * repeat_x)
        repeat_theta = self.exp_data.n_x

        # Loop over all possible theta combinations of 2
        for i in range(len(mesh_combos)):
            # Create a copy of the true values to change the mehsgrid valus on
            theta_set_copy = np.copy(theta_set)
            # Set the indeces of theta_set for evaluation as each row of mesh_combos
            idcs = mesh_combos[i]
            # define name of parameter set as tuple ("param_1,param_2")
            data_set_name = (
                self.simulator.theta_true_names[idcs[0]],
                self.simulator.theta_true_names[idcs[1]],
            )

            # Create a meshgrid of values of the 2 selected values of theta and reshape to the correct shape
            # Assume that theta1 and theta2 have equal number of points on the meshgrid
            theta1 = np.linspace(
                self.simulator.bounds_theta_reg[0][idcs[0]],
                self.simulator.bounds_theta_reg[1][idcs[0]],
                n_points,
            )
            theta2 = np.linspace(
                self.simulator.bounds_theta_reg[0][idcs[1]],
                self.simulator.bounds_theta_reg[1][idcs[1]],
                n_points,
            )
            theta12_mesh = np.array(np.meshgrid(theta1, theta2))
            theta12_vals = np.array(theta12_mesh).T.reshape(-1, 2)

            # Set initial values for evaluation (true values) to meshgrid values
            theta_set_copy[:, idcs] = theta12_vals

            # Put values into instance of data class
            # Create data set based on emulator status
            if self.method.is_emulator == True:
                # Repeat the theta vals for Type 2 methods to ensure that theta and x values are in the correct form for evaluation with gp_emulator.eval_gp_mean_heat_map()
                theta_vals = np.repeat(theta_set_copy, repeat_theta, axis=0)
                data_set = CandidateSet(
                    theta_vals,
                    x_vals,
                    bounds_theta=self.simulator.bounds_theta_reg,
                    bounds_x=self.simulator.bounds_x,
                    sep_fact=self.cs_params.sep_fact,
                )
            else:
                data_set = CandidateSet(
                    theta_set_copy,
                    norm_x_vals,
                    bounds_theta=self.simulator.bounds_theta_reg,
                    bounds_x=self.simulator.bounds_x,
                    sep_fact=self.cs_params.sep_fact,
                )

            # Append data set to dictionary with name
            heat_map_data_dict[data_set_name] = data_set

        return heat_map_data_dict

    def __augment_train_data(self, theta_best_data):
        """
        Augments training data given a new data point

        Parameters
        ----------
        theta_best_data: Data
            The parameter set data associated with the optimal acquisition function value
        """
        # Augment training theta, x, and y/sse data
        self.gp_emulator.append_training_point(theta_best_data)

    def create_data_instance_from_theta(self, theta_array, get_y=True, with_noise = False):
        """
        Creates instance of Data from an np.ndarray parameter set

        Parameters
        ----------
        theta_array: np.ndarray
            Array of parameter values to turn into an instance of Data
        get_y: bool, default True
            Whether to calculate y values for theta_array
        with_noise: bool, default False
            Whether to add noise to the y values

        Returns
        --------
        theta_arr_data: Data
            Data class instance for the theta_array

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value
        """
        return create_data_instance_from_theta(
            theta_array, self.method, self.simulator, self.exp_data,
            self.cs_params.sep_fact, get_y=get_y, with_noise=with_noise, rng=self.rng_set,
        )

    def __run_bo_iter(self, iteration):
        """
        Runs a single GPBO iteration

        Parameters
        ----------
        iteration: int, The iteration of BO in progress

        Returns
        --------
        iter_df: pd.DataFrame
            Dataframe containing the results from the GPBO Workflow for iteration
        iter_max_ei_terms: pd.DataFrame or None
            Contains ei calculation terms for max ei parameter set if self.cs_params.save_data
        gp_emulator_curr: GPEmulator
            The class used for this iteration of the GPBO workflow
        """
        # Start timer
        # Initialize iter_max_ei df to None
        iter_max_ei_terms = None
        
        #Initialize the iterations seed with start_seed as a backup
        iter_seed = self.simulator.start_seed

        #Generate a random number for the seed to generate initial LHS samples with that is not the same as the sim or val seeds 
        if self.cs_params.seed is not None:
            for i in range(10):
                seed_init = self.rng_set.integers(1, 1e8)
                if seed_init not in [self.simulator.sim_seed, self.simulator.val_seed]:
                    iter_seed = seed_init
                    break
        else:
            iter_seed = None
        
        time_start = time.time()

        # Train GP model (this step updates the model to a trained model)
        self.gp_emulator.fit()

        # Calcuate best error
        best_err_data, best_error_metrics = self.__get_best_error()

        # Add not log best error to ep_bias
        if iteration == 0 or self.ep_bias.ep_enum.value == 4:
            # Since best error is squared when used in Jasrasaria calculations, the value will always be >=0
            self.ep_bias.best_error = best_error_metrics[0]

        # Calculate mean of var for validation set if using Jasrasaria heuristic
        if self.ep_bias.ep_enum.value == 4:
            # Calculate average gp mean and variance of the validation set
            val_pred = self.gp_emulator.predict(target="val")
            val_gp_mean, val_gp_var = val_pred
            # For emulator methods, the mean of the variance should come from the sse variance
            if self.method.is_emulator == True:
                # Redefine gp_mean and gp_var to be the mean and variane of the sse. Reuses
                # `val_pred` instead of re-reading data.gp_mean/data.gp_covar off gp_val_data.
                val_gp_mean, val_gp_var = self.gp_emulator.predict_sse(
                    target="val", method=self.method, exp_data=self.exp_data, prediction=val_pred
                )

            # Check for ln(sse) values
            # For 1B, propogate errors associated with an unlogged sse value
            val_gp_var = val_gp_var * np.exp(val_gp_mean) ** 2

            # Set mean of sse variance
            mean_of_var = np.average(val_gp_var)
            self.ep_bias.mean_of_var = mean_of_var

        # Set initial exploration bias and bo_iter
        if self.ep_bias.ep_enum.value == 2:
            self.ep_bias.bo_iter = iteration

        # Calculate new ep. Note. It is extemely important to do this AFTER setting the ep_max
        self.ep_bias.update()

        # Set Optimization starting points for this iteration
        self.opt_start_pts = self.__make_starting_opt_pts(best_error_metrics, iter_seed)

        # Call optimize E[SSE] or log(E[SSE]) objective function
        # Note if we didn't want actual sse values, we would have to set compute_y_sse = False
        min_sse, min_theta_data, min_sse_prediction = self.__opt_with_scipy("sse", get_y = self.cs_params.compute_y_sse, with_noise = self.cs_params.with_noise)

        # Call optimize EI acquistion fxn (If not using E[SSE])
        if self.method.method_name.value != 7:
            opt_acq, acq_theta_data, best_prediction = self.__opt_with_scipy("neg_ei", get_y = True, with_noise = self.cs_params.with_noise)
            if self.method.is_emulator == True:
                ei_args = dict(
                    data=acq_theta_data,
                    exp_data=self.exp_data,
                    ep_bias=self.ep_bias,
                    best_error_metrics=best_error_metrics,
                    method=self.method,
                    sg_mc_samples=self.sg_mc_samples,
                    gp_prediction=best_prediction,
                )
            else:
                ei_args = dict(
                    data=acq_theta_data,
                    exp_data=self.exp_data,
                    ep_bias=self.ep_bias,
                    best_error_metrics=best_error_metrics,
                    gp_prediction=best_prediction,
                )
        else:
            opt_acq, acq_theta_data, best_prediction = self.__opt_with_scipy("E_sse", get_y = True, with_noise = self.cs_params.with_noise)

        # If type 2, turn it into sse_data
        # Set the best data to be in sse form if using a type 2 GP and find the min sse
        if self.method.is_emulator == True:
            # SSE at the min-SSE theta (y_vals is None unless compute_y_sse)
            min_sse_theta_data = self.simulator.to_sse_data(
                self.method,
                min_theta_data,
                self.exp_data,
                self.cs_params.sep_fact,
                False,
            )
            # SSE at the acquisition theta (always simulated)
            acq_sse_theta_data = self.simulator.to_sse_data(
                self.method,
                acq_theta_data,
                self.exp_data,
                self.cs_params.sep_fact,
                False,
            )

        # Otherwise the sse data is the original (scaled) data
        else:
            # Evaluate SSE & SSE stdev at max ei theta
            min_sse_theta_data = min_theta_data
            acq_sse_theta_data = acq_theta_data

        # Evaluate max EI terms at theta
        if self.cs_params.save_data and not self.method.method_name.value == 7:
            ei_max, iter_max_ei_terms = self.gp_emulator.expected_improvement(**ei_args)

        # Turn min_sse_sim value into a float (this makes analyzing data from csvs and dataframes easier)
        min_sse_gp = float(np.asarray(min_sse).item())
        opt_acq_sim = float(np.asarray(acq_sse_theta_data.y_vals).item())

        # calculate improvement if using Boyle's method to update the exploration bias
        # Improvement is true if the min sim sse found is lower than (not log) best error, otherwise it's false
        if min_sse_gp < best_error_metrics[0]:
            improvement = True
        else:
            improvement = False
        if self.ep_bias.ep_enum.value == 3:
            # Set ep improvement
            self.ep_bias.improvement = improvement

        # Create a copy of the GP Emulator Class for this iteration
        gp_emulator_curr = copy.deepcopy(self.gp_emulator)

        # Call __augment_train_data to append training data
        self.__augment_train_data(acq_theta_data)


        # Calc time/ iter
        time_end = time.time()
        time_per_iter = time_end - time_start

        # Create Results Pandas DataFrame for 1 iter
        num_exp_x = self.exp_data.n_x
        # Return SSE and not log(SSE) for 'Min Obj', 'sse_actual', 'sse_gp' when calculating MSE
        if self.cs_params.compute_y_sse:
            min_sse_sim = float(np.asarray(min_sse_theta_data.y_vals).item())
        else:
            min_sse_sim = None

        iter_df = build_iteration_row(
            best_error_metrics[0],
            self.ep_bias.ep_curr,
            acq_theta_data.theta_vals[0],
            opt_acq,
            opt_acq_sim,
            min_sse_theta_data.theta_vals[0],
            min_sse_gp,
            min_sse_sim,
            time_per_iter,
            self.method.log_scaled,
            num_exp_x,
        )

        return iter_df, iter_max_ei_terms, gp_emulator_curr

    def __run_bo_to_term(self, run_num, job = None):
        """
        Runs GPBO to termination

        Params:
        -------
        gp_model: gpflow.models.GPR, GP emulator for workflow

        Returns
        --------
        iter_df: pd.DataFrame
            Dataframe containing the results from the GPBO Workflow for all iterations
        max_ei_details_df: pd.DataFrame
            Contains ei data for max ei parameter sets for each bo iter if self.cs_params.save_data
        list_gp_emulator_class: list(GPEmulator)
            The classes used for all iterations of the GPBO workflow
        why_term: str
            String containing reasons for bo algorithm termination

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value
        """
        assert (
            0 < self.bo_iter_term_frac <= 1
        ), "self.bo_iter_term_frac must be between 0 and 1"
        # Initialize pandas dataframes
        results_df = pd.DataFrame(columns=ITERATION_COLUMNS)
        max_ei_details_df = pd.DataFrame()
        list_gp_emulator_class = []
        # Initialize count
        obj_counter = 0
        self.ep_bias.ep_curr = None

        cond1 = len(self.gpbo_res_GP) > 0
        cond2 = len(self.gpbo_res_GP) == run_num + 1
        cond3 = job is not None

        #Check for files in job for gpbo_res_simple and gpbo_res_GP
        if cond1 and cond2 and cond3:
            results_df = self.gpbo_res_simple[run_num].results_df
            self.ep_bias.ep_curr = self.gpbo_res_simple[run_num].results_df["alpha"].iloc[-1]
            #The obj_counter is set as why_term until termination happens
            obj_counter = self.gpbo_res_simple[run_num].why_term
            list_gp_emulator_class = self.gpbo_res_GP[run_num].list_gp_emulator_class

        #Start at the next iteration after data ends. If no data, start at 0
        iter_start = len(list_gp_emulator_class)

        # Initilize terminate flags
        acq_flag = False
        obj_flag = False
        terminate = False

        # Set why_term strings
        why_terms = ["acq", "obj", "max_budget"]

        # Do Bo iters while stopping criteria is not met
        while terminate == False:
            # Loop over number of max bo iters
            for i in range(iter_start, self.cs_params.bo_iter_tot, 1):
                # Output results of 1 bo iter and the emulator used to get the results
                iter_df, iter_max_ei_terms, gp_emulator_class = self.__run_bo_iter(
                    i
                ) 
                # Add results to dataframe
                results_df = pd.concat(
                    [results_df.astype(iter_df.dtypes), iter_df], ignore_index=True
                )
                if iter_max_ei_terms is not None:
                    max_ei_details_df = pd.concat(
                        [max_ei_details_df, iter_max_ei_terms]
                    )
                # At the first iteration
                if i == 0:
                    # improvement is defined as infinity on 1st iteration (something is always better than nothing)
                    improvement = np.inf
                elif results_df["sse_gp"].iloc[i] < float(
                    results_df["sse_gp"][:-1].min()
                ):
                    # And the improvement is defined as the difference between the last Min Obj Cum. and current Obj Min (unscaled)
                    if not self.method.log_scaled:
                        improvement = (
                            results_df["sse_gp"][:-1].min()
                            - results_df["sse_gp"].iloc[i]
                        )
                    else:
                        improvement = np.exp(
                            results_df["sse_gp"][:-1].min()
                        ) - np.exp(results_df["sse_gp"].iloc[i])
                # Otherwise
                else:
                    # And the improvement is defined as 0, since it must be non-negative
                    improvement = 0

                # Add gp emulator data from that iteration to list (before stopping criteria)
                list_gp_emulator_class.append(gp_emulator_class)
                # Call stopping criteria after 1st iteration and update improvement counter
                # If the improvement is negligible, add to counter
                if improvement < self.cs_params.obj_tol:
                    obj_counter += 1
                # Otherwise reset the counter
                else:
                    obj_counter = 0

                # Two independent early-stop signals (both require i>=4 so a handful of
                # iterations always run first, since acq_value/obj_counter are noisy early
                # on): acq_flag fires if the best acquisition value has stayed below
                # tolerance for 3 STRAIGHT iterations (the model sees nothing left worth
                # exploring); obj_flag fires if obj_counter (the consecutive-iteration
                # stagnation streak built up above) has run for bo_iter_term_frac (1/3) of
                # the total budget (sustained lack of SSE progress, not just one flat
                # iteration). set flag if opt acq. func val is less than the tolerance 3
                # times in a row
                if (
                    all(results_df["acq_value"].tail(3) < self.cs_params.acq_tol)
                    and i >= 4
                ):
                    acq_flag = True
                # set flag if small sse progress over 1/3 of total iteration budget
                if (
                    obj_counter
                    >= int(self.cs_params.bo_iter_tot * self.bo_iter_term_frac)
                    and i >= 4
                ):
                    obj_flag = True

                flags = [acq_flag, obj_flag]

                # Terminate on whichever fires first: (1) the iteration budget is
                # exhausted (guarantees the loop always ends); (2) both acq_flag and
                # obj_flag agree (a corroborated early stop, so a single noisy signal
                # can't cut a run short); or (3) obj_counter alone has stagnated for half
                # the budget on long-enough runs (bo_iter_tot>=5) -- a stricter,
                # single-criterion fallback that doesn't need acq_flag's corroboration.
                # Terminate if you meet 2 stopping criteria, hit the budget, or obj has not improved after 1/2 of iterations
                if i == self.cs_params.bo_iter_tot - 1:
                    terminate = True
                    why_term = why_terms[-1]
                    break
                elif flags.count(True) >= 2:
                    terminate = True
                    # Pull indecies of list that are true
                    term_flags = [
                        why_terms[index] for index, value in enumerate(flags) if value
                    ]
                    why_term = "-".join(term_flags)
                    break
                elif (
                    obj_counter >= int(self.cs_params.bo_iter_tot * 0.5)
                    and self.cs_params.bo_iter_tot >= 5
                ):
                    terminate = True
                    why_term = why_terms[1]
                    break
                # Continue if no stopping criteria are met
                else:
                    terminate = False
                
                #Save results
                #make a new list of emulator classes to save which includes a copy of the original list + newest GP object
                list_emulator_class_temp = copy.deepcopy(list_gp_emulator_class.copy())
                list_emulator_class_temp[-1] = copy.deepcopy(self.gp_emulator)
                #Make temporary BO results for this iter
                bo_results_res, bo_results_GPs = self.__make_BO_results_temp(results_df, obj_counter, max_ei_details_df, list_emulator_class_temp)
                # Add simulator class and save the rng seeds that are being used
                bo_results_res.simulator_class = copy.deepcopy(self.simulator)
                bo_results_GPs.driver_rng = copy.deepcopy(self.rng_set)
                bo_results_res.sim_rng = copy.deepcopy(self.simulator.rng_set)
                
                #Save results at each iteration so that if the job takes a while it can be continued
                if len(self.gpbo_res_simple) == len(self.gpbo_res_GP) != run_num + 1:
                    self.gpbo_res_simple.append(bo_results_res)
                    self.gpbo_res_GP.append(bo_results_GPs)
                else:
                    self.gpbo_res_simple[run_num] = bo_results_res
                    self.gpbo_res_GP[run_num] = bo_results_GPs

                #Save results
                if job is not None:
                    self.save_results_run(job)
                
        # Reset the index of the pandas df
        results_df = results_df.reset_index()

        # Fill Cumulative value columns based on results
        # Initialize cum columns as the same as the original columns
        results_df.rename(columns={"index": "bo_iter"}, inplace=True)
        results_df["bo_iter"] += 1
        results_df["method"] = self.method.report_name
        results_df["max_evals"] = len(results_df)
        results_df["theta_best_at_acq"] = results_df["theta_at_acq"]
        results_df["theta_best_gp"] = results_df["theta_at_min"]
        results_df["theta_best_actual"] = results_df["theta_at_min"]
        results_df["termination_reason"] = why_term
        results_df["total_time"] = float(results_df["time_per_iter"].sum())

        results_df["best_sse_gp"] = np.minimum.accumulate(results_df["sse_gp"])
        if self.cs_params.compute_y_sse == True:
            results_df["best_sse_actual"] = np.minimum.accumulate(results_df["sse_actual"])
        else:
            results_df["best_sse_actual"] = None
        results_df["best_sse_at_acq"] = np.minimum.accumulate(results_df["sse_at_acq"])

        # Add cumulative values to the dataframe
        for i in range(len(results_df)):
            if i > 0:
                if (
                    results_df["best_sse_at_acq"].iloc[i]
                    >= results_df["best_sse_at_acq"].iloc[i - 1]
                ):
                    results_df.at[i, "theta_best_at_acq"] = (
                        results_df["theta_best_at_acq"].iloc[i - 1].copy()
                    )
                #If we are tracking actual values, update as normal, otherwise follow the same trend os the GP SSE
                if (self.cs_params.compute_y_sse == True and
                    results_df["best_sse_actual"].iloc[i]
                    >= results_df["best_sse_actual"].iloc[i - 1] 
                ) or (self.cs_params.compute_y_sse == False and results_df["best_sse_gp"].iloc[i]
                    >= results_df["best_sse_gp"].iloc[i - 1]):
                    results_df.at[i, "theta_best_actual"] = (
                        results_df["theta_best_actual"].iloc[i - 1].copy()
                    )
                if (
                    results_df["best_sse_gp"].iloc[i]
                    >= results_df["best_sse_gp"].iloc[i - 1]
                ):
                    results_df.at[i, "theta_best_gp"] = (
                        results_df["theta_best_gp"].iloc[i - 1].copy()
                    )

        # Create df for ei and add those results here
        if iter_max_ei_terms is not None:
            max_ei_details_df.columns = iter_max_ei_terms.columns.tolist()
            max_ei_details_df = max_ei_details_df.reset_index(drop=True)


        return results_df, max_ei_details_df, list_gp_emulator_class, why_term

    def __run_bo_workflow(self, run_num, job = None):
        """
        Runs a GPBO method through all bo iterations and reports the data for that run of the method

        Returns
        --------
        bo_results_res: BOResults
            Includes table of results, exp_Data and why term for the GPBO workflow
        bo_results_GPs: BOResults
            Includes the GP emulator classes used and max ei details for each iteration of the BO workflow

        Notes
        ------
        Two instances of BOResults are used since opening the GP files is often tedious and we may not need to open them to analyze the results
        """
        
        #If a results object for this run exists, load it
        cond1 = len(self.gpbo_res_GP) > 0
        cond2 = len(self.gpbo_res_GP) == run_num + 1
        cond3 = job is not None

        #If results exist for this run and is being saved, use the emulator class from the last iteration
        if cond1 and cond2 and cond3:
            self.gp_emulator = self.gpbo_res_GP[run_num].list_gp_emulator_class[-1]
            self.rng_set = self.gpbo_res_GP[run_num].driver_rng
            self.simulator.rng_set = self.gpbo_res_simple[run_num].sim_rng
        #If results do not exist for this run, initialize the emulator class
        else:
            #Reset driver rng at each run to update seed for driver class
            self.reset_rng()
            # Initialize gp_emualtor class
            gp_emulator = self.__gen_emulator()
            self.gp_emulator = gp_emulator

            # Choose training data
            train_data, test_data = self.gp_emulator.split_train_test(
                self.cs_params.sep_fact, self.cs_params.seed
            )

        ##Call bo_iter
        results_df, max_ei_details_df, list_gp_emulator_class, why_term = (
            self.__run_bo_to_term(run_num, job)
        )

        # # Set results for all compiled iterations for that run
        bo_results_res, bo_results_GPs = self.__make_BO_results_temp(results_df, why_term, max_ei_details_df, list_gp_emulator_class)

        # return bo_results_res, bo_results_GPs
        return bo_results_res, bo_results_GPs
    
    def reset_rng(self):
        """
        Resets the random number generator to the seed value
        """
        if self.cs_params.seed is not None:
            self.rng_set = np.random.default_rng(self.cs_params.seed)
        if self.simulator.sim_seed is not None:
            self.simulator.rng_set = np.random.default_rng(self.simulator.sim_seed)
    
    def run(self, job = None):
        """
        Runs multiple GPBO restarts

        Returns
        --------
        gpbo_res_simple: list(BOResults)
            Includes the most relevant results related to a set of BO iters for all restarts
        gpbo_res_GP: list(BOResults)
            Includes the GP emulator classes used and max ei details for each iteration of the BO workflow

        Notes
        ------
        gpbo_res_simple includes the Configuration, Simulator class, Experiment Data Results DataFrame, and termination criteria results
        """
        gpbo_res_simple = []
        gpbo_res_GP = []
        run_start = 0

        if job is not None:
            #Check for files in job for gpbo_res_simple and gpbo_res_GP
            # if os.path.exists("BO_Results.gz") and os.path.exists("BO_Results_GPs.gz"):
            if job.isfile("BO_Results.gz") and job.isfile("BO_Results_GPs.gz"):
                #Load the data from the files
                fileObj1 = gzip.open(job.fn("BO_Results.gz"), "rb")
                # fileObj1 = gzip.open("BO_Results.gz", "rb")
                gpbo_res_simple = pickle.load(fileObj1)
                fileObj1.close()
                fileObj2 = gzip.open(job.fn("BO_Results_GPs.gz"), "rb")
                # fileObj2 = gzip.open("BO_Results_GPs.gz", "rb")
                gpbo_res_GP = pickle.load(fileObj2)
                fileObj2.close()

        self.gpbo_res_simple = gpbo_res_simple
        self.gpbo_res_GP = gpbo_res_GP
        
        simulator_class = self.simulator
        configuration = {
            "Created At String": self.cs_params.created_at,
            "Method Name Enum Value": self.method.method_name.value,
            "Case Study Name": self.cs_params.cs_name,
            "Number of Parameters": len(self.simulator.theta_true_names),
            "Number of State Points": self.exp_data.n_x,
            "Exploration Bias Method Value": self.ep_bias.ep_enum.value,
            "Separation Factor": self.cs_params.sep_fact,
            "Normalize": self.cs_params.normalize,
            "Initial Kernel": self.cs_params.kernel,
            "Initial Lengthscale": self.cs_params.lengthscale,
            "Initial Outputscale": self.cs_params.outputscale,
            "Retrain GP": self.cs_params.retrain_gp,
            "Reoptimize Obj": self.cs_params.reoptimize_obj,
            "Heat Map Points Generated": self.cs_params.gen_heat_map_data,
            "Max BO Iters": self.cs_params.bo_iter_tot,
            "Number of Workflow Restarts": self.cs_params.bo_run_tot,
            "Seed": self.cs_params.seed,
            "Acq Tolerance": self.cs_params.acq_tol,
            "MC SG Max Points": self.sg_mc_samples,
            "Obj Improvement Tolerance": self.cs_params.obj_tol,
            "Theta Generation Enum Value": self.gen_meth_theta.value,
            "Gen y with Noise": self.cs_params.with_noise,
            "Gen y for Minimized SSE": self.cs_params.compute_y_sse,
        }

        #If some runs have already been completed
        if len(self.gpbo_res_simple) > 0:
            # Check if all of the iterations of that runs have been completed 
            if len(self.gpbo_res_GP[-1].list_gp_emulator_class) < self.cs_params.bo_iter_tot:
                #If not, complete the last run before continuing
                run_start = len(gpbo_res_simple) -1
            else:
                #If the run is complete, start from the next run
                run_start = len(gpbo_res_simple)

        #Get the seed based on the run number
        if self.cs_params.seed is not None:
            self.cs_params.seed += run_start

        #Complete remaining runs
        for i in range(run_start, self.cs_params.bo_run_tot, 1):
            #Run the bo workflow and get the results
            bo_results_res, bo_results_GPs = self.__run_bo_workflow(i, job)

            # Update the seed in configuration
            configuration["Seed"] = self.cs_params.seed
            # Add this copy of configuration with the new seed to the bo_results
            bo_results_res.configuration = configuration.copy()
            # # Add simulator class after rng changes (allows us to restart from the next run)
            bo_results_res.simulator_class = copy.deepcopy(simulator_class)
            # On the 1st iteration of the first run, create heat map data if we are actually generating the data
            if i == 0:
                if self.cs_params.gen_heat_map_data == True:
                    # Generate heat map data for each combination of parameter values stored in a dictionary
                    heat_map_data_dict = self.create_heat_map_param_data()
                    # Save these heat map values in the bo_results object
                    # Only store in first list entry to avoid repeated data which stays the same for each iteration.
                    bo_results_GPs.heat_map_data_dict = heat_map_data_dict

            #Save the results to the gpbo_res_simple and gpbo_res_GP lists.
            # The per-iteration loop in __run_bo_to_term normally appends an entry for this run,
            # but when bo_iter_tot == 1 the loop terminates before that append runs, leaving the
            # lists short. Append-or-assign handles both cases (fixes the bo_iter_tot==1 crash).
            if i < len(self.gpbo_res_simple):
                self.gpbo_res_simple[i] = bo_results_res
                self.gpbo_res_GP[i] = bo_results_GPs
            else:
                self.gpbo_res_simple.append(bo_results_res)
                self.gpbo_res_GP.append(bo_results_GPs)

            # #At each restart, resave gpbo_res_simple and gpbo_res_GP to the data file
            if job is not None:
                self.save_results_run(job)

            # Add 1 to the seed to get different seeds when the seeds are set at each restart
            if self.cs_params.seed is not None:
                self.cs_params.seed += 1

        return self.gpbo_res_simple, self.gpbo_res_GP

    def save_results_run(self, job):
        """
        Defines where to save data to and saves data accordingly

        Parameters
        ----------
        restart_bo_results: list of class instances of BO_results, The results of all restarts of the BO workflow for reproduction
        """
        ##Define a path for the data. (Use the name of the case study and date)
        #Get Date only from Created At String
        savepath1 = job.fn("BO_Results.gz")
        # savepath1 = "BO_Results.gz"
        fileObj1 = gzip.open(savepath1, "wb", compresslevel=1)
        pickled_results1 = pickle.dump(self.gpbo_res_simple, fileObj1)
        fileObj1.close()

        savepath2 = job.fn("BO_Results_GPs.gz")
        # savepath2 = "BO_Results_GPs.gz"
        fileObj2 = gzip.open(savepath2, "wb", compresslevel=2)
        pickled_results2 = pickle.dump(self.gpbo_res_GP, fileObj2)
        fileObj2.close()


def create_data_instance_from_theta(
    theta_array, method, simulator, exp_data, sep_fact, get_y=True, with_noise=False,
    rng=None,
):
    """
    Creates instance of Data from an np.ndarray parameter set

    Parameters
    ----------
    theta_array: np.ndarray
        Array of parameter values to turn into an instance of Data
    method: GPBOMethod
        Class containing GPBO method information
    simulator: Simulator
        Class containing values of simulation parameters
    exp_data: Data
        Experimental data containing at least exp_data.x_vals
    sep_fact: float or int
        The separation factor that decides what percentage of data will be training data
    get_y: bool, default True
        Whether to calculate y values for theta_array
    with_noise: bool, default False
        Whether to add noise to the y values
    rng: np.random.Generator or None, default None
        The random number generator used for noise generation

    Returns
    --------
    theta_arr_data: Data
        Data class instance for the theta_array

    Raises
    ------
    AssertionError
        If any of the required parameters are missing or not of the correct type or value
    """
    assert isinstance(theta_array, np.ndarray), "theta_array must be np.ndarray"
    assert len(theta_array.shape) == 1, "theta_array must be 1D"
    assert isinstance(
        exp_data.x_vals, (np.ndarray)
    ), "exp_data.x_vals must be np.ndarray"

    # Repeat the theta best array once for each x value
    # Need to repeat theta_best such that it can be evaluated at every x value in exp_data using simulator.evaluate_model
    theta_arr_repeated = np.repeat(
        theta_array.reshape(1, -1), exp_data.n_x, axis=0
    )
    # Add instance of Data class to theta_best
    theta_arr_data = CandidateSet(
        theta_arr_repeated,
        exp_data.x_vals,
        bounds_theta=simulator.bounds_theta_reg,
        bounds_x=simulator.bounds_x,
        sep_fact=sep_fact,
    )
    if get_y:
        if with_noise:
            # Calculate y values and sse for theta_best with noise
            theta_arr_data.y_vals = simulator.evaluate_model(
                theta_arr_data, simulator.noise_mean, simulator.noise_std, rng
            )
        else:
            # Calculate y values and sse for theta_best without noise
            theta_arr_data.y_vals = simulator.evaluate_model(
                theta_arr_data, simulator.noise_mean, 0, rng
            )

    # Set the best data to be in sse form if using a type 1 GP
    if method.is_emulator == False:
        theta_arr_data = simulator.to_sse_data(
            method,
            theta_arr_data,
            exp_data,
            sep_fact,
            not get_y,
        )

    return theta_arr_data
