import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional

# import emcal
from .GPBO_Classes_New import Simulator

# NOTE: pyomo is imported lazily inside the Müller case study solve method
# (__solve_pyomo_Muller_min) so the rest of the case studies / the package import
# without pyomo installed. Pyomo + an `ipopt` solver are only needed for CS2/CS3.


@dataclass
class CalibrationProblem:
    """
    A model-calibration problem: a model to calibrate, its parameters, and the
    data (or a synthetic ground truth) to calibrate against.

    This is the user-facing way to define a problem. Two modes:

    - **Real calibration** — supply ``experimental_data=(x, y)`` measured from the
      system; there is no known ground truth.
    - **Synthetic benchmark** — supply ``true_params`` (the "reference" parameters);
      experimental data is generated from the model at those parameters. This is the
      mode used by the paper's built-in case studies (see :func:`get_case_study`).

    Parameters
    ----------
    model : Callable
        The (expensive) model to calibrate, called as ``model(theta, x) -> y`` for a
        parameter vector ``theta`` and experimental condition(s) ``x``. Stateful models
        (solvers, fixed data) should capture that state in a closure or callable object.
    param_names : list[str]
        Names of the calibrated parameters, in order.
    param_bounds : list[tuple]
        ``(lo, hi)`` bounds per parameter (same length/order as ``param_names``).
    x_bounds : list[tuple] or None
        ``(lo, hi)`` bounds per experimental-design dimension, if conditions are sampled.
    x_values : np.ndarray or None
        Fixed experimental conditions, if the design is not sampled (e.g. VLE data).
    experimental_data : tuple or None
        ``(x, y)`` measured data for real calibration. Mutually exclusive with ``true_params``.
    true_params : np.ndarray or None
        Reference parameters for a synthetic benchmark. Mutually exclusive with ``experimental_data``.
    noise : float or None
        Optional noise level associated with the problem (informational; the synthetic
        data-generation noise is set when building the simulator).
    name : str
        Human-readable problem name.
    indices_to_consider : list[int] or None
        Advanced: subset of parameter indices actually calibrated (defaults to all).
        Used by the built-in Müller studies, which vary only a subset of 24 parameters.
    validate_model : bool, default True
        Run one trial ``model(theta, x)`` call at construction. Set False for expensive
        models (each call may be a full simulation).

    Raises
    ------
    AssertionError
        If the fields are inconsistent (see __post_init__).
    """

    model: Callable
    param_names: list
    param_bounds: list
    x_bounds: Optional[list] = None
    x_values: Optional[np.ndarray] = None
    experimental_data: Optional[tuple] = None
    true_params: Optional[np.ndarray] = None
    noise: Optional[float] = None
    name: str = ""
    indices_to_consider: Optional[list] = None
    validate_model: bool = True

    def __post_init__(self):
        assert callable(self.model), "model must be callable as model(theta, x)"
        assert len(self.param_names) == len(self.param_bounds), (
            "param_names and param_bounds must have the same length "
            f"({len(self.param_names)} vs {len(self.param_bounds)})"
        )
        for i, b in enumerate(self.param_bounds):
            assert len(b) == 2 and b[0] < b[1], (
                f"param_bounds[{i}] must be (lo, hi) with lo < hi, got {b}"
            )
        # Exactly one of experimental_data / true_params (real vs synthetic mode).
        assert (self.experimental_data is None) != (self.true_params is None), (
            "provide exactly one of experimental_data (real calibration) or "
            "true_params (synthetic benchmark)"
        )
        if self.true_params is not None:
            tp = np.asarray(self.true_params, dtype=float)
            assert len(tp) == len(self.param_names), (
                f"true_params length ({len(tp)}) must match param_names ({len(self.param_names)})"
            )
            for i, (lo, hi) in enumerate(self.param_bounds):
                assert lo <= tp[i] <= hi, (
                    f"true_params[{i}]={tp[i]} is outside its bound ({lo}, {hi})"
                )
        if self.experimental_data is not None:
            assert len(self.experimental_data) == 2, "experimental_data must be (x, y)"
            x_exp, y_exp = self.experimental_data
            assert len(x_exp) == len(y_exp), (
                f"experimental_data x and y must have the same length "
                f"({len(x_exp)} vs {len(y_exp)})"
            )
        # Trial model evaluation: catches a theta-dimension mismatch between the model
        # and param_* by actually calling it. Use true_params when available (known
        # evaluable), else the mid-point of the parameter bounds.
        if self.true_params is not None:
            trial_theta = np.asarray(self.true_params, dtype=float)
        else:
            trial_theta = np.array([(lo + hi) / 2 for lo, hi in self.param_bounds])
        trial_x = None
        if self.x_values is not None and len(self.x_values):
            trial_x = np.asarray(self.x_values)[0]
        elif self.x_bounds is not None:
            trial_x = np.array([(lo + hi) / 2 for lo, hi in self.x_bounds])
        if trial_x is not None and self.validate_model:
            try:
                y_trial = self.model(trial_theta, trial_x)
            except Exception as e:  # pragma: no cover - surfaced as a clear error
                raise AssertionError(
                    "model(theta, x) failed on a trial evaluation "
                    f"(theta dim {len(trial_theta)}, x={trial_x}): {e!r}. "
                    "Check that model's signature and theta dimension match param_names."
                )
            assert np.all(np.isfinite(np.asarray(y_trial, dtype=float))), (
                "model(theta, x) returned non-finite values at the trial evaluation"
            )


class _CaseStudyModel:
    """
    Picklable ``model(theta, x)`` wrapping a built-in ``calc_y_fxn(theta, x, args)``.

    A plain class (not a closure) so that a Simulator holding it -- and the BO_Results
    that pickle the Simulator -- remain picklable, matching the original design where the
    module-level calc_y_fxn and its args dict were stored directly.
    """

    def __init__(self, calc_y_fxn, args):
        self.calc_y_fxn = calc_y_fxn
        self.args = args

    def __call__(self, theta, x):
        return self.calc_y_fxn(theta, x, self.args)


class _CalcYFromModel:
    """
    Picklable 3-arg ``calc_y_fxn(theta, x, args)`` delegating to a ``model(theta, x)``.

    Lets a Simulator be built from any CalibrationProblem.model while keeping the
    3-argument calling convention Simulator expects (the extra ``args`` is ignored;
    model state is captured by the model object itself).
    """

    def __init__(self, model):
        self.model = model

    def __call__(self, theta, x, args=None):
        return self.model(theta, x)

    @property
    def evaluate_all(self):
        """The model's evaluate_all(theta, X), or None (Simulator then loops over rows)."""
        return getattr(self.model, "evaluate_all", None)


def _problem_from_cs_class(cs_class):
    """Build a CalibrationProblem from an internal CS* case-study class (synthetic mode)."""
    # Public model(theta, x); the built-in models take a third args dict, captured here
    # via a picklable adapter object (see _CaseStudyModel).
    model = _CaseStudyModel(cs_class.calc_y_fxn, cs_class.calc_y_fxn_args)

    return CalibrationProblem(
        model=model,
        param_names=list(cs_class.theta_names),
        param_bounds=list(zip(cs_class.bounds_theta_l, cs_class.bounds_theta_u)),
        x_bounds=list(zip(cs_class.bounds_x_l, cs_class.bounds_x_u)),
        true_params=cs_class.theta_ref,
        name=cs_class.name,
        indices_to_consider=list(cs_class.idcs_to_consider),
    )


def get_case_study(cs_num):
    """
    Return one of the paper's built-in case studies as a :class:`CalibrationProblem`.

    Parameters
    ----------
    cs_num : int
        Case-study identifier (1, 2, 3, 10, 11, 12, 13, 14, 15, 16, 17).

    Returns
    -------
    CalibrationProblem
        The built-in problem (synthetic-benchmark mode, with ``true_params`` set).
    """
    return _problem_from_cs_class(get_cs_class_from_val(cs_num))


def make_case_study_simulator(problem, noise_mean, noise_std, seed):
    """
    Build the internal :class:`Simulator` engine from a :class:`CalibrationProblem`.

    Parameters
    ----------
    problem : CalibrationProblem
        The problem definition (a built-in from :func:`get_case_study` or a user's own).
    noise_mean, noise_std : float, int, or None
        Noise mean / standard deviation for synthetic data generation (None std -> 5%
        of mean(Y_exp)).
    seed : int or None
        RNG seed for data generation.

    Returns
    -------
    Simulator
    """
    n_params = len(problem.param_names)
    idcs = (
        problem.indices_to_consider
        if problem.indices_to_consider is not None
        else list(range(n_params))
    )
    theta_l = [b[0] for b in problem.param_bounds]
    theta_u = [b[1] for b in problem.param_bounds]
    if problem.x_bounds is not None:
        x_l = [b[0] for b in problem.x_bounds]
        x_u = [b[1] for b in problem.x_bounds]
    elif problem.x_values is not None:
        arr = np.atleast_2d(np.asarray(problem.x_values))
        x_l = list(np.min(arr, axis=0))
        x_u = list(np.max(arr, axis=0))
    else:
        raise AssertionError("problem must define x_bounds or x_values")

    # 3-arg adapter for Simulator (picklable); problem.model already captures model state.
    calc_y_fxn = _CalcYFromModel(problem.model)

    return Simulator(
        idcs,
        problem.true_params,
        problem.param_names,
        theta_l,
        x_l,
        theta_u,
        x_u,
        noise_mean,
        noise_std,
        seed,
        calc_y_fxn,
        None,
    )


def get_cs_class_from_val(cs_num):
    """
    Returns the internal CS* class associated with the case study value.

    TODO(refactor): back-compat shim. Prefer :func:`get_case_study`, which returns a
    public :class:`CalibrationProblem`. This is kept only because a few plotter/paper
    callers read raw CS-class attributes (theta_ref, bounds_*, calc_y_fxn). Remove once
    those are migrated to CalibrationProblem.
    """
    assert cs_num in [
        1,
        2,
        3,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
    ], "cs_num must be 1,2,3,10,11,12,13,14,15,16,17 not {}".format(cs_num)
    # Get class based on cs number
    if cs_num == 1:
        cs_class = CS1()
    elif 2 <= cs_num <= 3:
        cs_class = CSMuller(cs_num)
    elif cs_num == 10:
        cs_class = CS10()
    elif cs_num == 11:
        cs_class = CS11()
    elif cs_num == 12:
        cs_class = CS12()
    elif cs_num == 13:
        cs_class = CS13()
    elif cs_num == 14:
        cs_class = CS14()
    elif cs_num == 15:
        cs_class = CS15()
    elif cs_num == 16:
        cs_class = CS16()
    elif cs_num == 17:
        cs_class = CS17()

    return cs_class


def simulator_helper_test_fxns(cs_num, noise_mean, noise_std, seed):
    """
    Sets the model for calculating y based off of the case study identifier.

    TODO(refactor): back-compat shim. Prefer the explicit two-step API
    ``make_case_study_simulator(get_case_study(cs_num), noise_mean, noise_std, seed)``,
    which this now delegates to. Remove once all call sites are migrated.

    Parameters
    ----------
    cs_num: int
        The number associated with the case study value.
    noise_mean:float, int
        The mean of the noise
    noise_std: float, int
        The standard deviation of the noise. If None, 5% of mean of Y-exp will be used
    seed: int or None
        Determines seed for randomizations. None if seed is random

    Returns
    --------
    Simulator(): Simulator
        Simulator() class object

    Raises
    ------
    AssertionError
        If any of the required parameters are missing or not of the correct type or value
    """
    # Backwards-compatible wrapper: build the case study as a CalibrationProblem and
    # then the Simulator from it (numerically identical to the old direct construction).
    return make_case_study_simulator(
        get_case_study(cs_num), noise_mean, noise_std, seed
    )


class CS1:
    """
    Class containing constants for Simple Linear Case Study

    Methods:
    --------
    __init__(): Initializes the class
    """

    def __init__(self):
        self.name = "Simple Linear"
        self.param_name_str = "t1t2"
        self.idcs_to_consider = [0, 1]
        self.theta_names = ["theta_1", "theta_2"]
        self.bounds_x_l = [-2]
        self.bounds_x_u = [2]
        self.bounds_theta_l = [-2, -2]
        self.bounds_theta_u = [2, 2]
        self.theta_ref = np.array([1.0, -1.0])
        self.calc_y_fxn = calc_cs1_polynomial
        self.calc_y_fxn_args = None


def calc_cs1_polynomial(true_model_coefficients, x, args=None):
    """
    Calculates the value of y for Simple Linear Case Study

    Parameters
    ----------
    true_model_coefficients: np.ndarray
        The array containing the true values of Theta1 and Theta2
    x: np.ndarray
        The list of xs that will be used to generate y
    args: dict, default None
        Extra arguments to pass to the function

    Returns
    --------
    y_poly: np.ndarray
        The noiseless values of y given theta_true and x

    Raises
    ------
    AssertionError
        If true_model_coefficients is not of length 2
    """

    assert len(true_model_coefficients) == 2, "true_model_coefficients must be length 2"

    y_poly = true_model_coefficients[0] * x + true_model_coefficients[1] * x**2 + x**3

    return y_poly


class CSMuller:
    """
    Class containing constants for The Muller x0 and y0 Case Studies

    Methods:
    --------
    __init__(): Initializes the class
    __set_param_str(): Sets the param_name_str based on the cs_number
    __set_idcs_to_consider(): Sets the idcs_to_consider based on the param_name_str
    __solve_pyomo_Muller_min(): Creates and Solves a Pyomo model for the minimum of the Muller potential
    """

    def __init__(self, cs_number):
        assert 2 <= cs_number <= 3
        self.cs_number = cs_number
        self.__set_param_str()
        self.name = "Muller " + self.param_name_str
        self.__set_idcs_to_consider()
        self.theta_names = ["x0_1", "x0_2", "x0_3", "x0_4"]
        self.theta_names = [
            "A_1",
            "A_2",
            "A_3",
            "A_4",
            "a_1",
            "a_2",
            "a_3",
            "a_4",
            "b_1",
            "b_2",
            "b_3",
            "b_4",
            "c_1",
            "c_2",
            "c_3",
            "c_4",
            "x0_1",
            "x0_2",
            "x0_3",
            "x0_4",
            "y0_1",
            "y0_2",
            "y0_3",
            "y0_4",
        ]
        self.bounds_x_l = [-1.5, -0.5]
        self.bounds_x_u = [1, 2]
        self.bounds_theta_l = [
            -300,
            -200,
            -250,
            5,
            -2,
            -2,
            -10,
            -2,
            -2,
            -2,
            5,
            -2,
            -20,
            -20,
            -10,
            -1,
            -2,
            -2,
            -2,
            -2,
            -2,
            -2,
            0,
            -2,
        ]
        self.bounds_theta_u = [
            -100,
            0,
            -150,
            20,
            2,
            2,
            0,
            2,
            2,
            2,
            15,
            2,
            0,
            0,
            0,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
        ]
        self.theta_ref = np.array(
            [
                -200,
                -100,
                -170,
                15,
                -1,
                -1,
                -6.5,
                0.7,
                0,
                0,
                11,
                0.6,
                -10,
                -10,
                -6.5,
                0.7,
                1,
                0,
                -0.5,
                -1,
                0,
                0.5,
                1.5,
                1,
            ]
        )
        self.calc_y_fxn = calc_muller
        self.calc_y_fxn_args = {"min muller": self.__solve_pyomo_Muller_min()}

    def __set_param_str(self):
        """Sets the param_name_str based on the cs_number."""
        if self.cs_number == 2:
            param_name_str = "x0"
        elif self.cs_number == 3:
            param_name_str = "y0"
        self.param_name_str = param_name_str

    def __set_idcs_to_consider(self):
        """Sets the idcs_to_consider based on the param_name_str."""
        # Set param_name_str
        indecies_to_consider = []
        all_param_idx = [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
        ]

        if "A" in self.param_name_str:
            indecies_to_consider += all_param_idx[0:4]
        if "x0" in self.param_name_str:
            indecies_to_consider += all_param_idx[16:20]
        if "y0" in self.param_name_str:
            indecies_to_consider += all_param_idx[20:]

        self.idcs_to_consider = indecies_to_consider

    def __solve_pyomo_Muller_min(self, verbose=False):
        """
        Creates and Solves a Pyomo model for the Muller potential
        Parameters:
        -----------
        verbose: bool
            If True, prints the solver status and termination condition. Default False

        Returns:
        --------
        model.obj(): float
            The minimum value of the Muller potential for the given sub problem defined by param_name_str
        """
        # Lazy pyomo import (only CS2/CS3 Müller need it; requires an ipopt solver on PATH)
        import pyomo.environ as pe
        ConcreteModel = pe.ConcreteModel
        Var = pe.Var
        Param = pe.Param
        Set = pe.Set
        Objective = pe.Objective
        minimize = pe.minimize
        exp = pe.exp
        SolverFactory = pe.SolverFactory

        # Create Model
        model = ConcreteModel()

        # Create a Set to represent the iterable set of variables A1-A4, b1-b4,...y01-y04
        index_set = range(1, 5)
        if "A" in self.param_name_str:
            model.A = Var(
                Set(initialize=index_set),
                initialize={1: -210, 2: -100, 3: -200, 4: 10},
                bounds={1: (-300, -100), 2: (-200, 0), 3: (-250, -150), 4: (5, 20)},
            )
        else:
            model.A = Param(
                Set(initialize=index_set), initialize={1: -200, 2: -100, 3: -170, 4: 15}
            )

        model.a = Param(
            Set(initialize=index_set), initialize={1: -1, 2: -1, 3: -6.5, 4: 0.7}
        )
        model.b = Param(
            Set(initialize=index_set), initialize={1: 0, 2: 0, 3: 11, 4: 0.6}
        )
        model.c = Param(
            Set(initialize=index_set), initialize={1: -10, 2: -10, 3: -6.5, 4: 0.7}
        )

        if "x0" in self.param_name_str:
            model.x0 = Var(
                Set(initialize=index_set),
                initialize={1: 0, 2: 0, 3: 0, 4: 0},
                bounds={1: (-2, 2), 2: (-2, 2), 3: (-2, 2), 4: (-2, 2)},
            )
        else:
            model.x0 = Param(
                Set(initialize=index_set), initialize={1: 1, 2: 0, 3: -0.5, 4: -1}
            )

        if "y0" in self.param_name_str:
            model.y0 = Var(
                Set(initialize=index_set),
                initialize={1: 0, 2: 0, 3: 1, 4: 0},
                bounds={1: (-2, 2), 2: (-2, 2), 3: (0, 2), 4: (-2, 2)},
            )
        else:
            model.y0 = Param(
                Set(initialize=index_set), initialize={1: 0, 2: 0.5, 3: 1.5, 4: 1}
            )

        model.x_index = Set(initialize=range(1, 3))
        model.x = Var(
            model.x_index,
            initialize={1: -1, 2: 0},
            bounds={1: (-1.5, 1.0), 2: (-0.5, 2)},
        )

        # Define Muller potential
        def calc_muller_pyo(model):
            # Calculate Muller Potential
            expression = sum(
                (
                    model.A[i]
                    * exp(
                        model.a[i] * (model.x[1] - model.x0[i]) ** 2
                        + model.b[i]
                        * (model.x[1] - model.x0[i])
                        * (model.x[2] - model.y0[i])
                        + model.c[i] * (model.x[2] - model.y0[i]) ** 2
                    )
                    for i in range(1, 5)
                )
            )

            return expression

        # Define objective
        model.obj = Objective(rule=calc_muller_pyo, sense=minimize)

        solver = SolverFactory("ipopt")
        solver.options["max_iter"] = 10000
        result = solver.solve(model, tee=verbose)

        if verbose:
            # Access solver status and results
            print("Solver Status:", result.solver.status)
            print("Termination Condition:", result.solver.termination_condition)
            # Print the variable value
            # Print model
            model.pprint()

        return model.obj()


def calc_muller(model_coefficients, x, args):
    """
    Caclulates the log-scaled and minimum shifted Muller Potential

    Parameters
    ----------
    model_coefficients: np.ndarray
        The array containing the values of Muller constants
    x: np.ndarray
        Values of X
    args: dict
        Extra arguments to pass to the function.

    Returns:
    --------
    y_mul_scl: float
        Value of log scaled and minimum value shifted Muller potential

    Raises
    ------
    AssertionError
        If "min muller" is not in args keys
    """
    assert "min muller" in list(args.keys())

    min_muller = args["min muller"]

    # Reshape x to matrix form
    # If array is not 2D, give it shape (len(array), 1)
    if not len(x.shape) > 1:
        x = x.reshape(-1, 1)

    assert x.shape[0] == 2, "Muller Potential x_data must be 2 dimensional"
    X1, X2 = x  # Split x into 2 parts by splitting the rows

    # Separate all model parameters into their appropriate pieces
    model_coefficients_reshape = model_coefficients.reshape(6, 4)

    # Calculate Muller Potential
    A, a, b, c, x0, y0 = model_coefficients_reshape
    term1 = a * (X1 - x0) ** 2
    term2 = b * (X1 - x0) * (X2 - y0)
    term3 = c * (X2 - y0) ** 2
    y_mul = np.sum(A * np.exp(term1 + term2 + term3))
    y_mul_scl = np.log(max(y_mul - min_muller + 1e-12, 1e-12))

    return y_mul_scl


class CS10:
    """
    Class containing constants for Large Linear Case Study
    Methods:
    --------
    __init__(): Initializes the class"""

    def __init__(self):
        self.param_name_str = "t1t2t3t4t5"
        self.name = "Large Linear"
        self.idcs_to_consider = [0, 1, 2, 3, 4]
        self.theta_names = ["theta_1", "theta_2", "theta_3", "theta_4", "theta_5"]
        self.bounds_x_l = [-2, -3]
        self.bounds_x_u = [2, 3]
        self.bounds_theta_l = [-5, -5, 0, 5, -5]
        self.bounds_theta_u = [5, 5, 1, 10, -1]
        self.theta_ref = np.array([1, -2, 0.5, 7, -3])
        self.calc_y_fxn = calc_cs8_10_polynomial
        self.calc_y_fxn_args = None


def calc_cs8_10_polynomial(true_model_coefficients, x, args=None):
    """
    Caclulates the simulated y-values for the Large Linear case study

    Parameters
    ----------
    model_coefficients: np.ndarray
        The array containing the true parameter values
    x: np.ndarray
        Values of X
    args: dict
        Extra arguments to pass to the function.

    Returns:
    --------
    y_model: float
        Value of the model

    Raises
    ------
    AssertionError
        If true_model_coefficients is not of length 5
    """
    assert len(true_model_coefficients) == 5, "true_model_coefficients must be length 5"
    t1, t2, t3, t4, t5 = true_model_coefficients

    # If array is not 2D, give it shape (len(array), 1)
    if not len(x.shape) > 1:
        x = x.reshape(-1, 1)

    assert x.shape[0] == 2, "Polynomial x_data must be 2 dimensional"
    x1, x2 = x  # Split x into 2 parts by splitting the rows

    y_model = t1 * x1 + t2 * x2 + t3 * x1 * x2 + t4 * x1**2 + t5 * x2**2

    return y_model


class CS11:
    """
    Class containing constants for BOD Curve Case Study

    Methods:
    --------
    __init__(): Initializes the class"""

    def __init__(self):
        self.theta_names = ["theta_1", "theta_2"]
        self.name = "BOD Curve"
        self.idcs_to_consider = [0, 1]
        self.bounds_x_l = [1]
        self.bounds_x_u = [7]
        self.bounds_theta_l = [10, 0]
        self.bounds_theta_u = [30, 1]
        self.theta_ref = np.array([19.143, 0.5311])
        self.calc_y_fxn = calc_cs11_BOD
        self.calc_y_fxn_args = None


def calc_cs11_BOD(true_model_coefficients, x, args=None):
    """
    Caclulates the simulated y-values for the BOD Curve case study

    Parameters
    ----------
    model_coefficients: np.ndarray
        The array containing the true parameter values
    x: np.ndarray
        Values of X
    args: dict
        Extra arguments to pass to the function.

    Returns:
    --------
    y_model: float
        Value of the model

    Raises
    ------
    AssertionError
        If true_model_coefficients is not of length 2
    """
    assert len(true_model_coefficients) == 2, "true_model_coefficients must be length 2"
    t1, t2 = true_model_coefficients
    y_model = t1 * (1 - np.exp(-t2 * x))

    return y_model


class CS12:
    """
    Class containing constants for Yield-Loss Case Study
    Methods:
    --------
    __init__(): Initializes the class"""

    def __init__(self):
        self.theta_names = ["theta_1", "theta_2", "theta_3"]
        self.name = "Yield-Loss"
        self.idcs_to_consider = [0, 1, 2]
        self.bounds_x_l = [0]
        self.bounds_x_u = [100]
        self.bounds_theta_l = [20, 5, 60]
        self.bounds_theta_u = [40, 15, 80]
        self.theta_ref = np.array([30.5, 8.25, 75.1])
        self.calc_y_fxn = calc_cs12_yield
        self.calc_y_fxn_args = None


def calc_cs12_yield(true_model_coefficients, x, args=None):
    """
    Caclulates the simulated y-values for the Yield-Loss case study

    Parameters
    ----------
    model_coefficients: np.ndarray
        The array containing the true parameter values
    x: np.ndarray
        Values of X
    args: dict
        Extra arguments to pass to the function.

    Returns:
    --------
    y_model: float
        Value of the model

    Raises
    ------
    AssertionError
        If true_model_coefficients is not of length 3
    """
    assert len(true_model_coefficients) == 3, "true_model_coefficients must be length 3"
    t1, t2, t3 = true_model_coefficients
    y_model = t1 * (1 - t2 * x / (100 * (1 + t2 * x / t3)))

    return y_model


class CS13:
    """
    Class containing constants for Log Logistic Case Study
    Methods:
    --------
    __init__(): Initializes the class"""

    def __init__(self):
        self.theta_names = ["theta_1", "theta_2", "theta_3", "theta_4"]
        self.name = "Log Logistic"
        self.idcs_to_consider = [0, 1, 2, 3]
        self.bounds_x_l = [0]
        self.bounds_x_u = [15]
        self.bounds_theta_l = [0, 3, 0.01, 0]
        self.bounds_theta_u = [1, 10, 5, 5]
        self.theta_ref = np.array([0.35, 4.54, 2.47, 1.45])
        self.calc_y_fxn = calc_cs13_logit
        self.calc_y_fxn_args = None


def calc_cs13_logit(true_model_coefficients, x, args=None):
    """
    Caclulates the simulated y-values for the Log Logistic case study

    Parameters
    ----------
    model_coefficients: np.ndarray
        The array containing the true parameter values
    x: np.ndarray
        Values of X
    args: dict
        Extra arguments to pass to the function.

    Returns:
    --------
    y_model: float
        Value of the model

    Raises
    ------
    AssertionError
        If true_model_coefficients is not of length 4
    """
    assert len(true_model_coefficients) == 4, "true_model_coefficients must be length 4"
    t1, t2, t3, t4 = true_model_coefficients
    y_model = t1 + (t2 - t1) / (1 + (x / t3) ** t4)

    return y_model


class CS14:
    """
    Class containing constants for 2D Log Logistic Case Study
    Methods:
    --------
    __init__(): Initializes the class"""

    def __init__(self):
        self.theta_names = ["theta_1", "theta_2", "theta_3", "theta_4"]
        self.name = "2D Log Logistic"
        self.idcs_to_consider = [0, 1, 2, 3]
        self.bounds_x_l = [-5, 0]
        self.bounds_x_u = [5, 15]
        self.bounds_theta_l = [0, 3, 0.01, 0]
        self.bounds_theta_u = [1, 10, 5, 5]
        self.theta_ref = np.array([0.35, 4.54, 2.47, 1.45])
        self.calc_y_fxn = calc_cs14_logit2D
        self.calc_y_fxn_args = None


def calc_cs14_logit2D(true_model_coefficients, x, args=None):
    """
    Caclulates the simulated y-values for the 2D Log Logistic case study

    Parameters
    ----------
    model_coefficients: np.ndarray
        The array containing the true parameter values
    x: np.ndarray
        Values of X
    args: dict
        Extra arguments to pass to the function.

    Returns:
    --------
    y_model: float
        Value of the model

    Raises
    ------
    AssertionError
        If true_model_coefficients is not of length 4
    """
    assert len(true_model_coefficients) == 4, "true_model_coefficients must be length 4"
    t1, t2, t3, t4 = true_model_coefficients

    # If array is not 2D, give it shape (len(array), 1)
    if not len(x.shape) > 1:
        x = x.reshape(-1, 1)

    assert x.shape[0] == 2, "Isotherm x_data must be 2 dimensional"
    x1, x2 = x  # Split x into 2 parts by splitting the rows

    t1, t2, t3, t4 = true_model_coefficients
    y_model = x1 * t1**2 + (t2 - t1 * x1) / (1 + (x2 / t3) ** t4)

    return y_model

class CS15:
    """
    Class containing constants for Simple Linear Case Study

    Methods:
    --------
    __init__(): Initializes the class
    """

    def __init__(self):
        self.name = "Simple Multimodal"
        self.param_name_str = "t1t2"
        self.idcs_to_consider = [0, 1]
        self.theta_names = ["theta_1", "theta_2"]
        self.bounds_x_l = [-2]
        self.bounds_x_u = [1.5]
        self.bounds_theta_l = [-2, -2]
        self.bounds_theta_u = [2, 2]
        self.theta_ref = np.array([-1.5, 0.5 ])
        self.calc_y_fxn = calc_cs15_polynomial
        self.calc_y_fxn_args = None


def calc_cs15_polynomial(true_model_coefficients, x, args=None):
    """
    Calculates the value of y for Simple Linear Case Study

    Parameters
    ----------
    true_model_coefficients: np.ndarray
        The array containing the true values of Theta1 and Theta2
    x: np.ndarray
        The list of xs that will be used to generate y
    args: dict, default None
        Extra arguments to pass to the function

    Returns
    --------
    y_poly: np.ndarray
        The noiseless values of y given theta_true and x

    Raises
    ------
    AssertionError
        If true_model_coefficients is not of length 2
    """

    assert len(true_model_coefficients) == 2, "true_model_coefficients must be length 2"

    y_poly = (true_model_coefficients[0] * x**3 - true_model_coefficients[1] * x**2 + 2*x - 1)**2 + (true_model_coefficients[0] - true_model_coefficients[1])**2 + (x**2 - 1)**2

    return y_poly

class CS16:
    """
    Class containing constants for the Water + Glycerol VLE Case Study

    Methods:
    --------
    __init__(): Initializes the class
    """

    def __init__(self):
        self.name = "Water-Glycerol"
        self.param_name_str = "t1t2"
        self.idcs_to_consider = [0, 1]
        self.theta_names = ["tau_{12}", "tau_{21}"]
        self.bounds_x_l = [0]
        self.bounds_x_u = [1]
        self.bounds_theta_l = [-1e3,-1e3]
        self.bounds_theta_u = [1.2e3,1.2e3]
        self.theta_ref = np.array([27.584,-195.9166])
        self.calc_y_fxn = uniquac_model
        self.calc_y_fxn_args =  {"r" :[0.92, 3.5857], #H2O + Glycerol
                                "q" :[1.4, 3.06],
                                "T" : 100+273.15, #K
                                "R" : 1.98721 , #cal/molK
                                "A": [8.07225,7.10850],
                                "B": [1730.63, 1537.78],
                                "C": [233.426, 210.39],
                                "mode": "P"
                                }

class CS17:
    """
    Class containing constants for the Acetonitrile (ACN) + Water VLE Case Study

    Methods:
    --------
    __init__(): Initializes the class
    """

    def __init__(self):
        self.name = "ACN-Water"
        self.param_name_str = "t1t2"
        self.idcs_to_consider = [0, 1]
        self.theta_names = ["tau_{12}", "tau_{21}"]
        self.bounds_x_l = [0]
        self.bounds_x_u = [1]
        self.bounds_theta_l = [-1e4,-5e3]
        self.bounds_theta_u = [1e4,1e4]
        self.theta_ref = np.array([436.4803,225.3647])
        self.calc_y_fxn = uniquac_model
        self.calc_y_fxn_args =  {"r" :[1.8701,0.92], #ACN, H2O
                                "q" :[1.7240,1.4],
                                "T" : 50+273.15, #K
                                "R" : 1.98721 , #cal/molK
                                "A": [7.33986,8.07131],
                                "B": [1482.29,1730.63],
                                "C": [250.523,233.426],
                                "mode": "y"
                                } 



def uniquac_model(unknown_params, xP, args):
    """
    Compute activity coefficients using the UNIQUAC model for a binary mixture.

    Parameters:
    unknown_params : np.array
        A vector containing the unknown interaction energy parameters Δu_ij.
    xP : np.array or float
        Mole fractions x1 (x2 is inferred).
    args : dict
        A dictionary containing necessary additional parameters:
        - "r": np.array, volume parameters for components
        - "q": np.array, surface area parameters for components
        - "R": float, gas constant
        - "T": float, temperature
        - "z": float, coordination number (default 10)
        - "A", "B", "C": Antoine equation parameters for vapor pressure
        - "mode": str, one of "P" (vapor pressure), "gamma" (activity coefficient of
          component 1), or "y" (vapor mole fraction of component 1) -- selects what is
          returned

    Returns:
    np.array or float
        Vapor pressure P, activity coefficient gamma1, or vapor mole fraction y1,
        depending on args["mode"] ("P", "gamma", or "y" respectively).
    """
    # Extract parameters
    r = np.array(args["r"])
    q = np.array(args["q"])
    z = args.get("z", 10)
    R = args["R"]
    T = args["T"]
    A, B, C = np.array(args["A"]), np.array(args["B"]), np.array(args["C"])
    mode = args["mode"]
    
    # Precompute constants (l_param is the UNIQUAC "l_i" parameter, l_i = (z/2)(r_i-q_i)-(r_i-1))
    l_param = (z / 2) * (r - q) - (r - 1)
    tau = np.exp(-unknown_params / (R * T))
    psat = 10 ** (A - B / (C + (T - 273.15)))

    # Ensure xP is at least 1D
    x1 = np.atleast_2d(xP).reshape(-1,1)
    x2 = 1 - x1
    x = np.hstack([x1, x2])

    # Initialize gamma with ones
    gamma = np.ones_like(x, dtype=float)

    # Identify valid indices where both x1 and x2 are nonzero
    valid_mask = (x1.flatten() > 0) & (x2.flatten() > 0)

    if np.any(valid_mask):
        # Apply valid_mask correctly to both dimensions
        valid_x = x[valid_mask, :]  # Shape (M, 2) where M is number of valid rows

        sum_xq = np.dot(valid_x, q)
        sum_xr = np.dot(valid_x, r)

        theta = (valid_x * q) / sum_xq[:, None]
        psi = (valid_x * r) / sum_xr[:, None]

        lngC = (
            np.log(psi / valid_x) + (z / 2) * q * np.log(theta / psi) + psi[:, ::-1] * (l_param - r * l_param[::-1] / r[::-1])
        )

        lngR = (
            -q * np.log(theta + theta[:, ::-1] * tau[::-1]) + theta[:, ::-1] * q * (
                tau[::-1] / (theta + theta[:, ::-1] * tau[::-1]) - tau / (theta[:, ::-1] + theta * tau)
            )
        )

        gamma[valid_mask, :] = np.exp(lngC + lngR)
        
    # Handle infinite dilution cases
    if np.any(~valid_mask):
        # Compute gamma at infinite dilution for both components
        gamma_inf = np.zeros(2)

        # term1 = 1- (r[0]/r[1]) +np.log(r[0]/r[1])
        # term2 = -5*q[0]*(1-(r[0]*q[1])/(r[1]*q[0]) + np.log((r[0]*q[1])/(r[1]*q[0])))

        term1 = np.log(r[0]/r[1])
        term2a = 5*np.log((q[0]*r[1])/(q[1]*r[0])) - np.log(tau[1]) + 1 -tau[0]
        term2 = q[0]*term2a
        term3 = l_param[0]-(r[0]/r[1])*l_param[1]
        gamma_inf[0] = np.exp(term1 + term2 + term3)

        term1_x2 = np.log(r[1]/r[0])
        term2a_x2 = 5*np.log((q[1]*r[0])/(q[0]*r[1])) - np.log(tau[0]) + 1 -tau[1]
        term2_x2 = q[1]*term2a_x2
        term3_x2 = l_param[1]-(r[1]/r[0])*l_param[0]
        gamma_inf[1] = np.exp(term1_x2 + term2_x2 + term3_x2)

    gamma1 = gamma[:, 0]
    gamma2 = gamma[:, 1]

    #Manually alter gamma values at infinite dilution
    if np.any(x2.flatten() == 0):
        gamma2[-1] = gamma_inf[1]
    if np.any(x1.flatten() == 0):
        gamma1[0] = gamma_inf[0]
        

    P = np.sum(x * gamma * psat, axis=1)
    y = x*gamma*psat/P[:, None]
    if mode == "P":
        var = P # Return scalar if input was scalar-like
    elif mode == "gamma":
        var = gamma1 #Return gamma1
    elif mode == "y":
        var = y[:,0] #Return y1

    # return var[0] if var.shape == (1,) else var # Return scalar if input was scalar-like
    return var