"""emcal — Gaussian-process Bayesian optimization for nonlinear model calibration.

Public API for the core algorithm (signac-free). The analysis/plotting modules
(``analysis``, ``plotting``) are intentionally NOT imported here
because they currently depend on signac; import them explicitly if needed (and see
the ``analysis`` optional dependency group). They will be decoupled in a later phase.
"""

from .GPBO_Classes_New import (
    # enums
    MethodName,
    Kernel,
    GenMethod,
    EpSchedule,
    # core classes
    GPBOMethod,
    BOConfig,
    Simulator,
    Data,
    ExperimentalData,
    SimulationData,
    ObjectiveData,
    CandidateSet,
    GPPrediction,
    GPEmulator,
    ObjectiveGP,
    EmulatorGP,
    ExpectedImprovement,
    ExplorationBias,
    BOResults,
    GPBODriver,
)

from .case_studies import (
    CalibrationProblem,
    get_case_study,
    make_case_study_simulator,
    simulator_helper_test_fxns,
    get_cs_class_from_val,
)

# GP-diagnostics API (examine a fitted GP before running BO). Submodule import is light
# (gpflow loads lazily only when a GP is actually built).
from . import diagnostics

# External-simulator support (stdlib + numpy only).
from . import external

try:  # populated by versioningit at build/install time
    from ._version import __version__
except Exception:  # editable/source checkout without a built version file
    __version__ = "0.0.0+unknown"

__all__ = [
    "MethodName",
    "Kernel",
    "GenMethod",
    "EpSchedule",
    "GPBOMethod",
    "BOConfig",
    "Simulator",
    "Data",
    "ExperimentalData",
    "SimulationData",
    "ObjectiveData",
    "CandidateSet",
    "GPPrediction",
    "GPEmulator",
    "ObjectiveGP",
    "EmulatorGP",
    "ExpectedImprovement",
    "ExplorationBias",
    "BOResults",
    "GPBODriver",
    "CalibrationProblem",
    "get_case_study",
    "make_case_study_simulator",
    "simulator_helper_test_fxns",
    "get_cs_class_from_val",
    "diagnostics",
    "external",
    "__version__",
]
