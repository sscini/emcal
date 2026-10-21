"""Calibrate external simulators (separate programs) with emcal.

Wrap a simulator as an ExternalModel, then pass an ExternalModelAdapter as
``CalibrationProblem.model``. Only the standard library and numpy are needed.
"""
from .executors import ExecResult, Executor, LocalExecutor
from .failures import SimulationFailed
from .model import EvaluationResult, ExternalModel, ExternalModelAdapter
from .params import Parameter, ParameterSpace
from .store import EvaluationStore

__all__ = [
    "EvaluationResult",
    "EvaluationStore",
    "ExecResult",
    "Executor",
    "ExternalModel",
    "ExternalModelAdapter",
    "LocalExecutor",
    "Parameter",
    "ParameterSpace",
    "SimulationFailed",
]
