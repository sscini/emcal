"""ExternalModel interface, and the adapter that plugs one into emcal as ``model(theta, x)``."""
import dataclasses
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .executors import LocalExecutor
from .failures import FAILURE_POLICIES, SimulationFailed
from .params import ParameterSpace
from .store import EvaluationStore


class ExternalModel(ABC):
    """
    A simulator that runs once per parameter set and returns all outputs on a fixed
    x grid. Subclasses implement the four abstract members; ``check`` is optional.
    """

    name = "external"
    version = "0"  # part of the cache key: bump it when the simulator changes

    @property
    @abstractmethod
    def x_grid(self):
        """(n_x, x_dim) array of conditions, in the order parse_outputs returns values."""

    @abstractmethod
    def write_inputs(self, params, run_dir):
        """Write the simulator's input files for ``params`` ({name: value}) into run_dir."""

    @abstractmethod
    def command(self, run_dir):
        """Command to run, as a list of str (no shell). Runs with cwd=run_dir."""

    @abstractmethod
    def parse_outputs(self, run_dir):
        """Read the run's outputs; return an (n_x,) array ordered like x_grid."""

    def check(self, run_dir, returncode, y):
        """Return "ok" or "failed". Default: zero exit code and finite outputs."""
        ok = returncode == 0 and y is not None and bool(np.all(np.isfinite(y)))
        return "ok" if ok else "failed"


@dataclass
class EvaluationResult:
    """One evaluation. ``y`` is None unless status is "ok"."""

    theta: np.ndarray
    params: dict
    y: object
    status: str  # "ok", "failed", or "timeout"
    wall_time: float
    run_dir: Path
    cached: bool
    message: str = ""


class ExternalModelAdapter:
    """
    Makes an ExternalModel usable as ``CalibrationProblem.model``.

    Each parameter set is simulated once; later calls for the same values (any x, any
    process sharing the store) reuse the stored outputs.

    Parameters
    ----------
    model: ExternalModel
        The simulator wrapper
    space: ParameterSpace
        Calibrated parameters (optimizer order) and fixed values
    store: EvaluationStore, str, or Path
        Where evaluations are stored
    executor: Executor or None, default None
        How commands are run; None means LocalExecutor()
    failure: str, default "raise"
        "raise" raises SimulationFailed; "penalty" returns ``penalty`` for every output
    penalty: float or None, default None
        Output value used when failure="penalty"
    rerun_failed: bool, default False
        Re-run parameter sets whose stored evaluation failed
    """

    def __init__(self, model, space, store, executor=None, failure="raise", penalty=None,
                 rerun_failed=False):
        assert isinstance(model, ExternalModel), "model must be an ExternalModel"
        assert isinstance(space, ParameterSpace), "space must be a ParameterSpace"
        assert failure in FAILURE_POLICIES, f"failure must be one of {FAILURE_POLICIES}"
        if failure == "penalty":
            assert penalty is not None and np.isfinite(penalty), "failure='penalty' needs a finite penalty"
        self.model = model
        self.space = space
        self.store = store if isinstance(store, EvaluationStore) else EvaluationStore(store)
        self.executor = executor if executor is not None else LocalExecutor()
        self.failure = failure
        self.penalty = penalty
        self.rerun_failed = rerun_failed
        grid = np.asarray(model.x_grid, dtype=float)
        self._x_grid = grid.reshape(-1, 1) if grid.ndim == 1 else grid
        self._cache = {}
        self.n_runs = 0  # simulator launches by this adapter

    @property
    def x_grid(self):
        return self._x_grid.copy()

    @property
    def n_x(self):
        return len(self._x_grid)

    # ---- emcal entry points (called by Simulator.evaluate_model) ----
    def __call__(self, theta, x):
        return float(self._outputs(theta)[self._x_index(x)])

    def evaluate_all(self, theta, X):
        """Outputs at every row of X, from a single simulation."""
        y = self._outputs(theta)
        return np.array([y[self._x_index(x)] for x in np.asarray(X, dtype=float)])

    # ---- direct use ----
    def evaluate(self, theta):
        """Evaluate one parameter vector (calibration space). Returns an EvaluationResult."""
        theta = np.asarray(theta, dtype=float).reshape(-1)
        params = self.space.to_physical(theta)
        key = self.store.key(self.model.name, self.model.version, params)
        if key in self._cache:
            return dataclasses.replace(self._cache[key], theta=theta, cached=True)
        record = self.store.load(key)
        if record is not None and (record["status"] == "ok" or not self.rerun_failed):
            result = EvaluationResult(theta, params, record["y"], record["status"], record["wall_time"],
                                      self.store.run_dir(key), True, record.get("message", ""))
        else:
            result = self._run(key, theta, params)
        self._cache[key] = result
        return result

    def evaluate_many(self, thetas):
        """Evaluate several parameter vectors, running up to executor.max_workers at once."""
        thetas = np.atleast_2d(np.asarray(thetas, dtype=float))
        keys = [self.store.key(self.model.name, self.model.version, self.space.to_physical(t))
                for t in thetas]
        first = {}
        for i, k in enumerate(keys):
            first.setdefault(k, i)
        unique = sorted(first.values())
        workers = max(1, int(getattr(self.executor, "max_workers", 1)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            done = dict(zip(unique, pool.map(lambda i: self.evaluate(thetas[i]), unique)))
        return [done[i] if i in done else dataclasses.replace(done[first[k]], theta=thetas[i], cached=True)
                for i, k in enumerate(keys)]

    # ---- internals ----
    def _outputs(self, theta):
        result = self.evaluate(theta)
        if result.status == "ok":
            return result.y
        if self.failure == "penalty":
            return np.full(self.n_x, float(self.penalty))
        raise SimulationFailed(f"{self.model.name} {result.status}: {result.message}",
                               result.run_dir, result.status)

    def _run(self, key, theta, params):
        run_dir = self.store.prepare(key, params)
        self.model.write_inputs(dict(params), run_dir)
        exe = self.executor.run(self.model.command(run_dir), run_dir)
        self.n_runs += 1
        y, problems = None, []
        if exe.timed_out:
            status = "timeout"
            problems.append(f"timed out after {getattr(self.executor, 'timeout', None)} s")
        else:
            if exe.returncode != 0:
                problems.append(f"return code {exe.returncode}")
            try:
                y = np.asarray(self.model.parse_outputs(run_dir), dtype=float).reshape(-1)
            except Exception as e:
                problems.append(f"parse_outputs failed: {e!r}")
            if y is not None and y.size != self.n_x:
                problems.append(f"expected {self.n_x} outputs, got {y.size}")
                y = None
            status = self.model.check(run_dir, exe.returncode, y)
            if status == "ok" and y is None:
                status = "failed"
            if status != "ok" and not problems:
                problems.append("rejected by check()")
        message = "; ".join(problems) if status != "ok" else ""
        self.store.save(key, params, y if status == "ok" else None, status, exe.wall_time, message)
        return EvaluationResult(theta, params, y if status == "ok" else None, status, exe.wall_time,
                                run_dir, False, message)

    def _x_index(self, x):
        x = np.asarray(x, dtype=float).reshape(-1)
        assert x.size == self._x_grid.shape[1], f"x has {x.size} values; x_grid rows have {self._x_grid.shape[1]}"
        hits = np.flatnonzero(np.all(np.isclose(self._x_grid, x, rtol=1e-9, atol=1e-12), axis=1))
        if hits.size == 0:
            raise KeyError(f"x={x} is not on the model's x_grid")
        return int(hits[0])
