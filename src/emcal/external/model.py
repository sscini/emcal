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
        if returncode != 0:
            return "failed"
        if y is None:
            return "failed"
        if not np.all(np.isfinite(y)):
            return "failed"
        return "ok"


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

        if not isinstance(store, EvaluationStore):
            store = EvaluationStore(store)
        if executor is None:
            executor = LocalExecutor()

        self.model = model
        self.space = space
        self.store = store
        self.executor = executor
        self.failure = failure
        self.penalty = penalty
        self.rerun_failed = rerun_failed

        # Keep the grid as (n_x, x_dim), also for a 1D grid
        grid = np.asarray(model.x_grid, dtype=float)
        if grid.ndim == 1:
            grid = grid.reshape(-1, 1)
        self._x_grid = grid

        self._cache = {}  # key -> EvaluationResult, for this adapter only
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
        values = []
        for x in np.asarray(X, dtype=float):
            values.append(y[self._x_index(x)])
        return np.array(values)

    # ---- direct use ----
    def evaluate(self, theta):
        """Evaluate one parameter vector (calibration space). Returns an EvaluationResult."""
        theta = np.asarray(theta, dtype=float).reshape(-1)
        params = self.space.to_physical(theta)
        key = self._key(params)

        # Already evaluated by this adapter
        if key in self._cache:
            return _as_cached(self._cache[key], theta)

        # Finished earlier, possibly by another process sharing the store
        record = self.store.load(key)
        if self._can_reuse(record):
            result = EvaluationResult(
                theta=theta,
                params=params,
                y=record["y"],
                status=record["status"],
                wall_time=record["wall_time"],
                run_dir=self.store.run_dir(key),
                cached=True,
                message=record.get("message", ""),
            )
        else:
            result = self._run(key, theta, params)

        self._cache[key] = result
        return result

    def evaluate_many(self, thetas):
        """Evaluate several parameter vectors, running up to executor.max_workers at once."""
        thetas = np.atleast_2d(np.asarray(thetas, dtype=float))

        # Find each row's key, and the first row with each key
        keys = []
        first_row_of_key = {}
        for row in range(len(thetas)):
            key = self._key(self.space.to_physical(thetas[row]))
            keys.append(key)
            if key not in first_row_of_key:
                first_row_of_key[key] = row

        # Evaluate only the first row of each key
        rows_to_evaluate = sorted(first_row_of_key.values())
        results_by_row = self._evaluate_in_parallel(thetas, rows_to_evaluate)

        # Repeated rows reuse the result of the first row with the same key
        results = []
        for row in range(len(thetas)):
            if row in results_by_row:
                results.append(results_by_row[row])
            else:
                first_row = first_row_of_key[keys[row]]
                results.append(_as_cached(results_by_row[first_row], thetas[row]))
        return results

    # ---- internals ----
    def _key(self, params):
        """Store key for one set of physical parameter values."""
        return self.store.key(self.model.name, self.model.version, params)

    def _can_reuse(self, record):
        """True if a stored record can be returned instead of running again."""
        if record is None:
            return False
        if record["status"] == "ok":
            return True
        # A failed run is reused unless rerun_failed is set
        return not self.rerun_failed

    def _evaluate_in_parallel(self, thetas, rows):
        """Call evaluate(thetas[row]) for each row on a thread pool. Returns {row: result}."""
        n_workers = max(1, int(getattr(self.executor, "max_workers", 1)))
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {}
            for row in rows:
                futures[row] = pool.submit(self.evaluate, thetas[row])
            results_by_row = {}
            for row in rows:
                results_by_row[row] = futures[row].result()
        return results_by_row

    def _outputs(self, theta):
        result = self.evaluate(theta)
        if result.status == "ok":
            return result.y
        if self.failure == "penalty":
            return np.full(self.n_x, float(self.penalty))
        raise SimulationFailed(f"{self.model.name} {result.status}: {result.message}",
                               result.run_dir, result.status)

    def _run(self, key, theta, params):
        """Run the simulator for one parameter set and save the outcome in the store."""
        run_dir = self.store.prepare(key, params)
        self.model.write_inputs(dict(params), run_dir)
        exe = self.executor.run(self.model.command(run_dir), run_dir)
        self.n_runs += 1

        if exe.timed_out:
            timeout = getattr(self.executor, "timeout", None)
            y = None
            status = "timeout"
            problems = [f"timed out after {timeout} s"]
        else:
            y, status, problems = self._read_outputs(run_dir, exe.returncode)

        # Only successful runs keep their outputs; failed runs get a message
        if status == "ok":
            message = ""
        else:
            y = None
            message = "; ".join(problems)

        self.store.save(key, params, y, status, exe.wall_time, message)
        return EvaluationResult(
            theta=theta,
            params=params,
            y=y,
            status=status,
            wall_time=exe.wall_time,
            run_dir=run_dir,
            cached=False,
            message=message,
        )

    def _read_outputs(self, run_dir, returncode):
        """Parse and check a finished run. Returns (y or None, status, list of problems)."""
        problems = []
        if returncode != 0:
            problems.append(f"return code {returncode}")

        y = None
        try:
            y = np.asarray(self.model.parse_outputs(run_dir), dtype=float).reshape(-1)
        except Exception as e:
            problems.append(f"parse_outputs failed: {e!r}")

        if y is not None and y.size != self.n_x:
            problems.append(f"expected {self.n_x} outputs, got {y.size}")
            y = None

        status = self.model.check(run_dir, returncode, y)
        # A custom check() could accept a run without usable outputs
        if status == "ok" and y is None:
            status = "failed"
        if status != "ok" and len(problems) == 0:
            problems.append("rejected by check()")
        return y, status, problems

    def _x_index(self, x):
        """Row of x_grid that matches x."""
        x = np.asarray(x, dtype=float).reshape(-1)
        assert x.size == self._x_grid.shape[1], f"x has {x.size} values; x_grid rows have {self._x_grid.shape[1]}"
        close = np.isclose(self._x_grid, x, rtol=1e-9, atol=1e-12)  # (n_x, x_dim)
        row_matches = np.all(close, axis=1)
        hits = np.flatnonzero(row_matches)
        if hits.size == 0:
            raise KeyError(f"x={x} is not on the model's x_grid")
        return int(hits[0])


def _as_cached(result, theta):
    """Copy of an EvaluationResult for ``theta``, marked as cached."""
    return dataclasses.replace(result, theta=theta, cached=True)
