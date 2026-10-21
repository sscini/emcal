"""Run commands as local subprocesses (any OS)."""
import os
import subprocess
import time
from pathlib import Path

from . import ExecResult, Executor


class LocalExecutor(Executor):
    """
    Runs commands with ``subprocess`` (no shell), writing stdout.txt and stderr.txt
    into the run directory.

    Parameters
    ----------
    timeout: float or None, default None
        Seconds before a run is killed and reported as timed out
    env: dict or None, default None
        Extra environment variables, added to the current environment
    max_workers: int, default 1
        Concurrent runs allowed in ExternalModelAdapter.evaluate_many
    """

    def __init__(self, timeout=None, env=None, max_workers=1):
        assert max_workers >= 1, "max_workers must be >= 1"
        self.timeout = timeout
        self.env = dict(env or {})
        self.max_workers = int(max_workers)

    def run(self, command, cwd):
        assert not isinstance(command, str), "command must be a list of arguments, not a string"
        cwd = Path(cwd)
        env = {**os.environ, **self.env} if self.env else None
        start = time.perf_counter()
        with open(cwd / "stdout.txt", "w") as out, open(cwd / "stderr.txt", "w") as err:
            try:
                proc = subprocess.run([str(c) for c in command], cwd=cwd, stdout=out, stderr=err,
                                      env=env, timeout=self.timeout, check=False)
                returncode, timed_out = proc.returncode, False
            except subprocess.TimeoutExpired:
                returncode, timed_out = None, True
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f"executable not found: {command[0]!r}. Check the path or PATH."
                ) from e
        return ExecResult(returncode, timed_out, time.perf_counter() - start)
