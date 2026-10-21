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
        if env is None:
            env = {}
        self.timeout = timeout
        self.env = dict(env)
        self.max_workers = int(max_workers)

    def run(self, command, cwd):
        """Run ``command`` in ``cwd`` and wait for it. Returns an ExecResult."""
        assert not isinstance(command, str), "command must be a list of arguments, not a string"
        cwd = Path(cwd)
        args = []
        for part in command:
            args.append(str(part))
        env = self._environment()

        start = time.perf_counter()
        with open(cwd / "stdout.txt", "w") as out, open(cwd / "stderr.txt", "w") as err:
            try:
                proc = subprocess.run(args, cwd=cwd, stdout=out, stderr=err,
                                      env=env, timeout=self.timeout, check=False)
                returncode = proc.returncode
                timed_out = False
            except subprocess.TimeoutExpired:
                returncode = None
                timed_out = True
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f"executable not found: {command[0]!r}. Check the path or PATH."
                ) from e
        wall_time = time.perf_counter() - start
        return ExecResult(returncode, timed_out, wall_time)

    def _environment(self):
        """Environment for the subprocess, or None to inherit the current one unchanged."""
        if not self.env:
            return None
        env = dict(os.environ)
        env.update(self.env)
        return env
