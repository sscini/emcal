"""Executors launch an external model's command in its run directory and wait for it."""
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ExecResult:
    """Outcome of one command. returncode is None if the command timed out."""

    returncode: object
    timed_out: bool
    wall_time: float


class Executor(ABC):
    """Runs one command in a directory. ``max_workers`` caps concurrent runs."""

    max_workers = 1

    @abstractmethod
    def run(self, command, cwd):
        """Run ``command`` (list of str) in ``cwd`` and return an ExecResult."""


from .local import LocalExecutor  # noqa: E402

__all__ = ["ExecResult", "Executor", "LocalExecutor"]
