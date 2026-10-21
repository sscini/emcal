"""Failure handling for external simulations."""

FAILURE_POLICIES = ("raise", "penalty")


class SimulationFailed(RuntimeError):
    """An external simulation failed, timed out, or produced unusable output."""

    def __init__(self, message, run_dir=None, status=None):
        super().__init__(message if run_dir is None else f"{message} (run dir: {run_dir})")
        self.run_dir = run_dir
        self.status = status
