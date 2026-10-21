"""A toy ExternalModel that runs ``_toy_simulator.py`` as a subprocess.

Useful for testing the external-model pipeline without a real simulator, and as a
template for wrapping one.
"""
import json
import sys
from pathlib import Path

import numpy as np

from .model import ExternalModel
from .params import Parameter, ParameterSpace

_SCRIPT = Path(__file__).with_name("_toy_simulator.py")

TOY_TRUTH = {"a": 1.0, "k": 0.5}


class ToyModel(ExternalModel):
    """
    y(x) = a*x + k*x**2 + x**3, computed by a separate Python process.

    Parameters
    ----------
    x_values: array-like, default np.linspace(-2, 2, 8)
        Conditions the simulator reports
    python: str or None, default None
        Python executable for the subprocess; None means sys.executable
    """

    name = "toy_cubic"
    version = "1"

    def __init__(self, x_values=None, python=None):
        if x_values is None:
            x = np.linspace(-2.0, 2.0, 8)
        else:
            x = np.asarray(x_values, dtype=float)
        self._x = x.reshape(-1, 1)

        if not python:  # None or ""
            python = sys.executable
        self.python = python

    @property
    def x_grid(self):
        return self._x

    def write_inputs(self, params, run_dir):
        """Write toy_input.json with a, k, the x values, and the optional crash threshold."""
        cfg = {
            "a": params["a"],
            "k": params["k"],
            "x": self._x[:, 0].tolist(),
        }
        if params.get("crash_above_a") is not None:
            cfg["crash_above_a"] = params["crash_above_a"]
        input_file = Path(run_dir) / "toy_input.json"
        input_file.write_text(json.dumps(cfg), encoding="utf-8")

    def command(self, run_dir):
        return [self.python, str(_SCRIPT), str(run_dir)]

    def parse_outputs(self, run_dir):
        """Read toy_output.txt ("x y" per line) and return the y column."""
        data = np.loadtxt(Path(run_dir) / "toy_output.txt", ndmin=2)
        y = data[:, 1]
        return y


def toy_space(crash_above_a=None):
    """Calibrate a (linear) and k (log10); optionally make the simulator crash above a value of a."""
    parameters = [
        Parameter("a", (-2.0, 2.0)),
        Parameter("k", (-2.0, 0.5), log10=True),
    ]
    fixed = {}
    if crash_above_a is not None:
        fixed["crash_above_a"] = crash_above_a
    return ParameterSpace(parameters, fixed=fixed)
