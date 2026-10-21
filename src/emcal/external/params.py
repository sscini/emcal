"""Parameter definitions for external models: names, bounds, log10 scaling, fixed values."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Parameter:
    """
    A calibrated parameter.

    Parameters
    ----------
    name: str
        Name passed to the model's input writer
    bounds: tuple(float, float)
        (low, high) in calibration space, i.e. log10 units when log10=True
    log10: bool, default False
        If True, the optimizer works on log10(value)
    """

    name: str
    bounds: tuple
    log10: bool = False

    def __post_init__(self):
        lo, hi = self.bounds
        assert lo < hi, f"{self.name}: bounds must be (low, high) with low < high, got {self.bounds}"


class ParameterSpace:
    """
    Maps between the optimizer's vector (calibration space) and the model's named
    physical values.

    Parameters
    ----------
    parameters: list(Parameter)
        Calibrated parameters, in optimizer order
    fixed: dict or None, default None
        Physical values for parameters that are not calibrated
    """

    def __init__(self, parameters, fixed=None):
        if fixed is None:
            fixed = {}
        self.parameters = list(parameters)
        self.fixed = dict(fixed)

        names = self.names
        assert len(set(names)) == len(names), "parameter names must be unique"
        overlap = set(names) & set(self.fixed)
        assert not overlap, f"parameters are both calibrated and fixed: {sorted(overlap)}"

    @property
    def names(self):
        return [p.name for p in self.parameters]

    @property
    def bounds(self):
        """[(low, high), ...] in calibration space, ready for CalibrationProblem."""
        return [tuple(p.bounds) for p in self.parameters]

    @property
    def dim(self):
        return len(self.parameters)

    def to_physical(self, theta):
        """Optimizer vector -> {name: physical value}, including the fixed values."""
        theta = np.asarray(theta, dtype=float).reshape(-1)
        assert theta.size == self.dim, f"expected {self.dim} values, got {theta.size}"
        values = dict(self.fixed)
        for i, p in enumerate(self.parameters):
            if p.log10:
                values[p.name] = float(10.0**theta[i])
            else:
                values[p.name] = float(theta[i])
        return values

    def to_calibration(self, physical):
        """{name: physical value} -> optimizer vector."""
        theta = []
        for p in self.parameters:
            if p.log10:
                theta.append(np.log10(physical[p.name]))
            else:
                theta.append(physical[p.name])
        return np.array(theta, dtype=float)
