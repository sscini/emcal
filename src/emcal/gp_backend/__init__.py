"""GP backend registry.

`get_backend(name)` returns a GPBackend implementation. Backend modules (gpflow/TF or
torch/gpytorch) are imported lazily, so `import emcal` needs neither.
"""
from .base import GPBackend

_BACKENDS = ("gpflow", "gpytorch")


def get_backend(name="gpflow"):
    """
    Return a GPBackend instance for `name` (default 'gpflow').

    Parameters
    ----------
    name: str or None, default "gpflow"
        Backend identifier: "gpflow" or "gpytorch". None is treated as "gpflow".

    Returns
    -------
    backend: GPBackend
        A new instance of the requested backend

    Raises
    ------
    ValueError
        If name does not match a registered backend
    """
    key = (name or "gpflow").lower()
    if key == "gpflow":
        from .gpflow_backend import GpflowBackend

        return GpflowBackend()
    if key == "gpytorch":
        from .gpytorch_backend import GPyTorchBackend

        return GPyTorchBackend()
    raise ValueError(
        f"Unknown GP backend {name!r}. Available backends: {_BACKENDS}."
    )


__all__ = ["GPBackend", "get_backend"]
