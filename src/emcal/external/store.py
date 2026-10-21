"""On-disk store of external-model evaluations: one folder per parameter set.

Layout::

    root/
      index.csv                 one line per finished evaluation
      evals/<key>/
        params.json             physical parameter values
        status.json             written last; its presence marks the evaluation finished
        outputs.npy             output vector (successful runs)
        ...                     model input/output files, stdout.txt, stderr.txt
"""
import csv
import hashlib
import json
import os
import shutil
import threading
import time
from pathlib import Path

import numpy as np


def _atomic_write_text(path, text):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _atomic_save_npy(path, array):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, array)
    os.replace(tmp, path)


class EvaluationStore:
    """
    Persistent cache of evaluations keyed by model name, version, and parameters.

    Parameters
    ----------
    root: str or Path
        Store directory (created if missing)
    """

    def __init__(self, root):
        self.root = Path(root)
        (self.root / "evals").mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_lock")  # locks can't be pickled
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.Lock()

    @staticmethod
    def key(model_name, model_version, params):
        """Stable 16-hex-digit key; values are rounded to 12 significant digits."""
        payload = json.dumps(
            {
                "model": str(model_name),
                "version": str(model_version),
                "params": {k: float(f"{float(v):.12g}") for k, v in params.items()},
            },
            sort_keys=True,
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]

    def run_dir(self, key):
        return self.root / "evals" / key

    def load(self, key):
        """Finished record as a dict (with "y" for successful runs), or None."""
        d = self.run_dir(key)
        status_file = d / "status.json"
        if not status_file.exists():
            return None
        record = json.loads(status_file.read_text(encoding="utf-8"))
        out_file = d / "outputs.npy"
        record["y"] = np.load(out_file) if out_file.exists() else None
        return record

    def prepare(self, key, params):
        """Fresh run directory with params.json; clears any unfinished earlier attempt."""
        d = self.run_dir(key)
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        _atomic_write_text(d / "params.json", json.dumps(params, indent=2, sort_keys=True))
        return d

    def save(self, key, params, y, status, wall_time, message=""):
        """Record a finished evaluation (successful or not)."""
        d = self.run_dir(key)
        if y is not None:
            _atomic_save_npy(d / "outputs.npy", np.asarray(y, dtype=float))
        record = {
            "key": key,
            "status": status,
            "wall_time": float(wall_time),
            "message": message,
            "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        _atomic_write_text(d / "status.json", json.dumps(record, indent=2))
        self._append_index(record, params)

    def _append_index(self, record, params):
        index = self.root / "index.csv"
        with self._lock:
            new = not index.exists()
            with open(index, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if new:
                    writer.writerow(["key", "status", "wall_time", "finished", "params"])
                writer.writerow([record["key"], record["status"], f"{record['wall_time']:.3f}",
                                 record["finished"], json.dumps(params, sort_keys=True)])
