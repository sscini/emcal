"""Stand-in external simulator used by ToyModel. Needs only numpy.

Usage: python _toy_simulator.py RUN_DIR

Reads RUN_DIR/toy_input.json and writes RUN_DIR/toy_output.txt ("x y" per line), with
y = a*x + k*x**2 + x**3. Exits with code 3 if ``crash_above_a`` is set and a exceeds it.
"""
import json
import sys
from pathlib import Path

import numpy as np


def main(run_dir):
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "toy_input.json").read_text(encoding="utf-8"))
    a, k = cfg["a"], cfg["k"]
    crash = cfg.get("crash_above_a")
    if crash is not None and a > crash:
        print(f"toy simulator: a={a} above {crash}, crashing", file=sys.stderr)
        return 3
    x = np.asarray(cfg["x"], dtype=float)
    y = a * x + k * x**2 + x**3
    np.savetxt(run_dir / "toy_output.txt", np.column_stack([x, y]), header="x y")
    print(f"toy simulator: wrote {len(x)} outputs")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
