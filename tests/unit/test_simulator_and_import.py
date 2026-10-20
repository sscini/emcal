"""Fast tests for Simulator data generation and the lazy-import guarantee.

No GP training here, so these run without the gpflow/Tasmanian extras.
"""
import subprocess
import sys

import numpy as np

from emcal import get_case_study, make_case_study_simulator, GenMethod


def test_experimental_data_shapes_and_determinism():
    problem = get_case_study(1)                 # 2 params, 1-D x
    sim = make_case_study_simulator(problem, 0, None, 1)
    exp1 = sim.generate_experimental_data(5, GenMethod.MESHGRID, None, 0.01)
    assert exp1.n_x == 5
    assert exp1.x_dim == 1
    # Same seed -> identical experimental data (data generation is deterministic).
    sim2 = make_case_study_simulator(problem, 0, None, 1)
    exp2 = sim2.generate_experimental_data(5, GenMethod.MESHGRID, None, 0.01)
    assert np.allclose(np.asarray(exp1.y_vals), np.asarray(exp2.y_vals))


def test_unseeded_simulator_draws_an_int_seed_in_range():
    problem = get_case_study(1)
    sim = make_case_study_simulator(problem, 0, None, None)
    assert type(sim.sim_x_seed) is int
    assert 1 <= sim.sim_x_seed <= 10**8
    assert sim.sim_seed is None and sim.val_seed is None and sim.start_seed is None


def test_unseeded_simulator_ignores_stdlib_random_state():
    # Re-seeding stdlib random must not make unseeded Simulators repeat.
    import random

    problem = get_case_study(1)
    seeds = set()
    for _ in range(3):
        random.seed(0)
        seeds.add(make_case_study_simulator(problem, 0, None, None).sim_x_seed)
    assert len(seeds) > 1


def test_simulation_data_dimensions():
    problem = get_case_study(1)
    sim = make_case_study_simulator(problem, 0, None, 1)
    sim.generate_experimental_data(5, GenMethod.MESHGRID, None, 0.01)
    n = len(sim.indices_to_consider)               # 2 for CS1
    sim_data = sim.generate_simulation_data(
        10 * n, 5, GenMethod.LHS, GenMethod.MESHGRID, 1.0, 1, False, None, with_noise=False
    )
    assert sim_data.theta_dim == n


def test_import_needs_no_heavy_deps():
    # `import emcal` must not pull in gpflow / tensorflow / Tasmanian
    # (they load lazily only when a GP model is built). Run in a fresh interpreter.
    code = (
        "import sys, emcal; "
        "heavy = [m for m in ('gpflow','tensorflow','Tasmanian') if m in sys.modules]; "
        "print('LOADED:' + ','.join(heavy))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "LOADED:" in out.stdout
    loaded = out.stdout.split("LOADED:")[1].strip()
    assert loaded == "", f"import emcal eagerly loaded heavy deps: {loaded}"
