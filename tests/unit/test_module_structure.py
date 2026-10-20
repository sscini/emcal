"""Guards the module split: every class is importable from its own submodule, and the
GPBO_Classes_New facade re-exports the SAME objects (so old import paths keep working)."""
import importlib

import pytest

# submodule -> classes that must live there after the split
LAYOUT = {
    "enums": ["MethodName", "Kernel", "GenMethod", "EpSchedule"],
    "exploration": ["ExplorationBias"],
    "config": ["BOConfig"],
    "data": ["Data"],
    "methods": ["GPBOMethod"],
    "simulator": ["Simulator"],
    "acquisition": ["ExpectedImprovement"],
    "emulators": ["GPEmulator", "ObjectiveGP", "EmulatorGP"],
    "results": ["BOResults"],
    "driver": ["GPBODriver"],
}


@pytest.mark.parametrize("submodule,classes", list(LAYOUT.items()))
def test_class_lives_in_submodule(submodule, classes):
    mod = importlib.import_module(f"emcal.{submodule}")
    for cls in classes:
        assert hasattr(mod, cls), f"{cls} missing from emcal.{submodule}"


def test_facade_reexports_same_objects():
    # GPBO_Classes_New is now a thin facade; the objects must be identical (is) to the
    # ones defined in the real submodules, so any old `from .GPBO_Classes_New import X` works.
    facade = importlib.import_module("emcal.GPBO_Classes_New")
    for submodule, classes in LAYOUT.items():
        mod = importlib.import_module(f"emcal.{submodule}")
        for cls in classes:
            assert getattr(facade, cls) is getattr(mod, cls), f"facade {cls} != {submodule}.{cls}"


def test_case_study_helpers_in_case_studies_module():
    cs = importlib.import_module("emcal.case_studies")
    for name in ["CalibrationProblem", "get_case_study", "make_case_study_simulator"]:
        assert hasattr(cs, name)


_ALLOWED_NP_RANDOM = {"default_rng", "Generator", "SeedSequence", "BitGenerator", "PCG64"}


def test_package_uses_only_numpy_generator_randomness():
    """No stdlib `random` or legacy global np.random API; only numpy Generators."""
    import ast
    import pathlib

    import emcal

    offenders = []
    for path in pathlib.Path(emcal.__file__).parent.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) and any(a.name == "random" for a in node.names):
                offenders.append(f"{path.name}:{node.lineno} import random")
            elif isinstance(node, ast.ImportFrom) and node.module == "random":
                offenders.append(f"{path.name}:{node.lineno} from random import ...")
            elif (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Attribute)
                and node.value.attr == "random"
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id in {"np", "numpy"}
                and node.attr not in _ALLOWED_NP_RANDOM
            ):
                offenders.append(f"{path.name}:{node.lineno} np.random.{node.attr}")
    assert not offenders, "non-Generator randomness found: " + ", ".join(offenders)


def test_public_api_matches_all():
    import emcal
    for name in emcal.__all__:
        assert hasattr(emcal, name), f"__all__ lists {name} but it is not importable"
