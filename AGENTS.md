# emcal: notes for coding agents

emcal is the Dowling lab's Gaussian-process Bayesian optimization package for model
calibration. Stephen Cini (scini@nd.edu) maintains it and is extending it to drive
external simulators (REM's Surface Evolver model, then a CUDA Drosophila wing-disc model).

## Layout

- `src/emcal/`: core (driver, simulator, emulators, acquisition, GP backends in `gp_backend/`)
- `src/emcal/external/`: external simulators: `ExternalModel`, `ExternalModelAdapter`,
  `EvaluationStore`, executors, `profiles.py` (curve resampling)
- `tests/unit/`: pytest; `examples/`; `devtools/conda-envs/`: environment files

## Environment

```bash
conda env create -f devtools/conda-envs/emcal-py313.yml
conda activate emcal-py313-fromrepo
pip install -e . --no-deps
```

- **macOS on Apple silicon:** use `devtools/conda-envs/emcal-py313-mac.yml` instead, then
  run `pip install --no-deps gpflow==2.9.2` before `pip install -e . --no-deps`. (gpflow
  2.9.2 asks for `tensorflow-macos` there, which has no Python 3.13 build.)
- The env file also installs the REM Surface Evolver packages (spatial-efd,
  similaritymeasures, openpyxl). Full lock: `devtools/conda-envs/emcal-py313-pip-freeze.txt`.
- Stephen's Windows laptop has the same env under its older name, `emcal-py313-pc`.
- Activate the env before running anything: the Müller case studies need `ipopt` on PATH.
  On Windows without activation, put `<env>\Library\bin` on PATH.
- Python packages come from pip only; conda supplies just Python and ipopt.
- Validated on Windows 11. macOS and Linux are untested with this env file.

## Tests

```bash
pytest -m "not slow"     # fast tier
pytest                   # everything, including end-to-end GP training
```

Known: `test_end_to_end`'s golden value is from an older TF/macOS stack and fails on
Python 3.13; Tasmanian (method C2) tests are skipped.

## Planning notes

The plan and progress notes are outside the repo, in OneDrive:

- Windows: `C:\Users\steph\OneDrive - nd.edu\Dowling_Lab\AI_coding_docs\emcal\_drosophila_integration`
- macOS: the same `Dowling_Lab/AI_coding_docs/emcal/_drosophila_integration` folder under
  `~/Library/CloudStorage/` (OneDrive)

Read `10_status_and_next_steps.md` and `06_questions_and_log.md` there before starting.
Record decisions and results there (dated entries in 06; new numbered notes listed in
`00_README.md`). `docs/_drosophila_integration/` in this repo is the old, gitignored
location; don't write there.

## Code style

- **Plain, step-by-step code.** Prefer explicit loops and named intermediate variables over
  compact idioms (nested comprehensions with conditions, `dict(zip(map(...)))`,
  `setdefault` tricks, dense one-line conditionals). Stephen maintains this code.
- **Short comments and docstrings.** One line where possible; longer reasoning goes in the
  notes or the commit message.
- Numpy-style docstrings as in the existing code (`name: type` under `Parameters`).
- Argument checks use `assert` with a clear message.
- Randomness: numpy Generators only (a test scans the package for stdlib `random`).

## Git

- Working branch: `ucr-collab-integrate-claude`; base branch: `main`.
- Don't commit, push, or switch branches unless asked. Commit only files you changed.
- **Unsupervised runs** (no one watching): before changing anything, note the starting
  commit (`git rev-parse HEAD`) and create a new branch from it, e.g.
  `claude/2026-10-24-<topic>`. Commit your work there in small steps. Don't merge it into
  the working branch, and don't push unless the brief says to. Record the branch name and
  starting hash in 06, so the before and after can be compared and merged selectively.
