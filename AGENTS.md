# Repository Guide

## Version Control
- Apply new changes on `develop`. Check the active branch before editing; switch to `develop` when needed without discarding local changes.
- Merge `develop` into `main` only when the user explicitly instructs you to do so.
- Whenever the user requests a merge into `main` for a new Git push, first increment the patch version (the third number, e.g. `1.0.7` to `1.0.8`) on `develop`, keeping `pyproject.toml` and `src/benchmark_systems/__init__.py` synchronized, and include that version bump in the merge.

## Environment
- Use Python 3.13; both `.python-version` and `pyproject.toml` require it.
- PDM owns the lockfile and build backend. Install the locked default and development groups with `pdm sync -G dev`; include the symbolic extra with `pdm sync -G dev -G sym`. The `anim` extra is already covered by the development group's Matplotlib dependency.
- Build distributions with `pdm build`. Do not edit generated `pdm.lock`, `.pdm-build/`, or `dist/` contents directly.

## Package Boundaries
- Source lives under `src/benchmark_systems`; notebooks in `examples/` are consumers, not package entrypoints.
- `benchmark_systems` and `benchmark_systems.models` expose numerical NumPy callables. ODEs return derivatives for SciPy; DAEs return residuals for `solve-dae` and therefore accept both state and state-derivative arguments.
- `benchmark_systems.symbolic` requires CasADi. Its model functions are factories that return CasADi functions or public `DAEFunction` wrappers; it also owns `solve_ivp_casadi` and `solve_dae_casadi`. Both integrators require SciPy-style `t_eval`, which may omit `t_span[0]` for piecewise simulations.
- `benchmark_systems.animations` requires Matplotlib. Helpers live in dedicated modules and are publicly re-exported from its `__init__.py`.
- Keep public exports synchronized in the relevant package `__init__.py`. Quadrotor implementations are deliberately exported as `quadrotor_ode` and `quadrotor_dae`.

## Verification
- Documentation is written in English for end users. Keep implementation history, conversation notes and migration-session details out of the public guides.
- Install documentation tooling with `pdm sync -G dev -G docs -G sym`. Build with `pdm run sphinx-build -W --keep-going -b html docs docs/_build/html` and execute guide examples with the `doctest` builder.
- Regenerate `docs/requirements.txt` with `pdm export -G docs -G all --without-hashes -o docs/requirements.txt` after documentation dependency changes. Keep build output under ignored `docs/_build/`.
- Run the test suite with `pdm run pytest`. Tests cover numerical/symbolic parity, integrator contracts, and public exports; no lint, formatter, typecheck, or CI configuration currently exists.
- Run a focused import smoke check for numerical work: `pdm run python -c "import benchmark_systems; import benchmark_systems.models"`.
- For symbolic changes, use `pdm run python -c "import benchmark_systems.symbolic"`; for animation changes, use `pdm run python -c "import benchmark_systems.animations"`.
- Before release/version work, keep `pyproject.toml` and `src/benchmark_systems/__init__.py` synchronized; both currently report `1.1.1`.
