# Repository Guide

## Environment
- Use Python 3.13; both `.python-version` and `pyproject.toml` require it.
- PDM owns the lockfile and build backend. Install the locked default and development groups with `pdm sync -G dev`; include the symbolic extra with `pdm sync -G dev -G sym`. The `anim` extra is already covered by the development group's Matplotlib dependency.
- Build distributions with `pdm build`. Do not edit generated `pdm.lock`, `.pdm-build/`, or `dist/` contents directly.

## Package Boundaries
- Source lives under `src/benchmark_systems`; notebooks in `examples/` are consumers, not package entrypoints.
- `benchmark_systems` and `benchmark_systems.models` expose numerical NumPy callables. ODEs return derivatives for SciPy; DAEs return residuals for `solve-dae` and therefore accept both state and state-derivative arguments.
- `benchmark_systems.symbolic` requires CasADi. Its model functions are factories that return CasADi functions or public `DAEFunction` wrappers; it also owns `solve_ivp_casadi` and `solve_dae_casadi`. Both integrators require SciPy-style `t_eval`, which may omit `t_span[0]` for piecewise simulations.
- `benchmark_systems.animations` requires Matplotlib and contains all animation helpers in its `__init__.py`.
- Keep public exports synchronized in the relevant package `__init__.py`. Quadrotor implementations are deliberately exported as `quadrotor_ode` and `quadrotor_dae`.

## Verification
- Run the test suite with `pdm run pytest`. Tests cover numerical/symbolic parity, integrator contracts, and public exports; no lint, formatter, typecheck, or CI configuration currently exists.
- Run a focused import smoke check for numerical work: `pdm run python -c "import benchmark_systems; import benchmark_systems.models"`.
- For symbolic changes, use `pdm run python -c "import benchmark_systems.symbolic"`; for animation changes, use `pdm run python -c "import benchmark_systems.animations"`.
- Before release/version work, keep `pyproject.toml` and `src/benchmark_systems/__init__.py` synchronized; both currently report `1.0.6`.
