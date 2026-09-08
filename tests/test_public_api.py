from pathlib import Path
import tomllib

import benchmark_systems
import benchmark_systems.models as models
import benchmark_systems.symbolic as symbolic


def test_runtime_version_matches_distribution_metadata():
    # Prevent the runtime/distribution mismatch that previously shipped in source.
    pyproject = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())

    assert benchmark_systems.__version__ == pyproject["project"]["version"]


def test_numeric_and_symbolic_model_exports_remain_in_sync():
    # Symbolic exports contain the shared models plus their wrapper and integrators.
    symbolic_support = {"DAEFunction", "solve_ivp_casadi", "solve_dae_casadi"}

    assert set(models.__all__) == set(symbolic.__all__) - symbolic_support


def test_dae_wrapper_exposes_dimensions():
    # Integrators and callers should not need private CasADi attributes for shapes.
    function = symbolic.double_cart_pendulum(
        M=0.6, m1=0.2, m2=0.2, L1=0.5, L2=0.5
    )

    assert isinstance(function, symbolic.DAEFunction)
    assert function.state_size == 6
    assert function.input_size == 1
    assert function.n_diff == 6
