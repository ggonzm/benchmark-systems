import casadi as ca
import numpy as np
import pytest

from benchmark_systems.symbolic import (
    double_cart_pendulum,
    solve_dae_casadi,
    solve_ivp_casadi,
)


def _time_dependent_ode():
    # This equation exposes integrators that repeatedly evaluate the first interval.
    t = ca.SX.sym("t")
    x = ca.SX.sym("x")
    u = ca.SX.sym("u")
    return ca.Function("time_dependent", [t, x, u], [t + 0 * x + 0 * u], ["t", "x", "u"], ["dx"])


def test_solve_ivp_includes_endpoint_and_advances_time():
    # Integral of dx/dt=t is x=t**2/2 and depends on absolute, not local, time.
    result = solve_ivp_casadi(
        _time_dependent_ode(), (0.0, 2.0), [0.0], t_eval=np.linspace(0.0, 2.0, 21)
    )

    assert result.success
    assert result.t.shape == (21,)
    assert result.y.shape == (1, 21)
    assert result.t[-1] == 2.0
    np.testing.assert_allclose(result.y[0], 0.5 * result.t**2, rtol=2e-3, atol=2e-3)


def test_solve_ivp_returns_exact_shifted_nonuniform_t_eval():
    # Omitting t0 prevents duplicated samples when piecewise results are stacked.
    t_eval = np.array([0.1, 0.35, 1.0, 1.7])

    result = solve_ivp_casadi(
        _time_dependent_ode(), (0.0, 2.0), [0.0], t_eval=t_eval
    )

    np.testing.assert_array_equal(result.t, t_eval)
    assert result.y.shape == (1, t_eval.size)
    np.testing.assert_allclose(result.y[0], 0.5 * t_eval**2, rtol=2e-3, atol=2e-3)


def test_solve_ivp_rejects_invalid_dimensions():
    # Dimension failures should be reported before entering CasADi internals.
    function = _time_dependent_ode()

    with pytest.raises(ValueError, match="y0 must contain 1 states"):
        solve_ivp_casadi(function, (0.0, 1.0), [0.0, 1.0], t_eval=[0.0, 1.0])
    with pytest.raises(ValueError, match="u must contain 1 inputs"):
        solve_ivp_casadi(
            function, (0.0, 1.0), [0.0], u=[1.0, 2.0], t_eval=[0.0, 1.0]
        )


@pytest.mark.parametrize(
    ("t_eval", "message"),
    [
        ([0.2, 0.1], "strictly increasing"),
        ([-0.1, 0.5], "within t_span"),
        ([0.5, 1.1], "within t_span"),
        ([0.5, np.nan], "finite"),
        ([], "non-empty one-dimensional"),
    ],
)
def test_solve_ivp_rejects_invalid_t_eval(t_eval, message):
    with pytest.raises(ValueError, match=message):
        solve_ivp_casadi(
            _time_dependent_ode(), (0.0, 1.0), [0.0], t_eval=t_eval
        )


def test_solve_dae_includes_endpoint_and_returns_small_residuals():
    # Returned derivatives must satisfy the original fully implicit equations.
    function = double_cart_pendulum(M=0.6, m1=0.2, m2=0.2, L1=0.5, L2=0.5)
    y0 = np.array([0.0, np.pi, np.pi, 0.0, 0.0, 0.0])
    yp0 = np.zeros(6)

    result = solve_dae_casadi(
        function, (0.0, 0.1), y0, yp0, u=0.0, t_eval=[0.0, 0.05, 0.1]
    )

    assert result.success
    assert result.t.shape == (3,)
    assert result.y.shape == (6, 3)
    assert result.yp.shape == (6, 3)
    assert result.t[-1] == 0.1
    for index, time in enumerate(result.t[1:], start=1):
        residual = np.asarray(
            function(time, result.y[:, index], result.yp[:, index], 0.0),
            dtype=float,
        ).reshape(-1)
        np.testing.assert_allclose(residual, 0.0, atol=1e-7)


def test_solve_dae_omits_initial_sample_when_t_eval_starts_later():
    function = double_cart_pendulum(M=0.6, m1=0.2, m2=0.2, L1=0.5, L2=0.5)
    y0 = np.array([0.0, np.pi, np.pi, 0.0, 0.0, 0.0])
    t_eval = np.array([0.05, 0.1])

    result = solve_dae_casadi(
        function, (0.0, 0.1), y0, np.zeros(6), u=0.0, t_eval=t_eval
    )

    np.testing.assert_array_equal(result.t, t_eval)
    assert result.y.shape == (6, 2)
    assert result.yp.shape == (6, 2)
