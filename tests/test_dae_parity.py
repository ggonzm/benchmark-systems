import numpy as np
import pytest

from benchmark_systems import models
from benchmark_systems import symbolic


@pytest.mark.parametrize(
    ("numeric", "factory", "parameters", "state", "derivative", "control"),
    [
        (
            models.double_cart_pendulum,
            symbolic.double_cart_pendulum,
            {"M": 0.7, "m1": 0.2, "m2": 0.15, "L1": 0.5, "L2": 0.4},
            [0.1, 2.8, 3.0, -0.4, 0.2, -0.3],
            [-0.4, 0.2, -0.3, 0.5, -0.6, 0.7],
            1.1,
        ),
        (
            models.quadrotor_dae,
            symbolic.quadrotor_dae,
            {"Ixx": 1.2, "Iyy": 1.3, "Izz": 2.1, "k": 0.9, "L": 0.25, "m": 2.0, "drag": 0.15},
            [1.0, -2.0, 3.0, 0.2, -0.3, 0.4, 0.7, -0.5, 0.9, 0.1, -0.2, 0.3],
            [0.7, -0.5, 0.9, 0.1, -0.2, 0.3, 0.4, -0.6, 0.8, -0.1, 0.2, -0.3],
            [4.0, 4.2, 3.8, 4.1],
        ),
    ],
)
def test_numeric_and_symbolic_dae_residuals_match(
    numeric, factory, parameters, state, derivative, control
):
    # Compare residual equations directly, independently of a DAE solver.
    expected = numeric(0.2, state, derivative, u=control, **parameters)
    function = factory(**parameters)
    actual = np.asarray(function(0.2, state, derivative, control), dtype=float).reshape(-1)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    ("body_rates", "control"),
    [
        pytest.param([0.0, 0.0, 0.0], [1.0, 2.0, 3.0, 4.0], id="body-torque"),
        pytest.param([0.1, 0.2, 0.3], [1.0, 1.0, 1.0, 1.0], id="torque-free-rotation"),
    ],
)
def test_quadrotor_dae_angular_dynamics_match_ode(body_rates, control):
    parameters = {
        "Ixx": 0.4, "Iyy": 0.5, "Izz": 0.7, "k": 1.0,
        "L": 0.2, "m": 2.0, "drag": 0.1,
    }
    angles = [0.4, 0.3, 0.2]
    phi, theta, _ = angles
    sp, cp = np.sin(phi), np.cos(phi)
    st, ct = np.sin(theta), np.cos(theta)
    rate_map = np.array([[1, 0, -st], [0, cp, ct * sp], [0, -sp, ct * cp]])

    ode_state = np.r_[np.zeros(3), angles, np.zeros(3), body_rates]
    ode_derivative = models.quadrotor_ode(0.0, ode_state, u=control, **parameters)
    euler_rates = ode_derivative[3:6]
    roll_rate, pitch_rate, _ = euler_rates
    rate_map_dot = np.array([
        [0, 0, -ct * pitch_rate],
        [0, -sp * roll_rate, ct * cp * roll_rate - st * sp * pitch_rate],
        [0, -cp * roll_rate, -ct * sp * roll_rate - st * cp * pitch_rate],
    ])
    # Differentiate body_rates = rate_map @ euler_rates.
    euler_acceleration = np.linalg.solve(
        rate_map, ode_derivative[9:12] - rate_map_dot @ euler_rates
    )
    dae_state = np.r_[np.zeros(3), angles, np.zeros(3), euler_rates]
    # Only the angular residual is checked, so linear accelerations are unused.
    dae_derivative = np.r_[np.zeros(3), euler_rates, np.zeros(3), euler_acceleration]

    numeric = models.quadrotor_dae(0.0, dae_state, dae_derivative, u=control, **parameters)
    symbolic_residual = symbolic.quadrotor_dae(**parameters)(
        0.0, dae_state, dae_derivative, control
    )
    np.testing.assert_allclose(numeric[9:12], 0.0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(symbolic_residual)[9:12], 0.0, atol=1e-12)


def test_multilevel_cart_pendulum_friction_matches_symbolic_backend():
    # Nonzero velocities exercise every element of the friction vector.
    parameters = {
        "M": 0.7,
        "m": [0.2, 0.15],
        "L": [0.5, 0.4],
        "deltas": [0.1, 0.2, 0.3],
    }
    state = [0.1, 2.8, 3.0, -0.4, 0.2, -0.3]
    derivative = [-0.4, 0.2, -0.3, 0.5, -0.6, 0.7]
    control = 1.1

    expected = models.multilevel_cart_pendulum(
        0.2, state, derivative, u=control, **parameters
    )
    function = symbolic.multilevel_cart_pendulum(**parameters)
    actual = np.asarray(function(0.2, state, derivative, control), dtype=float).reshape(-1)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("deltas", ([0.1], [0.1, 0.2]))
def test_multilevel_cart_pendulum_rejects_wrong_friction_count(deltas):
    # Both factories must reject a coefficient vector that cannot map to the DOFs.
    with pytest.raises(ValueError, match="deltas"):
        models.multilevel_cart_pendulum(
            0.0,
            np.zeros(6),
            np.zeros(6),
            M=1.0,
            m=[0.2, 0.3],
            L=[0.4, 0.5],
            deltas=deltas,
        )

    with pytest.raises(ValueError, match="deltas"):
        symbolic.multilevel_cart_pendulum(
            M=1.0, m=[0.2, 0.3], L=[0.4, 0.5], deltas=deltas
        )
