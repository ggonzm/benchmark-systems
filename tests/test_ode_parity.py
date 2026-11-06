import numpy as np
import pytest

from benchmark_systems import models
from benchmark_systems import symbolic


def _evaluate(function, state, control):
    # Normalize CasADi's column-shaped DM output to the numerical API's 1-D shape.
    return np.asarray(function(0.25, state, control), dtype=float).reshape(-1)


@pytest.mark.parametrize(
    ("numeric", "factory", "parameters", "state", "control"),
    [
        (
            models.pendulum,
            symbolic.pendulum,
            {"m": 1.2, "L": 0.7, "drag": 0.15},
            [0.4, -0.3],
            [0.2],
        ),
        (
            models.dc_motor,
            symbolic.dc_motor,
            {"J": 0.02, "b": 0.1, "Ke": 0.01, "Kt": 0.01, "R": 1.0, "L": 0.5},
            [0.2, -0.4, 0.7],
            [2.0],
        ),
        (
            models.cart_pendulum,
            symbolic.cart_pendulum,
            {"m": 0.2, "M": 0.8, "L": 0.5, "drag": 0.05},
            [0.1, -0.2, 2.4, 0.3],
            [1.5],
        ),
        (
            models.furuta_pendulum,
            symbolic.furuta_pendulum,
            {
                "m1": 0.3, "L1": 0.4, "J1": 0.004, "b1": 0.01,
                "m2": 0.2, "L2": 0.3, "J2": 0.003, "b2": 0.02,
                "l1": 0.2, "l2": 0.15, "K": 0.08,
            },
            [0.2, 1.0, -0.3, 0.4],
            [1.2],
        ),
        (
            models.multimass_spring,
            symbolic.multimass_spring,
            {"tau": [0.1, 0.2], "I": [0.3, 0.4], "K": [1.0, 1.2, 0.8], "d": [0.05, 0.07]},
            [0.1, -0.2, 0.3, -0.4, 0.2, -0.1],
            [0.5, -0.6],
        ),
        (
            models.batch_bioreactor,
            symbolic.batch_bioreactor,
            {"S_in": 2.0, "mu_max": 0.5, "K_m": 0.2, "K_i": 4.0, "v": 0.1, "Y_x": 0.6, "Y_p": 0.8},
            [0.3, 1.2, 0.2, 3.0],
            [0.05],
        ),
        (
            models.cstr,
            symbolic.cstr,
            {
                "K0_ab": 1.0, "Ea_ab": 2.0, "Hr_ab": -3.0,
                "K0_bc": 0.8, "Ea_bc": 2.5, "Hr_bc": -2.0,
                "K0_ad": 0.3, "Ea_ad": 1.5, "Hr_ad": -1.0,
                "rho": 1.0, "Cp": 4.0, "Cp_k": 3.0, "m_k": 2.0,
                "A": 1.5, "V": 2.0, "T_in": 20.0, "K_w": 0.4, "Ca_in": 1.0,
            },
            [0.7, 0.2, 35.0, 25.0],
            [0.4, -0.2],
        ),
        (
            models.quadrotor_ode,
            symbolic.quadrotor_ode,
            {"Ixx": 1.2, "Iyy": 1.3, "Izz": 2.1, "k": 0.9, "L": 0.25, "m": 2.0, "drag": 0.15},
            [1.0, -2.0, 3.0, 0.2, -0.3, 0.4, 0.7, -0.5, 0.9, 0.1, -0.2, 0.3],
            [4.0, 4.2, 3.8, 4.1],
        ),
    ],
)
def test_numeric_and_symbolic_odes_match(numeric, factory, parameters, state, control):
    # A shared state point catches equation drift without solver-dependent error.
    expected = numeric(0.25, state, u=control[0] if len(control) == 1 else control, **parameters)
    actual = _evaluate(factory(**parameters), state, control)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(("omega1", "omega2"), [(0.0, 2.0), (2.0, 0.0)])
def test_furuta_pendulum_satisfies_link_torque_balance(omega1, omega2):
    parameters = {
        "m1": 0.2, "L1": 0.3, "J1": 0.02, "b1": 0.0,
        "m2": 0.4, "L2": 0.5, "J2": 0.03, "b2": 0.0,
        "l1": 0.15, "l2": 0.25, "K": 0.1,
    }
    theta2 = 0.4
    state = [0.0, theta2, omega1, omega2]
    control = 0.5
    sin_theta, cos_theta = np.sin(theta2), np.cos(theta2)
    link_inertia = parameters["J2"] + parameters["m2"] * parameters["l2"]**2
    coupling = parameters["m2"] * parameters["L1"] * parameters["l2"]
    expected_torque = (
        link_inertia * omega1**2 * sin_theta * cos_theta
        - parameters["m2"] * 9.81 * parameters["l2"] * sin_theta
    )
    numeric = models.furuta_pendulum(0.25, state, u=control, **parameters)
    symbolic_result = _evaluate(symbolic.furuta_pendulum(**parameters), state, [control])

    # Check the pendulum equation before inverting the coupled inertia matrix.
    for derivative in (numeric, symbolic_result):
        torque = coupling * cos_theta * derivative[2] + link_inertia * derivative[3]
        assert torque == pytest.approx(expected_torque, abs=1e-12)


def test_johansson_backends_match_above_saturation():
    # The symbolic backend previously omitted the numerical model's upper clamp.
    parameters = {
        "A": [28.0, 32.0, 28.0, 32.0],
        "a": [0.071, 0.057, 0.071, 0.057],
        "K": [3.33, 3.35],
        "h_max": [20.0, 20.0, 20.0, 20.0],
    }
    state = [25.0, 7.0, 4.0, 3.0]
    pumps = [10.0, 15.0]
    gamma = [0.4, 0.6]

    expected = models.johansson(0.0, state, u=pumps, gamma=gamma, **parameters)
    actual = _evaluate(symbolic.johansson(**parameters), state, pumps + gamma)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
