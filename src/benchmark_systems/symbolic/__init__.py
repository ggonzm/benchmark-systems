"""
benchmark_systems.symbolic
==========================

CasADi-based symbolic counterparts of all benchmark ODE/DAE models.

Exports
-------
odes
    All ODE models as CasADi Functions: pendulum, dc_motor,
    cart_pendulum, multimass_spring, johansson,
    batch_bioreactor, cstr, quadrotor.

daes
    All DAE models as _DAEFunction wrappers: double_cart_pendulum,
    multilevel_cart_pendulum, quadrotor, oil_well, neutralization.

integrate
    Convenience integrators with a scipy-compatible return type:
    solve_ivp_casadi, solve_dae_casadi.
"""

from .odes import (
    pendulum,
    dc_motor,
    cart_pendulum,
    multimass_spring,
    johansson,
    batch_bioreactor,
    cstr,
    quadrotor as quadrotor_ode,
)
from .daes import (
    double_cart_pendulum,
    multilevel_cart_pendulum,
    quadrotor as quadrotor_dae,
    oil_well,
    neutralization,
)
from .integrate import solve_ivp_casadi, solve_dae_casadi

__all__ = [
    "pendulum",
    "dc_motor",
    "cart_pendulum",
    "multimass_spring",
    "johansson",
    "batch_bioreactor",
    "cstr",
    "quadrotor_ode",
    "double_cart_pendulum",
    "multilevel_cart_pendulum",
    "quadrotor_dae",
    "oil_well",
    "neutralization",
    "solve_ivp_casadi",
    "solve_dae_casadi",
]
