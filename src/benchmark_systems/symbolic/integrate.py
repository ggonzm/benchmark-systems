"""
CasADi-based integrators with an interface compatible with scipy's
``solve_ivp`` and ``scipy_dae``'s ``solve_dae``.
 
CasADi 3.6.x and the semi-explicit DAE reformulation
------------------------------------------------------
CasADi 3.6.x IDAS does not support the ``'xdot'`` key in the DAE dict.
Fully-implicit DAEs  G(t, x, ẋ, u) = 0  are handled by splitting the
state vector into differential and algebraic parts according to the
``n_diff`` attribute of the DAEFunction:
 
    x_diff  – first n_diff components  (have a true time derivative)
    z_alg   – remaining components are purely algebraic (no derivative)
              PLUS the derivatives of x_diff themselves
 
Concretely, for a DAE with n_diff differential states and n_alg algebraic:
 
    x  (CasADi 'x')  = x_diff          shape (n_diff,)
    z  (CasADi 'z')  = [xdot_diff ; x_alg]   shape (n_diff + n_alg,)
    ode               = z[:n_diff]      dx_diff/dt = z_diff
    alg               = G(t, [x_diff; z_alg], z_diff, u) = 0
 
For fully differential DAEs (n_alg = 0, all components differential) this
reduces to: z = xdot, ode = z, alg = G(t, x, z, u).
"""
 
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence
import numpy as np
import casadi as ca

from .daes import DAEFunction
 
 
# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
 
@dataclass
class IntegrationResult:
    """
    Lightweight result object compatible with scipy's OdeResult.
 
    Attributes
    ----------
    t   : np.ndarray  shape (n_times,)
    y   : np.ndarray  shape (n_states, n_times)  – full state [x_diff ; x_alg]
    yp  : np.ndarray  shape (n_states, n_times)  – ẋ trajectory; zeros for ODEs
    success : bool
    message : str
    """
    t:       np.ndarray
    y:       np.ndarray
    yp:      np.ndarray = field(default_factory=lambda: np.array([]))
    success: bool = True
    message: str  = "Integration successful"
 
 
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
 
def _to_dm(val) -> ca.DM:
    if isinstance(val, (int, float)):
        return ca.DM([val])
    return ca.DM(val)


def _validate_t_span(t_span: tuple[float, float]) -> tuple[float, float]:
    if len(t_span) != 2:
        raise ValueError("t_span must contain exactly two values")
    t0, tf = map(float, t_span)
    if not np.isfinite([t0, tf]).all():
        raise ValueError("t_span values must be finite")
    if tf <= t0:
        raise ValueError("t_span must satisfy tf > t0")
    return t0, tf


def _evaluation_grid(
    t_span: tuple[float, float], t_eval: Sequence[float]
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return the integration grid, requested output grid, and output offset."""
    t0, tf = _validate_t_span(t_span)
    output_times = np.asarray(t_eval, dtype=float)
    if output_times.ndim != 1 or output_times.size == 0:
        raise ValueError("t_eval must be a non-empty one-dimensional sequence")
    if not np.isfinite(output_times).all():
        raise ValueError("t_eval values must be finite")
    if np.any(np.diff(output_times) <= 0):
        raise ValueError("t_eval values must be strictly increasing")
    if output_times[0] < t0 or output_times[-1] > tf:
        raise ValueError("t_eval values must lie within t_span")

    if output_times[0] == t0:
        return output_times, output_times, 0

    # CasADi still needs to advance from y(t0), although SciPy-style t_eval may
    # request that the initial state be omitted from the returned solution.
    integration_times = np.concatenate(([t0], output_times))
    return integration_times, output_times, 1
 
 
# ---------------------------------------------------------------------------
# ODE integrator
# ---------------------------------------------------------------------------
 
def solve_ivp_casadi(f: ca.Function,
                     t_span: tuple[float, float],
                     y0: Sequence[float],
                     *,
                     t_eval: Sequence[float],
                     u: float | Sequence[float] = 0.0,
                     integrator: str = 'rk',
                     opts: dict | None = None) -> IntegrationResult:
    """
    Integrate an ODE  ẋ = f(t, x, u)  using a CasADi integrator.
 
    Parameters
    ----------
    f : ca.Function
        CasADi Function  f(t, x, u) → dx  from the symbolic ODE constructors.
    t_span : (t0, tf)
    y0 : array-like
    u : scalar or array-like.  Constant control input.  Default 0.
    t_eval : array-like.
        Times at which to return the solution. Values must be strictly
        increasing and lie within ``t_span``. Unlike the internal integration
        grid, this sequence does not need to include ``t_span[0]``.
    integrator : str.  ``'rk'`` (RK4), ``'cvodes'``, or ``'collocation'``.
    opts : dict, optional.  Extra options for ``ca.integrator``.
    """
    integration_times, output_times, output_start = _evaluation_grid(
        t_span, t_eval
    )
    y0_arr = np.asarray(y0, dtype=float).reshape(-1)
    if not np.isfinite(y0_arr).all():
        raise ValueError("y0 values must be finite")
    nx     = y0_arr.size
    expected_nx = f.size1_in('x')
    if nx != expected_nx:
        raise ValueError(f"y0 must contain {expected_nx} states, got {nx}")
    u_dm   = _to_dm(u)
    nu     = u_dm.numel()
    expected_nu = f.size1_in('u')
    if nu != expected_nu:
        raise ValueError(f"u must contain {expected_nu} inputs, got {nu}")
 
    t_sym = ca.SX.sym('t')
    x_sym = ca.SX.sym('x', nx)
    # Every cached integrator runs on local time [0, dt]. The extra parameter
    # shifts local time back to the absolute interval expected by f(t, x, u).
    p_sym = ca.SX.sym('p', nu + 1)
    u_sym = p_sym[:nu]
    time_offset = p_sym[nu]
    dx    = f(t_sym + time_offset, x_sym, u_sym)

    ode_dict = {'t': t_sym, 'x': x_sym, 'p': p_sym, 'ode': dx}
 
    default_opts: dict = {}
    if integrator == 'rk':
        default_opts['number_of_finite_elements'] = 4
    if opts:
        default_opts.update(opts)
 
    states = np.zeros((nx, integration_times.size))
    states[:, 0] = y0_arr
    x_k = ca.DM(y0_arr.reshape(-1, 1))
    integrators: dict[float, ca.Function] = {}

    for k, (time_start, time_end) in enumerate(
        zip(integration_times[:-1], integration_times[1:])
    ):
        dt = float(time_end - time_start)
        # Building a CasADi integrator is expensive. Reuse one for equal step
        # sizes, tolerating the round-off produced by np.linspace/np.diff.
        integ = next(
            (
                candidate
                for step, candidate in integrators.items()
                if np.isclose(dt, step, rtol=1e-12, atol=0.0)
            ),
            None,
        )
        if integ is None:
            integ = ca.integrator(
                f'F_{len(integrators)}', integrator, ode_dict, 0.0, [dt], default_opts
            )
            integrators[dt] = integ
        parameters = ca.vertcat(u_dm, float(time_start))
        res = integ(x0=x_k, p=parameters)
        x_k = res['xf']
        states[:, k + 1] = np.asarray(x_k).flatten()

    # output_start is one only when t0 was inserted as an internal-only node.
    output_states = states[:, output_start:]
    return IntegrationResult(
        t=output_times, y=output_states, yp=np.zeros_like(output_states)
    )
 
 
# ---------------------------------------------------------------------------
# DAE integrator
# ---------------------------------------------------------------------------
 
def solve_dae_casadi(F: DAEFunction,
                     t_span: tuple[float, float],
                     y0: Sequence[float],
                     yp0: Sequence[float],
                     *,
                     t_eval: Sequence[float],
                     u: float | Sequence[float] = 0.0,
                     opts: dict | None = None) -> IntegrationResult:
    """
    Integrate a fully-implicit DAE  G(t, x, ẋ, u) = 0  using CasADi / IDAS.
 
    Parameters
    ----------
    F : DAEFunction
        Object returned by the symbolic DAE constructors.  Its ``n_diff``
        attribute controls the differential/algebraic splitting.
    t_span : (t0, tf)
    y0 : array-like.  Initial state vector (full: differential + algebraic).
    yp0 : array-like.
        Initial derivatives for the differential part; algebraic initial
        guess for the algebraic part.
    u : scalar or array-like.  Constant control input.  Default 0.
    t_eval : array-like.
        Times at which to return the solution. Values must be strictly
        increasing and lie within ``t_span``. This sequence may omit
        ``t_span[0]`` to simplify concatenating piecewise simulations.
    opts : dict, optional.  Extra IDAS options.
 
    Returns
    -------
    IntegrationResult
        ``.yp`` is the derivative trajectory (zeros for algebraic components).
 
    Notes
    -----
    Semi-explicit reformulation used here (CasADi 3.6.x compatible):
 
      Let  nd = F.n_diff,  na = nx - nd.
 
      CasADi 'x'   = x_diff       (first nd components)
      CasADi 'z'   = [xdot_diff ; x_alg]   (nd + na components)
      ode           = z[:nd]       so  d(x_diff)/dt = z_diff = xdot_diff
      alg           = G(t, [x_diff ; z_alg], z_diff, u) = 0   (nx equations)
 
    y0[nd:]   is the initial guess for the algebraic states.
    yp0[:nd] is the initial guess for xdot_diff.
    IDAS will refine both z_diff and z_alg to satisfy consistency at t0.
    """
    integration_times, output_times, output_start = _evaluation_grid(
        t_span, t_eval
    )
    y0_arr = np.asarray(y0, dtype=float).reshape(-1)
    ydot0_arr = np.asarray(yp0, dtype=float).reshape(-1)
    if y0_arr.size != ydot0_arr.size:
        raise ValueError("y0 and yp0 must have the same length")
    if not np.isfinite(y0_arr).all() or not np.isfinite(ydot0_arr).all():
        raise ValueError("y0 and yp0 values must be finite")
    nx      = y0_arr.size
    expected_nx = F.fn.size1_in('x')
    if nx != expected_nx:
        raise ValueError(f"y0 must contain {expected_nx} states, got {nx}")
    nd      = F.n_diff           # number of differential states
    if not 0 < nd <= nx:
        raise ValueError("F.n_diff must be between 1 and the number of states")
    na      = nx - nd            # number of algebraic states
    u_dm    = _to_dm(u)
    nu      = u_dm.numel()
    expected_nu = F.fn.size1_in('u')
    if nu != expected_nu:
        raise ValueError(f"u must contain {expected_nu} inputs, got {nu}")
 
    t_sym  = ca.SX.sym('t')
    xd_sym = ca.SX.sym('xd', nd)                 # differential states
    # z = [xdot_diff (nd) ; x_alg (na)]
    z_sym  = ca.SX.sym('z',  nd + na)
    # As in the ODE solver, append the absolute interval start to the constant
    # controls so one local-time integrator can be reused over the whole grid.
    p_sym  = ca.SX.sym('p', nu + 1)
    u_sym  = p_sym[:nu]
    time_offset = p_sym[nu]
 
    xdot_diff = z_sym[:nd]
    x_alg     = z_sym[nd:]
 
    # Reconstruct the full state and xdot vectors for G
    x_full    = ca.vertcat(xd_sym, x_alg)          # [x_diff ; x_alg]
    xdot_full = ca.vertcat(xdot_diff, ca.SX.zeros(na))  # alg derivatives = 0
    
    G_sym = F.fn(t_sym + time_offset, x_full, xdot_full, u_sym)
 
    dae_dict = {
        't':   t_sym,
        'x':   xd_sym,
        'z':   z_sym,
        'p':   p_sym,
        'ode': xdot_diff,   # d(x_diff)/dt = z[:nd]
        'alg': G_sym,       # G(t, x_full, xdot_full, u) = 0
    }
 
    default_opts: dict = {'abstol': 1e-8, 'reltol': 1e-8}
    if opts:
        default_opts.update(opts)
 
    states = np.zeros((nx, integration_times.size))
    derivs = np.zeros((nx, integration_times.size))
 
    states[:, 0] = y0_arr
    derivs[:, 0] = ydot0_arr
 
    xd_k = ca.DM(y0_arr[:nd].reshape(-1, 1))
    # z0 = [xdot_diff_guess ; x_alg_guess]
    z_k  = ca.DM(np.concatenate([ydot0_arr[:nd], y0_arr[nd:]]).reshape(-1, 1))
    integrators: dict[float, ca.Function] = {}

    for k, (time_start, time_end) in enumerate(
        zip(integration_times[:-1], integration_times[1:])
    ):
        dt = float(time_end - time_start)
        # Non-uniform t_eval grids may require several compiled integrators,
        # while uniform grids normally reuse this first cached instance.
        integ = next(
            (
                candidate
                for step, candidate in integrators.items()
                if np.isclose(dt, step, rtol=1e-12, atol=0.0)
            ),
            None,
        )
        if integ is None:
            integ = ca.integrator(
                f'G_integ_{len(integrators)}', 'idas', dae_dict,
                0.0, [dt], default_opts
            )
            integrators[dt] = integ
        parameters = ca.vertcat(u_dm, float(time_start))
        res  = integ(x0=xd_k, z0=z_k, p=parameters)
        xd_k = res['xf']
        z_k  = res['zf']
 
        xd_np  = np.asarray(xd_k).flatten()
        z_np   = np.asarray(z_k).flatten()
 
        states[:nd,  k + 1] = xd_np
        states[nd:,  k + 1] = z_np[nd:]     # algebraic states from z
        derivs[:nd,  k + 1] = z_np[:nd]     # xdot_diff from z

    # Drop the internally inserted t0 column when it was not requested.
    return IntegrationResult(
        t=output_times,
        y=states[:, output_start:],
        yp=derivs[:, output_start:],
    )
