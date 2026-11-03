"""
CasADi integration helpers with SciPy-style output grids.

Both helpers accept a constant control input and requested sample times,
returning component-major NumPy trajectories. ODE models use a selected
CasADi integrator; DAE models use IDAS with the wrapper's state split.
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
    Sampled trajectory returned by the symbolic integration helpers.

    Attributes
    ----------
    t : numpy.ndarray
        Requested output times, with shape (n_times,).
    y : numpy.ndarray
        Full state trajectory with shape (n_states, n_times), including any
        algebraic states in their model-defined positions.
    yp : numpy.ndarray
        Derivative information with the same shape as y. ODE results contain
        zeros, not evaluated derivatives. DAE results contain differential
        derivatives and zero placeholders for algebraic entries at advanced
        times; an explicitly requested initial column retains the supplied yp0.
    success : bool
        True when the helper returns after successful integration.
    message : str
        Completion message.

    Notes
    -----
    Validation and backend failures can raise exceptions rather than return a
    result with success=False. The helpers use SciPy-style trajectory array
    orientation but do not implement every SciPy result field or solver option.
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
    Integrate a symbolic ODE with a constant input and requested sample times.

    Parameters
    ----------
    f : casadi.Function
        CasADi model with named inputs t, x, u and output dx, as returned by
        an ODE factory.
    t_span : tuple of float
        Finite forward time bounds (t0, tf), with tf > t0, in the model time
        unit.
    y0 : array-like
        Initial state at t0, with one finite value per model state.
    t_eval : sequence of float
        Nonempty, finite, strictly increasing output times inside t_span. The
        initial time may be omitted.
    u : float or sequence of float, optional
        Constant model input. Supply all input components; the scalar default
        0 does not broadcast to multi-input models.
    integrator : str, optional
        CasADi integrator plugin name. Default is 'rk'; alternatives include
        'cvodes' and 'collocation' when available.
    opts : dict or None, optional
        Backend-specific options passed to ca.integrator. None uses the helper
        defaults.

    Returns
    -------
    IntegrationResult
        Sampled solution with t of shape (n_times,) and y of shape (n_states,
        n_times), plus success, message and a zero-filled yp placeholder.

    Notes
    -----
    The model receives absolute time even though each interval is
    integrated using local backend time. Returned times are exactly t_eval;
    include tf when its state is needed. The input is constant for one call.
    For piecewise inputs, call successive intervals and carry their final
    states forward. Evaluate f separately if physical derivatives are needed;
    the ODE result's yp array is not a derivative trajectory.

    Invalid arguments and backend integration failures may raise exceptions.

    Examples
    --------
    >>> from benchmark_systems.symbolic import pendulum, solve_ivp_casadi
    >>> f = pendulum(m=1.0, L=1.0)
    >>> result = solve_ivp_casadi(f, (0.0, 1.0), [0.1, 0.0],
    ...                         t_eval=[0.0, 0.5, 1.0], u=0.0)
    >>> result.y.shape
    (2, 3)
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
    Integrate a symbolic DAE with IDAS and requested sample times.

    Parameters
    ----------
    F : DAEFunction
        Model wrapper returned by a DAE factory; n_diff identifies its leading
        differential-state block.
    t_span : tuple of float
        Finite forward time bounds (t0, tf), with tf > t0, in the model time
        unit.
    y0 : array-like
        Full initial state at t0. Differential values and algebraic-state
        guesses use the model state order.
    yp0 : array-like
        Full initial derivative vector, the same length as y0. Leading
        differential entries provide derivative guesses; trailing algebraic
        entries are not used as algebraic-state guesses.
    t_eval : sequence of float
        Nonempty, finite, strictly increasing output times inside t_span. The
        initial time may be omitted.
    u : float or sequence of float, optional
        Constant model input with all input components. The scalar default 0
        is intended for single-input models.
    opts : dict or None, optional
        CasADi IDAS options. None uses the helper tolerance defaults; choose
        initialization options appropriate to the supplied initial state.

    Returns
    -------
    IntegrationResult
        Sampled solution with t of shape (n_times,) and full state and
        derivative arrays y, yp of shape (n_states, n_times), plus success and
        message.

    Notes
    -----
    Use initial values consistent with the selected IDAS initialization
    options. Algebraic-state guesses come from y0, not yp0. Returned columns
    at an explicitly requested initial time copy y0 and yp0 unchanged.
    At advanced times, yp contains differential derivatives and zero
    placeholders for algebraic-state derivatives.

    Input u is constant for each call. For successive intervals, carry the
    final state and derivative estimates and include the preceding endpoint
    in its output grid. Invalid arguments and backend failures may raise
    exceptions.

    Examples
    --------
    >>> from benchmark_systems.symbolic import double_cart_pendulum, solve_dae_casadi
    >>> F = double_cart_pendulum(M=0.6, m1=0.2, m2=0.2, L1=0.5, L2=0.5)
    >>> result = solve_dae_casadi(F, (0.0, 0.1), [0.0]*6, [0.0]*6,
    ...                         t_eval=[0.0, 0.05, 0.1], u=0.0)
    >>> result.y.shape, result.yp.shape
    ((6, 3), (6, 3))
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
