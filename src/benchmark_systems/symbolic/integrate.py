"""
CasADi-based integrators with an interface compatible with scipy's
``solve_ivp`` and ``scipy_dae``'s ``solve_dae``.
 
CasADi 3.6.x and the semi-explicit DAE reformulation
------------------------------------------------------
CasADi 3.6.x IDAS does not support the ``'xdot'`` key in the DAE dict.
Fully-implicit DAEs  G(t, x, ẋ, u) = 0  are handled by splitting the
state vector into differential and algebraic parts according to the
``n_diff`` attribute of the _DAEFunction:
 
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
 
 
# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
 
@dataclass
class IntegrationResult:
    """
    Lightweight result object compatible with scipy's OdeResult.
 
    Attributes
    ----------
    t   : np.ndarray  shape (N+1,)
    y   : np.ndarray  shape (n_states, N+1)  – full state [x_diff ; x_alg]
    yp  : np.ndarray  shape (n_states, N+1)  – ẋ trajectory; zeros for ODEs
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
 
 
# ---------------------------------------------------------------------------
# ODE integrator
# ---------------------------------------------------------------------------
 
def solve_ivp_casadi(f: ca.Function,
                     t_span: tuple[float, float],
                     y0: Sequence[float],
                     *,
                     u: float | Sequence[float] = 0.0,
                     N: int = 1000,
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
    N : int.  Number of integration steps.  Default 1000.
    integrator : str.  ``'rk'`` (RK4), ``'cvodes'``, or ``'collocation'``.
    opts : dict, optional.  Extra options for ``ca.integrator``.
    """
    t0, tf = t_span
    nx     = len(y0)
    u_dm   = _to_dm(u)
    nu     = u_dm.numel()
    dt     = (tf - t0) / N
 
    t_sym = ca.SX.sym('t')
    x_sym = ca.SX.sym('x', nx)
    u_sym = ca.SX.sym('u', nu)
    dx    = f(t_sym, x_sym, u_sym)
 
    ode_dict = {'x': x_sym, 'p': u_sym, 'ode': dx}
 
    default_opts: dict = {}
    if integrator == 'rk':
        default_opts['number_of_finite_elements'] = 4
    if opts:
        default_opts.update(opts)
 
    integ  = ca.integrator('F', integrator, ode_dict, t0, [t0 + dt], default_opts)
    times  = np.linspace(t0, tf, N + 1)
    states = np.zeros((nx, N + 1))
    states[:, 0] = np.asarray(y0).flatten()
    x_k = ca.DM(np.asarray(y0, dtype=float).reshape(-1, 1))
 
    for k in range(N):
        res = integ(x0=x_k, p=u_dm)
        x_k = res['xf']
        states[:, k + 1] = np.asarray(x_k).flatten()
 
    return IntegrationResult(t=times[:-1], y=states[:, :-1])
 
 
# ---------------------------------------------------------------------------
# DAE integrator
# ---------------------------------------------------------------------------
 
def solve_dae_casadi(F,
                     t_span: tuple[float, float],
                     y0: Sequence[float],
                     yp0: Sequence[float],
                     *,
                     u: float | Sequence[float] = 0.0,
                     N: int = 1000,
                     opts: dict | None = None) -> IntegrationResult:
    """
    Integrate a fully-implicit DAE  G(t, x, ẋ, u) = 0  using CasADi / IDAS.
 
    Parameters
    ----------
    F : _DAEFunction
        Object returned by the symbolic DAE constructors.  Its ``n_diff``
        attribute controls the differential/algebraic splitting.
    t_span : (t0, tf)
    y0 : array-like.  Initial state vector (full: differential + algebraic).
    yp0 : array-like.
        Initial derivatives for the differential part; algebraic initial
        guess for the algebraic part.
    u : scalar or array-like.  Constant control input.  Default 0.
    N : int.  Number of output steps.  Default 1000.
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
    t0, tf  = t_span
    nx      = len(y0)
    nd      = F.n_diff           # number of differential states
    na      = nx - nd            # number of algebraic states
    u_dm    = _to_dm(u)
    nu      = u_dm.numel()
    dt_step = (tf - t0) / N
 
    t_sym  = ca.SX.sym('t')
    xd_sym = ca.SX.sym('xd', nd)                 # differential states
    # z = [xdot_diff (nd) ; x_alg (na)]
    z_sym  = ca.SX.sym('z',  nd + na)
    u_sym  = ca.SX.sym('u',  nu)
 
    xdot_diff = z_sym[:nd]
    x_alg     = z_sym[nd:]
 
    # Reconstruct the full state and xdot vectors for G
    x_full    = ca.vertcat(xd_sym, x_alg)          # [x_diff ; x_alg]
    xdot_full = ca.vertcat(xdot_diff, ca.SX.zeros(na))  # alg derivatives = 0
    
    G_sym = F.fn(t_sym, x_full, xdot_full, u_sym)
 
    dae_dict = {
        'x':   xd_sym,
        'z':   z_sym,
        'p':   u_sym,
        'ode': xdot_diff,   # d(x_diff)/dt = z[:nd]
        'alg': G_sym,       # G(t, x_full, xdot_full, u) = 0
    }
 
    default_opts: dict = {'abstol': 1e-8, 'reltol': 1e-8}
    if opts:
        default_opts.update(opts)
 
    integ = ca.integrator('G_integ', 'idas', dae_dict,
                          t0, [t0 + dt_step], default_opts)
 
    times  = np.linspace(t0, tf, N + 1)
    states = np.zeros((nx, N + 1))
    derivs = np.zeros((nx, N + 1))
 
    y0_arr    = np.asarray(y0,     dtype=float).flatten()
    ydot0_arr = np.asarray(yp0, dtype=float).flatten()
 
    states[:, 0] = y0_arr
    derivs[:, 0] = ydot0_arr
 
    xd_k = ca.DM(y0_arr[:nd].reshape(-1, 1))
    # z0 = [xdot_diff_guess ; x_alg_guess]
    z_k  = ca.DM(np.concatenate([ydot0_arr[:nd], y0_arr[nd:]]).reshape(-1, 1))
 
    for k in range(N):
        res  = integ(x0=xd_k, z0=z_k, p=u_dm)
        xd_k = res['xf']
        z_k  = res['zf']
 
        xd_np  = np.asarray(xd_k).flatten()
        z_np   = np.asarray(z_k).flatten()
 
        states[:nd,  k + 1] = xd_np
        states[nd:,  k + 1] = z_np[nd:]     # algebraic states from z
        derivs[:nd,  k + 1] = z_np[:nd]     # xdot_diff from z
 
    return IntegrationResult(t=times[:-1], y=states[:, :-1], yp=derivs[:, :-1])