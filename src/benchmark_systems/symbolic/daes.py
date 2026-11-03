"""
CasADi factories for benchmark systems in implicit form.

Factories bind physical parameters and return ``DAEFunction`` wrappers with
the state/input dimensions and differential-state count. The wrapper is
callable as ``F(t, x, xdot, u)``; its ``fn`` attribute exposes the underlying
CasADi function. Use the public imports from ``benchmark_systems.symbolic``,
including ``quadrotor_dae`` for the quadrotor.
"""
 
from typing import Sequence
import casadi as ca
from casadi import sin, cos, sqrt, log
from ..common import Const
 
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
 
def _sym(name: str, n: int = 1) -> ca.SX:
    return ca.SX.sym(name, n)
 
 
class DAEFunction:
    """
    Wrap a CasADi model and its differential/algebraic state split.

    Model factories create this wrapper automatically. Construct it directly
    when defining an additional model for ``solve_dae_casadi``.

    Parameters
    ----------
    name : str
        Name assigned to the underlying CasADi function.
    t : casadi.SX
        Scalar symbolic time.
    x : casadi.SX
        Symbolic full-state column vector, with differential states first.
    xdot : casadi.SX
        Symbolic full-state derivative column vector, the same size as x.
    u : casadi.SX
        Symbolic control-input column vector.
    G : casadi.SX
        Model expression with one equation per state, evaluated as G(t,x,xdot,u).
    n_diff : int or None, optional
        Size of the leading differential-state block. None means all states
        are differential; trailing states, when present, are algebraic.

    Attributes
    ----------
    fn : casadi.Function
        Underlying function with named inputs t, x, xdot, u and output G.
    n_diff : int
        Number of leading differential states. This is 3 for oil_well and
        neutralization, and the full state count for the mechanical DAE models.

    Notes
    -----
    ``state_size`` counts all states; ``input_size`` counts control entries.
    Call the wrapper positionally to evaluate its CasADi function. Use ``fn``
    for named evaluation or differentiation, and ``dae_dict()`` to obtain
    the representation accepted by a CasADi integrator.
    """
 
    def __init__(self, name: str, t, x, xdot, u, G, n_diff: int | None = None):
        self.fn = ca.Function(name, [t, x, xdot, u], [G],
                              ['t', 'x', 'xdot', 'u'], ['G'])
        nx = x.shape[0]
        self.n_diff = nx if n_diff is None else n_diff
        self._t    = t
        self._x    = x
        self._xdot = xdot
        self._u    = u
        self._G    = G
        self._nx   = nx
 
    def __call__(self, t_val, x_val, xdot_val, u_val):
        """
        Evaluate the model at the supplied time, state, derivatives and input.

        Parameters
        ----------
        t_val : float or casadi expression
            Time in the model's time unit.
        x_val : array-like or casadi expression
            Full state with state_size entries.
        xdot_val : array-like or casadi expression
            Full derivative vector, with state_size entries.
        u_val : scalar, array-like or casadi expression
            Control values with input_size entries.

        Returns
        -------
        casadi.DM, casadi.SX or casadi.MX
            Model-equation column vector, according to the argument types.
        """
        return self.fn(t_val, x_val, xdot_val, u_val)

    @property
    def state_size(self) -> int:
        """Number of state and residual equations."""
        return self._nx

    @property
    def input_size(self) -> int:
        """Number of control inputs."""
        return self._u.shape[0]
 
    def dae_dict(self) -> dict:
        """
        Build the semi-explicit representation accepted by CasADi integrators.

        Returns
        -------
        dict
            Entries t, x, z, p, ode and alg. The differential state is x, p holds
            control inputs, and z contains differential derivatives followed by
            algebraic states.

        Notes
        -----
        For nd=n_diff, initialize x0 with the first nd state entries and z0 with
        the nd derivative guesses followed by algebraic-state guesses. The public
        ``solve_dae_casadi`` helper manages this layout for you.

        Examples
        --------
        >>> import casadi as ca
        >>> from benchmark_systems.symbolic import double_cart_pendulum
        >>> F = double_cart_pendulum(M=0.6, m1=0.2, m2=0.2, L1=0.5, L2=0.5)
        >>> integrator = ca.integrator('example_dae', 'idas', F.dae_dict(), 0.0, [0.02])
        >>> result = integrator(x0=[0.0]*6, z0=[0.0]*6, p=0.0)
        >>> result['xf'].shape
        (6, 1)
        """
        nd = self.n_diff
        na = self._nx - nd
        t_s  = ca.SX.sym('_t')
        xd_s = ca.SX.sym('_xd', nd)
        z_s  = ca.SX.sym('_z',  nd + na)
        u_s  = ca.SX.sym('_u',  self._u.shape[0])
        xdot_diff = z_s[:nd]
        x_alg     = z_s[nd:]
        x_full    = ca.vertcat(xd_s, x_alg)
        xdot_full = ca.vertcat(xdot_diff, ca.SX.zeros(na))
        G_s = self.fn(t_s, x_full, xdot_full, u_s)
        return {
            't':   t_s,
            'x':   xd_s,
            'z':   z_s,
            'p':   u_s,
            'ode': xdot_diff,
            'alg': G_s,
        }
 
 
# ---------------------------------------------------------------------------
# Double Cart-Pendulum
# ---------------------------------------------------------------------------
 
def double_cart_pendulum(*,
                         M: float, m1: float, m2: float,
                         L1: float, L2: float) -> DAEFunction:
    """
    Build a symbolic double pendulum on a force-driven cart.

    Parameters
    ----------
    M : float
        Cart mass in kg.
    m1 : float
        First-link mass in kg.
    m2 : float
        Second-link mass in kg.
    L1 : float
        First-link length in m.
    L2 : float
        Second-link length in m.

    Returns
    -------
    DAEFunction
        Callable F(t, x, xdot, u) with six states, one input and n_diff=6; fn
        evaluates a six-component CasADi residual.

    Notes
    -----
    State order is ``[position, theta1, theta2, velocity, omega1,
    omega2]``. Position/velocity units are m and m/s; angles and angular
    velocities use rad and rad/s. Angles are absolute, with pi pointing up.
    Input u is cart force in N and time is in seconds. Link centres of mass
    are fixed at half length, with rod inertias computed from mass and length.
    Friction is omitted. All states are differential; supply corresponding
    derivatives when evaluating the returned wrapper.

    Examples
    --------
    >>> from benchmark_systems.symbolic import double_cart_pendulum
    >>> F = double_cart_pendulum(M=0.6, m1=0.2, m2=0.2, L1=0.5, L2=0.5)
    >>> F(0.0, [0.0]*6, [0.0]*6, 0.0).shape
    (6, 1)
    """
    t    = _sym('t')
    x    = _sym('x', 6)
    xdot = _sym('xdot', 6)
    u    = _sym('u')
 
    pos, theta1, theta2, dx, omega1, omega2 = (x[i] for i in range(6))
    x_dot, th1_dot, th2_dot, ddx, alpha1, alpha2 = (xdot[i] for i in range(6))
 
    g  = Const.GRAVITY
    l1 = L1 / 2
    l2 = L2 / 2
    J1 = m1 * l1**2 / 3
    J2 = m2 * l2**2 / 3
 
    p1 = M + m1 + m2
    p2 = m1 * l1 + m2 * L1
    p3 = m2 * l2
    p4 = m1 * l1**2 + m2 * L1**2 + J1
    p5 = m2 * l2 * L1
    p6 = m2 * l2**2 + J2
    p7 = (m1 * l1 + m2 * L1) * g
    p8 = m2 * l2 * g
 
    G = ca.vertcat(
        dx    - x_dot,
        omega1 - th1_dot,
        omega2 - th2_dot,
        p1 * ddx + p2 * alpha1 * cos(theta1) + p3 * alpha2 * cos(theta2)
            - (p2 * omega1**2 * sin(theta1)
               + p3 * omega2**2 * sin(theta2) + u),
        -p2 * cos(theta1) * ddx - p4 * alpha1
            - p5 * alpha2 * cos(theta1 - theta2)
            - (p7 * sin(theta1) + p5 * omega2**2 * sin(theta1 - theta2)),
        -p3 * cos(theta2) * ddx
            - p5 * alpha1 * cos(theta1 - theta2) - p6 * alpha2
            - (-p5 * omega1**2 * sin(theta1 - theta2)
               + p8 * sin(theta2)),
    )
    return DAEFunction('double_cart_pendulum', t, x, xdot, u, G)
 
 
# ---------------------------------------------------------------------------
# Multilevel Cart-Pendulum
# ---------------------------------------------------------------------------
 
def multilevel_cart_pendulum(*,
                              M: float,
                              m: Sequence[float],
                              L: Sequence[float],
                               deltas: Sequence[float] | None = None) -> DAEFunction:
    """
    Build a symbolic serial N-link cart-pendulum model.

    Parameters
    ----------
    M : float
        Cart mass in kg.
    m : sequence of float
        N link masses in kg.
    L : sequence of float
        N link lengths in m, matching m.
    deltas : sequence of float or None, optional
        N+1 friction-law coefficients: cart first, then rotational joints. For
        the implemented abs(deltas*velocities) magnitude law, use N s^2/m^2
        for the cart and N m s^2/rad^2 for rotational joints. None disables
        friction.

    Returns
    -------
    DAEFunction
        Callable F(t, x, xdot, u) with 2*(N+1) differential states, one input
        and an equally sized residual.

    Notes
    -----
    States are ``[position, theta1, ..., thetaN, velocity, omega1,
    ..., omegaN]``. Use m, m/s, rad and rad/s, respectively, and seconds for
    time. Input u is horizontal cart force in N. Angles are absolute with
    pi upright. Each link's centre of mass is at half length. The friction
    coefficients are used in a velocity-dependent magnitude law, not simply
    as a constant diagonal viscous-damping matrix. Evaluate the wrapper as
    ``F(t, x, xdot, u)`` with full-length state and derivative vectors.
    """
    np_pend = len(m)
    if len(L) != np_pend:
        raise ValueError("m and L must have the same length")
    if deltas is not None and len(deltas) != np_pend + 1:
        raise ValueError("deltas must contain one coefficient for the cart and each pendulum")
    half    = 1 + np_pend          # positions only
    nx      = 2 * half             # full state
 
    t    = _sym('t')
    xv   = _sym('x', nx)
    xdot = _sym('xdot', nx)
    u    = _sym('u')
 
    pos    = xv[0]
    thetas = [xv[1 + i] for i in range(np_pend)]
    dx     = xv[half]
    omegas = [xv[half + 1 + i] for i in range(np_pend)]
 
    x_dot    = xdot[0]
    th_dots  = [xdot[1 + i] for i in range(np_pend)]
    ddx      = xdot[half]
    alphas   = [xdot[half + 1 + i] for i in range(np_pend)]
 
    g  = Const.GRAVITY
    l  = [L[i] / 2 for i in range(np_pend)]
    J  = [m[i] * l[i]**2 / 3 for i in range(np_pend)]
 
    # Friction vector (simplified: proportional to velocity norm, always ≥ 0)
    if deltas is not None:
        # f[0]: cart–ground friction, f[i+1]: rotational friction of pendulum i
        all_vel = ca.vertcat(dx, *omegas)
        f = [float(deltas[i]) * ca.fabs(all_vel[i]) for i in range(np_pend + 1)]
    else:
        f = [0.0] * (np_pend + 1)
 
    # Model coefficients
    a0 = M + sum(m)
    a  = [
        m[i] * l[i] + sum(m[j] * L[i] for j in range(i + 1, np_pend))
        for i in range(np_pend)
    ]
    b  = [
        J[i] + m[i] * l[i]**2 + sum(m[j] * L[i]**2 for j in range(i + 1, np_pend))
        for i in range(np_pend)
    ]
 
    # H1  (symmetric inertia matrix)
    def h1_elem(i, j):
        # i, j ∈ {0 … np_pend}; i=0 → cart
        if i == j:
            return a0 if i == 0 else b[i - 1]
        ii, jj = min(i, j), max(i, j)
        if ii == 0:
            return -a[jj - 1] * cos(thetas[jj - 1])
        else:
            return a[jj - 1] * L[ii - 1] * cos(thetas[jj - 1] - thetas[ii - 1])
 
    H1 = ca.SX(np_pend + 1, np_pend + 1)
    for i in range(np_pend + 1):
        for j in range(np_pend + 1):
            H1[i, j] = h1_elem(i, j)
 
    # Build acceleration vector
    accel = ca.vertcat(ddx, *[-alphas[i] for i in range(np_pend)])
 
    # h3 (Const.GRAVIty vector)
    h3 = ca.vertcat(0.0, *[a[i] * g * sin(thetas[i]) for i in range(np_pend)])
 
    # H2 (velocity-dependent and friction terms) — replicated from numpy version
    # Upper triangular velocity coupling
    H2_sup_rows = []
    for i in range(np_pend + 1):
        row = ca.SX(1, np_pend + 1)
        for j in range(i, np_pend):
            if i == 0:
                row[0, j + 1] = -a[j] * sin(thetas[j]) * omegas[j]
            else:
                row[0, j + 1] = a[j] * L[i - 1] * omegas[j] * sin(thetas[j] - thetas[i - 1])
        H2_sup_rows.append(row)
    H2_sup = ca.vertcat(*H2_sup_rows)
 
    H2_inf = ca.SX(np_pend + 1, np_pend + 1)
    for j in range(1, np_pend + 1):
        for i in range(j, np_pend):
            H2_inf[i + 1, j] = -a[i] * L[j - 1] * sin(thetas[i] - thetas[j - 1]) * omegas[j - 1]
 
    # Friction diagonal (cart + pendulums)
    f_diag_vals = (
        [-f[0]]
        + [-f[i] - f[i + 1] for i in range(1, np_pend)]
        + [-f[-1]]
    )
    H2_diag = ca.diag(ca.vertcat(*f_diag_vals))
 
    H2 = H2_diag + H2_sup + H2_inf
    for i in range(1, np_pend):
        H2[i, i + 1] = H2[i, i + 1] + f[i]
        H2[i + 1, i] = H2[i + 1, i] + f[i]
 
    vel_vec = ca.vertcat(dx, *[-omegas[i] for i in range(np_pend)])
 
    h0 = ca.vertcat(1.0, *[0.0] * np_pend)
 
    # ODE part: dx/dt = velocities
    ode_residual = ca.vertcat(
        dx - x_dot,
        *[omegas[i] - th_dots[i] for i in range(np_pend)]
    )
    # Euler-Lagrange residual
    el_residual = H2 @ vel_vec + h3 + h0 * u - H1 @ accel
 
    G = ca.vertcat(ode_residual, el_residual)
    return DAEFunction('multilevel_cart_pendulum', t, xv, xdot, u, G)
 
 
# ---------------------------------------------------------------------------
# Quadrotor (Euler-Lagrange, DAE)
# ---------------------------------------------------------------------------
 
def quadrotor(*,
              Ixx: float, Iyy: float, Izz: float,
               k: float, L: float, m: float, drag: float) -> DAEFunction:
    """
    Build a symbolic Euler-Lagrange quadrotor model.

    Parameters
    ----------
    Ixx : float
        Body-frame principal inertia about the x axis in kg m^2.
    Iyy : float
        Body-frame principal inertia about the y axis in kg m^2.
    Izz : float
        Body-frame principal inertia about the z axis in kg m^2.
    k : float
        Thrust coefficient: k times squared rotor speed gives thrust in N.
    L : float
        Centre-to-rotor arm length in m.
    m : float
        Vehicle mass in kg.
    drag : float
        Rotor yaw-torque coefficient: drag times squared rotor speed gives
        torque in N m; not translational air drag.

    Returns
    -------
    DAEFunction
        Callable F(t, x, xdot, u) with 12 differential states, four inputs and
        a 12-component residual.

    Notes
    -----
    Use the public import ``benchmark_systems.symbolic.quadrotor_dae``.
    State order is ``[x, y, z, phi, theta, psi, dx, dy, dz, dphi, dtheta,
    dpsi]``: positions in m, angles in rad, inertial linear velocities in
    m/s and Euler-angle rates in rad/s. These velocity coordinates differ
    from the quadrotor ODE's body-frame velocities. Input contains four
    squared rotor speeds, in (rad/s)^2 with compatible coefficients; time
    is in seconds. Evaluate ``F(t, x, xdot, u)`` with all state derivatives.
    Euler coordinates are singular at cos(theta)=0; aerodynamic resistance
    and rotor dynamics are not included.
    """
    t    = _sym('t')
    xv   = _sym('x', 12)
    xdot = _sym('xdot', 12)
    u    = _sym('u', 4)
 
    pos_x, pos_y, pos_z     = xv[0],  xv[1],  xv[2]
    phi, theta, psi         = xv[3],  xv[4],  xv[5]
    dx, dy, dz              = xv[6],  xv[7],  xv[8]
    dphi, dtheta, dpsi      = xv[9],  xv[10], xv[11]
 
    x_dot, y_dot, z_dot       = xdot[0], xdot[1], xdot[2]
    phi_dot, th_dot, psi_dot  = xdot[3], xdot[4], xdot[5]
    ddx, ddy, ddz             = xdot[6], xdot[7], xdot[8]
    ddphi, ddth, ddpsi        = xdot[9], xdot[10], xdot[11]
 
    tau_phi   = L * k * (-u[1] + u[3])
    tau_theta = L * k * (-u[0] + u[2])
    tau_psi   = drag * (-u[0] + u[1] - u[2] + u[3])
    F         = k * (u[0] + u[1] + u[2] + u[3])
 
    cp, sp = cos(phi),   sin(phi)
    ct, st = cos(theta), sin(theta)
    cy, sy = cos(psi),   sin(psi)
 
    R = ca.vertcat(
        ca.horzcat(cy*ct, cy*st*sp - sy*cp, cy*st*cp + sy*sp),
        ca.horzcat(sy*ct, sy*st*sp + cy*cp, sy*st*cp - cy*sp),
        ca.horzcat(-st,          ct*sp,             ct*cp),
    )
    W = ca.vertcat(
        ca.horzcat(1,   0,        -st),
        ca.horzcat(0,  cp,  ct * sp),
        ca.horzcat(0, -sp,  ct * cp),
    )
 
    J_mat = ca.diag(ca.vertcat(Ixx, Iyy, Izz))
    M_mat = W.T @ J_mat @ W
 
    c11 = 0
    c12 = ((Iyy-Izz)*(dtheta*cp*sp + dpsi*sp**2*ct)
           + (Izz-Iyy)*dpsi*cp**2*ct - Ixx*dpsi*ct)
    c13 = (Izz-Iyy)*dpsi*cp*sp*ct**2
    c21 = -c12
    c22 = (Izz-Iyy)*dphi*cp*sp
    c23 = (-Ixx*dpsi*st*ct
           + Iyy*dpsi*sp**2*ct*st
           + Izz*dpsi*cp**2*ct*st)
    c31 = -c13 - Ixx*dtheta*ct
    c32 = ((Izz-Iyy)*(dtheta*cp*sp*st + dphi*sp**2*ct)
           + (Iyy-Izz)*dpsi*cp**2*ct - c23)
    c33 = ((Iyy-Izz)*dphi*cp*sp*ct**2
           - (Iyy*dtheta*sp**2*ct*st
              + Izz*dtheta*cp**2*ct*st
              - Ixx*dtheta*st*ct))
    C_mat = ca.vertcat(
        ca.horzcat(c11, c12, c13),
        ca.horzcat(c21, c22, c23),
        ca.horzcat(c31, c32, c33),
    )
 
    ang_vel = ca.vertcat(dphi, dtheta, dpsi)
 
    g_ode = ca.vertcat(
        dx - x_dot, dy - y_dot, dz - z_dot,
        dphi - phi_dot, dtheta - th_dot, dpsi - psi_dot,
    )
    g_lin = (ca.vertcat(0, 0, -Const.GRAVITY)
             + R @ ca.vertcat(0, 0, F / m)
             - ca.vertcat(ddx, ddy, ddz))
    g_ang = (ca.solve(M_mat,
                      ca.vertcat(tau_phi, tau_theta, tau_psi)
                      - C_mat @ ang_vel)
             - ca.vertcat(ddphi, ddth, ddpsi))
 
    G = ca.vertcat(g_ode, g_lin, g_ang)
    return DAEFunction('quadrotor', t, xv, xdot, u, G)
 
 
# ---------------------------------------------------------------------------
# Oil Well
# ---------------------------------------------------------------------------
 
def oil_well(*,
             T_a, V_a, L_a, K_gs, K_inj,
             M_G, P_gs,
             T_t, D_t, L_t, V_t, epsilon, K_pr, P_out,
             S_bh, L_bh,
             P_res, avg_w_res, PI, GOR,
             mu, rho_L) -> DAEFunction:
    """
    Build a symbolic gas-lift oil-well model.

    Parameters
    ----------
    T_a : float
        Annulus gas temperature in K.
    V_a : float
        Annulus gas volume in m^3.
    L_a : float
        Annulus length in m.
    K_gs : float
        Gas-lift choke coefficient, with effective area units m^2 for mass
        flow kg/s, density kg/m^3 and pressure difference Pa.
    K_inj : float
        Injection choke coefficient, with effective area units m^2 on the same
        SI mass-flow scale.
    M_G : float
        Gas molecular weight in kg/kmol.
    P_gs : float
        Gas-source pressure in bar; converted internally to Pa.
    T_t : float
        Tubing gas temperature in K.
    D_t : float
        Tubing diameter in m.
    L_t : float
        Tubing length in m.
    V_t : float
        Tubing volume in m^3.
    epsilon : float
        Pipe roughness in m.
    K_pr : float
        Production choke coefficient, with effective area units m^2 on the SI
        mass-flow scale.
    P_out : float
        Wellhead outlet pressure in bar; converted internally to Pa.
    S_bh : float
        Bottom-hole cross-sectional area in m^2.
    L_bh : float
        Bottom-hole length in m.
    P_res : float
        Reservoir pressure in bar; converted internally to Pa.
    avg_w_res : float
        Average reservoir mass flow in kg/s, used in friction correlations.
    PI : float
        Reservoir productivity index in kg/(s Pa).
    GOR : float
        Gas-to-liquid mass-flow ratio, dimensionless.
    mu : float
        Liquid dynamic viscosity in Pa s.
    rho_L : float
        Liquid density in kg/m^3.

    Returns
    -------
    DAEFunction
        Callable F(t, x, xdot, u) with 11 states, two inputs and n_diff=3; fn
        returns an 11-component residual.

    Notes
    -----
    State order is ``[x0, x1, x2, w_G_in, P_at, P_ab, P_bh, P_tt,
    w_out, rho_m_tt, alpha_L_tt]``. The first three states are gas mass in
    the annulus, gas mass in the tubing and liquid mass in the tubing (kg).
    Flow quantities use kg/s, pressure states bar, mixture density kg/m^3
    and liquid volume fraction no units. Only the masses are differential.

    Input is ``[production_choke, gas_lift_choke]``, with opening fractions
    in 0-1; time is in seconds. Evaluate ``F(t, x, xdot, u)`` using the full
    state and derivative vectors. Choke expressions produce mass flow in
    kg/s. Preserve the reference parameter scaling and positive physical
    gas volumes and densities. The symbolic friction expressions floor
    Reynolds numbers at 1; the numerical backend does not.
    """
    import math
 
    R      = Const.IDEAL_GAS * 1e6      # J/(kmol·K)
    g      = Const.GRAVITY
    P_gs_  = P_gs  * Const.BAR_2_PASCAL
    P_out_ = P_out * Const.BAR_2_PASCAL
    P_res_ = P_res * Const.BAR_2_PASCAL
 
    t    = _sym('t')
    xv   = _sym('x', 11)
    xdot = _sym('xdot', 11)
    u    = _sym('u', 2)
 
    x0, x1, x2 = xv[0], xv[1], xv[2]   # masses
 
    # ---- Annulus ----
    P_at     = R * T_a * x0 / (M_G * V_a)
    P_ab     = P_at + x0 * g * L_a / V_a
    rho_G_ab = P_ab * M_G / (R * T_a)
    rho_G_in = P_gs_ * M_G / (R * T_a)
    w_G_in   = K_gs * u[1] * sqrt(rho_G_in * ca.fmax(P_gs_ - P_at, Const.ZERO))
 
    # ---- Tubing ----
    rho_G_tt  = x1 / (V_t + S_bh * L_bh - x2 / rho_L)
    avg_rho_t = (x1 + x2 - rho_L * S_bh * L_bh) / V_t
 
    alpha_G_tb_m = GOR / (GOR + 1) if GOR + 1 != 0 else 0.0
 
    avg_U_sl_t = 4 * (1 - alpha_G_tb_m) * avg_w_res / (rho_L * math.pi * D_t**2)
    avg_U_sg_t = 4 * (w_G_in + alpha_G_tb_m * avg_w_res) / (rho_G_tt * math.pi * D_t**2)
    avg_U_t    = avg_U_sl_t + avg_U_sg_t
    Re_t       = avg_rho_t * avg_U_t * D_t / mu
 
    # Friction factor (Serghides / Colebrook): avoid log10 issues with fmax
    Re_safe = ca.fmax(Re_t, 1.0)
    lambda_t = 1 / (-1.82 * (log(
        (epsilon / D_t / 3.7)**1.11 + 6.9 / Re_safe
    ) / math.log(10)))**2
 
    avg_alpha_L = ca.fmax(x2 - rho_L * S_bh * L_bh, 0) / (V_t * rho_L)
    F_t  = (avg_alpha_L * lambda_t * avg_rho_t * avg_U_t**2 * L_t) / (2 * D_t)
    P_tt = rho_G_tt * R * T_t / M_G
    P_tb = P_tt + avg_rho_t * g * L_t + F_t
 
    # ---- Bottom hole ----
    D_bh        = sqrt(4 * S_bh / math.pi)
    avg_U_lb    = avg_w_res / (rho_L * S_bh)
    Re_bh       = rho_L * avg_U_lb * D_bh / mu
    Re_bh_safe  = ca.fmax(Re_bh, 1.0)
    lambda_b    = 1 / (-1.82 * (log(
        (epsilon / D_bh / 3.7)**1.11 + 6.9 / Re_bh_safe
    ) / math.log(10)))**2
    F_b  = lambda_b * rho_L * avg_U_lb**2 * L_bh / (2 * D_bh)
    P_bh = P_tb + F_b + rho_L * g * L_bh
 
    # ---- Inner flows ----
    w_G_inj  = K_inj * sqrt(rho_G_ab * ca.fmax(P_ab - P_tb, Const.ZERO))
    w_res    = PI * ca.fmax(P_res_ - P_bh, Const.ZERO)
    w_G_res  = alpha_G_tb_m * w_res
    w_L_res  = (1 - alpha_G_tb_m) * w_res
 
    # ---- Output flows ----
    rho_G_tb = P_tb * M_G / (R * T_t)
    denom_al = w_L_res * rho_G_tb + (w_G_inj + w_G_res) * rho_L
    alpha_L_tb = ca.if_else(
        ca.fabs(denom_al) > 1e-30,
        w_L_res * rho_G_tb / denom_al,
        0.0
    )
    alpha_L_tt = 2 * avg_alpha_L - alpha_L_tb
    rho_m_tt   = alpha_L_tt * rho_L + (1 - alpha_L_tt) * rho_G_tt
    alpha_G_tt_m = ((1 - alpha_L_tt) * rho_G_tt
                    / (alpha_L_tt * rho_L + (1 - alpha_L_tt) * rho_G_tt))
    w_out   = K_pr * u[0] * sqrt(rho_m_tt * ca.fmax(P_tt - P_out_, Const.ZERO))
    w_G_out = alpha_G_tt_m * w_out
    w_L_out = (1 - alpha_G_tt_m) * w_out
 
    G = ca.vertcat(
        # ODEs
        xdot[0] - (w_G_in - w_G_inj),
        xdot[1] - (w_G_inj + w_G_res - w_G_out),
        xdot[2] - (w_L_res - w_L_out),
        # Algebraic (intermediate variables)
        xv[3]  - w_G_in,
        xv[4]  - P_at / Const.BAR_2_PASCAL,
        xv[5]  - P_ab / Const.BAR_2_PASCAL,
        xv[6]  - P_bh / Const.BAR_2_PASCAL,
        xv[7]  - P_tt / Const.BAR_2_PASCAL,
        xv[8]  - w_out,
        xv[9]  - rho_m_tt,
        xv[10] - alpha_L_tt,
    )
    return DAEFunction('oil_well', t, xv, xdot, u, G, n_diff=3)
 
 
# ---------------------------------------------------------------------------
# pH Neutralization
# ---------------------------------------------------------------------------
 
def neutralization(*,
                   A: float, z: float, Cv4: float, n: float,
                   Wa1: float, Wa2: float, Wa3: float,
                   Wb1: float, Wb2: float, Wb3: float,
                   pK1: float, pK2: float,
                   q1: float, d: float) -> DAEFunction:
    """
    Build a symbolic pH neutralization tank with an alkaline-flow input.

    Parameters
    ----------
    A : float
        Tank cross-sectional area, documented on the source cm^2 scale.
    z : float
        Outlet/sensor height offset, on the source cm scale.
    Cv4 : float
        Outlet-flow coefficient for Cv4*(h+z)**n, in L/(s cm^n) on the
        documented source flow/length scale.
    n : float
        Dimensionless exponent in the outlet-flow law.
    Wa1 : float
        Reaction invariant a in the acid stream, in the source mol/L scale.
    Wa2 : float
        Reaction invariant a in the buffer stream, in the source mol/L scale.
    Wa3 : float
        Reaction invariant a in the alkaline stream, in the source mol/L
        scale.
    Wb1 : float
        Reaction invariant b in the acid stream, in the source mol/L scale.
    Wb2 : float
        Reaction invariant b in the buffer stream, in the source mol/L scale.
    Wb3 : float
        Reaction invariant b in the alkaline stream, in the source mol/L
        scale.
    pK1 : float
        First acid dissociation pK value, dimensionless.
    pK2 : float
        Second acid dissociation pK value, dimensionless.
    q1 : float
        Constant acid feed flow, documented on the source L/s scale.
    d : float
        Buffer feed flow q2, documented on the source L/s scale and bound as a
        disturbance parameter.

    Returns
    -------
    DAEFunction
        Callable F(t, x, xdot, u) with four states, one input and n_diff=3; fn
        returns a four-component residual.

    Notes
    -----
    State order is ``[Wa4, Wb4, h, pH]``: two reaction invariants,
    liquid height and outlet pH. The first three states are differential;
    pH is algebraic and its derivative is unused. Invariants may be negative.
    Input u is alkaline flow q3 on the same scale as q1 and d.

    Source units are mol/L, cm, cm^2 and L/s, but the equations apply no
    explicit litre-to-cm^3 conversion. Reconcile flow and geometry scaling
    when supplying a new dimensional parameter set. Use a positive liquid
    height and a consistent pH. Evaluate the wrapper as ``F(t, x, xdot, u)``;
    changing the bound disturbance d requires rebuilding the model.
    """
    t    = _sym('t')
    xv   = _sym('x', 4)
    xdot = _sym('xdot', 4)
    u    = _sym('u')
 
    Wa4, Wb4, h, pH = xv[0], xv[1], xv[2], xv[3]
 
    f1 = ca.vertcat(q1 * (Wa1 - Wa4) / h,
                    q1 * (Wb1 - Wb4) / h,
                    q1 - Cv4 * (h + z)**n)
    f2 = ca.vertcat((Wa3 - Wa4) / h,
                    (Wb3 - Wb4) / h,
                    1.0)
    f3 = ca.vertcat((Wa2 - Wa4) / h,
                    (Wb2 - Wb4) / h,
                    1.0)
 
    G_diff = xdot[:3] - (f1 + u * f2 + d * f3) / A
    G_alg  = (Wa4 + 10**(pH - 14)
              + Wb4 * (1 + 2 * 10**(pH - pK2))
                / (1 + 10**(pK1 - pH) + 10**(pH - pK2))
              - 10**(-pH))
 
    G = ca.vertcat(G_diff, G_alg)
    return DAEFunction('neutralization', t, xv, xdot, u, G, n_diff=3)
