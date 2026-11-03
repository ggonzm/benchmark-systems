"""
CasADi factories for the numerical benchmark ODE models.

Physical parameters are bound when a factory is called. Each returned
function has named inputs ``t``, ``x``, ``u`` and output ``dx``; there is no
separate runtime parameter-vector input. Use the public imports from
``benchmark_systems.symbolic``, including ``quadrotor_ode`` for the quadrotor.
"""

from typing import Sequence
import casadi as ca
from casadi import sin, cos, exp, sqrt
from ..common import Const

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sym(name: str, n: int = 1) -> ca.SX:
    return ca.SX.sym(name, n)


# ---------------------------------------------------------------------------
# Pendulum
# ---------------------------------------------------------------------------

def pendulum(*, m: float, L: float, drag: float = 0.0) -> ca.Function:
    """
    Build a symbolic torque-driven point-mass pendulum.

    Parameters
    ----------
    m : float
        Pendulum mass in kg; use a positive value.
    L : float
        Pivot-to-mass distance in m; use a positive value.
    drag : float, optional
        Rotational viscous damping in N m s/rad. Default is 0.

    Returns
    -------
    casadi.Function
        Function f(t, x, u) with named inputs t, x, u and output dx of shape
        (2, 1).

    Notes
    -----
    State order is ``[theta, omega]``, in rad and rad/s. Zero angle hangs
    downward; ``theta=pi`` points upward. Input ``u`` is scalar pivot torque
    in N m. Time is in seconds. Physical parameters are fixed by this factory;
    evaluate the returned function to obtain derivatives.

    Examples
    --------
    >>> from benchmark_systems.symbolic import pendulum
    >>> f = pendulum(m=1.0, L=1.0, drag=0.1)
    >>> f(0.0, [0.0, 0.0], 0.0).shape
    (2, 1)
    """
    t   = _sym('t')
    x   = _sym('x', 2)
    u   = _sym('u')

    theta, omega = x[0], x[1]

    dx = ca.vertcat(
        omega,
        -Const.GRAVITY / L * sin(theta) - drag / (m * L**2) * omega + 1 / (m * L**2) * u
    )
    return ca.Function('pendulum', [t, x, u], [dx],
                       ['t', 'x', 'u'], ['dx'])


# ---------------------------------------------------------------------------
# DC Motor
# ---------------------------------------------------------------------------

def dc_motor(*, J: float, b: float, Ke: float, Kt: float, R: float, L: float) -> ca.Function:
    """
    Build a symbolic DC motor with electrical and shaft dynamics.

    Parameters
    ----------
    J : float
        Rotor moment of inertia in kg m^2.
    b : float
        Shaft viscous friction coefficient in N m s/rad.
    Ke : float
        Back electromotive-force constant in V s/rad.
    Kt : float
        Motor torque constant in N m/A.
    R : float
        Armature resistance in ohm.
    L : float
        Armature inductance in H; distinct from a mechanical length.

    Returns
    -------
    casadi.Function
        Function f(t, x, u) with named inputs t, x, u and output dx of shape
        (3, 1).

    Notes
    -----
    State order is ``[theta, omega, current]``, in rad, rad/s and A.
    Input ``u`` is armature voltage in V; time is in seconds. Evaluate the
    returned function as ``f(t, x, u)``. Inertia and inductance must be
    nonzero. Voltage, current and speed are not automatically saturated.
    """
    t = _sym('t')
    x = _sym('x', 3)
    u = _sym('u')

    theta, omega, I = x[0], x[1], x[2]

    dx = ca.vertcat(
        omega,
        -b / J * omega + Kt / J * I,
        -Ke / L * omega - R / L * I + 1 / L * u,
    )
    return ca.Function('dc_motor', [t, x, u], [dx],
                       ['t', 'x', 'u'], ['dx'])


# ---------------------------------------------------------------------------
# Cart-Pendulum
# ---------------------------------------------------------------------------

def cart_pendulum(*, m: float, M: float, L: float, drag: float = 0.0) -> ca.Function:
    """
    Build a symbolic force-driven cart with a point-mass pendulum.

    Parameters
    ----------
    m : float
        Pendulum point mass in kg.
    M : float
        Cart mass in kg.
    L : float
        Pendulum length in m.
    drag : float, optional
        Cart viscous friction coefficient in N s/m. Default is 0.

    Returns
    -------
    casadi.Function
        Function f(t, x, u) with named inputs t, x, u and output dx of shape
        (4, 1).

    Notes
    -----
    State order is ``[position, velocity, theta, omega]``, in m, m/s,
    rad and rad/s. ``theta=pi`` is upright. Input ``u`` is cart force in N;
    time is in seconds. Evaluate the function as ``f(t, x, u)``. Pendulum
    inertia about its point mass, track stops and force saturation are not
    added to the model.
    """
    t  = _sym('t')
    x  = _sym('x', 4)
    u  = _sym('u')

    pos, vel, theta, omega = x[0], x[1], x[2], x[3]

    den = -M + m * ca.cos(theta)**2 - m

    dx = ca.vertcat(
        vel,
        (-L * m * omega**2 * sin(theta) + drag * vel
         - Const.GRAVITY * m * sin(theta) * cos(theta) - u) / den,
        omega,
        (L * m * omega**2 * sin(theta) * cos(theta)
         - drag * vel * cos(theta)
         + (M + m) * Const.GRAVITY * sin(theta)
         + u * cos(theta)) / (L * den),
    )
    return ca.Function('cart_pendulum', [t, x, u], [dx],
                       ['t', 'x', 'u'], ['dx'])

# ---------------------------------------------------------------------------
# Furuta Pendulum
# ---------------------------------------------------------------------------

def furuta_pendulum(*,
                    m1: float, L1: float, J1: float, b1: float,
                    m2: float, L2: float, J2: float, b2: float,
                    l1: float, l2 : float,
                    K: float) -> ca.Function:
    """
    Build a symbolic Furuta pendulum driven by motor voltage.

    Parameters
    ----------
    m1 : float
        Horizontal motor-arm mass in kg.
    L1 : float
        Horizontal motor-arm length in m.
    J1 : float
        Motor-arm moment of inertia about its centre of mass in kg m^2.
    b1 : float
        Motor-arm rotational damping coefficient in N m s/rad.
    m2 : float
        Pendulum-arm mass in kg.
    L2 : float
        Pendulum-arm length in m. The current derivative expressions use l2
        and J2 rather than L2.
    J2 : float
        Pendulum-arm moment of inertia about its centre of mass in kg m^2.
    b2 : float
        Pendulum-arm rotational damping coefficient in N m s/rad.
    l1 : float
        Motor-arm pivot-to-centre-of-mass distance in m; supplied explicitly.
    l2 : float
        Pendulum pivot-to-centre-of-mass distance in m; supplied explicitly.
    K : float
        Effective motor torque per applied voltage in N m/V; applied torque is
        K*u.

    Returns
    -------
    casadi.Function
        Function f(t, x, u) with named inputs t, x, u and output dx of shape
        (4, 1).

    Notes
    -----
    State order is ``[theta1, theta2, omega1, omega2]``. Angles are in
    rad and velocities in rad/s; ``theta2=0`` hangs downward and
    ``theta2=pi`` is upright. Input ``u`` is motor voltage in V, and time is
    in seconds. Centre-of-mass distances are not inferred from arm lengths.
    Evaluate the function as ``f(t, x, u)`` with nonsingular inertia terms.
    """
    
    t  = _sym('t')
    x  = _sym('x', 4)
    u_sym  = _sym('u')

    theta1, theta2, omega1, omega2 = x[0], x[1], x[2], x[3]

    # Total moments of inertia
    J1_hat = J1 + m1*l1**2 # Arm 1
    J2_hat = J2 + m2*l2**2 # Arm 2
    J0_hat = J1_hat + m2*L1**2 # motor rotor when pendulum is pointing down
    
    # alpha, beta, gamma, w0, cp1, cp2, Delta parameters
    alpha = (m2*L1*l2)/J2_hat
    beta = J0_hat/J2_hat
    gamma = K/(m2*Const.GRAVITY*l2)
    w0 = (m2*Const.GRAVITY*l2/J2_hat)**0.5
    cp1 = b1/J2_hat
    cp2 = b2/J2_hat
    Delta = beta + sin(theta2)**2 - alpha**2*cos(theta2)**2

    dx = ca.vertcat(
        omega1,
        omega2,
        (1/Delta) * (gamma*w0**2*u_sym - 2*omega1*omega2*sin(theta2)*cos(theta2) - omega1*cp1 + omega2**2*alpha*sin(theta2)
                    - omega1**2*alpha*sin(theta2)*cos(theta2)**2 + omega2*cp2*alpha*cos(theta2) + alpha*w0**2*sin(theta2)*cos(theta2)),
        (1/Delta) * (-alpha*gamma*w0**2*u_sym*cos(theta2) + 2*omega1*omega2*alpha*sin(theta2)*cos(theta2)**2 + omega1*cp1*alpha*cos(theta2)
                    - omega2**2*alpha**2*sin(theta2)*cos(theta2) + (beta + sin(theta2)**2)*omega2**2*sin(theta2)*cos(theta2)
                    - (beta + sin(theta2)**2)*omega2*cp2 - beta*w0**2*sin(theta2) - w0**2*sin(theta2)**3)
)
    return ca.Function('furuta_pendulum', [t, x, u_sym], [dx],
                       ['t', 'x', 'u'], ['dx'])

# ---------------------------------------------------------------------------
# Multimass Spring
# ---------------------------------------------------------------------------

def multimass_spring(*,
                     tau: Sequence[float],
                     I: Sequence[float],
                     K: Sequence[float],
                     d: Sequence[float]) -> ca.Function:
    """
    Build a symbolic spring-coupled disc chain with two filtered motors.

    Parameters
    ----------
    tau : sequence of float
        Two positive motor-position filter time constants in s.
    I : sequence of float
        N disc moments of inertia in kg m^2; use N >= 2.
    K : sequence of float
        N+1 torsional spring constants in N m/rad, including the two end
        springs.
    d : sequence of float
        N rotational viscous damping coefficients in N m s/rad.

    Returns
    -------
    casadi.Function
        Function f(t, x, u) with 2+2*N states, two inputs and output dx of
        shape (2+2*N, 1).

    Notes
    -----
    State order is ``[u1_true, u2_true, theta1, ..., thetaN, omega1,
    ..., omegaN]``. Motor positions and disc angles are in rad; disc speeds
    are in rad/s. Input ``u=[setpoint_left, setpoint_right]`` contains motor
    position commands in rad. The first two states obey first-order motor
    filters and are not instantaneous copies of the commands. Time is in
    seconds. Evaluate the model as ``f(t, x, u)`` with nonzero inertias and
    positive time constants.
    """
    n  = len(I)          # number of masses
    nx = 2 + 2 * n       # full state dimension

    t  = _sym('t')
    xv = _sym('x', nx)
    u  = _sym('u', 2)

    u_true = xv[:2]           # true motor positions
    theta  = xv[2:2 + n]      # mass angles
    omega  = xv[2 + n:]       # mass velocities

    du_list = [
        (u[i] - u_true[i]) / tau[i]
        for i in range(2)
    ]

    dtheta_list = [omega[i] for i in range(n)]

    domega_list = []
    for i in range(n):
        if i == 0:
            acc = (-K[i] * (theta[i] - u_true[0])
                   - K[i + 1] * (theta[i] - theta[i + 1])
                   - d[i] * omega[i]) / I[i]
        elif i == n - 1:
            acc = (-K[i] * (theta[i] - theta[i - 1])
                   - K[i + 1] * (theta[i] - u_true[1])
                   - d[i] * omega[i]) / I[i]
        else:
            acc = (-K[i] * (theta[i] - theta[i - 1])
                   - K[i + 1] * (theta[i] - theta[i + 1])
                   - d[i] * omega[i]) / I[i]
        domega_list.append(acc)

    dx = ca.vertcat(*du_list, *dtheta_list, *domega_list)
    return ca.Function('multimass_spring', [t, xv, u], [dx],
                       ['t', 'x', 'u'], ['dx'])


# ---------------------------------------------------------------------------
# Johansson four-tank
# ---------------------------------------------------------------------------

def johansson(*,
              A: Sequence[float],
              a: Sequence[float],
              K: Sequence[float],
              h_max: Sequence[float]) -> ca.Function:
    """
    Build a symbolic Johansson four-tank model with pump and valve inputs.

    Parameters
    ----------
    A : sequence of float
        Four tank cross-sectional areas in cm^2.
    a : sequence of float
        Four discharge-hole areas in cm^2.
    K : sequence of float
        Two pump gains converting each command to volumetric flow on the
        cm^3/s balance scale.
    h_max : sequence of float
        Four maximum liquid heights in cm.

    Returns
    -------
    casadi.Function
        Function f(t, x, u) with four height states, four inputs and output dx
        of shape (4, 1).

    Notes
    -----
    States are ``[h1, h2, h3, h4]`` in cm; time is in seconds. The
    symbolic input is ``[pump1, pump2, valve1, valve2]``: pump commands use
    the 0-100 percent scale and valve fractions the 0-1 scale. This combines
    the numerical model's separate ``u`` and ``gamma`` arguments. Pump 1
    feeds tank 1 and crossed tank 4; pump 2 feeds tank 2 and crossed tank 3.
    Height clipping and full-tank derivative guards are piecewise expressions,
    which should be considered when selecting a linearization point.

    Examples
    --------
    >>> from benchmark_systems.symbolic import johansson
    >>> f = johansson(A=[50.0]*4, a=[0.2]*4, K=[1.0]*2, h_max=[30.0]*4)
    >>> f(0.0, [10.0]*4, [25.0, 25.0, 0.5, 0.5]).shape
    (4, 1)
    """
    g = Const.GRAVITY * 100   # cm/s²

    t       = _sym('t')
    h       = _sym('h', 4)
    u       = _sym('u', 4)
    gamma   = u[2:]
    pumps   = u[:2]

    h_clip = [ca.fmin(ca.fmax(h[i], Const.ZERO), h_max[i]) for i in range(4)]

    dh = ca.vertcat(
        -(a[0] / A[0]) * sqrt(2 * g * h_clip[0]) + (a[2] / A[0]) * sqrt(2 * g * h_clip[2]) + (1 - gamma[0]) * K[0] * pumps[0] / A[0],
        -(a[1] / A[1]) * sqrt(2 * g * h_clip[1]) + (a[3] / A[1]) * sqrt(2 * g * h_clip[3]) + (1 - gamma[1]) * K[1] * pumps[1] / A[1],
        -(a[2] / A[2]) * sqrt(2 * g * h_clip[2]) + gamma[1] * K[1] * pumps[1] / A[2],
        -(a[3] / A[3]) * sqrt(2 * g * h_clip[3]) + gamma[0] * K[0] * pumps[0] / A[3],
    )

    # If tanks are full, the can only be emptied (dh<0) or remain full (dh=0)
    dh_sat = ca.vertcat(*[
        ca.if_else(h[i] >= h_max[i], ca.fmin(dh[i], 0), dh[i])
        for i in range(4)
    ])
    return ca.Function('johansson', [t, h, u], [dh_sat],
                       ['t', 'x', 'u'], ['dx'])


# ---------------------------------------------------------------------------
# Batch Bioreactor
# ---------------------------------------------------------------------------

def batch_bioreactor(*,
                     S_in: float,
                     mu_max: float,
                     K_m: float,
                     K_i: float,
                     v: float,
                     Y_x: float,
                     Y_p: float) -> ca.Function:
    """
    Build a symbolic fed batch bioreactor with Haldane growth kinetics.

    Parameters
    ----------
    S_in : float
        Substrate concentration in the feed, in mol/L.
    mu_max : float
        Maximum specific growth rate, on the chosen kinetic time scale (1/min
        for the documented feed units).
    K_m : float
        Michaelis concentration constant, in the same mol/L scale as S.
    K_i : float
        Substrate-inhibition concentration constant, in the same mol/L scale
        as S.
    v : float
        Product-formation coefficient in 1/min on the documented mol/L and
        minute scale.
    Y_x : float
        Dimensionless biomass/substrate yield ratio on the source
        concentration scale.
    Y_p : float
        Dimensionless product/substrate yield ratio on the source
        concentration scale.

    Returns
    -------
    casadi.Function
        Function f(t, x, u) with named inputs t, x, u and output dx of shape
        (4, 1).

    Notes
    -----
    State order is ``[X, S, P, V]``: biomass, substrate and product
    concentrations in the source mol/L scale, followed by volume in m^3.
    Input ``u`` is feed flow in m^3/min; use minutes for time with this scale.
    Evaluate the function as ``f(t, x, u)``. Positive volume is required for
    dilution terms. Lower-bound clipping and derivative guards are part of
    the expressions; they do not project a solver's state onto constraints.
    """
    t  = _sym('t')
    xv = _sym('x', 4)
    u  = _sym('u')

    # Negative concentrations and/or volume have no physical meaning
    X, S, P, V = (ca.fmax(xv[0], Const.ZERO), ca.fmax(xv[1], Const.ZERO), ca.fmax(xv[2], Const.ZERO), ca.fmax(xv[3], Const.ZERO))

    mu = mu_max * S / (K_m + S + S**2 / K_i)

    dX = mu * X - u * X / V
    dS = -mu * X / Y_x - v * X / Y_p + u / V * (S_in - S)
    dP = v * X - u * P / V
    dV = u

    # If concentrations and/or volume are 0, their derivatives must be greater or equal to 0
    dx = ca.vertcat(*[
        ca.if_else(xv[i] <= 0, ca.fmax(dxi, 0), dxi)
        for i, dxi in enumerate([dX, dS, dP, dV])
    ])
    return ca.Function('batch_bioreactor', [t, xv, u], [dx],
                       ['t', 'x', 'u'], ['dx'])


# ---------------------------------------------------------------------------
# CSTR
# ---------------------------------------------------------------------------

def cstr(*,
         K0_ab: float, Ea_ab: float, Hr_ab: float,
         K0_bc: float, Ea_bc: float, Hr_bc: float,
         K0_ad: float, Ea_ad: float, Hr_ad: float,
         rho: float, Cp: float,
         Cp_k: float, m_k: float,
         A: float, V: float, T_in: float, K_w: float, Ca_in: float) -> ca.Function:
    """
    Build a symbolic stirred-tank reactor with three reaction pathways.

    Parameters
    ----------
    K0_ab : float
        A-to-B Arrhenius pre-exponential coefficient, in 1/h on the documented
        balance scale.
    Ea_ab : float
        A-to-B activation energy in kJ/mol.
    Hr_ab : float
        A-to-B reaction enthalpy in kJ/mol.
    K0_bc : float
        B-to-products Arrhenius pre-exponential coefficient, in 1/h on the
        documented balance scale.
    Ea_bc : float
        B-to-products activation energy in kJ/mol.
    Hr_bc : float
        B-to-products reaction enthalpy in kJ/mol.
    K0_ad : float
        A self-reaction Arrhenius coefficient in L/(mol h) for mol/L
        concentrations and hour-based balances; its reaction term multiplies
        Ca squared.
    Ea_ad : float
        A self-reaction activation energy in kJ/mol.
    Hr_ad : float
        A self-reaction enthalpy in kJ/mol.
    rho : float
        Reactor-content density in kg/L.
    Cp : float
        Reactor-content heat capacity in kJ/(kg K).
    Cp_k : float
        Jacket coolant heat capacity in kJ/(kg K).
    m_k : float
        Jacket coolant mass in kg.
    A : float
        Heat-transfer wall area in m^2.
    V : float
        Reactor volume in L.
    T_in : float
        Feed temperature in degrees Celsius.
    K_w : float
        Wall heat-transfer coefficient in kJ/(m^2 K h).
    Ca_in : float
        Feed concentration of species A in mol/L.

    Returns
    -------
    casadi.Function
        Function f(t, x, u) with four states, two inputs and output dx of
        shape (4, 1).

    Notes
    -----
    States are ``[Ca, Cb, Tr, Tk]``: concentrations in mol/L and
    reactor/jacket temperatures in degrees Celsius. Arrhenius expressions
    convert temperatures to kelvin internally. Input is ``[F, dQ]``.

    Preserve the source parameter scale: ``F`` multiplies concentration and
    temperature differences directly, without division by ``V``; ``dQ`` is
    added to wall heat flow without a power-unit conversion. With hour-based
    balances, F is a dilution rate in 1/h and dQ is heat flow in kJ/h. The
    self-reaction coefficient then has units L/(mol h). Positive dQ heats
    the jacket. Evaluate the function as ``f(t, x, u)`` after reconciling any
    dimensional plant data with this balance scale.
    """
    t  = _sym('t')
    xv = _sym('x', 4)
    u  = _sym('u', 2)

    Ca, Cb, Tr, Tk = xv[0], xv[1], xv[2], xv[3]
    F, dQ = u[0], u[1]

    Ca_s = ca.fmax(Ca, Const.ZERO)
    Cb_s = ca.fmax(Cb, Const.ZERO)

    k1 = K0_ab * exp(-Ea_ab / ((Tr + Const.KELVIN) * Const.IDEAL_GAS))
    k2 = K0_bc * exp(-Ea_bc / ((Tr + Const.KELVIN) * Const.IDEAL_GAS))
    k3 = K0_ad * exp(-Ea_ad / ((Tr + Const.KELVIN) * Const.IDEAL_GAS))

    dCa = F * (Ca_in - Ca_s) - k1 * Ca_s - k3 * Ca_s**2
    dCb = -F * Cb_s + k1 * Ca_s - k2 * Cb_s
    dTr = ((k1 * Ca_s * Hr_ab + k2 * Cb_s * Hr_bc + k3 * Ca_s**2 * Hr_ad)
           / (-rho * Cp)
           + F * (T_in - Tr)
           + (K_w * A * (Tk - Tr)) / (V * rho * Cp))
    dTk = (dQ + K_w * A * (Tr - Tk)) / (m_k * Cp_k)

    # If concentrations are 0, their derivatives must be greater or equal to 0.
    dx = ca.vertcat(*[
        ca.if_else(xv[i] <= 0, ca.fmax(dxi, 0), dxi)
        for i, dxi in enumerate([dCa, dCb])
    ], dTr, dTk)

    return ca.Function('cstr', [t, xv, u], [dx],
                       ['t', 'x', 'u'], ['dx'])


# ---------------------------------------------------------------------------
# Quadrotor (Newton-Euler)
# ---------------------------------------------------------------------------

def quadrotor(*,
              Ixx: float, Iyy: float, Izz: float,
              k: float, L: float, m: float, drag: float) -> ca.Function:
    """
    Build a symbolic Newton-Euler quadrotor ODE.

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
    casadi.Function
        Function f(t, x, u) with 12 states, four inputs and output dx of shape
        (12, 1).

    Notes
    -----
    Use the public import ``benchmark_systems.symbolic.quadrotor_ode``.
    State order is ``[x, y, z, phi, theta, psi, s, v, w, p, q, r]``:
    inertial positions in m, Euler angles in rad, body-frame linear speeds
    in m/s and body-frame angular rates in rad/s. Input contains the four
    squared rotor angular speeds, in (rad/s)^2 with compatible coefficients;
    time is in seconds. Evaluate the function as ``f(t, x, u)``. Euler
    coordinates are singular at cos(theta)=0; rotor saturation, rotor
    dynamics and aerodynamic resistance are not included.
    """
    t  = _sym('t')
    xv = _sym('x', 12)
    u  = _sym('u', 4)

    pos_x, pos_y, pos_z     = xv[0], xv[1], xv[2]
    phi, theta, psi         = xv[3], xv[4], xv[5]
    s, v_b, w_b             = xv[6], xv[7], xv[8]
    p, q_r, r               = xv[9], xv[10], xv[11]

    tau_phi   = L * k * (-u[1] + u[3])
    tau_theta = L * k * (-u[0] + u[2])
    tau_psi   = drag * (-u[0] + u[1] - u[2] + u[3])
    F         = k * (u[0] + u[1] + u[2] + u[3])

    cp, sp = ca.cos(phi),   ca.sin(phi)
    ct, st = ca.cos(theta), ca.sin(theta)
    cy, sy = ca.cos(psi),   ca.sin(psi)

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

    body_vel   = ca.vertcat(s, v_b, w_b)
    body_force = (R.T @ ca.vertcat(0, 0, -m * Const.GRAVITY)
                  + ca.vertcat(0, 0, F))
    fx, fy, fz = body_force[0], body_force[1], body_force[2]

    dxyz  = R @ body_vel
    deuler = ca.solve(W, ca.vertcat(p, q_r, r))
    dsbody = (ca.vertcat(r * v_b - q_r * w_b,
                         p * w_b - r * s,
                         q_r * s - p * v_b)
              + (1 / m) * ca.vertcat(fx, fy, fz))
    dangular = (ca.vertcat((Iyy - Izz) / Ixx,
                            (Izz - Ixx) / Iyy,
                            (Ixx - Iyy) / Izz)
                * ca.vertcat(q_r * r, p * r, p * q_r)
                + ca.vertcat(tau_phi / Ixx,
                             tau_theta / Iyy,
                             tau_psi / Izz))

    dx = ca.vertcat(dxyz, deuler, dsbody, dangular)
    return ca.Function('quadrotor', [t, xv, u], [dx],
                       ['t', 'x', 'u'], ['dx'])
