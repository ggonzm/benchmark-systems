"""
CasADi symbolic formulations of all ODE benchmark models.

Each function returns a CasADi Function  f(t, x, u, p)  that evaluates the
right-hand side  ẋ = f(...)  of the ODE.  The same kwargs interface used in
the scipy models is replicated so that callers can swap backends with minimal
changes.

All physical parameters are collected into a named CasADi parameter vector
``p`` when building the function, enabling automatic differentiation, symbolic
manipulation (linearisation, sensitivity analysis…) and direct use with
CasADi integrators.

Typical usage
-------------
>>> import casadi as ca
>>> from benchmark_systems.symbolic.odes import pendulum
>>> f = pendulum(m=1.0, L=2.0, drag=0.0)
>>> # Numerical evaluation
>>> x0 = [3.04, 0.0]
>>> f_val = f(0, x0, 0)          # returns DM column vector
>>> # Build an integrator (RK4, 1000 steps over 20 s)
>>> integrator = pendulum_integrator(m=1.0, L=2.0, drag=0.0, T=20.0, N=1000)
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
    Simple pendulum.

    States : [theta, omega]   (theta=pi → pointing up)
    Inputs : u  – torque (scalar)

    Returns a CasADi Function  f(t, x, u)  with signature
        f : (1, 2, 1) → DM(2)
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
    DC motor.

    States : [theta, omega, current]
    Inputs : u – voltage (V)
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
    Cart-pendulum (Euler-Lagrange, no inertia, point mass).

    States : [x, dx, theta, omega]   (theta=pi → pointing up)
    Inputs : u – force on cart (N)
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
    Furuta pendulum (rotary inverted pendulum).

    States : [theta1, theta2, omega1, omega2]   (theta2=pi → pointing up)
    Inputs : u – DC Voltage (V)
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
    N discs connected via springs, driven by two stepper motors.

    States : [u1_true, u2_true, theta_1, …, theta_N, omega_1, …, omega_N]
    Inputs : u – [setpoint_motor1, setpoint_motor2]
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
    Johansson four-tank system.

    States : [h1, h2, h3, h4]  (water heights, cm)
    Inputs : u – [pump1 (0-100 %), pump2 (0-100 %), valve1 (0-1), valve2 (0-1)]
    """
    g = Const.GRAVITY * 100   # cm/s²

    t       = _sym('t')
    h       = _sym('h', 4)
    u       = _sym('u', 4)
    gamma   = u[2:]
    pumps   = u[:2]

    h_clip = [ca.fmax(h[i], 0.0) for i in range(4)]

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
    Batch bioreactor (Haldane kinetics).

    States : [X, S, P, V]
    Inputs : u – feed flow rate (m³/min)
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
    Continuous Stirred Tank Reactor (Arrhenius kinetics, 3 reactions).

    States : [Ca, Cb, Tr, Tk]
    Inputs : u – [F (L/h), dQ (kW)]
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
    Quadrotor – Newton-Euler formulation (ODE).

    States : [x, y, z, phi, theta, psi, s, v, w, p, q, r]
             (position, Euler angles, body-frame linear & angular velocities)
    Inputs : u – [w1², w2², w3², w4²]  (squared propeller speeds)
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
    dsbody = (ca.vertcat(q_r * v_b - r * w_b,
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
