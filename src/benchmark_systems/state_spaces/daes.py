import numpy as np
from numpy import sin, cos
from .common import Const

def double_cart_pendulum(t, x, x_dot, *,
                         M: float, m1: float, m2: float, L1: float, L2: float, u: float = 0.0) -> np.ndarray:
    '''
    Double cart-pendulum expressions obtained from Euler-Lagrange equations. No drag is considered.
    The equations are the same presented in https://www.do-mpc.com/en/latest/example_gallery/DIP.html,
    substituting theta1 and theta2 by pi - theta1 and pi - theta2, respectively.

    Parameters
    ----------
    t : float
        Time.
    x : Sequence[float]
        State variables [x, theta1, theta2, dx, omega1, omega2]. ThetaN = pi corresponds to the pendulum pointing up.
    x_dot : Sequence[float]
        Derivatives of the state variables [dx, omega1, omega2, ddx, alpha1, alpha2].
    M : float
        Mass of the cart.
    m1 : float
        Mass of the first pendulum.
    m2 : float
        Mass of the second pendulum.
    L1 : float
        Length of the first pendulum.
    L2 : float
        Length of the second pendulum.
    u : float, optional
        Force applied to the cart. Default is 0.0.
    '''
    x, theta1, theta2, dx, omega1, omega2 = x
    x_dot, theta1_dot, theta2_dot, ddx, alpha1, alpha2 = x_dot

    g = Const.GRAVITY
    l1 = L1/2 # Half of the length of the first pendulum
    l2 = L2/2 # Half of the length of the second pendulum
    J1 = (m1 * l1**2) / 3 # Inertia of the first pendulum
    J2 = (m2 * l2**2) / 3 # Inertia of the second pendulum

    # Some parameters to keep the equations clean
    p1 = M + m1 + m2
    p2 = m1*l1 + m2*L1
    p3 = m2*l2
    p4 = m1*l1**2 + m2*L1**2 + J1
    p5 = m2*l2*L1
    p6 = m2*l2**2 + J2
    p7 = (m1*l1 + m2*L1) * g
    p8 = m2*l2*g

    
    F = np.zeros(6)
    # ODEs ... Relation between the state variables and their derivatives
    F[0] = dx - x_dot
    F[1] = omega1 - theta1_dot
    F[2] = omega2 - theta2_dot
    # Euler-Lagrange equations ... All terms in one side of the equations (0 = g(x, z))
    F[3] =  p1*ddx + p2*alpha1*cos(theta1) + p3*alpha2*cos(theta2) - (
        p2*omega1**2*sin(theta1) + p3*omega2**2*sin(theta2) + u)
    F[4] = -p2*cos(theta1)*ddx - p4*alpha1 - p5*alpha2*cos(theta1 - theta2) - (
        p7*sin(theta1) + p5*omega2**2*sin(theta1 - theta2))
    F[5] = -p3*cos(theta2)*ddx - p5*alpha1*cos(theta1 - theta2) - p6*alpha2 - (
        -p5*omega1**2*sin(theta1 - theta2) + p8*sin(theta2))
    
    return F
