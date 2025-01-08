from typing import Sequence
import numpy as np
from .common import Const

def pendulum(t, x, *,
             m: float, L: float, drag: float = 0.0, u: float = 0.0) -> np.ndarray:
    '''
    Just a humble pendulum.

    Parameters
    ----------
    t : float
        Time.
    x : Sequence[float]
        State variables [theta, omega]. Theta = pi corresponds to the pendulum pointing up.
    m : float
        Mass of the pendulum.
    L : float
        Length of the pendulum.
    drag : float, optional
        Drag coefficient. Default is 0.0.
    u : float, optional
        Torque applied to the pendulum. Default is 0.0.
    '''

    g = Const.GRAVITY

    # State space
    dx = np.zeros(2)
    dx[0] = x[1]
    dx[1] = -g/L * np.sin(x[0]) - drag/m*x[1] + 1/(m*L**2)*u

    return dx

def dc_motor(t, x, *,
             J: float, b: float, Ke: float, Kt: float,  R: float, L: float, u: float = 0.0) -> np.ndarray:
    '''
    DC motor expressions obtained from Newton's second law and Kirchhoff's voltage law.

    Parameters
    ----------
    t : float
        Time.
    x : Sequence[float]
        State variables [theta, omega, current].
    J : float
        Moment of inertia of the rotor (in kg*m^2).
    b : float
        Viscous friction coefficient (in N*m*s/rad).
    Ke : float
        Electromotive force constant (in V*s/rad).
    Kt : float
        Torque constant (in N*m/A).
    R : float
        Armature resistance (in Ohm).
    L : float
        Armature inductance (in H).
    '''

    # State space
    dx = np.zeros(3)
    dx[0] = x[1]
    dx[1] = -b/J*x[1] + Kt/J*x[2]
    dx[2] = -Ke/L*x[1] - R/L*x[2] + 1/L*u

    return dx

def cart_pendulum(t, x, *,
            m: float, M: float, L: float, drag: float = 0.0, u: float = 0.0) -> np.ndarray:
    '''
    Cart-pendulum expressions obtained from Euler-Lagrange equations and Rayleigh dissipation equation.

    Parameters
    ----------
    t : float
        Time.
    x : Sequence[float]
        State variables [x, dx, theta, omega]. Theta = pi corresponds to the pendulum pointing up.
    m : float
        Mass of the pendulum.
    M : float
        Mass of the cart.
    L : float
        Length of the pendulum.
    drag : float, optional
        Drag coefficient. Default is 0.0.
    u : float, optional
        Force applied to the cart. Default is 0.0.
    '''
    sin_x3 = np.sin(x[2])
    cos_x3 = np.cos(x[2])
    den = -M + m*cos_x3**2 - m
    g = Const.GRAVITY

    # State space
    dx = np.zeros(4)
    dx[0] = x[1]
    dx[1] = (-L*m*x[3]**2*sin_x3 + drag*x[1] - g*m*sin_x3*cos_x3 - u) / den
    dx[2] = x[3]
    dx[3] = (L*m*x[3]**2*sin_x3*cos_x3 - drag*x[1]*cos_x3 + (M+m)*g*sin_x3 + u*cos_x3) / (L*den)

    return dx 


def spring_mass_damper(t, x, *,
                       I: Sequence[float], K: Sequence[float], d: Sequence[float], u: Sequence[float] = [0.0, 0.0]) -> np.ndarray:
    '''
    Spring-mass-damper expressions obtained from Euler-Lagrange equations.
    The equations are equivalent to the ones presented in https://www.do-mpc.com/en/latest/getting_started.html

    Parameters
    ----------
    t : float
        Time.
    x : Sequence[float]
        State variables [theta1, theta2 ... thetaN, omega1, omega2 ... omegaN]. ThetaN is the position (in radians) of the N-th mass,
        and omegaN is the velocity (in rad/s) of the N-th mass.
    I : Sequence[float]
        Inertia of the masses.
    K : Sequence[float]
        Spring constants.
    d : Sequence[float]
        Damping coefficients.
    u : Sequence[float], optional
        Stepper motor angles (in radians). Default is [0.0, 0.0].
    '''

    n = len(I) # Number of masses

    # State space
    dx = np.zeros(2*n)
    for i in range(n):
        dx[i] = x[i+n] # dTheta/dt = omega
        if i == 0: # First acceleration equation
            dx[i+n] = (-K[i]*(x[i] - u[0]) - K[i+1]*(x[i] - x[i+1]) - d[i]*x[i+n]) / I[i]
        elif i == n-1: # Last acceleration equation
            dx[i+n] = (-K[i]*(x[i] - x[i-1]) - K[i+1]*(x[i] - u[-1]) - d[i]*x[i+n]) / I[i]
        else: # Acceleration equations for the masses in between
            dx[i+n] = (-K[i]*(x[i] - x[i-1]) - K[i+1]*(x[i] - x[i+1]) - d[i]*x[i+n]) / I[i]

    return dx

def johansson(t, h, *, 
              u: Sequence[float], gamma: Sequence[float],
              A: Sequence[float], a: Sequence[float], K: Sequence[float], h_max: Sequence[float]) -> np.ndarray:
    '''
    K.H Johansson - DOI: 10.1109/87.845876
    
    Three-way valves work inversely in the current implementation.

    Parameters
    ----------
    t : float
        Time.
    h : Sequence[float]
        Water heights (in cm)
    u : Sequence[float]
        Pump setpoints (0-100%).
    gamma : Sequence[float]
        Valve openings (0-1).
    A : Sequence[float]
        Base areas of tanks.
    a : Sequence[float]
        Area of discharge holes.
    K : Sequence[float]
        Pump constants.
    h_max : Sequence[float]
        Tank saturations (in cm).
    '''

    g = Const.GRAVITY * 100 # cm/s^2

    # Tank saturations
    h_clip = [np.clip(h[i], Const.ZERO, h_max[i]) for i in range(4)]

    # State space
    dh = np.zeros(4)
    dh[0] = -(a[0]/A[0]) * np.sqrt(2*g*h_clip[0]) + (a[2]/A[0]) * np.sqrt(2*g*h_clip[2]) + (1-gamma[0])*K[0]*u[0]/A[0]
    dh[1] = -(a[1]/A[1]) * np.sqrt(2*g*h_clip[1]) + (a[3]/A[1]) * np.sqrt(2*g*h_clip[3]) + (1-gamma[1])*K[1]*u[1]/A[1]
    dh[2] = -(a[2]/A[2]) * np.sqrt(2*g*h_clip[2]) + gamma[1]*K[1]*u[1]/A[2]
    dh[3] = -(a[3]/A[3]) * np.sqrt(2*g*h_clip[3]) + gamma[0]*K[0]*u[0]/A[3]

    # If tanks are full, they can only be emptied (dh<0) or remain full (dh=0)
    for i in range(4):
        dh[i] = dh[i] if h_clip[i] < h_max[i] else min(Const.ZERO, dh[i])

    return dh

def batch_bioreactor(t, x, *,
                     S_in: float, mu_max: float, K_m: float, K_i: float, v: float, Y_x: float, Y_p: float, u: float = 0.0) -> np.ndarray:
    '''
    Batch bioreactor expressions obtained from mass balances and empirical models for substrate inhibition in bioreactors.
    The equations are the same presented in https://www.do-mpc.com/en/latest/example_gallery/batch_reactor.html.
    The empricial model is based on the Haldane equation (see https://en.wikipedia.org/wiki/Substrate_inhibition_in_bioreactors).

    Generic parameters
    ------------------
    t : float
        Time.
    x : Sequence[float]
        State variables [X, S, P, V]. X is the biomass concentration (in mol/L), S is the substrate concentration (in mol/L),
        P is the product concentration (in mol/L) and V is the volume of the bioreactor (in m^3).
    
    Haldane equation parameters
    ---------------------------
    mu_max : float
        Maximum specific growth rate.
    K_m : float
        Michaelis constant.
    K_i : float
        Inhibition constant.

    Bioreactor parameters
    ---------------------
    S_in : float
        Substrate concentration in the feed flow (in mol/L).
    v : float
        Rate of product formation.
    Y_x : float
        Yield coefficient of biomass.
    Y_p : float
        Yield coefficient of product.
    u : float
        Feed flow rate (in m^3/min).
    '''

    # Negative concentrations and/or volume have no physical meaning
    x = np.clip(x, Const.ZERO, None)

    X, S, P, V = x

    # Haldane equation
    mu = mu_max * S / (K_m + S + S**2/K_i)

    # State space
    dX = mu*X - u*X/V
    dS = -mu*X/Y_x - v*X/Y_p + u/V*(S_in - S)
    dP = v*X - u*P/V
    dV = u

    # If concentrations and/or volume are 0, their derivatives must be greater or equal to 0
    dx = np.array([dX, dS, dP, dV])
    for i in range(4):
        dx[i] = dx[i] if x[i] > Const.ZERO else max(dx[i], Const.ZERO)

    return dx

def cstr(t, x, *,
         K0_ab: float, Ea_ab: float, Hr_ab: float, 
         K0_bc: float, Ea_bc: float, Hr_bc: float,
         K0_ad: float, Ea_ad: float, Hr_ad: float,
         rho: float, Cp: float,
         Cp_k: float, m_k: float,
         A: float, V: float, T_in: float, K_w: float, Ca_in: float,
         u: Sequence[float] = [0.0, 0.0]) -> np.ndarray:
    '''
    Continuous stirred-tank reactor (CSTR) expressions obtained from mass balances, energy balances and empirical models for the reaction kinetics (Arrhenius equation).
    The equations are the same presented in https://www.do-mpc.com/en/latest/example_gallery/CSTR.html.

    Generic parameters
    ------------------
    t : float
        Time.
    x : Sequence[float]
        State variables [Ca, Cb, Tr, Tk]. Ca is the concentration of species A (in mol/L), Cb is the concentration of species B (in mol/L),
        Tr is the reactor temperature (in ºC) and Tk is the cooling jacket temperature (in ºC).

    Arrhenius parameters:
    --------------------   
    K0_ab : float
        Pre-exponential factor for the reaction A -> B (in 1/h).
    Ea_ab : float
        Activation energy for the reaction A -> B (in kJ/mol).
    Hr_ab : float
        Heat of reaction for the reaction A -> B (in kJ/mol).
    K0_bc : float
        Pre-exponential factor for the reaction B -> others (in 1/h).
    Ea_bc : float
        Activation energy for the reaction B -> others (in kJ/mol).
    Hr_bc : float
        Heat of reaction for the reaction B -> others (in kJ/mol).
    K0_ad : float
        Pre-exponential factor for the reaction of A with itself (in 1/h).
    Ea_ad : float
        Activation energy for the reaction of A with itself (in kJ/mol).
    Hr_ad : float
        Heat of reaction for the reaction of A with itself (in kJ/mol).

    Cooling parameters:
    -------------------
    Cp_k : float
        Heat capacity of the coolant (in kJ/(kg*K)).
    m_k : float
        Mass of the coolant (in kg).
    
    Reactor parameters:
    --------------------    
    rho : float
        Density of the reactor contents (in kg/L).
    Cp : float
        Heat capacity of the reactor contents (in kJ/(kg*K)).
    A : float
        Area of the reactor wall (in m^2).
    V : float
        Volume of the reactor (in L).
    T_in : float
        Temperature of the feed flow (in ºC).
    K_w : float
        Heat transfer coefficient (in kJ/(m^2*K*h)).
    Ca_in : float
        Concentration of species A in the feed flow (in mol/L).
    u : Sequence[float], optional
        Inputs [F, dQ]. F is the flow rate (in L/h) and dQ is the heat input (in kW). Default is [0.0, 0.0].
    '''

    Ca, Cb, Tr, Tk = x
    F, dQ = u

    # Negative concentrations have no physical meaning.
    Ca = max(Ca, Const.ZERO)
    Cb = max(Cb, Const.ZERO)

    # Arrhenius equations
    k1 = K0_ab * np.exp(-Ea_ab / ((Tr + Const.KELVIN) * Const.IDEAL_GAS))
    k2 = K0_bc * np.exp(-Ea_bc / ((Tr + Const.KELVIN) * Const.IDEAL_GAS))
    k3 = K0_ad * np.exp(-Ea_ad / ((Tr + Const.KELVIN) * Const.IDEAL_GAS))

    # State space
    dCa = F*(Ca_in - Ca) - k1*Ca - k3*Ca**2
    dCb = -F*Cb + k1*Ca - k2*Cb
    dTr = (k1*Ca*Hr_ab + k2*Cb*Hr_bc + k3*Ca**2*Hr_ad) / (-rho*Cp) + F*(T_in - Tr) + (K_w * A * (Tk - Tr)) / (V * rho * Cp)
    dTk = (dQ + K_w * A * (Tr - Tk)) / (m_k * Cp_k)

    # If concentrations are 0, their derivatives must be greater or equal to 0.
    dx = np.array([dCa, dCb, dTr, dTk])
    for i in range(2):
        dx[i] = dx[i] if x[i] > Const.ZERO else max(dx[i], Const.ZERO)

    return dx