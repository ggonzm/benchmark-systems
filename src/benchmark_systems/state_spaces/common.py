from enum import Enum

class Const(float, Enum):
    '''
    Generic constants for the state spaces.
    '''
    ZERO = 1e-40 # Perfect zero for numerical simulations
    IDEAL_GAS = 8.314e-3 # kJ/(mol*K)
    GRAVITY = 9.81 # m/s^2
    KELVIN = 273.15 # K
