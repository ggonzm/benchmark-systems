from enum import Enum
import numpy as np

class Const(float, Enum):
    '''
    Generic constants for the state spaces.
    '''
    ZERO = 1e-40 # Perfect zero for numerical simulations
    IDEAL_GAS = 8.314e-3 # kJ/(mol*K)
    GRAVITY = 9.81 # m/s^2
    KELVIN = 273.15 # K
    BAR_2_PASCAL = 1e5 # 1 bar = 1e5 Pa

def _cyclic(var, bounds: tuple[float, float]):
    '''
    Helper function to ensure cyclic variables are within bounds.

    Keep in mind tolerances when integrating these variables.
    '''
    return np.mod(var - bounds[0], bounds[1] - bounds[0]) + bounds[0]