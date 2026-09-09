from ._core import AutoAnimation
from .furuta import FurutaPendulum
from .mechanical import (
    CartPendulum,
    DCMotor,
    MultiCartPendulum,
    MultimassSpring,
    Pendulum,
)
from .processes import BatchBioreactor, CSTR, JohanssonTanks, OilWell, pHNeutralization
from .quadrotor import Quadrotor

__all__ = [
    "AutoAnimation",
    "Pendulum",
    "DCMotor",
    "CartPendulum",
    "FurutaPendulum",
    "MultiCartPendulum",
    "MultimassSpring",
    "JohanssonTanks",
    "BatchBioreactor",
    "CSTR",
    "OilWell",
    "pHNeutralization",
    "Quadrotor",
]
