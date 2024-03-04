import numpy as np
from dataclasses import dataclass


@dataclass
class OneLevelFactory:
    pphys: float
    dx: int
    dz: int
    dm: int
    error: np.float128
    failure_probability: float

    def qubits(self) -> int:
        raise NotImplementedError

    def code_cycles(self) -> np.float128:
        raise NotImplementedError

    def spacetime_cost(self) -> np.float128:
        raise NotImplementedError
