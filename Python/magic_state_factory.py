from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class MagicStateFactory:
    name: str
    distilled_magic_state_error_rate: float # Output
    space: Tuple[int, int] 
    qubits: int # qubits
    distillation_time_in_cycles: float # code cycles
    n_t_gates_produced_per_distillation: int = 1 # 1 for 15 to 1, 4 for 20 to 4
