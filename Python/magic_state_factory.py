from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class MagicStateFactory:
    name: str
    distilled_magic_state_error_rate: float # Output
    qubits: int # qubits
    distillation_time_in_cycles: float # code cycles
    n_t_gates_produced_per_distillation: int = 1 # 1 for 15 to 1, 4 for 20 to 4
    def __repr__(self):
        return f"""
{self.name}
Output error: {'{:.1e}'.format(self.distilled_magic_state_error_rate)}
Qubits: {self.qubits}
Code cycles: {self.distillation_time_in_cycles:.1f}
T-gates per distillation: {self.n_t_gates_produced_per_distillation}
Qubitcycles: {int(self.qubits*self.distillation_time_in_cycles/self.n_t_gates_produced_per_distillation)}
        """
