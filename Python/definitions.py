import numpy as np
from typing import List
from numpy.typing import NDArray

chpf = np.clongdouble # Complex high-precision float

pi = chpf(np.pi)

# Pauli matrices and projector |+><+|
x = np.array([[0, 1], [1, 0]], dtype=chpf)
y = np.array([[0, -1j], [1j, 0]], dtype=chpf)
z = np.array([[1, 0], [0, -1]], dtype=chpf)
one = np.eye(2, dtype=chpf)
projx = (one + x) / 2
rcp_sqrt2 = 1 / np.sqrt(2, dtype=chpf)
rcp_sqrt8 = 1 / np.sqrt(8, dtype=chpf)

# Density matrices of pure |+> states, pure magic states and pure CCZ states
plusstate = (
    np.array([[rcp_sqrt2], 
              [rcp_sqrt2]]) 
    @ 
    np.array([[rcp_sqrt2, rcp_sqrt2]])
)


magicstate = (
    np.array([[rcp_sqrt2], 
              [np.exp(1j * pi / 4) * rcp_sqrt2]]) 
    @ 
    np.array([[rcp_sqrt2, np.exp(-1j * pi / 4) * 1 / np.sqrt(2)]])
)

CCZstate = (
    np.array(
        [
            [rcp_sqrt8],
            [rcp_sqrt8],
            [rcp_sqrt8],
            [rcp_sqrt8],
            [rcp_sqrt8],
            [rcp_sqrt8],
            [rcp_sqrt8],
            [-rcp_sqrt8],
        ]
    )
    @
    np.array(
        [
            [rcp_sqrt8, rcp_sqrt8, rcp_sqrt8, rcp_sqrt8, rcp_sqrt8, rcp_sqrt8, rcp_sqrt8, -rcp_sqrt8]
        ]
    ),
)


def tensor_product(*args) -> NDArray[chpf]:
    """
    Computes the tensor product of a list of matrices
    """
    res = args[0]
    for mat in args[1:]:
        res = np.kron(res, mat)
    return res


# Density matrices of 5, 7 and 4 |+> states
init5qubit = tensor_product(plusstate, plusstate, plusstate, plusstate, plusstate)

init7qubit = tensor_product(plusstate, plusstate, plusstate, plusstate, plusstate, plusstate, plusstate)
init4qubit = tensor_product(plusstate, plusstate, plusstate, plusstate)

# Density matrices corresponding to the ideal output state of 15-to-1, 20-to-4 and 8-to-CCZ
ideal15to1 = tensor_product(magicstate, plusstate, plusstate, plusstate, plusstate)
ideal20to4 = tensor_product(magicstate, magicstate, magicstate, magicstate, plusstate, plusstate, plusstate)
ideal8toCCZ = tensor_product(CCZstate, plusstate)


def pauli_rot(axis: List[NDArray[chpf]], angle: chpf) -> NDArray[chpf]:
    """
    Pauli product rotation `e^(iP*phi)`, where the Pauli product `P` is specified by 'axis' and `phi` is the rotation angle
    """
    return np.cos(angle) * np.eye(2 ** len(axis)) + 1j * np.sin(angle) * tensor_product(*axis)



def apply_rot(state: NDArray[chpf], axis: List[NDArray[chpf]], p1: chpf, p2: chpf, p3: chpf) -> NDArray[chpf]:
    """
    Applies a `pi/8` Pauli product rotation specified by 'axis' with probability `1-p1-p2-p3`
    
    A `P_(pi/2) / P_(-pi/4) / P_(pi/4)` error occurs with probability `p1 / p2 / p3`
    """
    rot0 = pauli_rot(axis, pi / 8)
    rot1 = pauli_rot(axis, 5 * pi / 8)
    rot2 = pauli_rot(axis, -1 * pi / 8)
    rot3 = pauli_rot(axis, 3 * pi / 8)

    return (
            (1 - p1 - p2 - p3) * rot0 @ state @ rot0.conj().transpose()
            + p1               * rot1 @ state @ rot1.conj().transpose()
            + p2               * rot2 @ state @ rot2.conj().transpose()
            + p3               * rot3 @ state @ rot3.conj().transpose()
    )

def apply_pauli(state: NDArray[chpf], pauli: List[NDArray[chpf]], p: float) -> NDArray[chpf]:
    """
    Applies a Pauli operator to a state with probability `p`
    """
    return (1 - p) * state + p * tensor_product(*pauli) @ state @ tensor_product(*pauli)


def plog(pphys: chpf, d: int) -> chpf:
    """
    Estimate of the logical error rate of a surface-code patch with code distance `d` and circuit-level error rate `pphys`
    """
    return 0.1 * (100 * pphys) ** ((d + 1) / 2)


# For the 8-to-CCZ protocol, applies X/Z storage errors to qubits 1-4 with probabilities p1-p4
def storage_x_4(state, p1, p2, p3, p4):
    res = apply_pauli(state, [x, one, one, one], p1)
    res = apply_pauli(res, [one, x, one, one], p2)
    res = apply_pauli(res, [one, one, x, one], p3)
    res = apply_pauli(res, [one, one, one, x], p4)
    return res


def storage_z_4(state, p1, p2, p3, p4):
    res = apply_pauli(state, [z, one, one, one], p1)
    res = apply_pauli(res, [one, z, one, one], p2)
    res = apply_pauli(res, [one, one, z, one], p3)
    res = apply_pauli(res, [one, one, one, z], p4)
    return res


# For the 15-to-1 protocol, applies X/Z storage errors to qubits 1-5 with probabilities p1-p5
def storage_x_5(state, p1, p2, p3, p4, p5):
    res = apply_pauli(state, [x, one, one, one, one], p1)
    res = apply_pauli(res, [one, x, one, one, one], p2)
    res = apply_pauli(res, [one, one, x, one, one], p3)
    res = apply_pauli(res, [one, one, one, x, one], p4)
    res = apply_pauli(res, [one, one, one, one, x], p5)
    return res


def storage_z_5(state, p1, p2, p3, p4, p5):
    res = apply_pauli(state, [z, one, one, one, one], p1)
    res = apply_pauli(res, [one, z, one, one, one], p2)
    res = apply_pauli(res, [one, one, z, one, one], p3)
    res = apply_pauli(res, [one, one, one, z, one], p4)
    res = apply_pauli(res, [one, one, one, one, z], p5)
    return res


# For the 20-to-4 protocol, applies X/Z storage errors to qubits 1-7 with probabilities p1-p7
def storage_x_7(state, p1, p2, p3, p4, p5, p6, p7):
    res = apply_pauli(state, [x, one, one, one, one, one, one], p1)
    res = apply_pauli(res, [one, x, one, one, one, one, one], p2)
    res = apply_pauli(res, [one, one, x, one, one, one, one], p3)
    res = apply_pauli(res, [one, one, one, x, one, one, one], p4)
    res = apply_pauli(res, [one, one, one, one, x, one, one], p5)
    res = apply_pauli(res, [one, one, one, one, one, x, one], p6)
    res = apply_pauli(res, [one, one, one, one, one, one, x], p7)
    return res


def storage_z_7(state, p1, p2, p3, p4, p5, p6, p7):
    res = apply_pauli(state, [z, one, one, one, one, one, one], p1)
    res = apply_pauli(res, [one, z, one, one, one, one, one], p2)
    res = apply_pauli(res, [one, one, z, one, one, one, one], p3)
    res = apply_pauli(res, [one, one, one, z, one, one, one], p4)
    res = apply_pauli(res, [one, one, one, one, z, one, one], p5)
    res = apply_pauli(res, [one, one, one, one, one, z, one], p6)
    res = apply_pauli(res, [one, one, one, one, one, one, z], p7)
    return res
