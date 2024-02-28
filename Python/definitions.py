import mpmath
from mpmath import mp
from typing import List

# Pauli matrices and projector |+><+|
x = mp.matrix([[0, 1], [1, 0]])
y = mp.matrix([[0, -1j], [1j, 0]])
z = mp.matrix([[1, 0], [0, -1]])
one = mp.eye(2)
projx = (one + x) / 2
rcp_sqrt2 = 1 / mp.sqrt(2)
rcp_sqrt8 = 1 / mp.sqrt(8)

# Density matrices of pure |+> states, pure magic states and pure CCZ states
plusstate = (
    mp.matrix([[rcp_sqrt2], 
              [rcp_sqrt2]]) 
    * 
    mp.matrix([[rcp_sqrt2, rcp_sqrt2]])
)


magicstate = (
    mp.matrix([[rcp_sqrt2], 
              [mp.exp(1j * mp.pi / 4) * rcp_sqrt2]]) 
    * 
    mp.matrix([[rcp_sqrt2, mp.exp(-1j * mp.pi / 4) * 1 / mp.sqrt(2)]])
)

CCZstate = mp.matrix(
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
    ) * mp.matrix(
        [
            [rcp_sqrt8, rcp_sqrt8, rcp_sqrt8, rcp_sqrt8, rcp_sqrt8, rcp_sqrt8, rcp_sqrt8, -rcp_sqrt8]
        ]
    )



def kron(*args: mpmath.matrix) -> mpmath.matrix:
    """
    Calculates the tensor products of 2 or more matrices
    """
    new_rows: int = 1
    new_cols: int = 1
    for m in args:
        new_rows *= m.rows
        new_cols *= m.cols
    res = mp.ones(new_rows, new_cols)
    for i in range(new_rows):
        for j in range(new_cols):
            partition_rows: int = 1
            partition_cols: int = 1
            for m in args:
                partition_rows *= m.rows
                partition_cols *= m.cols
                res[i, j] *= m[
                        int(i * partition_rows / new_rows) % m.rows, 
                        int(j * partition_cols / new_cols) % m.cols
                    ]
    return res
    
def trace(m: mpmath.matrix) -> mpmath.mpc:
    res: int = 0
    for i in range(min(m.rows,m.cols)):
        res += m[i,i]
    return res


# Density matrices of 5, 7 and 4 |+> states
init5qubit = kron(plusstate, plusstate, plusstate, plusstate, plusstate)

init7qubit = kron(plusstate, plusstate, plusstate, plusstate, plusstate, plusstate, plusstate)
init4qubit = kron(plusstate, plusstate, plusstate, plusstate)

# Density matrices corresponding to the ideal output state of 15-to-1, 20-to-4 and 8-to-CCZ
ideal15to1 = kron(magicstate, plusstate, plusstate, plusstate, plusstate)
ideal20to4 = kron(magicstate, magicstate, magicstate, magicstate, plusstate, plusstate, plusstate)
ideal8toCCZ = kron(CCZstate, plusstate)


def pauli_rot(axis: List[mpmath.matrix], angle: mpmath.mpc) -> mpmath.matrix:
    """
    Pauli product rotation `e^(iP*phi)`, where the Pauli product `P` is specified by 'axis' and `phi` is the rotation angle
    """
    return mp.cos(angle) * mp.eye(2 ** len(axis)) + 1j * mp.sin(angle) * kron(*axis)



def apply_rot(state: mpmath.matrix, axis: List[mpmath.matrix], p1: mpmath.mpc, p2: mpmath.mpc, p3: mpmath.mpc) -> mpmath.matrix:
    """
    Applies a `pi/8` Pauli product rotation specified by 'axis' with probability `1-p1-p2-p3`
    
    A `P_(pi/2) / P_(-pi/4) / P_(pi/4)` error occurs with probability `p1 / p2 / p3`
    """
    rot0 = pauli_rot(axis, mp.pi / 8)
    rot1 = pauli_rot(axis, 5 * mp.pi / 8)
    rot2 = pauli_rot(axis, -1 * mp.pi / 8)
    rot3 = pauli_rot(axis, 3 * mp.pi / 8)

    return (
            (1 - p1 - p2 - p3) * rot0 * state * rot0.transpose_conj()
            + p1               * rot1 * state * rot1.transpose_conj()
            + p2               * rot2 * state * rot2.transpose_conj()
            + p3               * rot3 * state * rot3.transpose_conj()
    )

def apply_pauli(state: mpmath.matrix, pauli: List[mpmath.matrix], p: float) -> mpmath.matrix:
    """
    Applies a Pauli operator to a state with probability `p`
    """
    return (1 - p) * state + p * kron(*pauli) * state * kron(*pauli)


def plog(pphys: mpmath.mpf, d: int) -> mpmath.mpc:
    """
    Estimate of the logical error rate of a surface-code patch with code distance `d` and circuit-level error rate `pphys`
    """
    return 0.1 * (100 * pphys) ** ((d + 1) / 2)


def storage_x_4(state, p1, p2, p3, p4):
    """
    For the 8-to-CCZ protocol, applies X/Z storage errors to qubits 1-4 with probabilities p1-p4
    """
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



def storage_x_5(state, p1, p2, p3, p4, p5):
    """
    For the 15-to-1 protocol, applies X/Z storage errors to qubits 1-5 with probabilities p1-p5
    """
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


def storage_x_7(state, p1, p2, p3, p4, p5, p6, p7):
    """
    For the 20-to-4 protocol, applies X/Z storage errors to qubits 1-7 with probabilities p1-p7
    """
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
