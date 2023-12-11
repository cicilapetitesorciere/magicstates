from factory import OneLevelFactory
from dataclasses import dataclass
# from scipy import optimize
import mpmath as mp

from definitions import (
    z,
    one,
    projx,
    apply_rot,
    kron,
    trace,
    plog,
    storage_x_5,
    storage_z_5,
    init5qubit,
    ideal15to1
)

# @dataclass
# class OneLevelFactory15To1(OneLevelFactory):

#     def qubits(self) -> int:
#          return 2 * ((self.dx + 4 * self.dz) * 3 * self.dx + 2 * self.dm)
    
#     def code_cycles(self) -> np.float128:
#         return np.float128(6*self.dm) / (1 - self.failure_probability)

#     def spacetime_cost(self) -> np.float128:
#         return np.float128(2 * ((self.dx + 4 * self.dz) * 3 * self.dx + 2 * self.dm) * 6 * self.dm) / (1 - self.failure_probability)


def one_level_15to1_state(pphys: float | mp.mpc, dx: int, dz: int, dm: int) -> mp.matrix:
    
    """
    Generates the output-state density matrix of the 15-to-1 protocol

    `pphys`: The physical error rate

    `dx`, `dz`, `dm`: distance for x, z, and measurement errors respectively
    """

    pphys = mp.mpc(pphys)

    # Introduce shorthand notation for logical error rate with distances dx/dz/dm
    px = plog(pphys, dx)
    pz = plog(pphys, dz)
    pm = plog(pphys, dm)


    # Step 1 of 15-to-1 protocol applying rotations 1-3 and 5
    out = apply_rot(
        init5qubit,
        [one, z, one, one, one],
        pphys / 3 + 0.5 * (dm / dz) * pz * dm,
        pphys / 3 + 0.5 * dz * pm,
        pphys / 3,
    )

    out = apply_rot(
        out,
        [one, one, z, one, one],
        pphys / 3 + 0.5 * (dm / dz) * pz * dm,
        pphys / 3 + 0.5 * dz * pm,
        pphys / 3,
    )
    
    out = apply_rot(
        out,
        [one, one, one, z, one],
        pphys / 3 + 0.5 * (dm / dz) * pz * dm,
        pphys / 3 + 0.5 * dz * pm,
        pphys / 3,
    )
    
    out = apply_rot(
        out,
        [one, z, z, z, one],
        pphys / 3 + 0.5 * pm * dm,
        pphys / 3 + 0.5 * pm * dm + 0.5 * (3 * dz) * dx / dm * pm,
        pphys / 3,
    )

    # Apply storage errors for dm code cycles
    
    out = storage_x_5(
        out,
        0,
        0.5 * (dz / dx) * px * dm,
        0.5 * (dz / dx) * px * dm,
        0.5 * (dz / dx) * px * dm,
        0,
    )
    
    out = storage_z_5(
        out,
        0,
        0.5 * (dx / dz) * pz * dm,
        0.5 * (dx / dz) * pz * dm,
        0.5 * (dx / dz) * pz * dm,
        0,
    )

    # Step 2: apply rotations 6-7
    # Last operation: apply additional storage errors due to multi-patch measurements
    
    out = apply_rot(
        out,
        [z, z, z, one, one],
        pphys / 3 + 0.5 * pm * dm,
        pphys / 3 + 0.5 * pm * dm + 0.5 * (dx + 2 * dz) * dx / dm * pm,
        pphys / 3,
    )
    
    out = apply_rot(
        out,
        [z, z, one, z, one],
        pphys / 3 + 0.5 * pm * dm,
        pphys / 3 + 0.5 * pm * dm + 0.5 * (dx + 3 * dz) * dx / dm * pm,
        pphys / 3,
    )
    out = storage_z_5(
        out, 0.5 * ((dx + 2 * dz) + (dx + 3 * dz)) / dx * px * dm, 0, 0, 0, 0
    )

    # Apply storage errors for dm code cycles
    out = storage_x_5(
        out,
        0.5 * px * dm,
        0.5 * (dz / dx) * px * dm,
        0.5 * (dz / dx) * px * dm,
        0.5 * (dz / dx) * px * dm,
        0,
    )
    out = storage_z_5(
        out,
        0.5 * px * dm,
        0.5 * (dx / dz) * pz * dm,
        0.5 * (dx / dz) * pz * dm,
        0.5 * (dx / dz) * pz * dm,
        0,
    )

    # Step 3: apply rotation 4 and 8-9
    out = apply_rot(
        out,
        [z, one, z, z, one],
        pphys / 3 + 0.5 * pm * dm,
        pphys / 3 + 0.5 * pm * dm + 0.5 * (dx + 3 * dz) * dx / dm * pm,
        pphys / 3,
    )
    out = apply_rot(
        out,
        [z, one, one, z, z],
        pphys / 3 + 0.5 * pm * dm,
        pphys / 3 + 0.5 * pm * dm + 0.5 * (dx + 4 * dz) * dx / dm * pm,
        pphys / 3,
    )
    out = apply_rot(
        out,
        [one, one, one, one, z],
        pphys / 3 + 0.5 * (dm / dz) * pz * dm,
        pphys / 3 + 0.5 * dz * pm,
        pphys / 3,
    )
    out = storage_z_5(
        out, 0.5 * ((dx + 3 * dz) + (dx + 4 * dz)) / dx * px * dm, 0, 0, 0, 0
    )

    # Apply storage errors for dm code cycles
    out = storage_x_5(
        out,
        0.5 * px * dm,
        0.5 * (dz / dx) * px * dm,
        0.5 * (dz / dx) * px * dm,
        0.5 * (dz / dx) * px * dm,
        0.5 * (dz / dx) * px * dm,
    )
    out = storage_z_5(
        out,
        0.5 * px * dm,
        0.5 * (dx / dz) * pz * dm,
        0.5 * (dx / dz) * pz * dm,
        0.5 * (dx / dz) * pz * dm,
        0.5 * (dx / dz) * pz * dm,
    )

    # Step 4: apply rotations 10-11
    out = apply_rot(
        out,
        [z, z, one, one, z],
        pphys / 3 + 0.5 * pm * dm,
        pphys / 3 + 0.5 * pm * dm + 0.5 * (dx + 4 * dz) * dx / dm * pm,
        pphys / 3,
    )
    out = apply_rot(
        out,
        [z, one, z, one, z],
        pphys / 3 + 0.5 * pm * dm,
        pphys / 3 + 0.5 * pm * dm + 0.5 * (dx + 4 * dz) * dx / dm * pm,
        pphys / 3,
    )
    out = storage_z_5(
        out, 0.5 * ((dx + 4 * dz) + (dx + 4 * dz)) / dx * px * dm, 0, 0, 0, 0
    )

    # Apply storage errors for dm code cycles
    out = storage_x_5(
        out,
        0.5 * px * dm,
        0.5 * (dz / dx) * px * dm,
        0.5 * (dz / dx) * px * dm,
        0.5 * (dz / dx) * px * dm,
        0.5 * (dz / dx) * px * dm,
    )
    out = storage_z_5(
        out,
        0.5 * px * dm,
        0.5 * (dx / dz) * pz * dm,
        0.5 * (dx / dz) * pz * dm,
        0.5 * (dx / dz) * pz * dm,
        0.5 * (dx / dz) * pz * dm,
    )

    # Step 5: apply rotations 12-13
    out = apply_rot(
        out,
        [z, z, z, z, z],
        pphys / 3 + 0.5 * pm * dm,
        pphys / 3 + 0.5 * pm * dm + 0.5 * (dx + 4 * dz) * dx / dm * pm,
        pphys / 3,
    )
    out = apply_rot(
        out,
        [one, one, z, z, z],
        pphys / 3 + 0.5 * pm * dm,
        pphys / 3 + 0.5 * pm * dm + 0.5 * (3 * dz) * dx / dm * pm,
        pphys / 3,
    )
    out = storage_z_5(out, 0.5 * (dx + 4 * dz) / dx * px * dm, 0, 0, 0, 0)

    # Apply storage errors for dm code cycles
    # Qubit 1 is consumed as an output state: additional storage errors for dx code cycles
    out = storage_x_5(
        out,
        0.5 * px * (dm + 2 * dx),
        0.5 * (dz / dx) * px * dm,
        0.5 * (dz / dx) * px * dm,
        0.5 * (dz / dx) * px * dm,
        0.5 * (dz / dx) * px * dm,
    )
    out = storage_z_5(
        out,
        0.5 * px * (dm + 2 * dx),
        0.5 * (dx / dz) * pz * dm,
        0.5 * (dx / dz) * pz * dm,
        0.5 * (dx / dz) * pz * dm,
        0.5 * (dx / dz) * pz * dm,
    )

    # Step 6: apply rotation 14-15
    out = apply_rot(
        out,
        [one, z, one, z, z],
        pphys / 3 + 0.5 * pm * dm,
        pphys / 3 + 0.5 * pm * dm + 0.5 * (4 * dz) * dx / dm * pm,
        pphys / 3,
    )
    out = apply_rot(
        out,
        [one, z, z, one, z],
        pphys / 3 + 0.5 * pm * dm,
        pphys / 3 + 0.5 * pm * dm + 0.5 * (4 * dz) * dx / dm * pm,
        pphys / 3,
    )

    # Apply storage errors for dm code cycles
    out = storage_x_5(
        out,
        0,
        0.5 * (dz / dx) * px * dm,
        0.5 * (dz / dx) * px * dm,
        0.5 * (dz / dx) * px * dm,
        0.5 * (dz / dx) * px * dm,
    )
    out = storage_z_5(
        out,
        0,
        0.5 * (dx / dz) * pz * dm,
        0.5 * (dx / dz) * pz * dm,
        0.5 * (dx / dz) * pz * dm,
        0.5 * (dx / dz) * pz * dm,
    )

    return out

def cost_of_one_level_15to1(pphys: float | mp.mpc, dx: int, dz: int, dm: int):
    """
    Calculates the output error and cost of the 15-to-1 protocol with a physical error rate `pphys` and distances `dx`, `dz` and `dm`
    """

    pphys = mp.mpc(pphys)

    # Generate output state of 15-to-1 protocol
    out = one_level_15to1_state(pphys, dx, dz, dm)

    # Compute failure probability as the probability to measure qubits 2-5 in the |+> state
    pfail = (1 - trace(kron(one, projx, projx, projx, projx) * out)).real

    # Compute the density matrix of the post-selected output state, i.e., after projecting qubits 2-5 into |+>
    outpostsel = (1 / (1 - pfail)) * kron(one, projx, projx, projx, projx) * out * kron(one, projx, projx, projx, projx).transpose_conj()


    # Compute output error from the infidelity between the post-selected state and the ideal output state
    pout = (1 - trace(outpostsel * ideal15to1)).real

    # Full-distance computation: determine full distance required for a 100-qubit / 10000-qubit computation
    def logerr1(d):
       return mp.float64((231 / pout) * d * plog(pphys, d) - 0.01)

    def logerr2(d):
       return mp.float64((20284 / pout) * d * plog(pphys, d) - 0.01)

    #reqdist1 = int(2 * round(optimize.root(logerr1, 3, method="hybr").x[0] / 2) + 1)
    #reqdist2 = int(2 * round(optimize.root(logerr2, 3, method="hybr").x[0] / 2) + 1)

    # Print output error, failure probability, space cost, time cost and space-time cost
    print("15-to-1 with pphys=", pphys, ", dx=", dx, ", dz=", dz, ", dm=", dm, sep="")
    print("Output error: ", "%.4g" % pout, sep="")
    print("Failure probability: ", "%.3g" % pfail, sep="")
    print("Qubits: ", 2 * ((dx + 4 * dz) * 3 * dx + 2 * dm), sep="")
    print("Code cycles: ", "%.2f" % (6 * dm / (1 - pfail)), sep="")
    print(
        "Space-time cost: ",
        "%.0f" % (2 * ((dx + 4 * dz) * 3 * dx + 2 * dm) * 6 * dm / (1 - pfail)),
        " qubitcycles",
        sep="",
    )
    # print(
    #     "For a 100-qubit computation: ",
    #     (
    #         "%.3f"
    #         % (((dx + 4 * dz) * 3 * dx + 2 * dm) * 6 * dm / (1 - pfail) / reqdist1**3)
    #     ),
    #     "d^3 (d=",
    #     reqdist1,
    #     ")",
    #     sep="",
    # )
    # print(
    #     "For a 5000-qubit computation: ",
    #     (
    #         "%.3f"
    #         % (((dx + 4 * dz) * 3 * dx + 2 * dm) * 6 * dm / (1 - pfail) / reqdist2**3)
    #     ),
    #     "d^3 (d=",
    #     reqdist2,
    #     ")",
    #     sep="",
    # )
    print("")
