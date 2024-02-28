from magic_state_factory import MagicStateFactory
import mpmath
from mpmath import mp
from scipy import optimize
from definitions import (
    z,
    one,
    projx,
    kron,
    trace,
    apply_rot,
    plog,
    storage_x_5,
    storage_z_5,
    init5qubit,
    ideal15to1,
)
from onelevel15to1 import one_level_15to1_state

def cost_of_two_level_15to1(pphys: float | mpmath.mpf, dx: int, dz: int, dm: int, dx2: int, dz2: int, dm2: int, nl1: int) -> MagicStateFactory:

    """
    Calculates the output error and cost of the (15-to-1)x(15-to-1) protocol with a physical error rate pphys, level-1 distances dx, dz and dm, level-2 distances dx2, dz2 and dm2, using nl1 level-1 factories
    """
    
    # Introduce shorthand notation for logical error rate with distances dx2/dz2/dm2

    pphys = mp.mpf(pphys)

    px2 = plog(pphys, dx2)
    pz2 = plog(pphys, dz2)
    pm2 = plog(pphys, dm2)

    # Compute pl1, the output error of level-1 states
    out = one_level_15to1_state(pphys, dx, dz, dm)
    pfail = (1 - trace(kron(one, projx, projx, projx, projx) * out)).real

    outpostsel = (1 / (1 - pfail)) * kron(one, projx, projx, projx, projx) * out * kron(one, projx, projx, projx, projx).transpose_conj()
    
    pl1 = (1 - trace(outpostsel * ideal15to1)).real

    # Compute l1time, the speed at which level-2 rotations can be performed (t_{L1} in the paper)
    l1time = max(6 * dm / (nl1 / 2) / (1 - pfail), dm2)

    # Define lmove, the effective width-dm2 region a level-1 state needs to traverse
    # before reaching the level-2 block, picking up additional storage errors
    lmove = 10 * dm2 + nl1 / 4 * (dx + 4 * dz)

    # Step 1 of (15-to-1)x(15-to-1) protocol applying rotations 1-2
    out2 = apply_rot(
        init5qubit,
        [one, z, one, one, one],
        pl1 + 0.5 * lmove * pm2,
        0.5 * lmove * pm2 + 0.5 * (dx2 + dz2 + dm2) * dx2 / dm2 * pm2,
        mp.mpc(0),
    )
    out2 = apply_rot(
        out2,
        [one, one, z, one, one],
        pl1 + 0.5 * lmove * pm2,
        0.5 * lmove * pm2 + 0.5 * (3 * dz2 + dm2) * dx2 / dm2 * pm2,
        mp.mpc(0),
    )

    # Apply storage errors for l1time code cycles
    out2 = storage_x_5(
        out2,
        0,
        0.5 * (dz2 / dx2) * px2 * l1time,
        0.5 * (dz2 / dx2) * px2 * l1time,
        0,
        0,
    )
    out2 = storage_z_5(
        out2,
        0,
        0.5 * (dx2 / dz2) * pz2 * l1time,
        0.5 * (dx2 / dz2) * pz2 * l1time,
        0,
        0,
    )

    # Step 2: apply rotations 3-4
    out2 = apply_rot(
        out2,
        [one, one, one, z, one],
        pl1 + 0.5 * lmove * pm2,
        0.5 * lmove * pm2 + 0.5 * (dx2 + 3 * dz2 + dm2) * dx2 / dm2 * pm2,
        mp.mpc(0),
    )
    out2 = apply_rot(
        out2,
        [one, one, one, one, z],
        pl1 + 0.5 * lmove * pm2,
        0.5 * lmove * pm2 + 0.5 * (dz2 + dm2) * dx2 / dm2 * pm2,
        mp.mpc(0),
    )

    # Apply storage errors for l1time code cycles
    out2 = storage_x_5(
        out2,
        0,
        0.5 * (dz2 / dx2) * px2 * l1time,
        0.5 * (dz2 / dx2) * px2 * l1time,
        0.5 * (dz2 / dx2) * px2 * l1time,
        0.5 * (dz2 / dx2) * px2 * l1time,
    )
    out2 = storage_z_5(
        out2,
        0,
        0.5 * (dx2 / dz2) * pz2 * l1time,
        0.5 * (dx2 / dz2) * pz2 * l1time,
        0.5 * (dx2 / dz2) * pz2 * l1time,
        0.5 * (dx2 / dz2) * pz2 * l1time,
    )

    # Step 3: apply rotations 5-6
    # Last operation: apply additional storage errors due to multi-patch measurements
    out2 = apply_rot(
        out2,
        [z, z, z, one, one],
        pl1 + 0.5 * lmove * pm2,
        0.5 * lmove * pm2 + 0.5 * (dx2 + 2 * dz2 + dm2) * dx2 / dm2 * pm2,
        mp.mpc(0),
    )
    out2 = apply_rot(
        out2,
        [one, z, z, z, one],
        pl1 + 0.5 * lmove * pm2,
        0.5 * lmove * pm2 + 0.5 * (4 * dz2 + dm2) * dx2 / dm2 * pm2,
        mp.mpc(0),
    )
    out2 = storage_z_5(out2, 0.5 * (dx2 + 2 * dz2 + dm2) * dm2 / dx2 * px2, 0, 0, 0, 0)

    # Apply storage errors for l1time code cycles
    out2 = storage_x_5(
        out2,
        0.5 * px2 * l1time,
        0.5 * (dz2 / dx2) * px2 * l1time,
        0.5 * (dz2 / dx2) * px2 * l1time,
        0.5 * (dz2 / dx2) * px2 * l1time,
        0.5 * (dz2 / dx2) * px2 * l1time,
    )
    out2 = storage_z_5(
        out2,
        0.5 * px2 * l1time,
        0.5 * (dx2 / dz2) * pz2 * l1time,
        0.5 * (dx2 / dz2) * pz2 * l1time,
        0.5 * (dx2 / dz2) * pz2 * l1time,
        0.5 * (dx2 / dz2) * pz2 * l1time,
    )

    # Step 4: apply rotations 7-8
    out2 = apply_rot(
        out2,
        [z, one, z, z, one],
        pl1 + 0.5 * lmove * pm2,
        0.5 * lmove * pm2 + 0.5 * (dx2 + 3 * dz2 + dm2) * dx2 / dm2 * pm2,
        mp.mpc(0),
    )
    out2 = apply_rot(
        out2,
        [z, z, one, z, one],
        pl1 + 0.5 * lmove * pm2,
        0.5 * lmove * pm2 + 0.5 * (dx2 + 4 * dz2 + dm2) * dx2 / dm2 * pm2,
        mp.mpc(0),
    )
    out2 = storage_z_5(
        out2,
        0.5 * ((dx2 + 3 * dz2 + dm2) + (dx2 + 4 * dz2 + dm2)) * dm2 / dx2 * px2,
        0,
        0,
        0,
        0,
    )

    # Apply storage errors for l1time code cycles
    out2 = storage_x_5(
        out2,
        0.5 * px2 * l1time,
        0.5 * (dz2 / dx2) * px2 * l1time,
        0.5 * (dz2 / dx2) * px2 * l1time,
        0.5 * (dz2 / dx2) * px2 * l1time,
        0.5 * (dz2 / dx2) * px2 * l1time,
    )
    out2 = storage_z_5(
        out2,
        0.5 * px2 * l1time,
        0.5 * (dx2 / dz2) * pz2 * l1time,
        0.5 * (dx2 / dz2) * pz2 * l1time,
        0.5 * (dx2 / dz2) * pz2 * l1time,
        0.5 * (dx2 / dz2) * pz2 * l1time,
    )

    # Step 5: apply rotations 9-10
    out2 = apply_rot(
        out2,
        [z, z, one, one, z],
        pl1 + 0.5 * lmove * pm2,
        0.5 * lmove * pm2 + 0.5 * (dx2 + 4 * dz2 + dm2) * dx2 / dm2 * pm2,
        mp.mpc(0),
    )
    out2 = apply_rot(
        out2,
        [z, one, one, z, z],
        pl1 + 0.5 * lmove * pm2,
        0.5 * lmove * pm2 + 0.5 * (dx2 + 4 * dz2 + dm2) * dx2 / dm2 * pm2,
        mp.mpc(0),
    )
    out2 = storage_z_5(
        out2,
        0.5 * ((dx2 + 4 * dz2 + dm2) + (dx2 + 4 * dz2 + dm2)) * dm2 / dx2 * px2,
        0,
        0,
        0,
        0,
    )

    # Apply storage errors for l1time code cycles
    out2 = storage_x_5(
        out2,
        0.5 * px2 * l1time,
        0.5 * (dz2 / dx2) * px2 * l1time,
        0.5 * (dz2 / dx2) * px2 * l1time,
        0.5 * (dz2 / dx2) * px2 * l1time,
        0.5 * (dz2 / dx2) * px2 * l1time,
    )
    out2 = storage_z_5(
        out2,
        0.5 * px2 * l1time,
        0.5 * (dx2 / dz2) * pz2 * l1time,
        0.5 * (dx2 / dz2) * pz2 * l1time,
        0.5 * (dx2 / dz2) * pz2 * l1time,
        0.5 * (dx2 / dz2) * pz2 * l1time,
    )

    # Step 6: apply rotations 11-12
    out2 = apply_rot(
        out2,
        [z, one, z, one, z],
        pl1 + 0.5 * lmove * pm2,
        0.5 * lmove * pm2 + 0.5 * (dx2 + 4 * dz2 + dm2) * dx2 / dm2 * pm2,
        mp.mpc(0),
    )
    out2 = apply_rot(
        out2,
        [z, z, z, z, z],
        pl1 + 0.5 * lmove * pm2,
        0.5 * lmove * pm2 + 0.5 * (dx2 + 4 * dz2 + dm2) * dx2 / dm2 * pm2,
        mp.mpc(0),
    )
    out2 = storage_z_5(
        out2,
        0.5 * ((dx2 + 4 * dz2 + dm2) + (dx2 + 4 * dz2 + dm2)) * dm2 / dx2 * px2,
        0,
        0,
        0,
        0,
    )

    # Apply storage errors for l1time code cycles
    out2 = storage_x_5(
        out2,
        0.5 * px2 * l1time,
        0.5 * (dz2 / dx2) * px2 * l1time,
        0.5 * (dz2 / dx2) * px2 * l1time,
        0.5 * (dz2 / dx2) * px2 * l1time,
        0.5 * (dz2 / dx2) * px2 * l1time,
    )
    out2 = storage_z_5(
        out2,
        0.5 * px2 * l1time,
        0.5 * (dx2 / dz2) * pz2 * l1time,
        0.5 * (dx2 / dz2) * pz2 * l1time,
        0.5 * (dx2 / dz2) * pz2 * l1time,
        0.5 * (dx2 / dz2) * pz2 * l1time,
    )

    # Step 7: apply rotations 13-14
    out2 = apply_rot(
        out2,
        [one, z, one, z, z],
        pl1 + 0.5 * lmove * pm2,
        0.5 * lmove * pm2 + 0.5 * (dx2 + 4 * dz2 + dm2) * dx2 / dm2 * pm2,
        mp.mpc(0),
    )
    out2 = apply_rot(
        out2,
        [one, one, z, z, z],
        pl1 + 0.5 * lmove * pm2,
        0.5 * lmove * pm2 + 0.5 * (3 * dz2 + dm2) * dx2 / dm2 * pm2,
        mp.mpc(0),
    )
    out2 = storage_z_5(out2, 0.5 * (dx2 + 4 * dz2 + dm2) * dm2 / dx2 * px2, 0, 0, 0, 0)

    # Apply storage errors for l1time code cycles
    # Qubit 1 is consumed as an output state: additional storage errors for dx2 code cycles
    out2 = storage_x_5(
        out2,
        0.5 * px2 * (dm2 + 2 * dx2),
        0.5 * (dz2 / dx2) * px2 * l1time,
        0.5 * (dz2 / dx2) * px2 * l1time,
        0.5 * (dz2 / dx2) * px2 * l1time,
        0.5 * (dz2 / dx2) * px2 * l1time,
    )
    out2 = storage_z_5(
        out2,
        0.5 * px2 * (dm2 + 2 * dx2),
        0.5 * (dx2 / dz2) * pz2 * l1time,
        0.5 * (dx2 / dz2) * pz2 * l1time,
        0.5 * (dx2 / dz2) * pz2 * l1time,
        0.5 * (dx2 / dz2) * pz2 * l1time,
    )

    # Step 8: apply rotation 15
    out2 = apply_rot(
        out2,
        [one, z, z, one, z],
        pl1 + 0.5 * lmove * pm2,
        0.5 * lmove * pm2 + 0.5 * (4 * dz2 + dm2) * dx2 / dm2 * pm2,
        mp.mpc(0),
    )

    # Apply storage errors for l1time code cycles
    out2 = storage_x_5(
        out2,
        0,
        0.5 * (dz2 / dx2) * px2 * l1time,
        0.5 * (dz2 / dx2) * px2 * l1time,
        0,
        0.5 * (dz2 / dx2) * px2 * l1time,
    )
    out2 = storage_z_5(
        out2,
        0,
        0.5 * (dx2 / dz2) * pz2 * l1time,
        0.5 * (dx2 / dz2) * pz2 * l1time,
        0,
        0.5 * (dx2 / dz2) * pz2 * l1time,
    )

    # Compute level-2 failure probability as the probability to measure qubits 2-5 in the |+> state
    pfail2 = (1 - trace(kron(one, projx, projx, projx, projx) * out2)).real

    # Compute the density matrix of the post-selected output state, i.e., after projecting qubits 2-5 into |+>
    outpostsel2 = (1 / (1 - pfail2)) * kron(one, projx, projx, projx, projx) * out2 * kron(one, projx, projx, projx, projx).transpose_conj()

    # Compute level-2 output error from the infidelity between the post-selected state and the ideal output state
    pout = (1 - trace(outpostsel2 * ideal15to1)).real

    # Full-distance computation: determine full distance required for a 100-qubit / 10000-qubit computation
    def logerr1(d):
        return float(231 / pout * d * plog(pphys, d) - 0.01)

    def logerr2(d):
        return float(20284 / pout * d * plog(pphys, d) - 0.01)

    reqdist1 = int(2 * round(optimize.root(logerr1, 3, method="hybr").x[0] / 2) + 1)
    reqdist2 = int(2 * round(optimize.root(logerr2, 3, method="hybr").x[0] / 2) + 1)

    # Print output error, failure probability, space cost, time cost and space-time cost
    nqubits = 2 * int(
        (dx2 + 4 * dz2) * 3 * dx2
        + nl1 * ((dx + 4 * dz) * (3 * dx + dm2 / 2) + 2 * dm)
        + 20 * dm2 * dm2
        + 2 * dx2 * dm2
    )

    ncycles = 7.5 * l1time / (1 - pfail2)

    return MagicStateFactory(
        name=f'(15-to-1)x(15-to-1) with pphys={pphys}, dx={dx}, dz={dz}, dm={dm}, dx2={dx2}, dz2={dz2}, dm2={dm2}, nl1={nl1}',
        distilled_magic_state_error_rate=pout,
        space=(0, 0),
        qubits=nqubits,
        distillation_time_in_cycles=(nqubits * ncycles),
        n_t_gates_produced_per_distillation=1,
    )