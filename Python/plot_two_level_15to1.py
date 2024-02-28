from mpmath import mp

mp.prec = 128

import mpmath
from multiprocessing import Pool
import pandas as pd
from datetime import datetime

from magic_state_factory import MagicStateFactory
from twolevel15to1 import cost_of_two_level_15to1


def objective(factory: MagicStateFactory) -> mpmath.mpf:
    return mp.mpf(factory.distilled_magic_state_error_rate) / factory.qubits



step_size: int = 2
pphys = 10**-4


class SimulationTwoLevel15to1:

    def __init__(
        self, dx: int, dz: int, dm: int, dx2: int, dz2: int, dm2: int, nl1: int, tag: str = "Simulation"
    ):
        self.prec = mp.prec
        self.pphys = pphys
        self.dx = dx
        self.dz = dz
        self.dm = dm
        self.dx2 = dx2
        self.dz2 = dz2
        self.dm2 = dm2
        self.nl1 = nl1
        self.factory = cost_of_two_level_15to1(pphys, dx, dz, dm, dx2, dz2, dm2, nl1)
        print(f'{tag}: {self.factory.name}; rating={self.rating()}')

    def rating(self) -> mpmath.mpf:
        return (
            self.factory.distilled_magic_state_error_rate * self.factory.qubits
        )


df = pd.DataFrame(columns = ['date', 'precision_in_bits', 'pphys', 'dx', 'dz', 'dm', 'dx2', 'dz2', 'dm2', 'nl1', 'error_rate', 'qubits', 'code_cycles'])
def log_simulation(sim: SimulationTwoLevel15to1) -> None:
    new_row = {'date': datetime.now().strftime("%Y-%m-%d %H:%M"), 'pphys': pphys, 'precision_in_bits': mp.prec, 'dx': sim.dx, 'dz': sim.dz, 'dm': sim.dm, 'dx2': sim.dx2, 'dz2': sim.dz2, 'dm2': sim.dm2, 'nl1': sim.nl1, 'error_rate': sim.factory.distilled_magic_state_error_rate, 'qubits': sim.factory.qubits, 'code_cycles': sim.factory.distillation_time_in_cycles}
    df.loc[len(df)] = new_row # type: ignore


centre_dx = 9
centre_dz = 3
centre_dm = 3
centre_dx2 = 25
centre_dz2 = 9
centre_dm2 = 9
centre_nl1 = 8


while True:
    with Pool(processes=8) as pool:
        try:

            print('Starting round of optimization...')

            centre = pool.apply_async(
                SimulationTwoLevel15to1,
                (
                    centre_dx,
                    centre_dz,
                    centre_dm,
                    centre_dx2,
                    centre_dz2,
                    centre_dm2,
                    centre_nl1,
                    'centre',
                ),
            )

            dx_probe = pool.apply_async(
                SimulationTwoLevel15to1,
                (
                    centre_dx + step_size,
                    centre_dz,
                    centre_dm,
                    centre_dx2,
                    centre_dz2,
                    centre_dm2,
                    centre_nl1,
                    'dx_probe',
                ),
            )

            dz_probe = pool.apply_async(
                SimulationTwoLevel15to1,
                (
                    centre_dx,
                    centre_dz  + step_size,
                    centre_dm,
                    centre_dx2,
                    centre_dz2,
                    centre_dm2,
                    centre_nl1,
                    'dz_probe',
                ),
            )

            dm_probe = pool.apply_async(
                SimulationTwoLevel15to1,
                (
                    centre_dx,
                    centre_dz,
                    centre_dm   + step_size,
                    centre_dx2,
                    centre_dz2,
                    centre_dm2,
                    centre_nl1,
                    'dm_probe',
                ),
            )

            dx2_probe = pool.apply_async(
                SimulationTwoLevel15to1,
                (
                    centre_dx,
                    centre_dz ,
                    centre_dm,
                    centre_dx2 + step_size,
                    centre_dz2,
                    centre_dm2,
                    centre_nl1,
                    'dx2_probe',
                ),
            )

            dz2_probe = pool.apply_async(
                SimulationTwoLevel15to1,
                (
                    centre_dx,
                    centre_dz ,
                    centre_dm,
                    centre_dx2,
                    centre_dz2  + step_size,
                    centre_dm2,
                    centre_nl1,
                    'dz2_probe',
                ),
            )

            dm2_probe = pool.apply_async(
                SimulationTwoLevel15to1,
                (
                    centre_dx,
                    centre_dz ,
                    centre_dm,
                    centre_dx2,
                    centre_dz2,
                    centre_dm2  + step_size,
                    centre_nl1,
                    'dm2_probe',
                ),
            )

            nl1_probe = pool.apply_async(
                SimulationTwoLevel15to1,
                (
                    centre_dx,
                    centre_dz ,
                    centre_dm,
                    centre_dx2,
                    centre_dz2,
                    centre_dm2,
                    centre_nl1 + step_size,
                    'nl1_probe',
                ),
            )

            pool.close()

            centre_rating = centre.get().rating()

            log_simulation(centre.get())
            log_simulation(dx_probe.get())
            log_simulation(dz_probe.get())
            log_simulation(dm_probe.get())
            log_simulation(dx2_probe.get())
            log_simulation(dz2_probe.get())
            log_simulation(dm2_probe.get())
            log_simulation(nl1_probe.get())
            

            centre_dx = max(1, centre_dx + step_size * (1 if dx_probe.get().rating() <= centre_rating else -1))
            centre_dz = max(1, centre_dz + step_size * (1 if dz_probe.get().rating() <= centre_rating else -1))
            centre_dm = max(1, centre_dm + step_size * (1 if dm_probe.get().rating() <= centre_rating else -1))
            centre_dx2 = max(1, centre_dx2 + step_size * (1 if dx2_probe.get().rating() <= centre_rating else -1))
            centre_dz2 = max(1, centre_dz2 + step_size * (1 if dz2_probe.get().rating() <= centre_rating else -1))
            centre_dm2 = max(1, centre_dm2 + step_size * (1 if dm2_probe.get().rating() <= centre_rating else -1))
            centre_nl1 = max(2, centre_nl1 + step_size * (1 if nl1_probe.get().rating() <= centre_rating else -1))

        except KeyboardInterrupt:
            pool.terminate()
            df.to_csv(f'Simulation_Data/two_level_15to1_simulations {datetime.now().strftime("%Y-%m-%d %H:%M")}.csv', mode='a', index=False, header=True)
            break