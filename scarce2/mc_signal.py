import numpy as np
import matplotlib.pyplot as plt

from scarce2.sensor import Sensor
from scarce2 import plotting
from scarce2 import signals  # required for charge propagation
from tqdm import tqdm


def calc_total_charge(timestep, thickness, pitch, electorde_size, v_bias, n_eff=2e12):
    
    s = Sensor(n_pixel=7, pitch=pitch, electrode_size=electorde_size, thickness=thickness, n_eff=n_eff)
    s.generate_mesh(mesh_density=1)
    s.setup_w_potential()
    s.solve_w_potential()
    # We need the electric field to propagate the electrons/holes. For details, see the example electric_field.iypnb
    s.setup_e_potential()
    s.solve_e_potential(V_bias=v_bias)
    s.convert_to_numpy()  # Generate meshgrids for both fields to use fast numpy functions for further analysis

    shift = 300
    size = np.random.poisson(70*thickness)
    positions = np.random.randint(low=1, high=thickness, size=size)
    total_charge = np.zeros(100000)
    if size > 1:
        for pos in tqdm(positions):
            charge_placeholder = np.zeros(100000)
            initial_pos = (0, pos)  # horizontally centered below readout electrode, 50 um above backplane
            charge_e, charge_h = signals.collected_charge_vs_time_fast(sensor=s, initial_pos=initial_pos, timestep=timestep)

            # Fill shorter array with last value for display reasons and total charge
            if len(charge_e) > len(charge_h):
                charge_h = np.concatenate((charge_h, np.repeat(0, len(charge_e) - len(charge_h))))
            elif len(charge_h) > len(charge_e):
                charge_e = np.concatenate((charge_e, np.repeat(0, len(charge_h) - len(charge_e))))

            if len(charge_e) != len(charge_h):
                raise RuntimeError("Charge arrays have different length!")

            abs_time = np.linspace(0, len(charge_e) * timestep, len(charge_e))

            charge_placeholder[shift:len(abs_time)+shift] = (np.cumsum(charge_e) + np.cumsum(charge_h)) - 1
            total_charge = total_charge + charge_placeholder
    return total_charge