import numpy as np
import matplotlib.pyplot as plt

from scarce2.sensor import Sensor
from scarce2 import plotting
from scarce2 import signals  # required for charge propagation
from tqdm import tqdm

EPSILON_SI = 1.04e-10  # Permittivity of silicon [F/m]
ELECTRON_CHARGE = 1.602e-19
ELECTRON_MOBILITY = 1400

def calc_increase_leakage_current(phi, thickness, pitch):
    alpha = 3.99e-17
    return alpha*phi*pitch*pitch*thickness*1e-4*1e-4*1e-4

def calc_sensor_capacitance(pitch, v_bias, n_eff):
    v_bias = v_bias
    A = pitch*pitch*1e-2*1e-2
    return A*np.sqrt((EPSILON_SI*ELECTRON_CHARGE*np.abs(n_eff)/(2*np.abs(v_bias))))

def calc_resistivity(n_eff):
    return 1/(ELECTRON_CHARGE*ELECTRON_MOBILITY*np.abs(n_eff))

def calc_leakage_current(thickness, pitch, n_eff, v_bias):
    rho = calc_resistivity(n_eff)*thickness*1e-4/(pitch*pitch*1e-4*1e-4)
    return v_bias/rho

def calc_plate_capacitance(pitch, thickness):
    return EPSILON_SI*pitch*pitch*1e-4/thickness

def calc_eff_doping(phi, n_eff_0):
    'from the 1992 wunstorf'
    N_a = 2.4e10 #random value from wunstorf seems to work
    N_d = n_eff_0 - N_a
    c = 3.53e-13
    b = 7.94e-2
    if isinstance(phi, int) or isinstance(phi, float):
        phi = np.array(phi)
    before_inv = N_d*np.exp(-c*phi[phi<1e14]) - N_a - b*phi[phi<1e14]
    after_inv = -b*phi[phi>1e14]
    return np.concatenate([before_inv, after_inv])

def calc_annealing(t):
    'from the 1992 wunstorf'
    A_i = [0.214, 0.262, 0.118, 0.097, -0.107]
    tau_i = [9.4, 6.87e1, 3.43e2, 4e3, 7.52e4]
    ratio = 0
    for i in range(5):
        ratio += A_i[i]*np.exp(-t/tau_i[i])
    return ratio + 0.417
    # return n_eff_t0*np.sum([A_i[i]*np.exp(-t/tau_i[i]) for i in range(5)])
