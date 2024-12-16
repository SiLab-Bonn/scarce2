import numpy as np
import matplotlib.pyplot as plt

from scarce2.sensor import Sensor
from scarce2 import plotting
from scarce2 import signals  # required for charge propagation
from tqdm import tqdm

EPSILON_SI = 1.04e-10  # Permittivity of silicon [F/m]
ELECTRON_CHARGE = 1.602e-19
ELECTRON_MOBILITY = 1400

def calc_increase_leakage_current(alpha, phi, thickness, pitch):
    return alpha*phi*pitch*pitch*thickness*1e-4*1e-4*1e4

def calc_sensor_capacitance(pitch, v_bias, n_eff):
    eps_si = EPSILON_SI
    rho = calc_resistivity(n_eff)
    v_bias = v_bias
    A = pitch*pitch
    return np.sqrt((eps_si/(2*ELECTRON_MOBILITY*rho*np.abs(v_bias))))*A

def calc_resistivity(n_eff):
    e = ELECTRON_CHARGE
    return 1/(e*ELECTRON_MOBILITY*n_eff)

def calc_leakage_current(thickness, pitch, n_eff, v_bias):
    rho = calc_resistivity(n_eff)*thickness*1e-4/(pitch*pitch*1e-4*1e-4)
    return v_bias/rho

def calc_capacitance(pitch, thickness):
    return EPSILON_SI*pitch*pitch*1e-4/thickness