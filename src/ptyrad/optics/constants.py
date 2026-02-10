from typing import Literal

import numpy as np

# Physical Constants
PLANCKS = 6.62607015E-34 # m^2*kg / s
REST_MASS_E = 9.1093837015E-31 # kg
CHARGE_E = 1.602176634E-19 # coulomb 
SPEED_OF_LIGHT = 299792458 # m/s

# Useful constants in EM unit 
HC = PLANCKS * SPEED_OF_LIGHT / CHARGE_E*1E-3*1E10 # 12.398 keV-Ang, h*c
REST_ENERGY_E = REST_MASS_E*SPEED_OF_LIGHT**2/CHARGE_E*1E-3 # 511 keV, m0c^2

def get_EM_constants(kv: float, output_type: Literal['gamma', 'wavelength', 'sigma']):
    # acceleration_voltage: kV
    
    if output_type == 'gamma':
        return get_lorentz_factor_gamma(kv)
    elif output_type == 'wavelength':
        return get_wavelength_ang(kv)
    elif output_type == 'sigma':
        return get_interaction_parameter_sigma
    else:
        raise ValueError(f"output_type '{output_type}' not implemented yet, please use 'gamma', 'wavelength', or 'sigma'!")

def get_lorentz_factor_gamma(kv):
    gamma = 1 + kv / REST_ENERGY_E # m/m0 = 1 + e*V/m0c^2, dimensionless, Lorentz factor
    return gamma

def get_wavelength_ang(kv):
    wavelength = HC/np.sqrt((2*REST_ENERGY_E + kv)*kv) # Angstrom, lambda = hc/sqrt((2*m0c^2 + e*V)*e*V))
    return wavelength

def get_interaction_parameter_sigma(kv):
    gamma = get_lorentz_factor_gamma(kv)
    wavelength = get_wavelength_ang(kv)
    sigma = 2*np.pi*gamma*REST_MASS_E*CHARGE_E*wavelength/PLANCKS**2 * 1E-20 * 1E3 # interaction parameter, 2 pi*gamma*m0*e*lambda/h^2, 1/kV-Ang
    return sigma