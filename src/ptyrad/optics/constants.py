"""
Electron microscopy related physical constants

"""

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
    """Calculates a specific electron microscopy physical parameter.

    Args:
        kv (float): The acceleration voltage of the electron microscope in kilovolts (kV).
        output_type (Literal['gamma', 'wavelength', 'sigma']): The specific parameter 
            to calculate. Options are 'gamma' (Lorentz factor), 'wavelength' 
            (relativistic electron wavelength), or 'sigma' (interaction parameter).

    Returns:
        float: The calculated parameter value.

    Raises:
        ValueError: If an unsupported `output_type` is provided.
    """
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
    """Calculates the dimensionless relativistic Lorentz factor (gamma).

    The Lorentz factor accounts for the relativistic mass increase of the 
    accelerated electron. It is calculated as:
    gamma = 1 + (e * V) / (m0 * c^2)

    Args:
        kv (float): The acceleration voltage in kilovolts (kV).

    Returns:
        float: The dimensionless Lorentz factor.
    """
    gamma = 1 + kv / REST_ENERGY_E # m/m0 = 1 + e*V/m0c^2, dimensionless, Lorentz factor
    return gamma

def get_wavelength_ang(kv):
    """Calculates the relativistic electron wavelength (Ang).

    The wavelength is calculated using the relativistic de Broglie relationship:
    lambda = h * c / sqrt((2 * m0 * c^2 + e * V) * e * V)

    Args:
        kv (float): The acceleration voltage in kilovolts (kV).

    Returns:
        float: The electron wavelength in Angstroms.
    """
    wavelength = HC/np.sqrt((2*REST_ENERGY_E + kv)*kv) # Angstrom, lambda = hc/sqrt((2*m0c^2 + e*V)*e*V))
    return wavelength

def get_interaction_parameter_sigma(kv):
    """Calculates the electron interaction parameter (sigma).

    The interaction parameter governs the phase shift of the electron wave 
    per unit of projected electrostatic potential. It is calculated as:
    sigma = (2 * pi * gamma * m0 * e * lambda) / h^2

    Args:
        kv (float): The acceleration voltage in kilovolts (kV).

    Returns:
        float: The interaction parameter in units of 1 / (kV * Angstrom).
    """
    gamma = get_lorentz_factor_gamma(kv)
    wavelength = get_wavelength_ang(kv)
    sigma = 2*np.pi*gamma*REST_MASS_E*CHARGE_E*wavelength/PLANCKS**2 * 1E-20 * 1E3 # interaction parameter, 2 pi*gamma*m0*e*lambda/h^2, 1/kV-Ang
    return sigma