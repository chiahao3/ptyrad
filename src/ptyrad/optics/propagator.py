"""
Numpy-based propagator functions

"""

import numpy as np
from numpy.fft import ifftshift


# Propagator function used in init/initializer > init_H
def near_field_evolution(Npix_shape, dx, dz, lambd):
    r"""Generates the free-space propagation transfer function using the Angular Spectrum Method (ASM).

    This function calculates the exact wave propagator in Fourier space, often 
    referred to in literature as the Angular Spectrum Method, rather than the 
    paraxial Fresnel approximation. The transfer function is defined as:

    .. math::

        H(k_x, k_y) = \exp\left(i \Delta z \sqrt{k^2 - k_x^2 - k_y^2}\right)

    The output array is shifted via ``ifftshift`` so that the zero-frequency 
    component is located at the corners (index ``[0, 0]``). This allows it to be 
    directly multiplied with the output of standard unshifted FFT routines (e.g., ``fft2``).

    Note:
        Translated and simplified from Yi's fold_slice MATLAB implementation 
        into NumPy by Chia-Hao Lee.

    Args:
        Npix_shape (tuple of int): The dimensions of the 2D grid in pixels, 
            typically given as :math:`(N_y, N_x)`.
        dx (float): The real-space pixel size (assumed isotropic in :math:`x` and :math:`y`).
        dz (float): The propagation distance (e.g., slice thickness) along the 
            optical axis.
        lambd (float): The wavelength of the electron or illumination wave.

    Returns:
        numpy.ndarray: A 2D complex array of shape :math:`(N_y, N_x)` representing the 
        propagation transfer function in :math:`k`-space.
    """

    ygrid = (np.arange(-Npix_shape[0] // 2, Npix_shape[0] // 2) + 0.5) / Npix_shape[0]
    xgrid = (np.arange(-Npix_shape[1] // 2, Npix_shape[1] // 2) + 0.5) / Npix_shape[1]

    # Standard ASM
    k  = 2 * np.pi / lambd
    ky = 2 * np.pi * ygrid / dx
    kx = 2 * np.pi * xgrid / dx
    Ky, Kx = np.meshgrid(ky, kx, indexing="ij")
    H = ifftshift(np.exp(1j * dz * np.sqrt(k ** 2 - Kx ** 2 - Ky ** 2))) # H has zero frequency at the corner in k-space

    return H