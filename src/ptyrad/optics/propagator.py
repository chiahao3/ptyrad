import numpy as np
from numpy.fft import ifftshift


# Propagator function used in init/initializer > init_H
def near_field_evolution(Npix_shape, dx, dz, lambd):
    """ Fresnel propagator """
    #  Translated and simplified from Yi's fold_slice Matlab implementation into numPy by Chia-Hao Lee

    ygrid = (np.arange(-Npix_shape[0] // 2, Npix_shape[0] // 2) + 0.5) / Npix_shape[0]
    xgrid = (np.arange(-Npix_shape[1] // 2, Npix_shape[1] // 2) + 0.5) / Npix_shape[1]

    # Standard ASM
    k  = 2 * np.pi / lambd
    ky = 2 * np.pi * ygrid / dx
    kx = 2 * np.pi * xgrid / dx
    Ky, Kx = np.meshgrid(ky, kx, indexing="ij")
    H = ifftshift(np.exp(1j * dz * np.sqrt(k ** 2 - Kx ** 2 - Ky ** 2))) # H has zero frequency at the corner in k-space

    return H