from typing import Dict, Literal, Tuple, Union

import numpy as np
from numpy.fft import fft2, fftfreq, fftshift, ifft2, ifftshift

from ptyrad.optics.aberrations import Aberrations
from ptyrad.optics.constants import get_wavelength_ang
from ptyrad.runtime.logging import vprint # TODO: Remove the heavy torch-related vprint once we have a more modern logging


# Initialize probes
def make_aberration_surface_krivanek_polar(
    aberrations: Dict[Tuple[int, int], Dict[str, float]],
    kX: np.ndarray,
    kY: np.ndarray,
    wavelength: float
) -> np.ndarray:
    """Calculates the aberration phase surface chi(k) using Krivanek Polar form.

    Implements the standard polar expansion as defined in Kirkland Eqn. 2.22.

    Args:
        aberrations: A dictionary mapping order (n, m) to polar coefficients.
            Format: {(n, m): {'mag': float, 'phi': float}}
            'mag': Coefficient magnitude (e.g., C_s) in Angstroms.
            'phi': Azimuthal angle in degrees.
        kX: Spatial frequency coordinate X (1/Angstrom).
        kY: Spatial frequency coordinate Y (1/Angstrom).
        wavelength: Electron wavelength in Angstroms.

    Returns:
        np.ndarray: The aberration phase surface in radians.
    """
    
    alphaR = np.sqrt(kX**2 + kY**2) * wavelength
    alphaPhi = np.arctan2(kY, kX)
    chi = np.zeros_like(alphaR)
    
    for (n,m), coeffs in aberrations.items():
        
        if m == 0:
            C_nm = coeffs
            chi += (C_nm * alphaR**(n+1)) / (n+1)
        else:
            C_nm = coeffs['mag']
            phi_nm = np.radians(coeffs['phi'])
            # Kirkland Eq 2.22
            chi += (C_nm * alphaR**(n+1) * np.cos(m*(alphaPhi - phi_nm))) / (n+1)
    
    chi *= 2 * np.pi / wavelength
    
    return chi

def make_aberration_surface_krivanek_complex(
    aberrations: Dict[Tuple[int, int], complex],
    kX: np.ndarray,
    kY: np.ndarray,
    wavelength: float
) -> np.ndarray:
    """Calculates the aberration phase surface chi(k) using Krivanek Complex form.

    Implements the complex power series expansion (Kirkland Eqn. 2.19/2.20).
    This form utilizes the complex coordinate omega = alpha_x + i*alpha_y.
    Note that we swapped the exponents of omega and conj(omega) so the angle convention
    is consistent with Cartesian and Polar form.
    
    Args:
        aberrations: A dictionary mapping order (n, m) to complex coefficients.
            Format: {(n, m): complex_value}
        kX: Spatial frequency coordinate X (1/Angstrom).
        kY: Spatial frequency coordinate Y (1/Angstrom).
        wavelength: Electron wavelength in Angstroms.

    Returns:
        np.ndarray: The aberration phase surface in radians (real-valued).
    """
    
    alphaX = kX * wavelength # alphaX in radians
    alphaY = kY * wavelength
    chi = np.zeros_like(alphaX, dtype=complex)

    omega = alphaX + 1.0j*alphaY
    
    for (n,m), coeffs in aberrations.items():
        s = (n+m+1)//2
        C_nm = coeffs # This is complex valued
        chi += (C_nm * np.conj(omega)**s * omega**(n+1-s)) / (n+1) # Note that we swapped the exponenets of omega and conj(omega) so the angle conventions are consistent
    chi = 2 * np.pi / wavelength * chi.real
    
    return chi

def make_aberration_surface_krivanek_cartesian(
    aberrations: Dict[Tuple[int, int], Dict[str, float]],
    kX: np.ndarray,
    kY: np.ndarray,
    wavelength: float
) -> np.ndarray:
    r"""Calculates the aberration phase surface using recursive Cartesian polynomials.

    This method is significantly faster than polar or complex forms for large arrays
    as it avoids expensive trigonometric operations by using recursive multiplication.

    Mathematical Derivation:
    ------------------------
    1. Start with the standard Polar form (Kirkland Eq 2.22):
       $\chi(\alpha, \phi) = \frac{2\pi}{\lambda} \frac{1}{n+1} C_{nm} \alpha^{n+1} \cos[m(\phi - \phi_{nm})]$

    2. Expand the cosine term $\cos(m\phi - m\phi_{nm})$ and ignore the prefactor and summation for now:
       $\chi \propto \alpha^{n+1} [ \cos(m\phi)\cos(m\phi_{nm}) + \sin(m\phi)\sin(m\phi_{nm}) ]$

    3. Define Cartesian coefficients $C_{nma}, C_{nmb}$ to substitute $C_{nm}$ and $\phi_{nm}$:
       $C_{nma} = C_{nm} \cos(m\phi_{nm})$
       $C_{nmb} = C_{nm} \sin(m\phi_{nm})$

    4. Split the radial term $\alpha^{n+1}$ into $\alpha^{n+1-m} \cdot \alpha^m$ to isolate angular parts:
       $\chi \propto \alpha^{n+1-m} [ C_{nma} (\alpha^m \cos m\phi) + C_{nmb} (\alpha^m \sin m\phi) ]$

    5. Define Cartesian Angular Polynomials $X_m, Y_m$ using complex variable $Z = \alpha_x + i\alpha_y$:
       $Z^m = (\alpha e^{i\phi})^m = \alpha^m (\cos m\phi + i \sin m\phi)$
       
       Therefore:
       $X_m = \text{Re}(Z^m) = \alpha^m \cos(m\phi)$
       $Y_m = \text{Im}(Z^m) = \alpha^m \sin(m\phi)$

    6. Final Calculation:
       $X_m, Y_m$ are pre-calculated using the recurrence $Z_{m+1} = Z_m \cdot Z$.
       $\chi = \frac{2\pi}{\lambda} \sum \frac{1}{n+1} (\alpha^2)^{\frac{n+1-m}{2}} [ C_{nma}X_m + C_{nmb}Y_m ]$

    Args:
        aberrations: A dictionary mapping order (n, m) to Cartesian coefficients.
            Format: {(n, m): {'a': float, 'b': float}}
            'a': Cnma, cosine-like coefficient (Real part).
            'b': Cnmb, sine-like coefficient (Imaginary part).
        kX: Spatial frequency coordinate X (1/Angstrom).
        kY: Spatial frequency coordinate Y (1/Angstrom).
        wavelength: Electron wavelength in Angstroms.

    Returns:
        np.ndarray: The aberration phase surface in radians.
    """
    alphaX = kX * wavelength
    alphaY = kY * wavelength
    alpha_sq = alphaX**2 + alphaY**2
    
    # We scan the input dict to find the highest 'm' we need to generate
    max_m = 0
    if aberrations:
        max_m = max(m for (n, m) in aberrations.keys())
        
    # 3. Generate Angular Polynomials (X_m, Y_m) via Recursion
    # X_m = Real part of (ax + i*ay)^m
    # Y_m = Imag part of (ax + i*ay)^m
    
    # Storage for the basis functions
    X = {} 
    Y = {}
    
    # Base Case: m=0 (1, 0)
    X[0] = np.ones_like(alphaX)
    Y[0] = np.zeros_like(alphaX)
    
    # Recurrence Loop
    for m in range(max_m):
        # The Recurrence Relation: Z_{m+1} = Z_m * (x + iy)
        # Real: X_{m+1} = X_m * x - Y_m * y
        # Imag: Y_{m+1} = X_m * y + Y_m * x
        
        X[m+1] = X[m] * alphaX - Y[m] * alphaY
        Y[m+1] = X[m] * alphaY + Y[m] * alphaX
        
    chi = np.zeros_like(alphaX)
    
    for (n, m), val in aberrations.items():

        if m == 0:
            C_a = val
            C_b = 0.0
        else:
            C_a = val.get('a', 0.0)
            C_b = val.get('b', 0.0)
        
        if C_a == 0 and C_b == 0:
            continue
            
        # A. Radial Term: alpha^(n+1-m)
        # alpha^(n+1-m) = (alpha^2) ^ ((n+1-m)/2)
        power_rad = (n + 1 - m) / 2.0
        term_radial = alpha_sq ** power_rad
        
        # B. Angular Term: (C_a * X_m + C_b * Y_m)
        term_angular = C_a * X[m] + C_b * Y[m]
        
        chi += term_radial * term_angular / (n + 1)
        
    chi *= (2 * np.pi / wavelength)
    
    return chi
    
def make_stem_probe(
    kv: float, 
    conv_angle: float, 
    Npix: int, 
    dx: float, 
    aberrations: Union[dict, Aberrations], 
    method: Literal['polar', 'cartesian', 'complex'] = 'cartesian', 
    verbose: bool = True
) -> np.ndarray:
    """Simulates a STEM probe in real space using the specified methods for chi(k) calculations.
    The three methods (polar, cartesian, complex) give identical result within numerical precision, 
    while 'cartesian' is chosen as the default as it's the fastest (though they're all just few ms).

    Constructs the probe by defining the aperture and aberrations in Fourier space,
    applying the phase shift, and performing an inverse FFT to obtain the real-space
    complex wave function.

    Args:
        kv: Acceleration voltage in kilovolts (kV).
        conv_angle: Convergence semi-angle in milliradians (mrad).
        Npix: Number of pixels for the square simulation grid.
        dx: Real-space pixel size in Angstroms.
        aberrations: An Aberrations instance, or dictionary of aberration coefficients. 
            The dictionary can be in Haider (e.g., {'C1': 10}), or 
            Krivanek (e.g., {'C12': 10, 'phi12': 30}) notation in polar / cartesian / complex form.
            Mix-match and aliases like 'defocus', 'Cs' are supported.
        method: The computation approach for chi(k) calculation. Options:
            - 'polar': Standard Krivanek polar form (C_nm * alpha^(n+1) * cos[m(phi-phi_nm)]).
            - 'cartesian': Recursive Cartesian polynomials (C_nma * X[m] + C_nmb * Y[m]).
            - 'complex': Analytic complex power series (C_nm * w*^(n+1-s) * w^s).
        verbose: If True, prints simulation details to stdout.

    Returns:
        np.ndarray: A 2D complex array representing the probe wave function in 
        real space, normalized such that the total intensity sums to 1.
    """
    
    # Instantiate the Aberrations object if users are passing a dict
    if isinstance(aberrations, dict):
        ab = Aberrations(aberrations)
    else:
        ab = aberrations
    
    # Calculate some variables
    wavelength = get_wavelength_ang(kv) # wavelength in Ang
    k_aperture = conv_angle/1e3/wavelength
    dk = 1/(dx*Npix)
    
    # Make k space sampling and probe forming aperture
    k = fftshift(fftfreq(Npix, dx)) # k is now in unit of Ang-1
    kX,kY = np.meshgrid(k,k, indexing='xy')
    kR = np.sqrt(kX**2+kY**2)
    mask = (kR<=k_aperture)
    
    # Verbose printing
    if verbose:
        vprint("Start simulating STEM probe")
        vprint(f'  kv          = {kv} kV')    
        vprint(f'  wavelength  = {wavelength:.4f} Ang')
        vprint(f'  conv_angle  = {conv_angle} mrad')
        vprint(f'  Npix        = {Npix} px')
        vprint(f'  dk          = {dk:.4f} Ang^-1')
        vprint(f'  kMax        = {(Npix*dk/2):.4f} Ang^-1')
        vprint(f'  alpha_max   = {(Npix*dk/2*wavelength*1000):.4f} mrad')
        vprint(f'  dx          = {dx:.4f} Ang, Nyquist-limited dmin = 2*dx = {2*dx:.4f} Ang')
        vprint(f'  Rayleigh-limited resolution  = {(0.61*wavelength/conv_angle*1e3):.4f} Ang (0.61*lambda/alpha for focused probe )')
        vprint(f'  Real space probe extent = {dx*Npix:.4f} Ang')
        for line in ab.pretty_print().splitlines():
            vprint(line)

    # Choosing the computation method used for chi calculation
    if method == 'polar':
        aberrations_by_order = ab.export(notation='krivanek', style= 'polar', layout='nested')
        make_aberration_surface = make_aberration_surface_krivanek_polar
    elif method == 'cartesian':
        aberrations_by_order = ab.export(notation='krivanek', style= 'cartesian', layout='nested')
        make_aberration_surface = make_aberration_surface_krivanek_cartesian
    elif method == 'complex':
        aberrations_by_order = ab.export(notation='krivanek', style= 'complex', layout='nested')
        make_aberration_surface = make_aberration_surface_krivanek_complex
    else:
        raise ValueError(f"Unknown calculation method = {method}, please choose between 'polar', 'cartesian', or 'complex'")
    
    # Calculate chi(k) in unit of radians    
    chi = make_aberration_surface(aberrations=aberrations_by_order, kX=kX, kY=kY, wavelength=wavelength)

    # Make probe and normalize
    psi = np.exp(-1j*chi)
    probe = mask*psi # It's now the masked wave function at the aperture plane
    probe = fftshift(ifft2(ifftshift(probe))) # Propagate the wave function from aperture to the sample plane. 
    probe = probe/np.sqrt(np.sum((np.abs(probe))**2)) # Normalize the probe so sum(abs(probe)^2) = 1
    
    return probe

def make_fzp_probe(
    beam_kev: float, 
    Npix: int, 
    dx: float,
    Ls: float,
    Rn: float,
    dRn: float,
    D_FZP: float,
    D_H: float,
    verbose=True
) -> np.ndarray:
    """
    Generates a Fresnel zone plate probe with internal Fresnel propagation for x-ray ptychography simulations.

    Parameters:
        beam_kev (float): Energy of the x-ray photon.
        Npix (int): Number of pixels.
        dx (float): Pixel size (in meters) in the sample plane.
        Ls (float): Distance (in meters) from the focal plane to the sample.
        Rn (float): Radius of outermost zone (in meters).
        dRn (float): Width of outermost zone (in meters).
        D_FZP (float): Diameter of pinhole.
        D_H (float): Diameter of the central beamstop (in meters).

    Returns:
        ndarray: Calculated probe field in the sample plane.
    """

    lambda_ = 1.23984193e-9 / beam_kev # lambda_: m; energy: keV
    fl = 2 * Rn * dRn / lambda_  # focal length corresponding to central wavelength

    vprint("Start simulating FZP probe", verbose=verbose)

    dx_fzp = lambda_ * fl / Npix / dx  # pixel size in the FZP plane

    # Coordinate in the FZP plane
    lx_fzp = np.linspace(-dx_fzp * Npix / 2, dx_fzp * Npix / 2, Npix)
    x_fzp, y_fzp = np.meshgrid(lx_fzp, lx_fzp)

    
    T = np.exp(-1j * 2 * np.pi / lambda_ * (x_fzp**2 + y_fzp**2) / (2 * fl))
    C = (np.sqrt(x_fzp**2 + y_fzp**2) <= (D_FZP / 2)).astype(np.float64)  # circular function of FZP
    H = (np.sqrt(x_fzp**2 + y_fzp**2) >= (D_H / 2)).astype(np.float64)  # central block

    
    IN = C * T * H
    M, N = IN.shape
    k = 2 * np.pi / lambda_

    # Coordinate grid for input plane
    lx = np.linspace(-dx_fzp * M / 2, dx_fzp * M / 2, M)
    x, y = np.meshgrid(lx, lx)

    # Coordinate grid for output plane
    fc = 1 / dx_fzp
    fu = lambda_ * (fl + Ls) * fc
    lu = ifftshift(np.linspace(-fu / 2, fu / 2, M))
    u, v = np.meshgrid(lu, lu)

    z = fl + Ls
    if z > 0:
        # Propagation in the positive z direction
        pf = np.exp(1j * k * z) * np.exp(1j * k * (u**2 + v**2) / (2 * z))
        kern = IN * np.exp(1j * k * (x**2 + y**2) / (2 * z))
        
        kerntemp = fftshift(kern)
        cgh = fft2(kerntemp)
        probe = fftshift(cgh * pf)
    else:
        # Propagation in the negative z direction (or backward propagation)
        z = abs(z)
        pf = np.exp(1j * k * z) * np.exp(1j * k * (x**2 + y**2) / (2 * z))
        cgh = ifft2(ifftshift(IN) / np.exp(1j * k * (u**2 + v**2) / (2 * z)))
        probe = fftshift(cgh) / pf

    return probe

def make_mixed_probe(probe, pmodes, pmode_init_pows, verbose=True):
    ''' Make a mixed state probe from a single state probe '''
    # Input:
    #   probe: (Ny,Nx) complex array
    #   pmodes: number of incoherent probe modes, scaler int
    #   pmode_init_pows: Integrated intensity of modes. List of a value (e.g. [0.02]) or a couple values for the first few modes. sum(pmode_init_pows) must < 1. 
    # Output:
    #   mixed_probe: A mixed state probe with (pmode,Ny,Nx)
       
    # Prepare a mixed-state probe `mixed_probe`
    vprint(f"Start making mixed-state STEM probe with {pmodes} incoherent probe modes", verbose=verbose)
    M = np.ceil(pmodes**0.5)-1
    N = np.ceil(pmodes/(M+1))-1
    mixed_probe = hermite_like(probe, M,N)[:pmodes]
    
    # Normalize each pmode
    pmode_pows = np.zeros(pmodes)
    for ii in range(1,pmodes):
        if ii<np.size(pmode_init_pows):
            pmode_pows[ii] = pmode_init_pows[ii-1]
        else:
            pmode_pows[ii] = pmode_init_pows[-1]
    if sum(pmode_pows)>1:
        raise ValueError('Modes total power exceeds 1, check pmode_init_pows')
    else:
        pmode_pows[0] = 1-sum(pmode_pows)

    mixed_probe = mixed_probe * np.sqrt(pmode_pows)[:,None,None]
    vprint(f"Relative power of probe modes = {pmode_pows}", verbose=verbose)
    return mixed_probe

def hermite_like(fundam, M, N):
    # %HERMITE_LIKE
    # % Receives a probe and maximum x and y order M N. Based on the given probe
    # % and multiplying by a Hermitian function new modes are computed. The modes
    # % are then orthonormalized.
    
    # Input:
    #   fundam: base function
    #   X,Y: centered meshgrid for the base function
    #   M,N: order of the hermite_list basis
    # Output:
    #   H: 
    # Note:
    #   This function is a python implementation of `ptycho\+core\hermite_like.m` from PtychoShelves with some modification
    #   Most indexings arr converted from Matlab (start from 1) to Python (start from 0)
    #   The X, Y meshgrid are moved into the funciton
    #   The H is modified into (pmode, Ny, Nx) to be consistent with ptyrad
    #   Note that H would output (M+1)*(N+1) modes, which could be a bit more than the specified pmode
    
    
    # Initialize i/o
    M = M.astype('int')
    N = N.astype('int')
    m = np.arange(M+1)
    n = np.arange(N+1)
    H = np.zeros(((M+1)*(N+1), fundam.shape[-2], fundam.shape[-1]), dtype=fundam.dtype)
      
    # Create meshgrid
    rows, cols = fundam.shape[-2:]
    x = np.arange(cols) - cols / 2
    y = np.arange(rows) - rows / 2
    X, Y = np.meshgrid(x, y)
    
    cenx = np.sum(X * np.abs(fundam)**2) / np.sum(np.abs(fundam)**2)
    ceny = np.sum(Y * np.abs(fundam)**2) / np.sum(np.abs(fundam)**2)
    varx = np.sum((X - cenx)**2 * np.abs(fundam)**2) / np.sum(np.abs(fundam)**2)
    vary = np.sum((Y - ceny)**2 * np.abs(fundam)**2) / np.sum(np.abs(fundam)**2)

    counter = 0
    
    # Create basis
    for nii in n:
        for mii in m:
            auxfunc = ((X - cenx)**mii) * ((Y - ceny)**nii) * fundam
            if counter == 0:
                auxfunc = auxfunc / np.sqrt(np.sum(np.abs(auxfunc)**2))
            else:
                auxfunc = auxfunc * np.exp(-((X - cenx)**2 / (2*varx)) - ((Y - ceny)**2 / (2*vary)))
                auxfunc = auxfunc / np.sqrt(np.sum(np.abs(auxfunc)**2))

            # Now make it orthogonal to the previous ones
            for ii in range(counter): # The other ones
                auxfunc = auxfunc - np.dot(H[ii].reshape(-1), np.conj(auxfunc).reshape(-1)) * H[ii]

            # Normalize each mode so that their intensities sum to 1
            auxfunc = auxfunc / np.sqrt(np.sum(np.abs(auxfunc)**2))
            H[counter] = auxfunc
            counter += 1

    return H

def sort_by_mode_int_np(modes):
    spatial_axes = tuple(range(1, modes.ndim))
    modes_int = np.sum(np.abs(modes)**2, axis=spatial_axes)
    indices = np.argsort(modes_int)[::-1]  # sort descending
    modes = modes[indices]
    return modes

def orthogonalize_modes_vec_np(modes, sort=False):
    """
    Orthogonalize the modes using SVD-like procedure via eigen decomposition.

    Parameters
    ----------
    modes : np.ndarray
        Input modes of shape (Nmode, Ny, Nx), complex.
    sort : bool, optional
        Whether to sort modes by their intensity (norm), by default False.

    Returns
    -------
    np.ndarray
        Orthogonalized modes of the same shape as input.
    """

    orig_dtype = modes.dtype
    modes = modes.astype(np.complex128) # temporarily cast to complex128 for more precise orthogonalization

    input_shape = modes.shape
    n_modes = input_shape[0]

    # Reshape into (Nmode, Ny*Nx)
    modes_reshaped = modes.reshape(n_modes, -1)

    # Gram matrix A = M @ M^H (Nmode x Nmode)
    A = modes_reshaped @ modes_reshaped.conj().T

    # Eigen-decomposition
    eigvals, eigvecs = np.linalg.eig(A)

    # Project original modes into orthogonalized space
    ortho_modes = eigvecs.conj().T @ modes_reshaped
    ortho_modes = ortho_modes.reshape(input_shape)

    if sort:
        ortho_modes = sort_by_mode_int_np(ortho_modes)
    
    return ortho_modes.astype(orig_dtype)