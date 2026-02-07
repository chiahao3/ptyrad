"""
Physics-related functions of probes, propagators, and constants, etc.

"""

from typing import Dict, Literal, Optional, Tuple, Union

import numpy as np
import torch
from numpy.fft import fft2, fftfreq, fftshift, ifft2, ifftshift

from ptyrad.optics.aberrations import Aberrations
from .common import vprint


def infer_dx_from_params(
    dx: Optional[float] = None,
    dk: Optional[float] = None,
    kMax: Optional[float] = None,
    da: Optional[float] = None,
    angleMax: Optional[float] = None,
    RBF: Optional[float] = None,
    n_alpha: Optional[float] = None,
    conv_angle: Optional[float] = None,
    wavelength: Optional[float] = None,
    Npix: Optional[int] = None,
) -> float:
    """
    Infer the real-space pixel size (dx) based on available unit-related parameters.
    Accepts keyword arguments directly, or a dictionary via `infer_dx_from_params(**params)`.

    Args:
        dx (Optional[float], optional): Real space pixel size for object, probe, and scan position coordinates in unit of Ang (electron) or m (X-ray).
            This is used as the unified unit for calibration. Defaults to None.
            
        dk (Optional[float], optional): k-space pixel size for the measurments in unit of 1/Ang (electron) or 1/m (X-ray). Defaults to None.
        
        kMax (Optional[float], optional): Maximum collection angle in unit of 1/Ang for electron, or 1/m for X-ray. Defaults to None.
        
        da (Optional[float], optional): k-space pixel size for the measurments in unit of mrad. Defaults to None.
        
        angleMax (Optional[float], optional): Maximum collection angle in unit of mrad. Defaults to None.
        
        RBF (Optional[float], optional): Number of pixels within the bright field disk of the electron diffraction pattern. Defaults to None.
        
        n_alpha (Optional[float], optional): Collection angle in unit of convergence angles of the elctron probe (usually called "n-alpha"). Defaults to None.
        
        conv_angle (Optional[float], optional): Convergence angles of the electron probe. Unit: Ang. Defaults to None.
        
        wavelength (Optional[float], optional): Wavelength of the wave. Unit should be Ang (electron) or m (X-ray). Defaults to None.
        
        Npix (Optional[int], optional): Number of detector pixel. Defaults to None.

    Raises:
        ValueError: if required parameters are missing or input is ambiguous

    Returns:
        float: inferred dx (real-space pixel size)
    """    

    if dx is not None:
        return dx

    if dk is not None and Npix is not None:
        return 1 / (Npix * dk)

    if kMax is not None:
        return 1 / (2 * kMax)

    if da is not None and wavelength is not None and Npix is not None:
        dk = da / wavelength / 1e3  # mrad to rad
        return 1 / (Npix * dk)

    if angleMax is not None and wavelength is not None:
        kMax = angleMax / wavelength / 1e3  # mrad to rad
        return 1 / (2 * kMax)

    if all(v is not None for v in (RBF, conv_angle, wavelength, Npix)):
        da = conv_angle / RBF / 1e3  # radians
        dk = da / wavelength
        return 1 / (Npix * dk)

    if n_alpha is not None and wavelength is not None:
        angleMax = n_alpha * conv_angle
        kMax = angleMax / wavelength / 1e3  # mrad to rad
        return 1 / (2 * kMax)
    
    raise ValueError(
        "Insufficient or unrecognized parameters to infer dx. "
        "Please provide one of the following: "
        "'dx', or 'dk'+'Npix', or 'da'+'wavelength'+'Npix', or 'kMax', or "
        "'aMax'+'wavelength', or 'RBF'+'conv_angle'+'wavelength'+'Npix'."
    )

def get_EM_constants(acceleration_voltage, output_type):
    
    # acceleration_voltage: kV
    
    # Physical Constants
    PLANCKS = 6.62607015E-34 # m^2*kg / s
    REST_MASS_E = 9.1093837015E-31 # kg
    CHARGE_E = 1.602176634E-19 # coulomb 
    SPEED_OF_LIGHT = 299792458 # m/s
    
    # Useful constants in EM unit 
    hc = PLANCKS * SPEED_OF_LIGHT / CHARGE_E*1E-3*1E10 # 12.398 keV-Ang, h*c
    REST_ENERGY_E = REST_MASS_E*SPEED_OF_LIGHT**2/CHARGE_E*1E-3 # 511 keV, m0c^2
    
    # Derived values
    gamma = 1 + acceleration_voltage / REST_ENERGY_E # m/m0 = 1 + e*V/m0c^2, dimensionless, Lorentz factor
    wavelength = hc/np.sqrt((2*REST_ENERGY_E + acceleration_voltage)*acceleration_voltage) # Angstrom, lambda = hc/sqrt((2*m0c^2 + e*V)*e*V))
    sigma = 2*np.pi*gamma*REST_MASS_E*CHARGE_E*wavelength/PLANCKS**2 * 1E-20 * 1E3 # interaction parameter, 2 pi*gamma*m0*e*lambda/h^2, 1/kV-Ang
    
    if output_type == 'gamma':
        return gamma
    elif output_type == 'wavelength':
        return wavelength
    elif output_type == 'sigma':
        return sigma
    else:
        raise ValueError(f"output_type '{output_type}' not implemented yet, please use 'gamma', 'wavelength', or 'sigma'!")

def complex_object_interp3d(complex_object, zoom_factors, z_axis, use_np_or_cp='np'):
    """
    Interpolate a 3D complex object while preserving multiscattering behavior.

    Parameters:
        - complex_object (ndarray): Input complex object with shape (z, y, x).
        - zoom_factors (tuple): Tuple of zoom factors for (z, y, x).
        - z_axis: int indicating the z-axis posiiton
        - use_np_or_cp (str): Specify the library to use, 'np' for NumPy or 'cp' for CuPy.

    Returns:
        ndarray: Interpolated complex object with the same dtype as the input.

    Notes:
        - Amplitude and phase are treated separately as they obey different conservation laws.
        - Phase shift for multiple z-slices is additive, ensuring the sum of all z-slices remains the same.
        - Amplitude between each z-slice is multiplicative. Linear interpolation of log(amplitude) is performed while maintaining the conservation law.
        - The phase of the object should be unwrapped and smooth.
        - If possible, use cupy for 40x faster speed (I got 1 sec vs 40 sec for 320*320*420 target size in a one-shot calculation on my Quadro P5000)

    Example:
        ```python
        complex_object = np.random.rand(10, 10, 10) + 1j * np.random.rand(10, 10, 10)
        zoom_factors = (2, 2, 1.5)
        result = complex_object_interp3d(complex_object, zoom_factors, use_np_or_cp='np')
        ```
    """
    
    if use_np_or_cp == 'cp':
        import cupy as xp  # type: ignore
        from cupyx.scipy import ndimage  # type: ignore
        complex_object = xp.array(complex_object)
    else:
        import numpy as xp
        from scipy import ndimage
    
    if zoom_factors == (1,1,1):
        print(f"No interpolation is needed, returning original object with shape = {complex_object.shape}.")
        return complex_object

    else:
        obj_dtype = complex_object.dtype
        obj_a = xp.abs(complex_object)
        obj_p = xp.angle(complex_object)
        
        obj_a_interp = xp.exp(ndimage.zoom(xp.log(obj_a), zoom_factors) / zoom_factors[z_axis])
        obj_p_interp = ndimage.zoom(obj_p, zoom_factors) / zoom_factors[z_axis]
        
        complex_object_interp3d = obj_a_interp * xp.exp(obj_p_interp*1j)
        print(f"The object shape is interpolated to {complex_object_interp3d.shape}.")
        return complex_object_interp3d.astype(obj_dtype)

def complex_object_z_resample_torch(obj: Union[torch.Tensor, np.ndarray], 
                                    dz_now: float, 
                                    resample_mode: Literal['scale_Nlayer', 'scale_slice_thickness', 'target_Nlayer', 'target_slice_thickness'], 
                                    resample_value: Union[float, int], 
                                    output_type: Optional[Literal['complex', 'amplitude', 'phase', 'amp_phase']] = 'complex', 
                                    return_np: bool = True):
    """Resample a complex 3D object along the depth (z) axis while conserving
    amplitude product, phase sum, and total thickness.

    This function performs interpolation along the z-axis of a complex-valued
    object using PyTorch. The object is decomposed into amplitude and phase,
    resampled with conservation laws applied, and recombined into the desired
    output representation.

    Args:
        obj (ndarray or torch.Tensor): Input complex object with shape
            (..., Nz, Ny, Nx). Can be a NumPy array or a torch.Tensor.
        dz_now (float): Current slice thickness along the z-axis.
        resample_mode (str): Resampling mode for the depth axis. Must be one of:
            - "scale_Nlayer": Scale the number of layers by a float factor.
            - "scale_slice_thickness": Scale slice thickness by a float factor.
            - "target_Nlayer": Resample to a target integer number of layers.
            - "target_slice_thickness": Resample to a target slice thickness.
        resample_value (int or float): Parameter value for the resampling mode.
            - Positive float for "scale_Nlayer" or "scale_slice_thickness".
            - Positive integer (>=1) for "target_Nlayer".
            - Positive float for "target_slice_thickness".
        output_type (str, optional): Output representation. Must be one of:
            - "complex": Return recombined complex object (default).
            - "amplitude": Return amplitude only.
            - "phase": Return phase only.
            - "amp_phase": Return tuple (amplitude, phase).
        return_np (bool, optional): If True (default), convert outputs to NumPy
            arrays. If False, return PyTorch tensors.

    Returns:
        ndarray or torch.Tensor or tuple:
            The resampled object in the requested representation:
            - Complex ndarray/tensor if output_type == "complex".
            - Real ndarray/tensor if output_type == "amplitude" or "phase".
            - Tuple of (amplitude, phase) if output_type == "amp_phase".

            Type depends on `return_np`.

    Raises:
        ValueError: If `resample_mode` is invalid.
        ValueError: If the target number of layers is less than 1.
        ValueError: If the input object has unsupported dimensionality.
        ValueError: If `output_type` is not one of the allowed options.

    Examples:
        Resample by doubling the number of z-layers:

        >>> out = complex_object_z_resample_torch(
        ...     obj, dz_now=0.5, resample_mode="scale_Nlayer",
        ...     resample_value=2.0, output_type="complex"
        ... )
        >>> out.shape

        Resample to a target of 64 layers, keeping total thickness fixed:

        >>> out_amp, out_phase = complex_object_z_resample_torch(
        ...     obj, dz_now=0.5, resample_mode="target_Nlayer",
        ...     resample_value=64, output_type="amp_phase"
        ... )
    """
    import torch
    from torch.nn.functional import interpolate
    
    # Assign variables
    Nz_now, Ny_now, Nx_now = obj.shape[-3:]
    
    # Setup resampling modes and scaling constants
    if resample_mode == 'scale_Nlayer':
        scale_factors = [resample_value, 1, 1]
        sizes = None
        Nz_scale = resample_value
        
    elif resample_mode == 'scale_slice_thickness':
        scale_factors = [1/resample_value, 1, 1]
        sizes = None
        Nz_scale = 1/resample_value
        
    elif resample_mode == 'target_Nlayer':
        scale_factors = None
        sizes = [int(resample_value), Ny_now, Nx_now]
        Nz_scale = resample_value/Nz_now
        
    elif resample_mode == 'target_slice_thickness':
        scale_factors = [dz_now/resample_value, 1, 1]
        sizes = None
        Nz_scale = dz_now/resample_value
        
    else:
        raise ValueError(f"Supported obj_z_resample modes are 'scale_Nlayer', 'scale_slice_thickness', 'target_Nlayer', and 'target_slice_thickness', got {resample_mode}.")
    
    # Check scale factor validity
    if Nz_now * Nz_scale < 1:
        raise ValueError(f"Detected target Nlayer = {Nz_now * Nz_scale:.3f} < 1 (single slice), please check your 'obj_z_resampling' settings.")
    
    # Preprocess obj into torch tensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not isinstance(obj, torch.Tensor):
        obj_tensor = torch.tensor(obj, dtype=torch.complex64, device=device)
    else:
        obj_tensor = obj.to(dtype=torch.complex64, device=device)
    
    # Make it into 5D (1,omode,Nz,Ny,Nx) for 3D interpolation
    if obj_tensor.ndim == 3:
        orig_ndim = 3
        obj_tensor = obj_tensor.unsqueeze(0).unsqueeze(0)
    elif obj_tensor.ndim == 4:
        orig_ndim = 4
        obj_tensor = obj_tensor.unsqueeze(0)
    elif obj_tensor.ndim == 5:
        orig_ndim = 5
    else:
        raise ValueError(f"Complex object 3D interpolation only supports 3, 4, 5D tensor, got {obj_tensor.ndim}.")
    
    # Split into amplitude and phase parts
    obja = torch.abs(obj_tensor)
    objp = torch.angle(obj_tensor)
    
    # Apply resampling with proper value scaling to conserve prod(amp, axis='depth'), sum(phase, axis='depth'), and total thickness
    obja_resample = torch.exp(interpolate(torch.log(obja), size=sizes, scale_factor=scale_factors, mode='area') / Nz_scale)
    objp_resample = interpolate(objp, size=sizes, scale_factor=scale_factors, mode='area') / Nz_scale
    
    # Handle outputs
    if output_type == 'complex':
        out = torch.polar(obja_resample, objp_resample)
    elif output_type == 'amplitude':
        out = obja_resample
    elif output_type == 'phase':
        out = objp_resample
    elif output_type == 'amp_phase':
        out = (obja_resample, objp_resample)
    else:
        raise ValueError(
            f"output_type must be one of 'complex', 'amplitude', 'phase', 'amp_phase', "
            f"got {output_type}"
        )

    # Reduce back to original ndim
    if orig_ndim == 3:
        if isinstance(out, tuple):
            out = tuple(o.squeeze(0).squeeze(0) for o in out)
        else:
            out = out.squeeze(0).squeeze(0)
    elif orig_ndim == 4:
        if isinstance(out, tuple):
            out = tuple(o.squeeze(0) for o in out)
        else:
            out = out.squeeze(0)

    # Convert to numpy if requested
    if return_np:
        if isinstance(out, tuple):
            out = tuple(o.detach().cpu().numpy() for o in out)
        else:
            out = out.detach().cpu().numpy()

    return out

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
    wavelength = get_EM_constants(acceleration_voltage=kv, output_type='wavelength') #angstrom
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

# Propagator functions
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

def near_field_evolution_torch(Npix_shape, dx, dz, lambd, dtype=torch.complex64, device='cuda'):
    """ Fresnel propagator """
    # Translated and simplified from Yi's fold_slice Matlab implementation into PyTorch by Chia-Hao Lee
    # This is currently only used in 'obj_z_recenter' constraint to shift the probe defocus.
    # The forward pass uses the propagator direcly constructed in `PtychoModel.get_propagators`` for efficiency.
    from ptyrad.utils import ifftshift2

    ygrid = (torch.arange(-Npix_shape[0] // 2, Npix_shape[0] // 2, device=device) + 0.5) / Npix_shape[0]
    xgrid = (torch.arange(-Npix_shape[1] // 2, Npix_shape[1] // 2, device=device) + 0.5) / Npix_shape[1]

    # Standard ASM
    k  = 2 * torch.pi / lambd
    ky = 2 * torch.pi * ygrid / dx
    kx = 2 * torch.pi * xgrid / dx
    Ky, Kx = torch.meshgrid(ky, kx, indexing="ij")
    H = ifftshift2(torch.exp(1j * dz * torch.sqrt(k ** 2 - Kx ** 2 - Ky ** 2)), ) # H has zero frequency at the corner in k-space

    return H.to(dtype)