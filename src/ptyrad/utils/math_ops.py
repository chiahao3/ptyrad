"""
Math operations of FFTs, masks, affine transformation matrices, etc.

"""

from typing import Optional, Tuple

import numpy as np
import torch


def torch_phasor(phase):
    """
    Creates a complex tensor with unit magnitude using the phase.

    Args:
        phase (torch.Tensor): phase angle for the exp(i*theta)
        
    Note:
        This util function is created so torch.compile can properly handle complex tensors,
        because torch.exp(1j*phase) involves the 1j which is actually a Python built-in that can't be traced.
    """
    return torch.polar(torch.ones_like(phase), phase)
    
def exponential_decay(r, a, b):
    return a * np.exp(-b * r)

def power_law(r, a, b):
    return a * r**-b

def fftshift2(x):
    """ A wrapper over torch.fft.fftshift for the last 2 dims """
    # Note that fftshift and ifftshift are only equivalent when N = even 
    return torch.fft.fftshift(x, dim=(-2,-1))  

def ifftshift2(x):
    """ A wrapper over torch.fft.ifftshift for the last 2 dims"""
    # Note that fftshift and ifftshift are only equivalent when N = even 
    return torch.fft.ifftshift(x, dim=(-2,-1))  

def mfft2(im):
    # Periodic Artifact Reduction in Fourier Transforms of Full Field Atomic Resolution Images
    # https://doi.org/10.1017/S1431927614014639
    rows, cols = im.shape
    
    # Compute boundary conditions
    s = np.zeros_like(im)
    s[0, :] = im[0, :] - im[rows-1, :]
    s[rows-1, :] = -s[0, :]
    s[:, 0] += im[:, 0] - im[:, cols-1]
    s[:, cols-1] -= im[:, 0] - im[:, cols-1]

    # Create grid for computing Poisson solution
    cx, cy = np.meshgrid(2 * np.pi * np.arange(cols) / cols, 
                          2 * np.pi * np.arange(rows) / rows)

    # Generate smooth component from Poisson Eq with boundary condition
    D = 2 * (2 - np.cos(cx) - np.cos(cy))
    D[0, 0] = np.inf  # Enforce zero mean & handle division by zero
    S = np.fft.fft2(s) / D

    P = np.fft.fft2(im) - S  # FFT of periodic component
    return P, S

def make_sigmoid_mask(Npix: int, relative_radius: float = 2/3, relative_width: float = 0.2, center: Optional[Tuple[float, float]] = None):
    """
    Create a 2D circular mask with a sigmoid transition.

    Args:
        Npix (int): Size of the square mask (Npix x Npix).
        relative_radius (float): Relative radius of the circular mask where the sigmoid equals 0.5, 
            as a fraction of the image size.
        relative_width (float): Relative width of the sigmoid transition, as a fraction of the image size.
        center (Optional[Tuple[float, float]]): (y, x) coordinates of the center of the circle. 
            Defaults to the center of the image.

    Returns:
        torch.Tensor: A 2D circular mask with a sigmoid transition.
    
    Notes:
        - The default `relative_radius=2/3` is inspired by its use in abTEM to reduce edge artifacts 
          in diffraction patterns. It sets an antialias cutoff frequency at 2/3 of the simulated kMax. 
          https://abtem.readthedocs.io/en/latest/user_guide/appendix/antialiasing.html
        - The `relative_width` controls the steepness of the sigmoid transition. Smaller values result 
          in sharper transitions, while larger values produce smoother transitions.
    """

    def scaled_sigmoid(x, offset=0, scale=1):
        # If scale =  1, y drops from 1 to 0 between (-0.5,0.5), or effectively 1 px
        # If scale = 10, it takes roughly 10 px for y to drop from 1 to 0
        return 1 / (1 + torch.exp((x - offset) / scale * 10))

    # Set default center if not provided
    if center is None:
        center = (Npix // 2, Npix // 2)  # Use integer division for consistency

    # Create a grid of coordinates
    ky = torch.arange(Npix, dtype=torch.float32)
    kx = torch.arange(Npix, dtype=torch.float32)
    grid_ky, grid_kx = torch.meshgrid(ky, kx, indexing='ij')

    # Compute the distance from the specified center
    kR = torch.sqrt((grid_ky - center[0])**2 + (grid_kx - center[1])**2)

    # Apply the scaled sigmoid function
    sigmoid_mask = scaled_sigmoid(kR, offset=Npix * relative_radius / 2, scale=relative_width * Npix)

    return sigmoid_mask

def make_gaussian_mask(Npix: int, radius: float, std: float, center: Optional[Tuple[float, float]] = None):
    """
    Create a 2D Gaussian-blurred circular mask.

    Args:
        Npix (int): Size of the square mask (Npix x Npix).
        radius (float): Radius of the circular mask.
        std (float): Standard deviation of the Gaussian blur.
        center (tuple): (y, x) coordinates of the center of the circle.

    Returns:
        np.ndarray: A 2D Gaussian-blurred circular mask.
    """
    from scipy.ndimage import gaussian_filter

    # Set default center if not provided
    if center is None:
        center = (Npix / 2, Npix / 2)
    
    # Create a grid of coordinates
    y = np.linspace(0, Npix - 1, Npix)
    x = np.linspace(0, Npix - 1, Npix)
    grid_y, grid_x = np.meshgrid(y, x, indexing='ij')

    # Compute the distance from the center
    dist_from_center = np.sqrt((grid_y - center[0])**2 + (grid_x - center[1])**2)

    # Create a binary circular mask
    circular_mask = (dist_from_center <= radius).astype(float)

    # Apply Gaussian blur to the circular mask
    gaussian_mask = gaussian_filter(circular_mask, sigma=std)

    return gaussian_mask

def approx_torch_quantile(t, q, sample_size=16_000_000):
    """
    Approximated quantile to prevent the 2^24 element (roughly 16.7M) limitation of torch.quantile as of now.
    See https://github.com/pytorch/pytorch/issues/64947
    `RuntimeError: quantile() input tensor is too large`
    Note that this approximated quantile would have some randomness.

    Args:
        t (torch.Tensor): Input torch tensor
        q (float): Targeted quantile number [0,1]
        sample_size (int, optional): Number of randomly selected elements used to approximate the true quantile. Defaults to 16_000_000.

    Returns:
        float: The approximated quantile value for the input tensor
    """
    # flatten
    flat = t.view(-1)
    # random subsample if necessary
    if flat.numel() > sample_size:
        idx = torch.randint(0, flat.numel(), (sample_size,), device=flat.device)
        flat = flat[idx]
    return torch.quantile(flat, q)

# Affine
def compose_affine_matrix(scale, asymmetry, rotation, shear):
    # Adapted from PtychoShelves +math/compose_affine_matrix.m
    # The input rotation and shear is in unit of degree
    rotation_rad = np.radians(rotation)
    shear_rad = np.radians(shear)
    
    A1 = np.array([[scale, 0], [0, scale]])
    A2 = np.array([[1 + asymmetry/2, 0], [0, 1 - asymmetry/2]])
    A3 = np.array([[np.cos(rotation_rad), np.sin(rotation_rad)], [-np.sin(rotation_rad), np.cos(rotation_rad)]])
    A4 = np.array([[1, 0], [np.tan(shear_rad), 1]])
    
    affine_mat = A1 @ A2 @ A3 @ A4

    return affine_mat

def decompose_affine_matrix(input_affine_mat):
    from scipy.optimize import least_squares
    def err_fun(x):
        scale, asymmetry, rotation, shear = x
        fit_affine_mat = compose_affine_matrix(scale, asymmetry, rotation, shear)
        return (input_affine_mat - fit_affine_mat).ravel()

    # Initial guess
    initial_guess = np.array([1, 0, 0, 0])
    result = least_squares(err_fun, initial_guess)
    scale, asymmetry, rotation, shear = result.x

    return scale, asymmetry, rotation, shear

def get_decomposed_affine_matrix_from_bases(input, output):
    """ Fit the affine matrix components from input and output matrices A and B """
    # This util function is used to quickly estimate the needed affine transformation for scan positions
    # If we know the lattice constant and angle between lattice vectors, then we can easily correct the scale, asymmetry, and shear
    # The global rotation of the object is NOT defined by lattice constant/angle so we still need to compare with the actual CBED
    # Typical usage of this function is to first construct A by measuring the lattice vectors of a reconstructed object suffers from affine transformation
    # Then estimate ideal lattice vectors with prior knowledge (lattice constant and angle)
    # Lastly we use this function to estimate the needed F such that B = F @ A
    
    from scipy.optimize import minimize

    def objective(params, A, B):
        scale, asymmetry, rotation, shear = params
        F = compose_affine_matrix(scale, asymmetry, rotation, shear)
        return np.linalg.norm(B - F @ A)

    initial_guess = [1, 0, 0, 0]  # Initial guess for scale, asymmetry, rotation, shear
    result = minimize(objective, initial_guess, args=(input, output), method='L-BFGS-B')
    
    if result.success:
        (scale, asymmetry, rotation, shear) = result.x
        return (scale, asymmetry, rotation, shear)
    else:
        raise ValueError("Optimization failed")

# DCT/iDCT    
def dct_2d(x: torch.Tensor) -> torch.Tensor:
    """Computes a 2D DCT-II (orthonormalized except for constant factors) using FFT.

    Supports arbitrary batch dimensions. The DCT is applied over the last two
    dimensions (H, W).

    Args:
        x (torch.Tensor): Real-valued input tensor of shape (..., H, W).

    Returns:
        torch.Tensor: DCT coefficients of shape (..., H, W).
    """
    H, W = x.shape[-2:]

    # --- DCT along height (dim = -2) ---
    x_ext_h = torch.cat([x, x.flip(dims=[-2])], dim=-2)  # (..., 2H, W)
    X_h = torch.fft.fft(x_ext_h, dim=-2)

    n_h = torch.arange(H, device=x.device)
    scale_h = torch.exp(-1j * torch.pi * n_h / (2 * H))   # (H,)
    dct_h = (X_h[..., :H, :] * scale_h[:, None]).real * 2

    # --- DCT along width (dim = -1) ---
    x_ext_w = torch.cat([dct_h, dct_h.flip(dims=[-1])], dim=-1)  # (..., H, 2W)
    X_w = torch.fft.fft(x_ext_w, dim=-1)

    n_w = torch.arange(W, device=x.device)
    scale_w = torch.exp(-1j * torch.pi * n_w / (2 * W))   # (W,)
    dct_2d = (X_w[..., :W] * scale_w).real * 2

    return dct_2d

def idct_2d(x: torch.Tensor) -> torch.Tensor:
    """Computes a 2D inverse DCT-II (IDCT) using FFT.

    The inverse restores a real-valued signal and supports arbitrary batch
    dimensions.

    Args:
        x (torch.Tensor): DCT coefficients of shape (..., H, W).

    Returns:
        torch.Tensor: Reconstructed signal of shape (..., H, W).
    """
    H, W = x.shape[-2:]
    X = x.to(torch.complex64)

    # --- Undo width scaling ---
    n_w = torch.arange(W, device=x.device)
    scale_w = torch.exp(1j * torch.pi * n_w / (2 * W))

    Xw = X * scale_w / 2

    # Symmetric extension for width
    # Conjugate mirror excluding the DC term
    Xw_ext = torch.cat(
        [Xw, Xw[..., 1:].flip(dims=[-1]).conj()],
        dim=-1
    )  # (..., H, 2W)

    # IFFT along width
    x_w = torch.fft.ifft(Xw_ext, dim=-1)[..., :W].real

    # --- Undo height scaling ---
    n_h = torch.arange(H, device=x.device)
    scale_h = torch.exp(1j * torch.pi * n_h / (2 * H))

    Xh = x_w.to(torch.complex64) * scale_h[:, None] / 2

    # Symmetric extension for height
    Xh_ext = torch.cat(
        [Xh, Xh[..., 1:, :].flip(dims=[-2]).conj()],
        dim=-2
    )  # (..., 2H, W)

    # IFFT along height
    out = torch.fft.ifft(Xh_ext, dim=-2)[..., :H, :].real
    return out