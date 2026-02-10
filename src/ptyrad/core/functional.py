from typing import Literal, Optional, Tuple, Union

import numpy as np
import torch

# This is currently used in core/constraints and core/forward as FFT aliases
def fftshift2(x):
    """ A wrapper over torch.fft.fftshift for the last 2 dims """
    # Note that fftshift and ifftshift are only equivalent when N = even 
    return torch.fft.fftshift(x, dim=(-2,-1))  

def ifftshift2(x):
    """ A wrapper over torch.fft.ifftshift for the last 2 dims"""
    # Note that fftshift and ifftshift are only equivalent when N = even 
    return torch.fft.ifftshift(x, dim=(-2,-1))  

# This is currently only used in init/initializer
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

# This is currently used in core/constraint.py > get_obj_z_shift
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

# This is currently used in core/constraints.py > apply_obj_zblur
def get_gaussian1d(size, std, norm=False):
    from scipy.signal.windows import gaussian as gaussian1d

    k = gaussian1d(size, std)
    if norm:
        k /= k.sum()
    return k

def gaussian_blur_1d(tensor, kernel_size=5, sigma=0.5):
    # Note that the F.con1d does not have `padding_mode`, so it's default to be 0 padding, which is not ideal for obja
    # tensor_blur = F.conv1d(input=tensor.reshape(-1, 1, tensor.size(-1)), weight=k1d, padding='same').view(*tensor.shape)

    dtype  = tensor.dtype
    device = tensor.device 
    k = torch.from_numpy(get_gaussian1d(kernel_size, sigma, norm=True)).type(dtype).to(device)
    k1d = k.view(1, 1, -1)
    
    gaussian1d = torch.nn.Conv1d(1,1,kernel_size,padding='same', bias=False, padding_mode='replicate')
    gaussian1d.weight = torch.nn.Parameter(k1d)
    tensor_blur = gaussian1d(tensor.reshape(-1, 1, tensor.size(-1))).view(*tensor.shape)
    return tensor_blur

# This is currently used in core/constraints.py > kr_filter, probe_mask_k
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

# This is currently used in core/constraints.py > kr_thrsh
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

# This is currently used in 'obj_z_recenter' constraint to shift the probe defocus.
def near_field_evolution_torch(Npix_shape, dx, dz, lambd, dtype=torch.complex64, device='cuda'):
    """ Fresnel propagator """
    # Translated and simplified from Yi's fold_slice Matlab implementation into PyTorch by Chia-Hao Lee
    # The forward pass uses the propagator direcly constructed in `PtychoModel.get_propagators`` for efficiency.

    ygrid = (torch.arange(-Npix_shape[0] // 2, Npix_shape[0] // 2, device=device) + 0.5) / Npix_shape[0]
    xgrid = (torch.arange(-Npix_shape[1] // 2, Npix_shape[1] // 2, device=device) + 0.5) / Npix_shape[1]

    # Standard ASM
    k  = 2 * torch.pi / lambd
    ky = 2 * torch.pi * ygrid / dx
    kx = 2 * torch.pi * xgrid / dx
    Ky, Kx = torch.meshgrid(ky, kx, indexing="ij")
    H = ifftshift2(torch.exp(1j * dz * torch.sqrt(k ** 2 - Kx ** 2 - Ky ** 2)), ) # H has zero frequency at the corner in k-space

    return H.to(dtype)

# This is currently used in core/models/ptycho > PtychoModel for AD-optimizable propagators
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

# This is currently used in core/models/ptycho_model > get_probes
def imshift_batch(img, shifts, grid):
    """
    Generates a batch of shifted images from a single input image (..., Ny,Nx) with arbitray leading dimensions.
    
    This function shifts a complex/real-valued input image by applying phase shifts in the Fourier domain,
    achieving subpixel shifts in both x and y directions.

    Args:
        img (torch.Tensor): The input image to be shifted. 
                            img could be either a mixed-state complex probe (pmode, Ny, Nx) complex64 tensor, 
                            or a mixed-state pseudo-complex object stack (2,omode,Nz,Ny,Nx) float32 tensor.
        shifts (torch.Tensor): The shifts to be applied to the image. It should be a (Nb,2) tensor and each slice as (shift_y, shift_x).
        grid (torch.Tensor): The k-space grid used for computing the shifts in the Fourier domain. It should be a tensor with shape=(2, Ny, Nx),
                             where Ny and Nx are the height and width of the images, respectively. Note that the grid is normalized so the value spans
                             from [-0.5,0.5)

    Returns:
        shifted_img (torch.Tensor): The batch of shifted images. It has an extra dimension than the input image, i.e., shape=(Nb, ..., Ny, Nx),
                                    where Nb is the number of samples in the input batch.

    Note:
        - The shifts are in unit of pixel. For example, a shift of (0.5, 0.5) will shift the image by half a pixel in both y and x directions, positive is down/right-ward.
        - The function utilizes the fast Fourier transform (FFT) to perform the shifting operation efficiently.
        - Make sure to convert the input image and shifts tensor to the desired device before passing them to this function.
        - The fft2 and fftshifts are all applied on the last 2 dimensions, therefore it's only shifting along y and x directions
        - tensor[None, ...] would add an extra dimension at 0, so `*[None]*ndim` means unwrapping a list of ndim None as [None, None, ...]
        - The img is automatically broadcast to `(Nb, *img.shape)`, so if a batch of images are passed in, each image would be shifted independently
    """
    
    assert img.shape[-2:] == grid.shape[-2:], f"Found incompatible dimensions. img.shape[-2:] = {img.shape[-2:]} while grid.shape[-2:] = {grid.shape[-2:]}"
    
    ndim = img.ndim                                                                   # Get the total img ndim so that the shift is dimension-independent
    shifts = shifts[(...,) + (None,) * ndim]                                          # Expand shifts to (Nb,2,1,1,...) so shifts.ndim = ndim+2. It was written as `shifts = shifts[..., *[None]*ndim]` for Python 3.11 or above with better readability
    grid = grid[(slice(None),) + (None,) * (ndim - 1) + (...,)]                       # Expand grid to (2,1,1,...,Ny,Nx) so grid.ndim = ndim+2. It was written as `grid = grid[:,*[None]*(ndim-1), ...]` for Python 3.11 or above with better readability
    shift_y, shift_x = shifts[:, 0], shifts[:, 1]                                     # shift_y, shift_x are (Nb,1,1,...) with ndim singletons, so the shift_y.ndim = ndim+1
    ky, kx = grid[0], grid[1]                                                         # ky, kx are (1,1,...,Ny,Nx) with ndim-2 singletons, so the ky.ndim = ndim+1
    phase = -2*torch.pi * (shift_x * kx + shift_y * ky)
    w = torch_phasor(phase)                                                           # w = (Nb, 1,1,...,Ny,Nx) so w.ndim = ndim+1. The zero frequency term of w is at the corner.
    shifted_img = torch.fft.ifft2(torch.fft.fft2(img) * w)                            # For real-valued input, take shifted_img.real
    
    return shifted_img

# This is not used in PtyRAD yet, but could be useful for some analysis notebooks
def get_center_of_mass(image, corner_centered=False):
    """ Finds and returns the center of mass of an real-valued 2/3D tensor """
    # The expected input shape can be either (Ny, Nx) or (N, Ny, Nx)
    # The output center_y and center_x will be either (N,) or a scaler tensor
    # Note that for even-number sized arr (like [128,128]), even it's uniformly ones, the "center" would be between pixels like [63.5,63.5]
    # Note that the `corner_centered` flag idea is adapted from py4DSTEM, which is quite handy when we have corner-centered probe or CBED
    # https://github.com/py4dstem/py4DSTEM/blob/dev/py4DSTEM/process/utils/utils.py
    
    ndim = image.ndim
    assert ndim in [2, 3], f"image.ndim must be either 2 or 3, we've got {ndim}"
    
    # Create grid of coordinates
    device = image.device
    (ny, nx) = image.shape[-2:]

    if corner_centered:
        grid_y, grid_x = torch.meshgrid(torch.fft.fftfreq(ny, 1 / ny, device=device), 
                                        torch.fft.fftfreq(nx, 1 / nx, device=device), indexing='ij')
    else:
        grid_y, grid_x = torch.meshgrid(torch.arange(ny, device=device), 
                                        torch.arange(nx, device=device), indexing='ij')
    
    # Compute total intensity
    total_intensity = torch.sum(image, dim = (-2,-1)).mean()
    
    # Compute weighted sum of x and y coordinates
    center_y = torch.sum(grid_y * image, dim = (-2,-1)) / total_intensity
    center_x = torch.sum(grid_x * image, dim = (-2,-1)) / total_intensity
    
    return center_y, center_x