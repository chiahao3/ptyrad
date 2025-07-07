"""
Image processing tools for CoM, fitting, normalization, shifting, etc.

"""

import numpy as np
import torch
from scipy.optimize import minimize
from torch.fft import fft2, fftfreq, ifft2

from .common import vprint
from .math_ops import fftshift2, ifftshift2, make_gaussian_mask, torch_phasor


# Some quick estimation analysis tools
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
        grid_y, grid_x = torch.meshgrid(fftfreq(ny, 1 / ny, device=device), fftfreq(nx, 1 / nx, device=device), indexing='ij')
    else:
        grid_y, grid_x = torch.meshgrid(torch.arange(ny, device=device), torch.arange(nx, device=device), indexing='ij')
    
    # Compute total intensity
    total_intensity = torch.sum(image, dim = (-2,-1)).mean()
    
    # Compute weighted sum of x and y coordinates
    center_y = torch.sum(grid_y * image, dim = (-2,-1)) / total_intensity
    center_x = torch.sum(grid_x * image, dim = (-2,-1)) / total_intensity
    
    return center_y, center_x

def get_blob_size(dx, blob, output='d90', plot_profile=False, verbose=True):
    import matplotlib.pyplot as plt
    """ Get the probe / blob size

    Args:
        dx (float): px size in Ang
        blob (array): the probe/blob image, note that we assume the input is already directly measurable and no squaring is needed, centered, and background free
        plot_profile (bool): Flag for plotting the profile or not 

    Returns:
        D50*dx: D50 in Ang
        D90*dx: D90 in Ang
        radius_rms*dx: RMS radius in Ang
        radial_profile: radially averaged profile
        radial_sum: radial profile without normalizing by the ring area
        fig: Line profile figure
    """
    def get_radial_profile(data, center):
        # The radial intensity is calculated up to the corners
        # So len(radialprofile) will be len(data)/sqrt(2)
        # The bin width is set to be the same with original data spacing (dr = dx)
        y, x = np.indices((data.shape))
        r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        r = r.astype(int)
        tbin = np.bincount(r.ravel(), data.ravel())
        nr = np.bincount(r.ravel())
        radial_profile = tbin / nr
        radial_sum = tbin
        return radial_profile, radial_sum

    radial_profile, radial_sum = get_radial_profile(blob, (len(blob)//2, len(blob)//2))
    #print("sum(radial_sum) = %.5f " %(np.sum(radial_sum)))

    # Calculate the rms radius, in px
    x = np.arange(len(radial_profile))
    radius_rms = np.sqrt(np.sum(x**2*radial_profile*x)/np.sum(radial_profile*x))

    # Calculate FWHM
    
    HWHM = np.max(np.where((radial_profile / radial_profile.max()) >=0.5))
    
    # Calculate D50, D90
    cum_sum = np.cumsum(radial_sum)

    # R50, 90 without normalization
    R50 = np.min(np.where(cum_sum>=0.50*np.sum(radial_sum))[0])
    R90 = np.min(np.where(cum_sum>=0.90*np.sum(radial_sum))[0])
    R99 = np.min(np.where(cum_sum>=0.99*np.sum(radial_sum))[0])
    R995 = np.min(np.where(cum_sum>=0.995*np.sum(radial_sum))[0])
    R999 = np.min(np.where(cum_sum>=0.999*np.sum(radial_sum))[0])

    D50  = (2*R50+1)
    D90  = (2*R90+1)
    D99  = (2*R99+1)
    D995 = (2*R995+1)
    D999 = (2*R999+1)
    FWHM = (2*HWHM+1)

    if plot_profile:
        
        num_ticks = 11
        x = dx*np.arange(len(radial_profile))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.title("Radially averaged profile")
        plt.margins(x=0, y=0)
        ax.plot(x, radial_profile/np.max(radial_profile), label='Radially averaged profile')
        #plt.plot(x, cum_sum, 'k--', label='Integrated current')
        plt.vlines(x=R50*dx, ymin=0, ymax=1, color="tab:orange", linestyle=":", label='R50') #Draw vertical lines at the data coordinate, in this case would be Ang.
        plt.vlines(x=R90*dx, ymin=0, ymax=1, color="tab:red", linestyle=":", label='R90')
        plt.vlines(x=HWHM*dx, ymin=0, ymax=1, color="tab:blue", linestyle=":", label='FWHM')
        plt.vlines(x=radius_rms*dx, ymin=0, ymax=1, color="tab:green", linestyle=":", label='Radius_RMS')
        plt.xticks(np.arange(num_ticks)*np.round(len(radial_profile)*dx/num_ticks, decimals = 1-int(np.floor(np.log10(len(radial_profile)*dx)))))
        ax.set_xlabel(r"Distance from blob center ($\AA$)")
        ax.set_ylabel("Normalized intensity")
        plt.legend()
        plt.show()

    if output == 'd50':
        out = D50*dx
    elif output =='d90':
        out =  D90*dx
    elif output =='d99':
        out =  D99*dx
    elif output =='d995':
        out =  D995*dx
    elif output =='d999':
        out =  D999*dx
    elif output =='radius_rms':
        out =  radius_rms*dx
    elif output =='FWHM':
        out =  FWHM*dx
    elif output =='radial_profile':
        out =  radial_profile
    elif output =='radial_sum':
        out =  radial_sum
    elif output =='fig':
        out =  fig
    else:
        raise ValueError(f"output ={output} not implemented!")
    
    if output not in ['radial_profile', 'radial_sum', 'fig'] and verbose:
        vprint(f'{output} = {out/dx:.3f} px or {out:.3f} Ang')
    return out

def guess_radius_of_bright_field_disk(image: np.ndarray, thresh: float=0.5):
    """ Utility function that returns an estimate of the radius of rbf from CBED """
    # meas: 2D array of (ky,kx)
    # thresh: 0.5 for FWHM, 0.1 for Full-width at 10th maximum
    max_val = np.max(image)
    binary_img = image > (max_val * thresh)
    area = np.sum(binary_img)
    rbf = np.sqrt(area / np.pi) # Assume the region is circular
    return rbf

# Use in initial estimation of CBED geometry (center, radius, and edge blur)
def fit_cbed_pattern(image: np.ndarray, initial_guess=None, verbose=False):
    """
    Estimate the center, radius, and std of a CBED pattern by minimizing
    the difference between the observed image and a synthetic model.
    
    Args:
        image (np.ndarray): The input image to fit.
        initial_guess (dict, optional): Dictionary with initial guess parameters.
        verbose (bool): Whether to print detailed information during fitting.
        
    Returns:
        dict: Dictionary containing the fitted parameters as dict['center', 'radius', 'std'].
    """
    Npix = image.shape[0]
    image = image / image.max() # Make sure it's normalized to max at 1 like our mask
    assert image.shape[0] == image.shape[1], "Only square images supported for now."

    def loss(params):
        y0, x0, r, std = params  # Note: y0, x0 order to match center=(y,x) in make_gaussian_mask
        model = make_gaussian_mask(Npix, radius=r, std=std, center=(y0, x0))
        return np.mean((image - model) ** 2)  # Mean Squared Error

    # Set initial guess
    if initial_guess is None:
        # Try to estimate initial parameters from the image
        # Find approximate center by calculating the center of mass
        y_indices, x_indices = np.indices(image.shape)
        total_mass = np.sum(image)
        if total_mass > 0:
            y0_guess = np.sum(y_indices * image) / total_mass
            x0_guess = np.sum(x_indices * image) / total_mass
        else:
            y0_guess, x0_guess = Npix / 2, Npix / 2
            
        r_guess = guess_radius_of_bright_field_disk(image)
        std_guess = 0.5  # Start with a reasonable Gaussian blur
    else:
        # Use provided initial guess
        center = initial_guess.get("center", (Npix / 2, Npix / 2))
        y0_guess, x0_guess = center
        r_guess = initial_guess.get("radius", Npix / 4)
        std_guess = initial_guess.get("std", 0.5)
    
    p0 = [y0_guess, x0_guess, r_guess, std_guess]
    
    vprint(f"Initial guess: center=({y0_guess:.2f}, {x0_guess:.2f}), radius={r_guess:.2f}, Gaussian blur std={std_guess:.2f}", verbose=verbose)
        
    # Use tighter bounds for optimization
    bounds = [(0, Npix-1), (0, Npix-1), (1, Npix/2), (0, 5)]

    # Run optimization with more iterations and a higher tolerance
    options = {'maxiter': 1000, 'disp': verbose}
    result = minimize(loss, p0, bounds=bounds, method='L-BFGS-B', options=options)
    counts = 1
    
    # Try multiple starting points if the first optimization doesn't succeed
    if not result.success or result.fun > 0.01:
        vprint("First optimization attempt didn't converge well, trying different starting points", verbose=verbose)
        
        # Try a few different starting points
        best_result = result
        shift_range = np.linspace(-Npix/10,  Npix/10, 10)
        for shift_y in shift_range:
            for shift_x in shift_range:
                counts += 1
                
                new_p0 = [y0_guess + shift_y, x0_guess + shift_x, r_guess, std_guess]
                new_result = minimize(loss, new_p0, bounds=bounds, method='L-BFGS-B', options=options)
                
                if new_result.fun < best_result.fun:
                    best_result = new_result
                    if verbose:
                        vprint(f"Found better solution with starting point at ({new_p0[0]:.2f}, {new_p0[1]:.2f})")
        vprint(f"Total fitting trials with different initial guesses = {counts}", verbose=verbose)
        result = best_result

    y0, x0, r, std = result.x
    vprint(f"Final fit: center=({y0:.2f}, {x0:.2f}), radius={r:.2f}, Gaussian blur std={std:.2f}", verbose=verbose)
    return {
        "center": (y0, x0),
        "radius": r,
        "std": std,
        "success": result.success,
        "fun": result.fun
    }

def get_local_obj_tilts(pos, objp, dx, slice_thickness, slice_indices, blob_params, window_size=9):
    """ Estimate the local obj tilts from relative atomic column shifts """
    # objp (Nz, Ny, Nx)
    # pos: probe position at integer px sites, (N,2)

    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata
    from scipy.ndimage import center_of_mass
    from scipy.optimize import curve_fit
    from skimage.feature import blob_log

    # Choose the 2 slices from objp and detect blobs from the top slice
    slice_t, slice_b = slice_indices
    height = (slice_b - slice_t)*slice_thickness
    print(f"The height difference between slices {(slice_t, slice_b)} is {height:.2f} Ang")

    target_stack = objp[[slice_t,slice_b]]
    blobs = blob_log(target_stack[0], **blob_params)
    print(f"Found {len(blobs)} blobs with mean radius of {1.414*blobs.mean(0)[-1]:.2f} px or {dx*1.414*blobs.mean(0)[-1]:.2f} Ang")
    
    # Plot the detected blobs
    fig, ax = plt.subplots(figsize=(18,16))
    ax.imshow(target_stack[0])
    for blob in blobs:
        y, x, r = blob
        c = plt.Circle((x, y), r, linewidth=2, fill=False)
        ax.add_patch(c)
    plt.show()
    
    # Get the CoM of each atomic column for both top and bottom slices
    row_start = np.uint32(blobs[:,0]-window_size//2)
    row_end   = np.uint32(blobs[:,0]+window_size//2+1)
    col_start = np.uint32(blobs[:,1]-window_size//2)
    col_end   = np.uint32(blobs[:,1]+window_size//2+1)
    coord_t   = np.zeros((len(blobs),2))
    coord_b   = np.zeros((len(blobs),2))

    for i in range(len(blobs)):
        crop_img_t = target_stack[0][row_start[i]:row_end[i], col_start[i]:col_end[i]]
        crop_img_b = target_stack[1][row_start[i]:row_end[i], col_start[i]:col_end[i]]
        coord_t[i] = center_of_mass(crop_img_t) + blobs[i,:-1] - window_size//2
        coord_b[i] = center_of_mass(crop_img_b) + blobs[i,:-1] - window_size//2
    shift_vecs = coord_b - coord_t # This is the needed tilt to correct the obj tilt so it's pointing from top to bottom

    # Plot the detected CoM
    fig, axs = plt.subplots(1,2, figsize=(8,4))
    im0 = axs[0].imshow(crop_img_t)
    im1 = axs[1].imshow(crop_img_b)
    axs[0].set_title(f"crop_img_t \n {coord_t[-1].round(2)}")
    axs[1].set_title(f"crop_img_b \n {coord_b[-1].round(2)}")
    fig.colorbar(im0, shrink=0.7)
    fig.colorbar(im1, shrink=0.7)
    plt.show()
    
    # Plot the tilt vectors
    X = coord_t[:,1]
    Y = coord_t[:,0]
    U = shift_vecs[:,1]
    V = shift_vecs[:,0]
    M = np.arctan(np.hypot(U,V)*dx/height)*1e3

    fig, ax = plt.subplots(figsize=(16,12))
    plt.title("Needed local object tilts", fontsize=16)
    ax.imshow(target_stack[0], cmap='gray')
    q = ax.quiver(X, Y, U, V, M, pivot='mid', angles='xy', scale_units='xy')
    cbar = fig.colorbar(q, shrink=0.75)
    cbar.ax.set_ylabel('mrad')
    plt.show()
    
    # Interpolate tilt_y, tilt_x map
    tilt_y = np.arctan(V*dx/height)*1e3
    tilt_x = np.arctan(U*dx/height)*1e3

    xnew, ynew= np.mgrid[0:target_stack.shape[-2]:1, 0:target_stack.shape[-1]:1]
    tilt_y_interp = griddata(np.stack([Y,X], -1), tilt_y ,(xnew, ynew), method='cubic')
    tilt_x_interp = griddata(np.stack([Y,X], -1), tilt_x ,(xnew, ynew), method='cubic')

    fig, axs = plt.subplots(1,2, figsize=(12,6))
    im0=axs[0].imshow(tilt_y_interp)
    im1=axs[1].imshow(tilt_x_interp)
    axs[0].set_title("tilt_y_interp")
    axs[1].set_title("tilt_x_interp")
    cbar0 = fig.colorbar(im0, shrink=0.7)
    cbar0.ax.set_ylabel('mrad')
    cbar1 = fig.colorbar(im1, shrink=0.7)
    cbar1.ax.set_ylabel('mrad')
    plt.show()
    
    # Use curve_fit to extrapolate to the entire FOV
    def surface_fn(t, a1, b1, c1, d):
        y,x = t
        return  a1*x + b1*y + c1*x*y + d

    xdata = np.vstack((Y,X))
    ydata_tilt_y = tilt_y
    ydata_tilt_x = tilt_x
    popt_tilt_y, _ = curve_fit(surface_fn, xdata, ydata_tilt_y)
    popt_tilt_x, _ = curve_fit(surface_fn, xdata, ydata_tilt_x)
    
    # Implanting griddata interpolated values into the fitted background
    surface_tilt_y = surface_fn(np.stack((ynew,xnew)), *popt_tilt_y)
    surface_tilt_x = surface_fn(np.stack((ynew,xnew)), *popt_tilt_x)

    mask_tilt_y = ~np.isnan(tilt_y_interp)
    surface_tilt_y[mask_tilt_y] = tilt_y_interp[mask_tilt_y]
    mask_tilt_x = ~np.isnan(tilt_x_interp)
    surface_tilt_x[mask_tilt_x] = tilt_x_interp[mask_tilt_x]

    fig, axs = plt.subplots(1,2, figsize=(12,6))
    im0=axs[0].imshow(surface_tilt_y)
    im1=axs[1].imshow(surface_tilt_x)
    axs[0].set_title("surface_tilt_y")
    axs[1].set_title("surface_tilt_x")
    cbar0 = fig.colorbar(im0, shrink=0.7)
    cbar0.ax.set_ylabel('mrad')
    cbar1 = fig.colorbar(im1, shrink=0.7)
    cbar1.ax.set_ylabel('mrad')
    plt.show()
    
    # Sample the surface with our probe position
    tilt_ys = surface_tilt_y[pos[:,0], pos[:,1]]
    tilt_xs = surface_tilt_x[pos[:,0], pos[:,1]]
    obj_tilts = np.stack([tilt_ys, tilt_xs], axis=-1)

    fig, axs = plt.subplots(1,2, figsize=(12,4))
    im0=axs[0].scatter(x=pos[:,1], y=pos[:,0], c=obj_tilts[:,0])
    im1=axs[1].scatter(x=pos[:,1], y=pos[:,0], c=obj_tilts[:,1])
    axs[0].invert_yaxis()
    axs[1].invert_yaxis()

    axs[0].set_title("tilt_ys")
    axs[1].set_title("tilt_xs")
    cbar0 = fig.colorbar(im0, shrink=0.7)
    cbar0.ax.set_ylabel('mrad')
    cbar1 = fig.colorbar(im1, shrink=0.7)
    cbar1.ax.set_ylabel('mrad')
    plt.show()

    return obj_tilts

# This is used across the paper figure notebook but not really in the package
def center_crop(image, crop_height, crop_width, offset = (0,0)):
    """
    Center crops a 2D or 3D array (e.g., an image).

    Args:
        image (numpy.ndarray): The input array to crop. Can be 2D (H, W) or 3D (H, W, C).
        crop_height (int): The desired height of the crop.
        crop_width (int): The desired width of the crop.

    Returns:
        numpy.ndarray: The cropped image.
    """
    if len(image.shape) not in [2, 3]:
        raise ValueError("Input image must be a 2D or 3D array.")

    height, width = image.shape[-2:]

    if crop_height > height or crop_width > width:
        raise ValueError("Crop size must be smaller than the input image size.")

    start_y = (height - crop_height) // 2 + offset[0]
    start_x = (width - crop_width) // 2 + offset[0]

    return image[..., start_y:start_y + crop_height, start_x:start_x + crop_width]

# These are called during save.save_results()
def normalize_from_zero_to_one(arr):
    norm_arr = (arr - arr.min())/(arr.max()-arr.min())
    return norm_arr

def normalize_by_bit_depth(arr, bit_depth):

    if bit_depth == '8':
        norm_arr_in_bit_depth = np.uint8(255*normalize_from_zero_to_one(arr))
    elif bit_depth == '16':
        norm_arr_in_bit_depth = np.uint16(65535*normalize_from_zero_to_one(arr))
    elif bit_depth == '32':
        norm_arr_in_bit_depth = np.float32(normalize_from_zero_to_one(arr))
    elif bit_depth == 'raw':
        norm_arr_in_bit_depth = np.float32(arr)
    else:
        print(f'Unsuported bit_depth :{bit_depth} was passed into `result_modes`, `raw` is used instead')
        norm_arr_in_bit_depth = np.float32(arr)
    
    return norm_arr_in_bit_depth

# These are called inside constraints.py / CombinedConstraint > apply_obj_zblur
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

# These are used for meas_pad
def create_one_hot_mask(image, percentile):
    threshold = np.percentile(image, percentile)
    mask = image <= threshold
    vprint(f"Using percentile = {percentile:.2f}% to create an one-hot mask for measurements amplitude background fitting")
    radius_px = np.sqrt(np.abs(1-mask).sum() / np.pi)
    radius_r  = radius_px / (len(mask)//2)
    vprint(f"The mask has roughly {radius_px:.2f} px in radius, or {radius_r:.2f} of the distance from center to edge of the image")
    return mask.astype(int)

def fit_background(image, mask, fit_type='exp'):
    from scipy.optimize import curve_fit

    from ptyrad.utils.math_ops import exponential_decay, power_law
    
    y, x = np.indices(image.shape)
    center = np.array(image.shape) // 2
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2) + 1e-10
    
    masked_r = r[mask == 1]
    masked_image = image[mask == 1]
    
    if fit_type == 'exp':
        initial_guess = [np.max(masked_image), 0.1]  # [a_guess, b_guess]
        bounds = ([0, 0], [np.inf, np.inf])  # a > 0, b > 0
        popt, _ = curve_fit(exponential_decay, masked_r, masked_image, p0=initial_guess, bounds=bounds,maxfev=10000)
        vprint(f"Fitted a = {popt[0]:.4f}, b = {popt[1]:.4f} for exponential decay: y = a*exp(-b*r)")
    elif fit_type == 'power':
        initial_guess = [np.max(masked_image), 1]  # [a_guess, b_guess]
        bounds = ([0, 0], [np.inf, np.inf])  # a > 0, b > 0
        popt, _ = curve_fit(power_law, masked_r, masked_image, p0=initial_guess, bounds=bounds, maxfev=10000)
        vprint(f"Fitted a = {popt[0]:.4f}, b = {popt[1]:.4f} for power law decay: y = a*r^-b")
    else:
        raise ValueError("fit_type must be 'exp' or 'power'")
    
    return popt

# This is only called inside `models.py / PtychoAD`
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
                             from 0 to 1

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
    
    ndim = img.ndim                                                                   # Get the total img ndim so that the shift is dimension-indepent
    shifts = shifts[(...,) + (None,) * ndim]                                          # Expand shifts to (Nb,2,1,1,...) so shifts.ndim = ndim+2. It was written as `shifts = shifts[..., *[None]*ndim]` for Python 3.11 or above with better readability
    grid = grid[(slice(None),) + (None,) * (ndim - 1) + (...,)]                       # Expand grid to (2,1,1,...,Ny,Nx) so grid.ndim = ndim+2. It was written as `grid = grid[:,*[None]*(ndim-1), ...]` for Python 3.11 or above with better readability
    shift_y, shift_x = shifts[:, 0], shifts[:, 1]                                     # shift_y, shift_x are (Nb,1,1,...) with ndim singletons, so the shift_y.ndim = ndim+1
    ky, kx = grid[0], grid[1]                                                         # ky, kx are (1,1,...,Ny,Nx) with ndim-2 singletons, so the ky.ndim = ndim+1
    phase = -2*torch.pi * (shift_x * kx + shift_y * ky)
    w = torch_phasor(phase)                                                           # w = (Nb, 1,1,...,Ny,Nx) so w.ndim = ndim+1. w is at the center.
    shifted_img = ifft2(ifftshift2(fftshift2(fft2(img)) * w))                         # For real-valued input, take shifted_img.real
    
    # Note that for imshift, it's better to keep fft2(img) than fft2(ifftshift2(img))
    # While fft2(img).angle() might seem serrated, it's indeed better to keep it as is, which is essentially setting the center as the origin for FFT.
    
    return shifted_img