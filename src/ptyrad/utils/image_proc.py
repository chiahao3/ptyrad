"""
Image processing tools for fitting, cropping, normalization, etc.

"""

from typing import Optional, Tuple

import numpy as np
import logging

logger = logging.getLogger(__name__)


# Some quick estimation analysis tools
def get_blob_size(dx, blob, output='d90', plot_profile=False):
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
    
    if output not in ['radial_profile', 'radial_sum', 'fig']:
        logger.info(f'{output} = {out/dx:.3f} px or {out:.3f} Ang')
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
def fit_cbed_pattern(image: np.ndarray, initial_guess=None):
    """
    Estimate the center, radius, and std of a CBED pattern by minimizing
    the difference between the observed image and a synthetic model.
    
    Args:
        image (np.ndarray): The input image to fit.
        initial_guess (dict, optional): Dictionary with initial guess parameters.
        
    Returns:
        dict: Dictionary containing the fitted parameters as dict['center', 'radius', 'std'].
    """
    
    from scipy.optimize import minimize
    
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
    
    logger.info(f"Initial guess: center=({y0_guess:.2f}, {x0_guess:.2f}), radius={r_guess:.2f}, Gaussian blur std={std_guess:.2f}")
        
    # Use tighter bounds for optimization
    bounds = [(0, Npix-1), (0, Npix-1), (1, Npix/2), (0, 5)]

    # Run optimization with more iterations and a higher tolerance
    options = {'maxiter': 1000, 'disp': False}
    result = minimize(loss, p0, bounds=bounds, method='L-BFGS-B', options=options)
    counts = 1
    
    # Try multiple starting points if the first optimization doesn't succeed
    if not result.success or result.fun > 0.01:
        logger.info("First optimization attempt didn't converge well, trying different starting points")
        
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
                    logger.info(f"Found better solution with starting point at ({new_p0[0]:.2f}, {new_p0[1]:.2f})")
        logger.info(f"Total fitting trials with different initial guesses = {counts}")
        result = best_result

    y0, x0, r, std = result.x
    logger.info(f"Final fit: center=({y0:.2f}, {x0:.2f}), radius={r:.2f}, Gaussian blur std={std:.2f}")
    return {
        "center": (y0, x0),
        "radius": r,
        "std": std,
        "success": result.success,
        "fun": result.fun
    }

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

# This is used across the paper figure notebook but not really in the package
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

# These are used for meas_pad
def create_one_hot_mask(image, percentile):
    threshold = np.percentile(image, percentile)
    mask = image <= threshold
    logger.info(f"Using percentile = {percentile:.2f}% to create an one-hot mask for measurements amplitude background fitting")
    radius_px = np.sqrt(np.abs(1-mask).sum() / np.pi)
    radius_r  = radius_px / (len(mask)//2)
    logger.info(f"The mask has roughly {radius_px:.2f} px in radius, or {radius_r:.2f} of the distance from center to edge of the image")
    return mask.astype(int)

def fit_background(image, mask, fit_type='exp'):
    from scipy.optimize import curve_fit
    
    y, x = np.indices(image.shape)
    center = np.array(image.shape) // 2
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2) + 1e-10
    
    masked_r = r[mask == 1]
    masked_image = image[mask == 1]
    
    if fit_type == 'exp':
        initial_guess = [np.max(masked_image), 0.1]  # [a_guess, b_guess]
        bounds = ([0, 0], [np.inf, np.inf])  # a > 0, b > 0
        popt, _ = curve_fit(exponential_decay, masked_r, masked_image, p0=initial_guess, bounds=bounds,maxfev=10000)
        logger.info(f"Fitted a = {popt[0]:.4f}, b = {popt[1]:.4f} for exponential decay: y = a*exp(-b*r)")
    elif fit_type == 'power':
        initial_guess = [np.max(masked_image), 1]  # [a_guess, b_guess]
        bounds = ([0, 0], [np.inf, np.inf])  # a > 0, b > 0
        popt, _ = curve_fit(power_law, masked_r, masked_image, p0=initial_guess, bounds=bounds, maxfev=10000)
        logger.info(f"Fitted a = {popt[0]:.4f}, b = {popt[1]:.4f} for power law decay: y = a*r^-b")
    else:
        raise ValueError("fit_type must be 'exp' or 'power'")
    
    return popt

def exponential_decay(r, a, b):
    return a * np.exp(-b * r)

def power_law(r, a, b):
    return a * r**-b