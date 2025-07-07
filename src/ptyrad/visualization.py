"""
Plotting functions for output figures including forward passes, loss curves, positions, etc.

"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
from numpy.fft import fft2, fftshift, ifftshift

from ptyrad.utils import make_sigmoid_mask, safe_filename, vprint

def plot_sigmoid_mask(Npix, relative_radius, relative_width, img=None, show_circles=False):
    """ Plot a sigmoid mask overlay on img with a line profile """
    # Note that relative_radius ranges from 0 - 1 for center -> edge. radius = 1 corresponds to a inscribed circle
    # While relative_width also ranges from 0 - 1 for Npix * relative_width. width = 0.05 corresponds to a width of 5% of the image width would have sigmoid value change from 0 - 1
    mask = make_sigmoid_mask(Npix, relative_radius, relative_width).detach().cpu().numpy()
    img = np.ones((Npix,Npix)) if img is None else img/img.max()
    masked_img = mask * img
    fig, axs = plt.subplots(1,2, figsize=(13,6))
    fig.suptitle(f"Sigmoid mask with radius = {relative_radius}, width = {relative_width}", fontsize=18)
    im = axs[0].imshow(masked_img)
    axs[0].axhline(y=Npix//2, xmin=0.5, c='r', linestyle='--')
    axs[1].plot(mask[Npix//2, Npix//2:], c='r', label='mask')
    if img is not None:
        axs[1].plot(img[Npix//2, Npix//2:], label='image')
        axs[1].plot(masked_img[Npix//2, Npix//2:], label='masked_img')
    
    # Draw circles on the imshow
    if show_circles:
        circle1 = plt.Circle((Npix // 2, Npix // 2), (relative_radius-relative_width) * Npix/2, color='k', fill=False, linestyle='--')
        circle2 = plt.Circle((Npix // 2, Npix // 2), (relative_radius+relative_width) * Npix/2, color='k', fill=False, linestyle='--')
        axs[0].add_artist(circle1)
        axs[0].add_artist(circle2)
        axs[1].axvline(x=(relative_radius-relative_width) * Npix/2, color='k', linestyle='--')
        axs[1].axvline(x=(relative_radius+relative_width) * Npix/2, color='k', linestyle='--')
    
    fig.colorbar(im, shrink=0.7)
    plt.legend()
    plt.show()

def plot_forward_pass(model, indices, dp_power, show_fig=True, pass_fig=False):
    """ Plot the forward pass for the input torch model """
    # The input is expected to be torch object and the attributes are all torch tensors and will be converted to numpy
    
    # probes_int = (N_i, Ny, Nx), float32 np array
    # obj_ROI = (N_i, omode, Nz, Ny, Nx) -> (N_i, Nz, Ny, Nx), float32 np array
    # For probe, only plot the intensity of incoherently summed mixed-state probe
    # For object, only plot the phase of the weighted sum object mode and sums over z-slices
    # The dp_power here is for visualization purpose, the actual loss function has its own param field
    
    with torch.no_grad():
        probes      = model.get_probes(indices)
        probes_int  = probes.abs().pow(2).sum(1)
        model_DP    = model(indices)
        obj_patches = model.get_obj_patches(indices) # The cache would be cleared right after the mini-batch update so we have to re-calculate it here
        omode_occu  = model.omode_occu
        measured_DP = model.get_measurements(indices)
        
        probes_int  = probes_int.detach().cpu().numpy()
        obja_ROI    = (obj_patches[...,0] * omode_occu[:,None,None,None]).sum(1).detach().cpu().numpy() # obj_ROI = (N_i, Nz,Ny,Nx)
        objp_ROI    = (obj_patches[...,1] * omode_occu[:,None,None,None]).sum(1).detach().cpu().numpy() # obj_ROI = (N_i, Nz,Ny,Nx)
        model_DP    = model_DP.detach().cpu().numpy()
        measured_DP = measured_DP.detach().cpu().numpy()
    
    plt.ioff() # Temporaily disable the interactive plotting mode
    fig, axs = plt.subplots(len(indices), 5, figsize=(24, 5*len(indices)))
    plt.suptitle("Forward pass", fontsize=24)
    
    for i, idx in enumerate(indices):
        # Looping over the N_i dimension
        im00 = axs[i,0].imshow(probes_int[i]) 
        axs[i,0].set_title(f"Probe intensity idx{idx}", fontsize=16)
        fig.colorbar(im00, shrink=0.6)

        im01 = axs[i,1].imshow(obja_ROI[i].prod(0))
        axs[i,1].set_title(f"Object amp. (osum, zprod) idx{idx}", fontsize=16)
        fig.colorbar(im01, shrink=0.6)
        
        im02 = axs[i,2].imshow(objp_ROI[i].sum(0))
        axs[i,2].set_title(f"Object phase (osum, zsum) idx{idx}", fontsize=16)
        fig.colorbar(im02, shrink=0.6)

        im03 = axs[i,3].imshow((model_DP[i]**dp_power))
        axs[i,3].set_title(f"Model DP^{dp_power} idx{idx}", fontsize=16)
        fig.colorbar(im03, shrink=0.6)
        
        im04 = axs[i,4].imshow((measured_DP[i]**dp_power))
        axs[i,4].set_title(f"Data DP^{dp_power} idx{idx}", fontsize=16)
        fig.colorbar(im04, shrink=0.6)
    plt.tight_layout()
    if show_fig:
        plt.show()
    if pass_fig:
        return fig

def plot_obj_tilts_avg(avg_tilt_iters, last_n_iters=2, show_fig=True, pass_fig=False):
    last_n_iters = int(last_n_iters)
    
    # Unpack iteration numbers and tilt values
    iters, tilts = zip(*avg_tilt_iters)  # Separates into two tuples
    tilts = np.vstack(tilts)  # Converts list of (1,2) arrays to (N,2) array
    iters = np.array(iters)  # Convert iteration numbers to a NumPy array

    plt.ioff()  # Temporarily disable interactive mode
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 10), sharex=True)

    # Plot first component (tilt_y)
    axes[0].plot(iters, tilts[:, 0], marker='o', color='C0')
    axes[0].set_ylabel('Avg Obj tilt_y (mrad)', fontsize=16)
    axes[0].set_title(f'Avg Obj tilt_y (mrad): {tilts[-1,0]:.3f} at iter {iters[-1]}', fontsize=16)
    axes[0].grid(True)

    # Plot second component (tilt_x)
    axes[1].plot(iters, tilts[:, 1], marker='o', color='C1')
    axes[1].set_xlabel('Iterations', fontsize=16)
    axes[1].set_ylabel('Avg Obj tilt_x (mrad)', fontsize=16)
    axes[1].set_title(f'Avg Obj tilt_x (mrad): {tilts[-1,1]:.3f} at iter {iters[-1]}', fontsize=16)
    axes[1].grid(True)

    for i, ax in enumerate(axes):
        # Plot the last n iters as an inset
        if len(iters) > 20 and last_n_iters is not None:
            axins = ax.inset_axes([0.45, 0.3, 0.4, 0.5])

            # Correctly match inset plots to main plots
            axins.plot(iters[-last_n_iters:], tilts[-last_n_iters:, i], marker='o', color = f'{"C0" if i == 0 else "C1"}')

            axins.set_xlabel('Iterations', fontsize=12)
            axins.set_ylabel(f'Avg Obj tilt_{"y" if i == 0 else "x"} (mrad)', fontsize=12)
            axins.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.3f}'))
            ax.indicate_inset_zoom(axins, edgecolor="gray")
            axins.set_title(f'Last {last_n_iters} iterations', fontsize=12, pad=10)
            axins.grid(True)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()

    if show_fig:
        plt.show()
    if pass_fig:
        return fig

def plot_obj_tilts(pos, tilts, figsize=(16,16), show_fig=True, pass_fig=False):
    """ Plot the obj tilts given the probe position and pos-dependent tilts """
    
    plt.ioff() # Temporaily disable the interactive plotting mode
    fig = plt.figure(figsize = figsize)
    ax = plt.gca() # There's only 1 ax for plt.figure(), and plt.title is an Axes-level attribute so I need to pass the Axes out because I like plt.title layout better
    plt.title("Object tilts", fontsize=16)
    
    tilts = np.broadcast_to(tilts, shape=(len(pos),2))
    if np.allclose(tilts[:,0], 0, atol=1e-3):
        # All tilts are effectively zero; skip quiver plot and annotate
        ax.text(
            0.5, 0.5, "All tilts are effectively zero (<1e-3) mrad, no quiver plot",
            ha="center", va="center", fontsize=18, color="gray", transform=ax.transAxes
        )
    else:
        M = np.hypot(tilts[:,0], tilts[:,1])
        q = ax.quiver(pos[:,1], pos[:,0], tilts[:,1], tilts[:,0], M, pivot='mid', angles='xy', scale_units='xy', label='Obj tilts')
        cbar = fig.colorbar(q, shrink=0.75)
        cbar.ax.set_ylabel('mrad')
        cbar.ax.get_yaxis().labelpad = 15
    
    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().invert_yaxis()  # Flipped y-axis if there's only scatter plot
    plt.xlabel('X (obj coord, px)')
    plt.ylabel('Y (obj coord, px)')
    
    plt.tight_layout()
    if show_fig:
        plt.show()
    if pass_fig:
        return fig, ax

def plot_scan_positions(pos, init_pos=None, img=None, offset=None, figsize=(16,16), dot_scale=0.001, show_arrow=True, show_fig=True, pass_fig=False):
    """ Plot the scan positions given an array of (N,2) """
    # The array is expected to have shape (N,2)
    # Each row is rendered as (y, x), or equivalently (height, width)
    # The dots are plotted with asending size and color changes to represent the relative order
    
    plt.ioff() # Temporaily disable the interactive plotting mode
    fig = plt.figure(figsize = figsize)
    ax = plt.gca() # There's only 1 ax for plt.figure(), and plt.title is an Axes-level attribute so I need to pass the Axes out because I like plt.title layout better
    plt.title("Scan positions", fontsize=16)
    
    if img is not None:
        plt.imshow(img)
        pos = np.array(pos) + np.array(offset)
        plt.gca().invert_yaxis()  # Pre-flip y-axis so the y-axis is image-like no matter what
    
    if init_pos is None:
        plt.scatter(x=pos[:,1], y=pos[:,0], c=np.arange(len(pos)), s=dot_scale*np.arange(len(pos)), label='Scan positions')
    else:
        plt.scatter(x=init_pos[:,1], y=init_pos[:,0], c='C0', s=dot_scale, label='Init scan positions')
        plt.scatter(x=pos[:,1],      y=pos[:,0],      c='C1', s=dot_scale, label='Opt scan positions')
        plt.ylim(init_pos[:,0].min()-10, init_pos[:,0].max()+10)
        plt.xlim(init_pos[:,1].min()-10, init_pos[:,1].max()+10)
    
    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().invert_yaxis()  # Flipped y-axis if there's only scatter plot
    plt.xlabel('X (obj coord, px)')
    plt.ylabel('Y (obj coord, px)')
    
    # Draw arrow from 1st position to 10th position
    if show_arrow:
        plt.arrow(pos[0, 1], pos[0, 0], pos[9, 1] - pos[0, 1], pos[9, 0] - pos[0, 0],
                color='red', head_width=2.5, head_length=5)
    plt.legend()
    plt.tight_layout()
    if show_fig:
        plt.show()
    if pass_fig:
        return fig, ax
    
def plot_affine_transformation(scale, asymmetry, rotation, shear):
    from .utils import compose_affine_matrix
    # Example
    # plot_affine_transformation(2,0,45,0)
    A = np.eye(2)
    Af = compose_affine_matrix(scale, asymmetry, rotation, shear)
    
    plt.figure()
    plt.title(f"Visualize affine transformation \n (scale, asym, rot, shear) = {scale, asymmetry, rotation, shear}", fontsize=14)

    # Add origin and scatter points
    plt.scatter(0, 0, color='gray', marker='o', s=3)
    plt.scatter(A[:,1], A[:,0], label='Original')
    plt.scatter(Af[:,1], Af[:,0], label='Transformed')

    # Adding arrows
    plt.quiver(A[0,1], A[0,0], angles='xy', scale_units='xy', scale=1, color='C0', alpha=0.5)
    plt.quiver(A[1,1], A[1,0], angles='xy', scale_units='xy', scale=1, color='C0', alpha=0.5)
    plt.quiver(Af[0,1], Af[0,0], angles='xy', scale_units='xy', scale=1, color='C1', alpha=0.5)
    plt.quiver(Af[1,1], Af[1,0], angles='xy', scale_units='xy', scale=1, color='C1', alpha=0.5)

    # Adding grid lines
    plt.grid(True, linestyle='--', color='gray', linewidth=0.5)

    plt.ylim(-2,2)
    plt.xlim(-2,2)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().invert_yaxis()  # Flipped y-axis if there's only scatter plot
    
    plt.xlabel('X')
    plt.ylabel('Y')
    
    plt.legend()
    plt.show()

def plot_pos_grouping(pos, batches, circle_diameter=False, diameter_type='90%', figsize=(16,8), dot_scale=1, show_fig=True, pass_fig=False):
    
    plt.ioff() # Temporaily disable the interactive plotting mode
    fig, axs = plt.subplots(1,2, figsize = figsize)
    
    for i, ax in enumerate(axs):
        if i == 0:
            axs[0].set_title(f"Scan positions for all {len(batches)} groups", fontsize=18)
            for batch in batches:
                ax.scatter(x=pos[batch, 1], y=pos[batch, 0], s=dot_scale)
        else:
            axs[1].set_title("Scan positions from group 0", fontsize=18)
            ax.scatter(x=pos[batches[0], 1], y=pos[batches[0], 0], s=dot_scale)
            
            # Draw a circle at the first point with the given diameter
            if circle_diameter:
                first_point = pos[batches[0][0]]
                circle = plt.Circle((first_point[1], first_point[0]), circle_diameter / 2, fill=False, color='r', linestyle='--')
                ax.scatter(x=first_point[1], y=first_point[0], s=dot_scale, color='r')
                ax.add_artist(circle)
                
                # Add annotation for "90% probe intensity"
                annotation_text = f"{diameter_type} probe intensity"
                annotation_x = first_point[1]
                annotation_y = first_point[0] #+ circle_diameter / 2 + 10  # Adjust the vertical offset as needed
                ax.annotate(annotation_text, xy=(annotation_x-circle_diameter/2, annotation_y-circle_diameter/2-3))
            
        ax.set_xlabel('X (obj coord, px)')
        ax.set_ylabel('Y (obj coord, px)')
        ax.set_xlim(pos[:,1].min()-10, pos[:,1].max()+10) # Show the full range to better visualize if a sub-group (like 'center') is selected
        ax.set_ylim(pos[:,0].min()-10, pos[:,0].max()+10)
        ax.invert_yaxis()
        ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    if show_fig:
        plt.show(block=False)
    if pass_fig:
        return fig
    
def plot_loss_curves(loss_iters, last_n_iters=10, show_fig=True, pass_fig=False):
    last_n_iters = int(last_n_iters)
    data = np.array(loss_iters)

    plt.ioff() # Temporaily disable the interactive plotting mode
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))

    # Plot all loss values
    axs.plot(data[:,0], data[:,1], marker='o')

    # Plot the last n iters as an inset
    if len(data) > 20 and last_n_iters is not None:
        # Create inset subplot for zoomed-in plot
        axins = axs.inset_axes([0.45, 0.3, 0.4, 0.5])
        axins.plot(data[-last_n_iters:,0], data[-last_n_iters:,1], marker='o')
        axins.set_xlabel('Iterations', fontsize=12)
        axins.set_ylabel('Loss value', fontsize=12)
        axins.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.5f}'))
        axs.indicate_inset_zoom(axins, edgecolor="gray")
        axins.set_title(f'Last {last_n_iters} iterations', fontsize=12, pad=10)

    # Set labels and title for the main plot
    axs.set_xlabel('Iterations', fontsize=16)
    axs.set_ylabel('Loss value', fontsize=16)
    axs.set_title(f'Loss value: {data[-1,1]:.5f} at iter {int(data[-1,0])}', fontsize=16)
    axs.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.tight_layout()
    if show_fig:
        plt.show()
    if pass_fig:
        return fig

def plot_slice_thickness(dz_iters, last_n_iters=10, show_fig=True, pass_fig=False):
    last_n_iters = int(last_n_iters)
    data = np.array(dz_iters)

    plt.ioff() # Temporaily disable the interactive plotting mode
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))

    # Plot all loss values
    axs.plot(data[:,0], data[:,1], marker='o')
    axs.grid(True)

    # Plot the last n iters as an inset
    if len(data) > 20 and last_n_iters is not None:
        # Create inset subplot for zoomed-in plot
        axins = axs.inset_axes([0.45, 0.3, 0.4, 0.5])
        axins.plot(data[-last_n_iters:,0], data[-last_n_iters:,1], marker='o')
        axins.set_xlabel('Iterations', fontsize=12)
        axins.set_ylabel('Slice thickness (Ang)', fontsize=12)
        axins.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.5f}'))
        axs.indicate_inset_zoom(axins, edgecolor="gray")
        axins.set_title(f'Last {last_n_iters} iterations', fontsize=12, pad=10)

    # Set labels and title for the main plot
    axs.set_xlabel('Iterations', fontsize=16)
    axs.set_ylabel('Slice thickness (Ang)', fontsize=16)
    axs.set_title(f'Slice thickness (Ang): {data[-1,1]:.5f} at iter {int(data[-1,0])}', fontsize=16)
    axs.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.tight_layout()
    if show_fig:
        plt.show()
    if pass_fig:
        return fig

def plot_probe_modes(init_probe, opt_probe, amp_or_phase='amplitude', real_or_fourier='real', phase_cmap=None, amplitude_cmap=None, show_fig=True, pass_fig=False):
    # The input probes are expected to be numpy array
    # This is for visualization so each mode has its own colorbar.
    # See the actual probe amplitude output for absolute scale visualizaiton
    
    # Get the power distribution
    # Although init_probe_int wouldn't change, calculate it takes < 0.2 ms so realy no need to pre-calculate it
    init_probe_int = np.abs(init_probe)**2 
    opt_probe_int = np.abs(opt_probe)**2
    init_probe_pow = np.sum(init_probe_int, axis=(-2,-1))/np.sum(init_probe_int) 
    opt_probe_pow = np.sum(opt_probe_int, axis=(-2,-1))/np.sum(opt_probe_int)
    
    if real_or_fourier == 'fourier':
    # While it might seem redundant, the sandwitch fftshift(fft(ifftshift(probe)))) is needed for the following reason:
    # Although probe_fourier = fft2(ifftshift(probe)) and probe_fourier = fft2(probe) gives the same abs(probe_fourier),
    # pre-fftshifting the probe back to corner gives more accurate phase angle while plotting the angle(probe_fourier)
    # On the other hand, fft2(probe) would generate additional phase shifts that looks like checkerboard artifact in angle(probe_fourier)
        init_probe = fftshift(fft2(ifftshift(init_probe, axes=(-2,-1)), norm='ortho'), axes=(-2,-1))
        opt_probe  = fftshift(fft2(ifftshift(opt_probe,  axes=(-2,-1)), norm='ortho'), axes=(-2,-1))
    elif real_or_fourier =='real':
        pass
    else:
        raise ValueError("Please use 'real' or 'fourier' for probe mode visualization!")
        
    if amp_or_phase == 'phase':
        # Negative sign for consistency with chi(k), because psi = exp(-i*chi(k)). Overfocus should give positive phase shift near the edge of aperture
        # Scale the plotted phase by the amplitude so we can focus more on the relevant phases
        init_probe = -np.angle(init_probe)*np.abs(init_probe) 
        opt_probe  = -np.angle(opt_probe)*np.abs(opt_probe)
        cmap = phase_cmap if phase_cmap else 'twilight'
    elif amp_or_phase == 'amplitude' or amp_or_phase == 'amp':
        init_probe = np.abs(init_probe)
        opt_probe  = np.abs(opt_probe)
        cmap = amplitude_cmap if amplitude_cmap else 'viridis'
    else:
        raise ValueError("Please use 'amplitude' or 'phase' for probe mode visualization!")
    
    plt.ioff() # Temporaily disable the interactive plotting mode
    fig, axs = plt.subplots(2, len(opt_probe), figsize=(len(opt_probe)*2.5, 6))
    axs = axs[:,None] if len(opt_probe) == 1 else axs # Expand axs to (2,1) to handle the special case of 1 probe mode
    plt.suptitle(f"Probe modes {amp_or_phase} in {real_or_fourier} space", fontsize=18)
                
    for i in range(len(opt_probe)):
        ax_init = axs[0, i]
        ax_init.set_title(f"Init pmode {i}: {init_probe_pow[i]:.1%}")
        im_init = ax_init.imshow(init_probe[i], cmap=cmap)
        ax_init.axis('off')
        plt.colorbar(im_init, ax=ax_init, shrink=0.6)

        ax_opt = axs[1, i]
        ax_opt.set_title(f"Opt pmode {i}: {opt_probe_pow[i]:.1%}")
        im_opt = ax_opt.imshow(opt_probe[i], cmap=cmap)
        ax_opt.axis('off')
        plt.colorbar(im_opt, ax=ax_opt, shrink=0.6)
    plt.tight_layout()
    if show_fig:
        plt.show()
    if pass_fig:
        return fig

@torch.compiler.disable
def plot_summary(output_path, model, niter, indices, init_variables, selected_figs=['loss', 'forward', 'probe_r_amp', 'probe_k_amp', 'probe_k_phase', 'pos'], collate_str='', show_fig=True, save_fig=False, verbose=True):
    """ Wrapper function for most visualization function """
    # selected_figs can take 'loss', 'forward', 'probe_r_amp', 'probe_k_amp', 'probe_k_phase', 'pos', 'tilt', or 'all'
    # Note: Set show_fig=False and save_fig=True if you just want to save the figure without showing
    
    # Sets figure saving to be True if you accidiently disable both show_fig and save_fig
    if show_fig is False and save_fig is False:
        save_fig = True 
        
    if save_fig and verbose:
        vprint(f"Saving summary figures for iter {niter}")
    
    iter_str = '_iter' + str(niter).zfill(4)
    
    # loss curves
    if 'loss' in selected_figs or 'all' in selected_figs:
        fig_loss = plot_loss_curves(model.loss_iters, last_n_iters=10, show_fig=show_fig, pass_fig=True)
        if show_fig:
            fig_loss.show()
        if save_fig:
            fig_loss.savefig(safe_filename(output_path + f"/summary_loss{collate_str}{iter_str}.png"))
    
    # Forward pass
    if 'forward' in selected_figs or 'all' in selected_figs:
        n = int(len(indices)**0.5)
        n2 = int(len(indices))
        plot_indices = indices[np.int32([n2/2+n/4, n2/2+3*n/4])] # The idea is to get 2 regions of (N/2)x(N/2) that are +-N/4 from the center of the FOV.
        fig_forward = plot_forward_pass(model, plot_indices, 0.5, show_fig=False, pass_fig=True)
        fig_forward.suptitle(f"Forward pass at iter {niter}", fontsize=24)
        if show_fig:
            fig_forward.show()
        if save_fig:
            fig_forward.savefig(safe_filename(output_path + f"/summary_forward_pass{collate_str}{iter_str}.png"))
    
    # Probe modes in real and reciprocal space
    init_probe = init_variables['probe']
    opt_probe = model.get_complex_probe_view().detach().cpu().numpy()

    if 'probe_r_amp' in selected_figs or 'all' in selected_figs:
        fig_probe_modes_real_amp      = plot_probe_modes(init_probe, opt_probe, real_or_fourier='real',    amp_or_phase='amplitude', show_fig=False, pass_fig=True)
        fig_probe_modes_real_amp.suptitle(f"Probe modes amplitude in real space at iter {niter}", fontsize=18)
        if show_fig:
            fig_probe_modes_real_amp.show()
        if save_fig:
            fig_probe_modes_real_amp.savefig(safe_filename(output_path + f"/summary_probe_modes_real_amp{collate_str}{iter_str}.png"),bbox_inches='tight')
            

    if 'probe_k_amp' in selected_figs or 'all' in selected_figs:
        fig_probe_modes_fourier_amp   = plot_probe_modes(init_probe, opt_probe, real_or_fourier='fourier', amp_or_phase='amplitude', show_fig=False, pass_fig=True)
        fig_probe_modes_fourier_amp.suptitle(f"Probe modes amplitude in fourier space at iter {niter}", fontsize=18)
        if show_fig:
            fig_probe_modes_fourier_amp.show()
        if save_fig:
            fig_probe_modes_fourier_amp.savefig(safe_filename(output_path + f"/summary_probe_modes_fourier_amp{collate_str}{iter_str}.png"),bbox_inches='tight')
            

    if 'probe_k_phase' in selected_figs or 'all' in selected_figs:
        fig_probe_modes_fourier_phase = plot_probe_modes(init_probe, opt_probe, real_or_fourier='fourier', amp_or_phase='phase', show_fig=False, pass_fig=True)
        fig_probe_modes_fourier_phase.suptitle(f"Probe modes phase in fourier space at iter {niter}", fontsize=18)
        if show_fig:
            fig_probe_modes_fourier_phase.show()
        if save_fig:
            fig_probe_modes_fourier_phase.savefig(safe_filename(output_path + f"/summary_probe_modes_fourier_phase{collate_str}{iter_str}.png"),bbox_inches='tight')
            
            
    # Scan positions and tilts
    init_pos = init_variables['crop_pos'] + init_variables['probe_pos_shifts']
    pos = (model.crop_pos + model.opt_probe_pos_shifts).detach().cpu().numpy()
    tilts = model.opt_obj_tilts.detach().cpu().numpy()
    tilts = np.broadcast_to(tilts, (len(pos), 2)) # tilts has to be (N_scan, 2)
    
    if 'pos' in selected_figs or 'all' in selected_figs:
        fig_scan_pos, ax = plot_scan_positions(pos=pos[indices], init_pos=init_pos[indices], dot_scale=1, show_fig=False, pass_fig=True)
        ax.set_title(f"Scan positions at iter {niter}", fontsize=16)
        if show_fig:
            fig_scan_pos.show()
        if save_fig:
            fig_scan_pos.savefig(safe_filename(output_path + f"/summary_scan_pos{collate_str}{iter_str}.png"))
    
    if 'tilt' in selected_figs or 'all' in selected_figs:
        fig_obj_tilts, ax = plot_obj_tilts(pos=pos[indices], tilts=tilts[indices], show_fig=False, pass_fig=True)
        ax.set_title(f"Object tilts at iter {niter}", fontsize=16)
        if show_fig:
            fig_obj_tilts.show()
        if save_fig:
            fig_obj_tilts.savefig(safe_filename(output_path + f"/summary_obj_tilts{collate_str}{iter_str}.png"))
            
    if 'tilt_avg' in selected_figs or 'all' in selected_figs:
        fig_avg_obj_tilts = plot_obj_tilts_avg(model.avg_tilt_iters, last_n_iters=10, show_fig=show_fig, pass_fig=True)
        if show_fig:
            fig_avg_obj_tilts.show()
        if save_fig:
            fig_avg_obj_tilts.savefig(safe_filename(output_path + f"/summary_obj_tilts_avg{collate_str}{iter_str}.png"))
    
    # Slice thickness
    if 'slice_thickness' in selected_figs or 'all' in selected_figs:
        fig_slice_thickness = plot_slice_thickness(model.dz_iters, last_n_iters=10, show_fig=show_fig, pass_fig=True)
        if show_fig:
            fig_slice_thickness.show()
        if save_fig:
            fig_slice_thickness.savefig(safe_filename(output_path + f"/summary_slice_thickness{collate_str}{iter_str}.png"))
    
    # Close figures after saving
    plt.close('all')
