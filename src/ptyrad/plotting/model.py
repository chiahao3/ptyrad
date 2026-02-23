"""
Plotting functions related to PyTorch models
"""
# This isolates the heavy torch import (3-6 sec) and is not promoted via __init__

import logging

import matplotlib.pyplot as plt
import numpy as np
import torch

from ptyrad.io.save import safe_filename

from .basic import (
    plot_loss_curves,
    plot_obj_tilts,
    plot_obj_tilts_avg,
    plot_probe_modes,
    plot_scan_positions,
    plot_slice_thickness,
)

logger = logging.getLogger(__name__)

@torch.compiler.disable
def plot_summary(output_path, model, niter, indices, init_variables, selected_figs=['loss', 'forward', 'probe_r_amp', 'probe_k_amp', 'probe_k_phase', 'pos'], collate_str='', show_fig=True, save_fig=False):
    """ Wrapper function for most visualization function """
    # selected_figs can take 'loss', 'forward', 'probe_r_amp', 'probe_k_amp', 'probe_k_phase', 'pos', 'tilt', or 'all'
    # Note: Set show_fig=False and save_fig=True if you just want to save the figure without showing
    
    # Sets figure saving to be True if you accidiently disable both show_fig and save_fig
    if show_fig is False and save_fig is False:
        save_fig = True 
        
    if save_fig:
        logger.info(f"Saving summary figures for iter {niter}")
    
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
        obja_ROI    = (obj_patches[0] * omode_occu[:,None,None,None]).sum(1).detach().cpu().numpy() # obj_ROI = (N_i, Nz,Ny,Nx)
        objp_ROI    = (obj_patches[1] * omode_occu[:,None,None,None]).sum(1).detach().cpu().numpy() # obj_ROI = (N_i, Nz,Ny,Nx)
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