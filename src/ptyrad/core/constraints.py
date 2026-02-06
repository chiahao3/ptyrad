"""
Physical constraints that directly modify optimizable tensors with specified intervals of iterations

"""

import torch
from torch.fft import fft2, fftfreq, fftn, ifft2, ifftn
from torch.nn.functional import interpolate
from torchvision.transforms.functional import gaussian_blur

from ptyrad.utils.image_proc import gaussian_blur_1d
from ptyrad.utils.physics import near_field_evolution_torch
from ptyrad.utils.logging import vprint
from ptyrad.utils.math_ops import (
    dct_2d,
    fftshift2,
    idct_2d,
    ifftshift2,
    make_sigmoid_mask,
)
from ptyrad.utils.math_ops import approx_torch_quantile

@torch.compiler.disable # TorchRuntimeError: Dynamo failed to run FX node with fake tensors: call_function <Wrapped method <original sub>>(*(FakeTensor(..., size=(5,), dtype=torch.float64), 2.0), **{}): got AttributeError("'ndarray' object has no attribute 'sub'")
class CombinedConstraint(torch.nn.Module):
    """Applies iteration-wise in-place constraints on optimizable tensors.

    This class is designed to apply various constraints to a model's parameters during the optimization process.
    The constraints are applied at specific iteration frequencies, as determined by the `constraint_params` dictionary.
    These constraints include orthogonality, probe amplitude constraints in Fourier space, intensity constraints, Gaussian
    blurring, Fourier filtering, and more.

    Args:
        constraint_params (dict): A dictionary containing the configuration for each constraint. Each constraint should have a 
            frequency and other parameters necessary for its application.
        device (str, optional): The device on which the tensors are located (e.g., 'cuda' or 'cpu'). Defaults to 'cuda'.
        verbose (bool, optional): If True, prints messages during the application of constraints. Defaults to True.
    """
    def __init__(self, constraint_params, device='cuda', verbose=True):
        super(CombinedConstraint, self).__init__()
        self.device = device
        self.constraint_params = constraint_params
        self.verbose = verbose

    def _should_apply_at_iter(self, constraint_name, niter):
        """Check if the constraint should be applied at the current iteration."""
        start = self.constraint_params[constraint_name]['start_iter']
        step  = self.constraint_params[constraint_name]['step']
        end   = self.constraint_params[constraint_name]['end_iter']
        
        if start is None:
            return False
        if niter < start:
            return False
        if end is not None and niter >= end:
            return False
        return (niter - start) % step == 0

    def apply_ortho_pmode(self, model, niter):
        ''' Apply orthogonality constraint to probe modes '''
        
        if self._should_apply_at_iter('ortho_pmode', niter):
            model.opt_probe.data = torch.view_as_real(orthogonalize_modes_vec(model.get_complex_probe_view(), sort=True).contiguous()) # Note that model stores the complex probe as a (pmode, Ny, Nx, 2) float tensor (real view) so we need to do some real-complex view conversion.
            probe_int = model.get_complex_probe_view().abs().pow(2)
            probe_pow = (probe_int.sum((1,2))/probe_int.sum()).detach().cpu().numpy().round(3)
            vprint(f"Apply ortho pmode constraint at iter {niter}, relative pmode power = {probe_pow}, probe int sum = {probe_int.sum():.4f}", verbose=self.verbose)

    def apply_probe_mask_k(self, model, niter):
        ''' Apply probe amplitude constraint in Fourier space '''
        # Note that this will change the total probe intensity, please use this with `fix_probe_int`
        # Although the mask wouldn't change during the iteration, making a mask takes only ~0.5us on CPU so really no need to pre-calculate it
        # The sandwitch fftshift(fft(ifftshift(probe))) is needed to properly handle the complex probe without serrated phase
        # fft2 is for real->fourier, while fftshift2 is for corner->center
        
        if self._should_apply_at_iter('probe_mask_k', niter):
            relative_radius   = self.constraint_params['probe_mask_k']['radius']
            relative_width    = self.constraint_params['probe_mask_k']['width']
            power_thresh      = self.constraint_params['probe_mask_k']['power_thresh']
            
            probe = model.get_complex_probe_view()
            Npix = probe.size(-1)
            powers = probe.abs().pow(2).sum((-2,-1)) / probe.abs().pow(2).sum()
            powers_cumsum = powers.cumsum(0)
            pmode_index = (powers_cumsum > power_thresh).nonzero()[0].item() # This gives the pmode index that the cumulative power along mode dimension is greater than the power_thresh and should have mask extend to this index
            mask = torch.ones_like(probe, dtype=torch.float32, device=model.device)
            mask_value = make_sigmoid_mask(Npix, relative_radius, relative_width).to(model.device)
            mask[:pmode_index+1] = mask_value
            probe_k = fftshift2 (fft2(ifftshift2(probe), norm='ortho')) # probe_k at center for later masking
            probe_r = fftshift2(ifft2(ifftshift2(mask * probe_k),  norm='ortho')) # probe_r at center. Note that the norm='ortho' is explicitly specified but not needed for a round-trip
            probe_int = model.get_complex_probe_view().abs().pow(2)
            # Re-sort the probe modes, note that the masked strong modes might be swapping order with unmasked weak modes
            model.opt_probe.data = torch.view_as_real(sort_by_mode_int(probe_r))
            vprint(f"Apply Fourier-space probe amplitude constraint at iter {niter}, pmode_index = {pmode_index} when power_thresh = {power_thresh}, probe int sum = {probe_int.sum():.4f}", verbose=self.verbose)
    
    def apply_fix_probe_int(self, model, niter):
        ''' Apply probe intensity constraint '''
        # Note that the probe intensity fluctuation (std/mean) is typically only 0.5%, there's very little point to do a position-dependent probe intensity constraint
        # Therefore, a mean probe intensity is used here as the target intensity
        
        if self._should_apply_at_iter('fix_probe_int', niter):
            probe = model.get_complex_probe_view()
            current_amp = probe.abs().pow(2).sum().pow(0.5)
            target_amp  = model.probe_int_sum**0.5   
            model.opt_probe.data = torch.view_as_real(probe * target_amp/current_amp)
            probe_int = model.get_complex_probe_view().abs().pow(2)
            vprint(f"Apply fix probe int constraint at iter {niter}, probe int sum = {probe_int.sum():.4f}", verbose=self.verbose)
            
    def apply_obj_rblur(self, model, niter):
        ''' Apply Gaussian blur to object, this only applies to the last 2 dimension (...,H,W) '''
        # Note that it's not clear whether applying blurring after every iteration would ever reach a steady state
        # However, this is at least similar to PtychoShelves' eng. reg_mu
        
        if self._should_apply_at_iter('obj_rblur', niter) and self.constraint_params['obj_rblur']['std'] !=0:
            obj_type       = self.constraint_params['obj_rblur']['obj_type']
            obj_rblur_ks   = self.constraint_params['obj_rblur']['kernel_size']
            obj_rblur_std  = self.constraint_params['obj_rblur']['std']
            
            if obj_type in ['amplitude', 'both']:
                model.opt_obja.data = gaussian_blur(model.opt_obja, kernel_size=obj_rblur_ks, sigma=obj_rblur_std)
                vprint(f"Apply lateral (y,x) Gaussian blur with std = {obj_rblur_std} px on obja at iter {niter}", verbose=self.verbose)
            if obj_type in ['phase', 'both']:
                model.opt_objp.data = gaussian_blur(model.opt_objp, kernel_size=obj_rblur_ks, sigma=obj_rblur_std)
                vprint(f"Apply lateral (y,x) Gaussian blur with std = {obj_rblur_std} px on objp at iter {niter}", verbose=self.verbose)
    
    def apply_obj_zblur(self, model, niter):
        ''' Apply Gaussian blur to object, this only applies to the last dimension (...,L) '''

        if self._should_apply_at_iter('obj_zblur', niter) and self.constraint_params['obj_zblur']['std'] !=0:
            obj_type       = self.constraint_params['obj_zblur']['obj_type']
            obj_zblur_ks   = self.constraint_params['obj_zblur']['kernel_size']
            obj_zblur_std  = self.constraint_params['obj_zblur']['std']
            
            if obj_type in ['amplitude', 'both']:
                tensor = model.opt_obja.permute(0,2,3,1)
                model.opt_obja.data = gaussian_blur_1d(tensor, kernel_size=obj_zblur_ks, sigma=obj_zblur_std).permute(0,3,1,2).contiguous() # contiguous() returns a contiguous memory layout so that DDP wouldn't complain about the stride mismatch of grad and params
                vprint(f"Apply z-direction Gaussian blur with std = {obj_zblur_std} px on obja at iter {niter}", verbose=self.verbose)
            if obj_type in ['phase', 'both']:
                tensor = model.opt_objp.permute(0,2,3,1)
                model.opt_objp.data = gaussian_blur_1d(tensor, kernel_size=obj_zblur_ks, sigma=obj_zblur_std).permute(0,3,1,2).contiguous() 
                vprint(f"Apply z-direction Gaussian blur with std = {obj_zblur_std} px on objp at iter {niter}", verbose=self.verbose)
    
    def apply_kr_filter(self, model, niter):
        ''' Apply kr Fourier filter constraint on object '''
        # Note that the `kr_filter` is applied on stacked 2D FFT of object, so it's applying on (omode,z,ky,kx)
        # The kr filter is similar to a top-hat, so it's more like a cut-off, instead of the weak lateral Gaussian blurring (alpha) included in the `kz_filter` 

        if self._should_apply_at_iter('kr_filter', niter):
            obj_type         = self.constraint_params['kr_filter']['obj_type']
            relative_radius  = self.constraint_params['kr_filter']['radius']
            relative_width   = self.constraint_params['kr_filter']['width']
            
            if obj_type in ['amplitude', 'both']:
                model.opt_obja.data = kr_filter(model.opt_obja, relative_radius, relative_width)
                vprint(f"Apply kr_filter constraint with kr_radius = {relative_radius} on obja at iter {niter}", verbose=self.verbose)
            if obj_type in ['phase', 'both']:
                model.opt_objp.data = kr_filter(model.opt_objp, relative_radius, relative_width)
                vprint(f"Apply kr_filter constraint with kr_radius = {relative_radius} on objp at iter {niter}", verbose=self.verbose)
        
    def apply_kz_filter(self, model, niter):
        ''' Apply kz Fourier filter constraint on object '''
        # Note that the `kz_filter`` behaves differently for 'amplitude' and 'phase', see `kz_filter` implementaion for details

        if self._should_apply_at_iter('kz_filter', niter):
            obj_type               = self.constraint_params['kz_filter']['obj_type']
            beta_regularize_layers = self.constraint_params['kz_filter']['beta']
            alpha_gaussian         = self.constraint_params['kz_filter']['alpha']
            
            if obj_type in ['amplitude', 'both']:
                model.opt_obja.data = kz_filter(model.opt_obja, beta_regularize_layers, alpha_gaussian, obj_type='amplitude')
                vprint(f"Apply kz_filter constraint with beta = {beta_regularize_layers} on obja at iter {niter}", verbose=self.verbose)
            if obj_type in ['phase', 'both']:
                model.opt_objp.data = kz_filter(model.opt_objp, beta_regularize_layers, alpha_gaussian, obj_type='phase')
                vprint(f"Apply kz_filter constraint with beta = {beta_regularize_layers} on objp at iter {niter}", verbose=self.verbose)
    
    def apply_kr_thresh(self, model, niter):
        ''' Apply kr threshold constraint on object '''

        if self._should_apply_at_iter('kr_thresh', niter):
            obj_type         = self.constraint_params['kr_thresh']['obj_type']
            thresh           = self.constraint_params['kr_thresh']['thresh']
            
            if obj_type in ['amplitude', 'both']:
                model.opt_obja.data = dct_threshold_filter(model.opt_obja, threshold_ratio=thresh).contiguous()
                vprint(f"Apply kr_filter constraint with threshold = {thresh} (ratio of spatial frequencies to keep) on obja at iter {niter}", verbose=self.verbose)
            if obj_type in ['phase', 'both']:
                model.opt_objp.data = dct_threshold_filter(model.opt_objp, threshold_ratio=thresh).contiguous()
                vprint(f"Apply kr_filter constraint with threshold = {thresh} (ratio of spatial frequencies to keep) on objp at iter {niter}", verbose=self.verbose)
    
    def apply_complex_ratio(self, model, niter):
        ''' Apply complex constraint on object '''
        # Original paper seems to apply this constraint at each position. I'll try an iteration-wise constraint first

        if self._should_apply_at_iter('complex_ratio', niter):
            obj_type           = self.constraint_params['complex_ratio']['obj_type']
            alpha1             = self.constraint_params['complex_ratio']['alpha1']
            alpha2             = self.constraint_params['complex_ratio']['alpha2']
            
            objac, objpc, Cbar = complex_ratio_constraint(model, alpha1, alpha2)
            if obj_type in ['amplitude', 'both']:
                model.opt_obja.data = objac
                amin, amax = model.opt_obja.min().item(), model.opt_obja.max().item()
                vprint(f"Apply complex ratio constraint with alpha1: {alpha1}, alpha2: {alpha2}, and Cbar: {Cbar.item():.3f} on obja at iter {niter}. obja range becomes ({amin:.3f}, {amax:.3f})", verbose=self.verbose)
            if obj_type in ['phase', 'both']:
                model.opt_objp.data = objpc
                pmin, pmax = model.opt_objp.min().item(), model.opt_objp.max().item()
                vprint(f"Apply complex ratio constraint with alpha1: {alpha1}, alpha2: {alpha2}, and Cbar: {Cbar.item():.3f} on objp at iter {niter}. objp range becomes ({pmin:.3f}, {pmax:.3f})", verbose=self.verbose)  
    
    def apply_mirrored_amp(self, model, niter):
        '''Apply mirrored amplitude constraint on obja at voxel level'''
        # The idea is to replace the amplitude with Amp' = exp(-scale*phase^2), because the absorptive potential should scale with V^2

        if self._should_apply_at_iter('mirrored_amp', niter):
            relax           = self.constraint_params['mirrored_amp']['relax']
            scale           = self.constraint_params['mirrored_amp']['scale']
            power           = self.constraint_params['mirrored_amp']['power']
            
            v_power = model.opt_objp.clamp(min=0).pow(power)
            # amp_new = torch.exp(-scale*v_power)
            amp_new = 1-scale*v_power
            model.opt_obja.data = relax * model.opt_obja + (1-relax) * amp_new
            amin, amax = model.opt_obja.min().item(), model.opt_obja.max().item()
            relax_str = f'relaxed ({relax}*obj + ({1-relax}*obj_new))' if relax != 0 else 'hard'
            vprint(f"Apply {relax_str} mirrored amplitude constraint with scale = {scale} and power = {power} on obja at iter {niter}. obja range becomes ({amin:.3f}, {amax:.3f})", verbose=self.verbose)
    
    def apply_obj_z_recenter(self, model, niter):
        '''Apply object z-recentering along depth dimension '''
        # The idea is to recenter the object within the object tensor using CoM along depth
        # Because fundamentally there's an ambiguity between probe defocus and object z positioning,
        # so we'll have to shift the object and adjust the probe accordingly.
        # I thought I saw this idea in other packages but I couldn't seem to find it anymore......
        
        if self._should_apply_at_iter('obj_z_recenter', niter):
            threshold = self.constraint_params['obj_z_recenter'].get('thresh', 90)
            scale     = self.constraint_params['obj_z_recenter'].get('scale', 0.5)
            max_shift = self.constraint_params['obj_z_recenter'].get('max_shift', 5)
            unit_str = model.length_unit
            dz = model.opt_slice_thickness.detach().item()
            
            probe = model.get_complex_probe_view()
            dx = model.dx
            lambd = model.lambd
            obja = model.opt_obja
            objp = model.opt_objp
            objc = torch.polar(obja, objp)
            
            # Shift the obj along z 
            z_shift = get_obj_z_shift(objp, threshold, scale, max_shift) # unit: px
            objc = shift_obj_along_z(objc, z_shift)
            
            # Update model obj
            model.opt_obja.data = torch.abs(objc).contiguous()
            model.opt_objp.data = torch.angle(objc).contiguous()
            
            # Update model probe
            H = near_field_evolution_torch(probe.shape[-2:], dx, -z_shift*dz, lambd, device=model.device) # If the object is shifted along +z, then the probe should be back-propagated along -z 
            model.opt_probe.data = torch.view_as_real(ifft2(H[None,] * fft2(probe)).contiguous())
            vprint(f"Apply obj z-recenter constraint. Complex object and probe defocus are shifted by {z_shift:.3f} slice ({(z_shift*dz):.3f} {unit_str}) along depth dimension. threshold = {threshold}, scale = {scale}, and max_shift = {max_shift}.", verbose=self.verbose)
    
    def apply_obja_thresh(self, model, niter):
        ''' Apply thresholding on obja at voxel level '''
        # Although there's a lot of code repitition with `apply_postiv`, phase positivity itself is important enough as an individual operation

        if self._should_apply_at_iter('obja_thresh', niter):
            relax            = self.constraint_params['obja_thresh']['relax']
            thresh           = self.constraint_params['obja_thresh']['thresh']
            
            model.opt_obja.data = relax * model.opt_obja + (1-relax) * model.opt_obja.clamp(min=thresh[0], max=thresh[1])
            relax_str = f'relaxed ({relax}*obj + ({1-relax}*obj_clamp))' if relax != 0 else 'hard'
            vprint(f"Apply {relax_str} threshold constraint with thresh = {thresh} on obja at iter {niter}", verbose=self.verbose)

    def apply_objp_postiv(self, model, niter):
        ''' Apply positivity constraint on objp at voxel level '''
        # Note that this `relax` is defined oppositly to PtychoShelves's `positivity_constraint_object` in `ptycho_solver`. 
        # Here, relax=1 means fully relaxed and essentially no constraint.

        if self._should_apply_at_iter('objp_postiv', niter):
            relax            = self.constraint_params['objp_postiv']['relax']
            mode             = self.constraint_params['objp_postiv'].get('mode', 'clip_neg')
            
            original_min = model.opt_objp.min()
            if mode == 'subtract_min':
                modified_objp = model.opt_objp - original_min
            else: # 'clip_neg'
                modified_objp = model.opt_objp.clamp(min=0)
            model.opt_objp.data = relax * model.opt_objp + (1-relax) * modified_objp
            omin, omax = model.opt_objp.min().item(), model.opt_objp.max().item()
            relax_str = f'relaxed ({relax}*obj + ({1-relax}*obj_postiv))' if relax != 0 else 'hard'
            vprint(f"Apply {relax_str} positivity constraint on objp with '{mode}' mode at iter {niter}. Original min = {original_min.item():.3f}. objp range becomes ({omin:.3f}, {omax:.3f})", verbose=self.verbose)           

    def apply_pos_recenter(self, model, niter):
        ''' Apply position recentering constraint on probe positions '''
        # Here, relax=1 means fully relaxed and essentially no constraint.
        # Usually we start from reasonable probe and positions, so the object would get reconstructed in place instantly.
        # This makes object, probe, and probe positions remain relatively aligned so this constraint isn't critically needed.
        # For certain use cases (i.e., large position learning rates or misplaced object during initialization), the probe positions might accumulate large global offsets.
        # Then we can use this constraint to remove the global offset, which will soon recenter the object as well given that probe CoM is relatively stable.
        # TODO: Technically we can shift the object accordingly to accelerate the convergence but it doesn't seem to be critically needed at the moment

        if self._should_apply_at_iter('pos_recenter', niter):
            relax            = self.constraint_params['pos_recenter']['relax']
            
            pos_shifts = model.opt_probe_pos_shifts # float32
            orig_mean = pos_shifts.mean(0)

            model.opt_probe_pos_shifts.data = pos_shifts - (1 - relax) * orig_mean
            relax_str = f'relaxed (pos_shifts - ({1-relax:.3f}*original_mean))' if relax != 0 else 'hard'
            vprint(f"Apply {relax_str} position recentering constraint at iter {niter}. Original mean = {orig_mean.detach().cpu().numpy().round(3)}. probe_pos_shifts.mean(0) becomes {model.opt_probe_pos_shifts.mean(0).detach().cpu().numpy().round(3)}", verbose=self.verbose)

    def apply_tilt_smooth(self, model, niter):
        ''' Apply Gaussian blur to object tilts '''
        # Note that the smoothing is applied along the last 2 axes, which are scan dimensions, so the unit of std is "scan positions"
        # Besides, the relative position of the obj_tilts are neglected for simplicity

        if self._should_apply_at_iter('tilt_smooth', niter) and self.constraint_params['tilt_smooth']['std'] !=0:
            tilt_smooth_std  = self.constraint_params['tilt_smooth']['std']
            N_scan_slow = model.N_scan_slow
            N_scan_fast = model.N_scan_fast
            
            if model.opt_obj_tilts.shape[0] == 1: # obj_tilts.shape = (1,2) for tilt_type: 'all', and (N,2) for 'each'
                vprint("`tilt_smooth` constraint requires `tilt_type':'each'`, skip this constraint", verbose=self.verbose)
                return 
            obj_tilts = (model.opt_obj_tilts.reshape(N_scan_slow, N_scan_fast, 2)).permute(2,0,1)
            model.opt_obj_tilts.data = gaussian_blur(obj_tilts, kernel_size=5, sigma=tilt_smooth_std).permute(1,2,0).reshape(-1,2).contiguous() # contiguous() returns a contiguous memory layout so that DDP wouldn't complain about the stride mismatch of grad and params
            vprint(f"Apply Gaussian blur with std = {tilt_smooth_std} scan positions on obj_tilts at iter {niter}", verbose=self.verbose)
    
    def forward(self, model, niter):
        """
        Applies constraints to the optimizable `model` parameters if `niter` satisfies the pre-determined conditions (start_iter, step, end_iter)
        """
        # Note that the if check blocks are included in each apply methods so that it's cleaner, and I can print the info with niter
        
        with torch.no_grad():
            # Probe constraints
            self.apply_ortho_pmode   (model, niter)
            self.apply_probe_mask_k  (model, niter)
            self.apply_fix_probe_int (model, niter)
            # Object constraints
            self.apply_obj_rblur     (model, niter)
            self.apply_obj_zblur     (model, niter)
            self.apply_kr_filter     (model, niter)
            self.apply_kz_filter     (model, niter)
            self.apply_kr_thresh     (model, niter)
            self.apply_complex_ratio (model, niter)
            self.apply_mirrored_amp  (model, niter)
            self.apply_obj_z_recenter(model, niter)
            self.apply_obja_thresh   (model, niter)
            self.apply_objp_postiv   (model, niter)
            # Position constraints
            self.apply_pos_recenter  (model, niter)
            # Local tilt constraint
            self.apply_tilt_smooth   (model, niter)

###### Filter and helper functions for constraints ######
def sort_by_mode_int(modes):
    modes_int =  modes.abs().pow(2).sum(tuple(range(1,modes.ndim))) # Sum every but 1st dimension
    _, indices = torch.sort(modes_int, descending=True)
    modes = modes[indices]
    return modes

def orthogonalize_modes_vec(modes, sort = False):
    ''' orthogonalize the modes using SVD'''
    # Input:
    #   modes: input function with multiple modes
    # Output:
    #   ortho_modes: 
    # Note:
    #   This function is a highly vectorized PyTorch implementation of `ptycho\+core\probe_modes_ortho.m` from PtychoShelves
    #   It's numerically equivalent with the following for-loop version but is ~ 10x faster on small complex64 tensors (10,164,164) 
    #   Most indexings arr converted from Matlab (start from 1) to Python (start from 0)
    #   The expected shape of `modes` input is modified into (pmode, Ny, Nx) to be consistent with ptyrad
    #   If you check the orthoganality of each mode, make sure to change the input into complex128 or to modify the default tolerance of torch.allclose.
    #   Note that Matlab's dot(p2,p1) for complex input would implictly apply with the complex conjugate, 
    #   so Matlab's dot() != torch.dot because torch.dot doesn't automatically apply the complex conjugate.
    #   This is pointed out by @dong-zehao in issue #11.
        
    orig_modes_dtype = modes.dtype
    if orig_modes_dtype != torch.complex64:
        modes = torch.complex(modes, torch.zeros_like(modes))
    input_shape = modes.shape
    modes_reshaped = modes.reshape(input_shape[0], -1) # Reshape modes to have a shape of (Nmode, X*Y)
    A = torch.matmul(modes_reshaped, modes_reshaped.H) # A = M @ M^T.conj() = M @ M^H, H is the conjugate transpose

    if A.device.type == 'mps': # Temporary hack because PyTorch MPS backend doesn't seem to implement linalg.eig yet.
        _, evecs = torch.linalg.eig(A.to('cpu'))
        evecs = evecs.to('mps')
    else:
        _, evecs = torch.linalg.eig(A)
   
    # Matrix-multiplication version (N,N) @ (N,YX) = (N,YX)
    ortho_modes = torch.matmul(evecs.H, modes_reshaped).reshape(input_shape)

    # sort modes by their contribution
    if sort:
        ortho_modes = sort_by_mode_int(ortho_modes)
        
    return ortho_modes.to(orig_modes_dtype)

def dct_threshold_filter(
    x: torch.Tensor,
    threshold_ratio: float = 0.05,
) -> torch.Tensor:
    """Applies hard-threshold filtering in the DCT domain.

    Keeps only the largest-magnitude DCT coefficients according to
    ``threshold_ratio`` and zeros out the rest.

    Works for any input shape (..., H, W).

    Args:
        x (torch.Tensor): Input real-valued tensor of shape (..., H, W).
        threshold_ratio (float): Fraction of coefficients to keep. Must be in
            [0, 1]. For example, 0.05 retains the top 5% coefficients.

    Returns:
        torch.Tensor: Filtered tensor of same shape as ``x``.

    Raises:
        ValueError: If ``threshold_ratio`` is not in [0, 1].
    """
    if not (0.0 <= threshold_ratio <= 1.0):
        raise ValueError("threshold_ratio must be between 0 and 1.")

    # Compute DCT
    dct = dct_2d(x)

    # Flatten all but keep batch shape implicit
    dct_abs = dct.abs()
    flat = dct_abs.reshape(-1)

    # Determine number of coefficients to keep
    k = max(1, int(threshold_ratio * flat.numel()))

    # kth largest = (N-k+1)-th smallest
    kth = torch.kthvalue(flat, flat.numel() - k + 1).values

    # Hard thresholding
    mask = dct_abs >= kth
    dct_filtered = dct * mask

    # Inverse DCT
    return idct_2d(dct_filtered)

def kr_filter(obj, radius, width):
    ''' Apply kr_filter using the 2D sigmoid filter '''
    
    # Create the filter function W, note that the W has to be corner-centered
    Ny, Nx = obj.shape[-2:]
    mask = make_sigmoid_mask(min(Ny,Nx), radius, width).to(obj.device)
    W = ifftshift2(interpolate(mask[None,None,], size=(Ny,Nx))).squeeze() # interpolate needs 2 additional dimension (N,C,...) for the input than the output dimension
        
    # Filter the obj with filter function Wa, take the real part because Fourier-filtered obj could contain negative values    
    fobj = torch.real(ifft2(fft2(obj) * W[None,None,])) # Apply fft2/ifft2 for only the r(y,x) dimension so the omode and z would be broadcasted
    
    return fobj

def kz_filter(obj, beta_regularize_layers=1, alpha_gaussian=1, obj_type='phase'):
    ''' Apply kz_filter using the arctan filter '''
    # Note: Calculate force of regularization based on the idea that DoF = resolution^2/lambda
        
    device = obj.device
    
    # Generate 1D grids along each dimension
    Npix = obj.shape[-3:]
    kz = fftfreq(Npix[0]).to(device) 
    ky = fftfreq(Npix[1]).to(device) 
    kx = fftfreq(Npix[2]).to(device) 
    
    # Generate 3D coordinate grid using meshgrid
    grid_kz, grid_ky, grid_kx = torch.meshgrid(kz, ky, kx, indexing='ij')

    # Create the filter function Wa. W and Wa is exactly the same as PtychoShelves for now
    W = 1 - torch.atan((beta_regularize_layers * torch.abs(grid_kz) / torch.sqrt(grid_kx**2 + grid_ky**2 + 1e-3))**2) / (torch.pi/2)
    Wa = W * torch.exp(-alpha_gaussian * (grid_kx**2 + grid_ky**2))

    # Filter the obj with filter function Wa, take the real part because Fourier-filtered obj could contain negative values    
    fobj = torch.real(ifftn(fftn(obj, dim=(-3,-2,-1)) * Wa[None,], dim=(-3,-2,-1))) # Apply fftn/ifftn for only spatial dimension so the omode would be broadcasted
    
    if obj_type == 'amplitude':
        fobj = 1+0.9*(fobj-1) # This is essentially a soft obja threshold constraint built into the kz_filter routine for obja
        
    return fobj

def get_obj_z_shift(obj_phase, threshold=95, scale=1, max_shift=10):
    """
    Compute z-shift from the center-of-mass (CoM) of the object phase.
    
    Args:
        obj_phase: tensor (omode, z, y, x), phase values in radians
        threshold: threshold factor used to remove weak intensities in the image
        scale: scaling factor applied to the measured shift
        max_shift: maximum allowed shift in pixels
    
    Returns:
        float, signed shift (positive = shift down in z)
    """

    nz = obj_phase.shape[1]
    if max_shift is None:
        max_shift = (nz-1)/2
    else:
        max_shift = min(max_shift, (nz-1)/2)

    # Ensure no negative phase value
    obj_phase = torch.clamp(obj_phase, min=0)
    
    # Threshold to focus on actual signal
    if threshold is not None:
        cutoff = approx_torch_quantile(obj_phase, q=threshold/100) # quantile is between [0,1]
        obj_phase[obj_phase < cutoff] = 0 # Value smaller than cutoff is set to 0
    
    # Collapse omode,y,x → get mean phase per z-slice
    phase_z = obj_phase.mean(dim=(0, 2, 3))  # shape (z,)
    
    # Calculate CoM and shift
    z_coords = torch.arange(nz, device=obj_phase.device, dtype=obj_phase.dtype)
    com_z = (z_coords * phase_z).sum() / (phase_z.sum() + 1e-8)
    center_z = (nz - 1) / 2.0
    shift = center_z - com_z
    
    # Apply scaling and clip
    shift *= scale
    shift = torch.clamp(shift, min=-max_shift, max=max_shift)
    
    return shift.item()

def shift_obj_along_z(objc, z_shift):
    """
    Apply a subpixel shift along z using Fourier shift theorem.
    
    Args:
        objc: tensor (omode, z, y, x), complex
        z_shift: float, shift in pixels along +z direction
    Returns:
        shifted tensor, same shape
    """
    if abs(z_shift) < 1e-3:
        return objc
    
    nz = objc.shape[1]
    freq_z = torch.fft.fftfreq(nz, d=1.0, device=objc.device)  # cycles/pixel
    phase_ramp = torch.exp(-2j * torch.pi * freq_z * z_shift)  # (nz,)
    
    # FFT along z only
    obj_f = torch.fft.fft(objc, dim=1)
    obj_f = obj_f * phase_ramp[None, :, None, None]
    obj_shifted = torch.fft.ifft(obj_f, dim=1)
    
    return obj_shifted

def complex_ratio_constraint(model, alpha1, alpha2):
    # https://doi.org/10.1016/j.ultramic.2024.114068
    # https://doi.org/10.1364/OE.18.001981
    # Suggested values for alpha1, alpha2 are 1 and 0
    # For alpha1 = 1, alpha2 = 0, it's suggesting a phase object and phase would not be updated.
    # Namely, objac = exp(-alpha1*Cbar*objp); objpc = objp
    # NOTE that my implementaiton is slightly different from the papers
    # Because for electron ptychography we usually defines a positive phase shift in the transmission function, obj(r) = T(r) = exp(i*sigma*V(r))
    # Hence the object phase = angle(obj) = i*sigma*V(r)
    # So when electron being scattered by the nuclei, it accumulates positive phase shift and slightly less than 1 amplitude
    # Hence the constraint foumula is slightly modified accordingly, so we have positive phase associated with slightly less than 1 amplitude
    
    obja = model.opt_obja
    objp = model.opt_objp
    
    log_obja = torch.log(obja)
    
    # Compute Cbar for the entire object across (omode, z, y, x)
    # Although we can consider repeat this across z slices or omode
    Cbar = (log_obja.abs().sum()) / (objp.abs().sum() + 1e-8)  # Avoid division by zero

    # Compute updated amplitude, note the negative sign for the second term!
    objac = torch.exp((1 - alpha1) * log_obja - alpha1 * Cbar * objp)

    # Compute updated phase, note the negative sign for the second term!
    objpc = (1 - alpha2) * objp - alpha2 / (Cbar + 1e-8) * log_obja # Avoid division by zero
    return objac, objpc, Cbar