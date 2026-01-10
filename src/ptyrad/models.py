"""
Optimizable model of the ptychographic reconstruction using automatic differentiation (AD)

This is the PyTorch model that holds optimizable tensors and interacts with loss and constraints.

"""

import torch
from torch.fft import fft2, ifft2
import torch.nn as nn
from torchvision.transforms.functional import gaussian_blur

from ptyrad.dataloader import MeasDataLoader
from ptyrad.forward import multislice_forward_model_vec_all
from ptyrad.utils import imshift_batch, torch_phasor, vprint

# The obj_ROI_grid is modified from precalculation to on-the-fly generation for memory consumption
# It has very little performance impact but saves lots of memory for large 4D-STEM data
# Added create_grids and print_model_summary for readability and decoupling
# set_optimizer function is called at the end of the initializaiton, while this can also be called if you want to update the optimizer params without initializing the object
# obj optimization is now split into objp and obja
# mixed object modes are normalized by the init_omode_occu. By design this is a fixed value because optimizing omode_occu with obj simultaneously could be unstable
# obj_ROI cropping is done with vectorization and the obj_ROI_grid is only generated once
# probe with sub-px shifts are calculated only when probe_pos_shifts are enables for optimization
# All the sub-px shifted probes in a batch are processed together with vectorizaiton
# Likewise, the multislice forward model is also fully vectorized across samples (in batch), pmode, and omode
# Note that it's possible to reduce the peak-memory consumption by reducing the level of vectorizaiton and roll back to for loops
# Lastly, the forward pass of this model would output the dp_fwd (N, Ky, Kx) and objp_patches (N, omode, Nz, Ny, Nx) in float32 for later loss calculation

class PtychoAD(torch.nn.Module):
    """
    Main optimization class for ptychographic reconstruction using automatic differentiation (AD).

    This class is responsible for initializing the model parameters, setting up the optimizer, and
    performing forward passes to compute diffraction patterns based on the given input indices.

    Attributes:
        device (str): Device to run computations on ('cuda:0' by default).
        verbose (bool): If True, prints model summary (True by default).
        detector_blur_std (float): Standard deviation for detector blur, or None if no blur.
        obj_preblur_std (float): Standard deviation for object pre-blur, or None if no pre-blur.
        lr_params (dict): Learning rate parameters for optimizable tensors.
        opt_obja (torch.Tensor): Amplitude of the object.
        opt_objp (torch.Tensor): Phase of the object.
        opt_obj_tilts (torch.Tensor): Tilts of the object.
        opt_probe (torch.Tensor): Probe function.
        opt_probe_pos_shifts (torch.Tensor): Shifts for the probe positions.
        omode_occu (torch.Tensor): Occupation mode.
        H (torch.Tensor): Propagator matrix.
        measurements (torch.Tensor): Measurements for the ptychographic reconstruction.
        N_scan_slow (torch.Tensor): Number of scans in the slow direction.
        N_scan_fast (torch.Tensor): Number of scans in the fast direction.
        crop_pos (torch.Tensor): Cropping positions.
        slice_thickness (torch.Tensor): slice thickness (dz) parameter.
        dx (torch.Tensor): Pixel size in the x direction.
        dk (torch.Tensor): K-space sampling interval.
        scan_affine (affine.Affine): Affine transformation for scan.
        tilt_obj (bool): Whether object tilts are being optimized.
        shift_probes (bool): Whether probe shifts are being optimized.
        probe_int_sum (torch.Tensor): Sum of squared probe intensities.
        optimizable_tensors (dict): Dictionary of tensors that can be optimized.

    Args:
        init_variables (dict): Dictionary of initial variables required for the model.
        model_params (dict): Dictionary of model parameters including learning rates and blur stds.
        device (str): Device to run computations on. Default is 'cuda:0'.
        verbose (bool): If True, prints model summary. Default is True.
    """

    def __init__(self, init_variables, model_params, device='cuda', verbose=True):
        super(PtychoAD, self).__init__()
        with torch.no_grad():
            
            vprint('### Initializing PtychoAD model ###', verbose=verbose)
            
            # Setup model behaviors
            self.device                 = device
            self.verbose                = verbose
            self.detector_blur_std      = model_params['detector_blur_std']
            self.obj_preblur_std        = model_params['obj_preblur_std']
            self.preload_data           = model_params.get('preload_data', True)
            self.meas_loader            = MeasDataLoader(
                init_variables['measurements'], 
                preload_data=model_params.get('preload_data', True), 
                device=self.device,
                meas_padded=init_variables.get('on_the_fly_meas_padded', None),
                meas_padded_idx=init_variables.get('on_the_fly_meas_padded_idx', None),
                meas_scale_factors=init_variables.get('on_the_fly_meas_scale_factors', None),
            )

            # Parse the learning rate and start iter for optimizable tensors
            start_iter_dict = {}
            end_iter_dict = {}
            lr_dict = {}
            for key, params in model_params['update_params'].items():
                start_iter_dict[key] = params.get('start_iter')
                end_iter_dict[key] = params.get('end_iter')
                lr_dict[key] = params['lr']
            self.optimizer_params       = model_params['optimizer_params']
            self.start_iter             = start_iter_dict
            self.end_iter               = end_iter_dict
            self.lr_params              = lr_dict
            
            # Optimizable parameters
            self.opt_obja               = nn.Parameter(torch.abs(torch.tensor(init_variables['obj'],    device=device)).to(torch.float32))
            self.opt_objp               = nn.Parameter(torch.angle(torch.tensor(init_variables['obj'],  device=device)).to(torch.float32))
            self.opt_obj_tilts          = nn.Parameter(torch.tensor(init_variables['obj_tilts'],                dtype=torch.float32, device=device))
            self.opt_slice_thickness    = nn.Parameter(torch.tensor(init_variables['slice_thickness'],          dtype=torch.float32, device=device))
            self.opt_probe              = nn.Parameter(torch.view_as_real(torch.tensor(init_variables['probe'], dtype=torch.complex64, device=device))) # The `torch.view_as_real` allows correct handling of DDP via NCCL even in PyTorch 2.4
            self.opt_probe_pos_shifts   = nn.Parameter(torch.tensor(init_variables['probe_pos_shifts'],         dtype=torch.float32, device=device))
            
            # Buffers are used during forward pass
            self.register_buffer      ('omode_occu',      torch.tensor(init_variables['omode_occu'],       dtype=torch.float32, device=device))
            self.register_buffer      ('H',               torch.tensor(init_variables['H'],                dtype=torch.complex64, device=device))
            self.register_buffer      ('N_scan_slow',     torch.tensor(init_variables['N_scan_slow'],      dtype=torch.int32, device=device))# Saving this for reference, the cropping is based on self.obj_ROI_grid.
            self.register_buffer      ('N_scan_fast',     torch.tensor(init_variables['N_scan_fast'],      dtype=torch.int32, device=device))# Saving this for reference, the cropping is based on self.obj_ROI_grid.
            self.register_buffer      ('crop_pos',        torch.tensor(init_variables['crop_pos'],         dtype=torch.int32, device=device))# Saving this for reference, the cropping is based on self.obj_ROI_grid.
            self.register_buffer      ('slice_thickness', torch.tensor(init_variables['slice_thickness'],  dtype=torch.float32, device=device))# Saving this for reference
            self.register_buffer      ('dx',              torch.tensor(init_variables['dx'],               dtype=torch.float32, device=device))# Saving this for reference
            self.register_buffer      ('dk',              torch.tensor(init_variables['dk'],               dtype=torch.float32, device=device))# Saving this for reference
            self.register_buffer      ('lambd',           torch.tensor(init_variables['lambd'],            dtype=torch.float32, device=device))
            
            self.random_seed            = init_variables['random_seed']                                                           # Saving this for reference
            self.length_unit            = init_variables['length_unit']                                                           # Saving this for reference
            self.scan_affine            = init_variables['scan_affine']                                                           # Saving this for reference
            self.tilt_obj               = bool(self.lr_params['obj_tilts']        != 0 or torch.any(self.opt_obj_tilts))          # Set tilt_obj to True if lr_params['obj_tilts'] is not 0 or we have any none-zero tilt values
            self.shift_probes           = bool(self.lr_params['probe_pos_shifts'] != 0)                                           # Set shift_probes to True if lr_params['probe_pos_shifts'] is not 0
            self.change_thickness       = bool(self.lr_params['slice_thickness']  != 0)
            self.probe_int_sum          = self.get_complex_probe_view().abs().pow(2).sum() # This is only used for the `fix_probe_int`
            self.loss_iters             = []
            self.iter_times             = []
            self.dz_iters               = []
            self.avg_tilt_iters         = []

            # Create grids for shifting
            self.create_grids()

            # Create a dictionary to store the optimizable tensors
            self.optimizable_tensors = {
                'obja'            : self.opt_obja,
                'objp'            : self.opt_objp,
                'obj_tilts'       : self.opt_obj_tilts,
                'slice_thickness' : self.opt_slice_thickness,
                'probe'           : self.opt_probe,
                'probe_pos_shifts': self.opt_probe_pos_shifts}
            self.create_optimizable_params_dict(self.lr_params, self.verbose)

            # Initialize propagator-related variables
            self.init_propagator_vars()
            
            # Initialize iteration numbers that require torch.compile
            self.init_compilation_iters()
            
            vprint('### Done initializing PtychoAD model ###', verbose=verbose)
            vprint(' ', verbose=verbose)
            
    def get_complex_probe_view(self):
        """ Retrieve complex view of the probe """
        # This is a post-processing to ensure minimal code changes in PtyRAD for the DDP (multiGPU) via NCCL due to limited support for Complex value
        return torch.view_as_complex(self.opt_probe)
        
    def create_grids(self):
        """ Create the grids for shifting probes, selecting obj ROI, and Fresnel propagator in a vectorized approach """
        # Note that the shift_object_grid is pre-generated for potential future usage of sub-px object shifts
        
        device = self.device
        probe = self.get_complex_probe_view()
        Npy, Npx = probe.shape[-2:] # Number of probe pixels in y and x directions
        Noy, Nox = self.opt_objp.shape[-2:] # Number of object pixels in y and x directions
        
       # Grids for Fresnel propagator
       # Note that this grid has a half-bin shift that avoids exact Nyquist frequency (k = -0.5) which would generate NaNs inside sqrt.
       # Due to the half-bin shift, the 0th element is also not exactly 0.
       # If we decided to go with the small angle approximated propagator, then we may unify this grid with the reciprocal probe gird.
        ygrid = (torch.arange(-Npy // 2, Npy // 2, device=device) + 0.5) / Npy
        xgrid = (torch.arange(-Npx // 2, Npx // 2, device=device) + 0.5) / Npx
        ky = torch.fft.ifftshift(2 * torch.pi * ygrid / self.dx) # Use ifftshift to shift the 0 frequency to the corner
        kx = torch.fft.ifftshift(2 * torch.pi * xgrid / self.dx)
        Ky, Kx = torch.meshgrid(ky, kx, indexing="ij")
        self.propagator_grid = torch.stack([Ky,Kx], dim=0) # (2,Ky,Kx), k-space grid for Fresnel propagator with 2pi absorbed
        
        # Grids for obj_ROI selection
        rpy, rpx = torch.meshgrid(torch.arange(Npy, dtype=torch.int32, device=device), 
                                  torch.arange(Npx, dtype=torch.int32, device=device), indexing='ij') # real space grid for probe in y and x directions
        self.rpy_grid = rpy # real space grid with y-indices spans across probe extent
        self.rpx_grid = rpx
        
        # Grids for shifting probes and objects
        # This is the typical fftfreq grid that ranges from [-0.5, 0.5) and the 0th element is exactly 0
        kpy, kpx = torch.meshgrid(torch.fft.fftfreq(Npy, dtype=torch.float32, device=device),
                                  torch.fft.fftfreq(Npx, dtype=torch.float32, device=device), indexing='ij')
        koy, kox = torch.meshgrid(torch.fft.fftfreq(Noy, dtype=torch.float32, device=device),
                                  torch.fft.fftfreq(Nox, dtype=torch.float32, device=device), indexing='ij')
        self.shift_probes_grid = torch.stack([kpy, kpx], dim=0) # (2,Npy,Npx), normalized k-space grid stack for sub-px probe shifting 
        self.shift_object_grid = torch.stack([koy, kox], dim=0) # (2,Noy,Nox), normalized k-space grid stack for sub-px object shifting (Implemented for completeness, not used in PtyRAD)
    
    def create_optimizable_params_dict(self, lr_params, verbose=True):
        """ Sets the optimizer with lr_params """
        # # Use this to edit learning rate if needed some refinement

        # model.set_optimizer(lr_params={'obja'            : 5e-4,
        #                                'objp'            : 5e-4,
        #                                'obj_tilts'       : 1e-4,
        #                                'probe'           : 1e-4, 
        #                                'probe_pos_shifts': 1e-4})
        # optimizer=torch.optim.Adam(model.optimizer_params)
        
        self.lr_params = lr_params
        self.optimizable_params = []
        for param_name, lr in lr_params.items():
            if param_name not in self.optimizable_tensors:
                raise ValueError(f"WARNING: '{param_name}' is not a valid parameter name, check your `update_params` and choose from 'obja', 'objp', 'obj_tilts', 'slice_thickness', 'probe', and 'probe_pos_shifts'")
            else:
                self.optimizable_tensors[param_name].requires_grad = (lr != 0) and (self.start_iter[param_name] ==1) # Set requires_grad based on learning rate and start_iter
                if lr != 0:
                    self.optimizable_params.append({'params': [self.optimizable_tensors[param_name]], 'lr': lr})               
        if verbose:
            self.print_model_summary()

    def init_propagator_vars(self):
        """ Initialize propagator related variables """
        
        # Initialize propagator for fixed non-zero tilts and fixed thickness that could be position dependent
        # It's better to calculate the full one during initialization and slice it later given indices so we can use torch.compile later
        dz = self.opt_slice_thickness.detach()
        Ky, Kx = self.propagator_grid 
        tilts_y_full = self.opt_obj_tilts[:,0,None,None] / 1e3 # mrad, tilts_y = (N,Y,X)
        tilts_x_full = self.opt_obj_tilts[:,1,None,None] / 1e3
        self.H_fixed_tilts_full = self.H * torch_phasor(dz * (Ky * torch.tan(tilts_y_full) + Kx * torch.tan(tilts_x_full))) # (1,Y,X) or (N,Y,X)

        # Initialize other relevant variables
        self.k = 2 * torch.pi / self.lambd
        self.Kz = torch.sqrt(self.k ** 2 - Kx ** 2 - Ky ** 2) # Upper case K indicates it's a 2D tensor (Y,X)
    
    def init_compilation_iters(self):
        """ Initialize iteration numbers that require torch.compile """
        compilation_iters = {1}  # Always compile at first iteration
        
        for param_name in self.optimizable_tensors.keys():
            start_iter = self.start_iter.get(param_name)
            end_iter = self.end_iter.get(param_name)
            
            # Add start_iter compilation points
            if start_iter is not None and start_iter >= 1:
                compilation_iters.add(start_iter)
            
            # Add end_iter compilation points
            if end_iter is not None and end_iter >= 1:
                # Compile at end_iter to handle the transition. end_iter is exclusive for grad calculation.
                compilation_iters.add(end_iter)
        
        # Store as sorted list
        self.compilation_iters = sorted(compilation_iters)
        
    def print_model_summary(self):
        """ Prints a summary of the model's optimizable variables and statistics. """
        # Set all the print as vprint so that it'll only print once in DDP, the actual `if verbose` is set outside of the function
        vprint('### PtychoAD optimizable variables ###')
        for name, tensor in self.optimizable_tensors.items():
            vprint(f"{name.ljust(16)}: {str(tensor.shape).ljust(32)}, {str(tensor.dtype).ljust(16)}, device:{tensor.device}, grad:{str(tensor.requires_grad).ljust(5)}, lr:{self.lr_params[name]:.0e}")
        total_var = sum(tensor.numel() for _, tensor in self.optimizable_tensors.items() if tensor.requires_grad)
        # When you create a new model, make sure to pass the optimizer_params to optimizer using "optimizer = torch.optim.Adam(model.optimizer_params)"
        vprint(" ")        
        
        vprint('### Optimizable variables statitsics ###')
        vprint(f"Total measurement values  : {self.meas_loader.meas_arr.size:,d}")
        vprint(f"Total optimizing variables: {total_var:,d}")
        vprint(f"Overdetermined ratio      : {self.meas_loader.meas_arr.size/total_var:.2f}")
        vprint(" ")
        
        vprint('### Model behavior ###')
        vprint(f"Obj preblur               : {True if self.obj_preblur_std is not None else False}")
        vprint(f"Tilt propagator           : {self.tilt_obj}")
        vprint(f"Change slice thickness    : {self.change_thickness}")
        vprint(f"Sub-px probe shift        : {self.shift_probes}")
        vprint(f"Detector blur             : {True if self.detector_blur_std is not None else False}")
        vprint(f"Preload data              : {self.preload_data}")
        vprint(f"On-the-fly meas padding   : {True if self.meas_loader.meas_padded is not None else False}")
        vprint(f"On-the-fly meas resample  : {True if self.meas_loader.meas_scale_factors is not None else False}")
        vprint(" ")
    
    def get_obj_ROI(self, indices):
        """ Get object ROI with integer coordinates """
        # It's strongly recommended to do integer version of get_obj_ROI
        # opt_obj.shape = (B,D,H,W,C) = (omode,D,H,W,2)
        # object_roi = (N,B,D,H,W,2), N is the additional sample index within the input batch, B is now used for omode.
        
        # rpy_grid is the y-grid (Ny,Nx), by adding the y coordinates from init_crop_pos (N,1) in a broadcast way, it becomes (N,Ny,Nx)
        # obj_ROI_grid_y = (N,Ny,Nx)
        
        opt_obj = torch.stack([self.opt_obja, self.opt_objp], dim=-1)
        obj_ROI_grid_y = self.rpy_grid[None,:,:] + self.crop_pos[indices, None, None, 0]
        obj_ROI_grid_x = self.rpx_grid[None,:,:] + self.crop_pos[indices, None, None, 1]
        
        object_roi = opt_obj[:,:,obj_ROI_grid_y,obj_ROI_grid_x,:].permute(2,0,1,3,4,5)
        return object_roi
    
    def get_obj_patches(self, indices):
        """
        Get object patches from specified indices
        
        """
        
        object_patches = self.get_obj_ROI(indices)
        
        if self.obj_preblur_std is None or self.obj_preblur_std == 0:
            return object_patches
        
        else:
            # Permute and reshape approach, this is much faster than the stack/list comprehension version
            obj = object_patches.permute(5,0,1,2,3,4) # Move the r/i to the front so it's (2,N,B,D,H,W)
            obj_shape = obj.shape
            obj = obj.reshape(-1, obj_shape[-2], obj_shape[-1])
            object_patches = gaussian_blur(obj, kernel_size=5, sigma=self.obj_preblur_std).reshape(obj_shape).permute(1,2,3,4,5,0) # The torchvision Gaussian blur only acts on last 2 dimensions
            return object_patches
        
    def get_probes(self, indices):
        """ Get probes for each position """
        # This function will return a probe tensor with (N, pmode, Ny, Nx)
        # If you're not trying to optimize probe positions, there's not much point using sub-px shifted stationary probes
        # So the function would broadcast the same probe across the batch dimension,
        # and would only be returning multiple sub-px shifted probes if you're optimizing self.opt_probe_pos_shifts

        probe = self.get_complex_probe_view()
        
        if self.shift_probes:
            probes = imshift_batch(probe, shifts = self.opt_probe_pos_shifts[indices], grid = self.shift_probes_grid)
        else:
            probes = torch.broadcast_to(probe, (indices.shape[0], *probe.shape)) # Broadcast a batch dimension, essentially using same probe for all samples
        return probes
    
    def get_propagators(self, indices):
        """ Get propagators for each position """
        # self.tilt_obj is True as long as we're optimizing the opt_obj_tilts or we have non-zero initial tilt values
        # This function will return a single propagator (H) if self.opt_obj_tilts has shape = (1,2) (single tilt_y, tilt_x) 
        # If self.opt_obj_tilts has shape = (N,2), it'll return multiple propagtors stacked at axis 0 (N,Y,X)
        # Note that 0 tilts is numerically equivalent to the H and can be verified by "torch.allclose(model.H, model.get_propagators([0]))"
        # The exp(2pi * i * sqrt(k^2 - kx^2 - ky^2)) approach is equivalent to the common exp(-i * pi * lambda * dz * k^2) for small angles, 
        # see J. Goodman, Introduction to Fourier Optics (McGraw-Hill, 1968) (PDF page 88, eqn 4-20, 4-21 as attached).
        # https://www.hlevkin.com/hlevkin/90MathPhysBioBooks/Physics/Physics/Mix/Introduction%20to%20Fourier%20Optics.pdf
        
        # Note that torch.exp(1j*phase_shift) is not compatible with torch.compile because the 1j is a Python built-in and not a tensor,
        # so I've replaced them with torch.polar(torch.ones_like(phase), phase), which is wrapped as a util function `torch_phasor(phase)` 
        
        # Setup boolean flags
        tilt_obj = self.tilt_obj                         # Whether we need to apply tilt to the Fresnel propagator
        global_tilt = (self.opt_obj_tilts.shape[0] == 1) # 'tilt_type' = 'all' or 'each'
        change_tilt = (self.lr_params['obj_tilts'] != 0) # Whether tilts are optimized or not
        change_thickness = self.change_thickness         # Whether thickness is optimized or not
        
        # Setup tilts and other variables
        dz       = self.opt_slice_thickness
        Kz       = self.Kz # kz = torch.sqrt(k ** 2 - Kx ** 2 - Ky ** 2), k = 2pi/lambda.
        Ky, Kx   = self.propagator_grid
        
        # tilts can be either (1,2) or (N,2) depends on global_tilt flag
        if global_tilt:
            tilts = self.opt_obj_tilts 
        else: 
            tilts = self.opt_obj_tilts[indices] 
        tilts_y  = tilts[:,0,None,None] / 1e3 # mrad, tilts_y = (N,Y,X)
        tilts_x  = tilts[:,1,None,None] / 1e3
                
        if tilt_obj and change_thickness:
            # Case 1: Tilts are either non-zero or optimizing, while thickness is optimizing 
            H_opt_dz = torch_phasor(dz * Kz) # H has zero frequency at the corner in k-space
            return H_opt_dz * torch_phasor(dz * (Ky * torch.tan(tilts_y) + Kx * torch.tan(tilts_x)))

        elif tilt_obj and not change_thickness:
            if change_tilt:
                # Case 2A: Tilts are optimizing, while thickness is fixed
                return self.H * torch_phasor(dz * (Ky * torch.tan(tilts_y) + Kx * torch.tan(tilts_x)))
            else:
                # Case 2B: Tilts are fixed non-zero values (1,2) or (N,2), while thickness is fixed
                # Propagator is pre-calculated during init_propagator_vars
                return self.H_fixed_tilts_full if global_tilt else self.H_fixed_tilts_full[indices]
        
        elif not tilt_obj and change_thickness: 
            # Case 3: Tilt is fixed at 0 and thickness is optimizing
            H_opt_dz = torch_phasor(dz * Kz)
            return H_opt_dz[None,]
            
        else: 
            # Case 4: Tilt is fixed at 0 and thickness is fixed
            return self.H[None,]

    def get_propagated_probe(self, index):
        probe = self.get_probes(index)[0].detach() # (pmode, Ny, Nx), just grab the probe at 1st index
        H = self.get_propagators(index)[[0]].detach() # (1, Ny, Nx) or (N, Ny, Nx) depends on tilt_type ('all' or 'each'), so we need to grab the 1st index without reducing dimension
        n_slices = self.opt_objp.shape[1]
        probe_prop = torch.zeros((n_slices, *probe.shape), dtype=probe.dtype, device=probe.device)
        
        psi = probe # (z, pmode, Ny, Nx)
        for n in range(n_slices):
            probe_prop[n] = psi
            psi = ifft2(H[None,] * fft2(psi))
        
        return probe_prop
    
    def get_forward_meas(self, object_patches, probes, propagators):
        
        dp_fwd = multislice_forward_model_vec_all(object_patches, probes, propagators, omode_occu=self.omode_occu)
        
        if self.detector_blur_std is not None and self.detector_blur_std != 0:
            kernel_size = max(5, 2*round(3*self.detector_blur_std)+1) # Kernel size would have minimum 5, and scale with 6sigma+1
            dp_fwd = gaussian_blur(dp_fwd, kernel_size=kernel_size, sigma=self.detector_blur_std)
            
        return dp_fwd

    @torch.compiler.disable
    def get_measurements(self, indices=None):
        """ Get measurements for each position through the data loader"""
        # Return the selected measurements based on input indices
        # If no indices are passed, return the original numpy arr of measurements
        
        if indices is not None:
            return self.meas_loader[indices]
        else:
            return self.meas_loader.meas_arr
    
    def clear_cache(self):
        """Clear temporary attributes like cached object patches."""
        self._current_object_patches = None    
        
    def forward(self, indices):
        """ Doing the forward pass and get an output diffraction pattern for each input index """
        # The indices are passed in as an array and representing the whole batch
        # Note that detector blur is a physical process and should be included in the forward method
        # It's a design choice to put it here, instead of putting it under optimization.py

        object_patches = self.get_obj_patches(indices)
        probes         = self.get_probes(indices)
        propagators    = self.get_propagators(indices)
        dp_fwd         = self.get_forward_meas(object_patches, probes, propagators)
        
        # Keep the object_patches for later object-specific loss
        self._current_object_patches = object_patches
        
        return dp_fwd