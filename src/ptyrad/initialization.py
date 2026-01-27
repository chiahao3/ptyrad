"""
Initialize measurements, object, probe, probe positions, tilts, and other variables

This module is completely using NumPy for interoperability, 

so users can initialize their data with PtyRAD first and reconstruct with other approaches later.

"""

import os
from copy import deepcopy
from math import floor

import numpy as np
from scipy.io.matlab import matfile_version as get_matfile_version
from scipy.ndimage import gaussian_filter, zoom

from ptyrad.load import load_array_from_file, load_hdf5, load_mat, load_ptyrad
from ptyrad.save import save_array
from ptyrad.utils import (
    complex_object_z_resample_torch,
    compose_affine_matrix,
    create_one_hot_mask,
    exponential_decay,
    fit_background,
    fit_cbed_pattern,
    get_EM_constants,
    get_nested,
    guess_radius_of_bright_field_disk,
    infer_dx_from_params,
    make_fzp_probe,
    make_mixed_probe,
    make_stem_probe,
    near_field_evolution,
    orthogonalize_modes_vec_np,
    power_law,
    set_random_seed,
    sort_by_mode_int_np,
    vprint,
)


class Initializer:
    def __init__(self, init_params, seed=None, verbose=True):
        
        # A deepcopy creates a new object so modifying self.init_params won't affect the original init_params dict that was outside the class
        # This is important because self.init_params might get updated if there's cropping, padding, or resampling of the measurements
        # The original params file will could be directly saved to the output dir with `save.copy_params_to_dir`,
        # while we also keep a digital copy of the original params in self.init_params_original
        
        self.init_params = deepcopy(init_params) # This is the central params dict that will be used for initialization
        self.init_params_original = deepcopy(init_params)
        self.init_variables = {'random_seed': seed} # This dict stores all the variables that will be used for the later ptychography reconstruction
        self.random_seed = seed
        self.verbose=verbose
        self.print_init_params()
    
    ##### Public methods for initializing everything #####
    def print_init_params(self):
        ''' Print the current init_params in the Initialzier object '''
        
        vprint("init_params are displayed below:", verbose=self.verbose)
        for key, value in self.init_params.items():
            vprint(f"  {key}: {value}", verbose=self.verbose)
        vprint(" ", verbose=self.verbose)

    def init_cache(self):
        """ Check if the source paths are the same, if so, we may cache that field to reduce file loading time """
        # Note:
        # For caching, at least 2 out of 3 fields are using the same file path
        # Therefore, there's only one possible source for the self.cache_contents
        # With 2 file source posibilities, the self.cache_contents is either caching from 'PtyRAD' or 'PtyShv'
        # Even we add more file type supports in the future (py4dstem or ptypy), the cache would still be a single file type
        
        vprint("### Initializing cache ###", verbose=self.verbose)
        
        # Initialize flags for cached fields
        self.cache_source = None
        self.cache_path = None
        self.cache_contents = None
        self.use_cached_obj = False
        self.use_cached_probe = False
        self.use_cached_pos = False
        
        # Set cache_source, cache_path, and use_cached_xxx flags iteratively
        for source in ['PtyRAD', 'PtyShv', 'py4DSTEM']:
            self._set_use_cached_flags(source)
            
        # Set cache_contents
        if any([self.use_cached_obj, self.use_cached_probe, self.use_cached_pos]):
            if self.cache_source == 'PtyRAD':
                vprint(f"Loading 'PtyRAD' file from {self.cache_path} for caching", verbose=self.verbose)
                self.cache_contents = load_ptyrad(self.cache_path)
            elif self.cache_source == 'PtyShv':
                vprint(f"Loading 'PtyShv' file from {self.cache_path} for caching", verbose=self.verbose)
                self.cache_contents = load_mat(self.cache_path, key=['object', 'probe', 'outputs.probe_positions'], delimiter='.') # flattend dict with key using delimiter
            elif self.cache_source == 'py4DSTEM':
                vprint(f"Loading 'py4DSTEM' file from {self.cache_path} for caching", verbose=self.verbose)
                self.cache_contents = load_hdf5(self.cache_path, key=None)
            else:
                raise ValueError(f"File type {source} not implemented for caching yet, please use 'PtyRAD', or 'PtyShv'!")
            
        # Cache is only used when 2 out of 3 fields have the same source and path, so the following flags could only be all false, 1 false 2 true, or 3 true.
        vprint(f"use_cached_obj   = {self.use_cached_obj}", verbose=self.verbose)
        vprint(f"use_cached_probe = {self.use_cached_probe}", verbose=self.verbose)
        vprint(f"use_cached_pos   = {self.use_cached_pos}", verbose=self.verbose)
        vprint(" ", verbose=self.verbose)
        
    def init_measurements(self):
        vprint("### Initializing measurements ###", verbose=self.verbose)

        meas = self._load_meas()
        meas = self._process_meas(meas)

        meas_avg_pattern = np.mean(meas, axis=0, dtype=np.float32) # (ky, kx), this is equivalent to PACBED in electron microscopy. Note that if pad/resample are set to "on_the_fly", this would be different from the final one used for reconstruction.
        meas_total_ints = np.sum(meas, axis=(1,2), dtype=np.float32) # (N,), this is the integrated intensities for each individual pattern
        
        pad_mode = get_nested(self.init_params, key=['meas_pad', 'mode'], safe=True, default=None)
        if pad_mode == 'on_the_fly':
            padded = self.init_variables.get('on_the_fly_meas_padded')
            padded_int_sum = padded.sum() if padded is not None else 0
            vprint(f"Adjusting `meas_total_ints` by adding {padded_int_sum:.4f} for on_the_fly meas padding", verbose=self.verbose)
            meas_total_ints += padded_int_sum # meas_total_ints is used to normalize the probe intensity. 
            # Because the meas could gain intensity during on_the_fly padding, 
            # we need to consider the extra intensity from the padded region here. 
        
        self.init_variables['meas_total_ints'] = meas_total_ints
        self.init_variables['meas_avg_pattern'] = meas_avg_pattern
        self.init_variables['measurements'] = meas
        
        export_params = self.init_params.get('meas_export') # Ture, False, None, dict (could be {})
        if export_params is True or isinstance(export_params, dict):
            vprint(f"Exporting measurements with `meas_export` = {export_params}")
            self._export_meas(export_params if isinstance(export_params, dict) else {})
        
        # Print out some measurements statistics
        vprint(f"Pattern total int. statistics (min, mean, max)        = ({meas_total_ints.min():.4f}, {meas_total_ints.mean():.4f}, {meas_total_ints.max():.4f}), with min/max = {(meas_total_ints.min()/meas_total_ints.max()):.1%}", verbose=self.verbose)
        vprint(f"Global meausrements int. statistics (min, mean, max)  = ({meas.min():.4f}, {meas.mean():.4f}, {meas.max():.4f})", verbose=self.verbose)
        vprint(f"measurements                              (N, Ky, Kx) = {meas.dtype}, {meas.shape}", verbose=self.verbose)
        vprint(" ", verbose=self.verbose)

    def init_calibration(self):
        vprint("### Setting up calibration ###", verbose=self.verbose)

        calib_dict  = self.init_params['meas_calibration']
        calib_mode  = calib_dict['mode'] # One of 'dx', 'dk', 'kMax', 'da', 'angleMax', 'RBF', 'n_alpha', or 'fitRBF'
        calib_value = calib_dict.get('value') # fitRBF doesn't need a value here
        Npix        = self.init_params_original.get('meas_Npix') # Load the original Npix because init_params['meas_Npix'] could have been modified in init_measurements
        conv_angle  = self.init_params.get('probe_conv_angle')
        illum_type  = self.init_params.get('probe_illum_type') or 'electron'
        vprint(f"meas_calibration mode = '{calib_mode}', value = {calib_value}", verbose=self.verbose) # No need to add :.4f to value because it could be None, also it's user input so won't have too many digits
        
        # Load the meas_raw_avg first to ensure measurement is initialized
        try: 
            meas_raw_avg = self.init_variables['meas_raw_avg'] # This is the averaged measurements with only simple permuting/reshaping/flipping
        except KeyError:
            vprint("Warning: 'init_variables['meas_raw_avg]' not found. Initializing measurements first for calibration...", verbose=self.verbose)
            vprint(" ", verbose=self.verbose)
            self.init_measurements()
            meas_raw_avg = self.init_variables['meas_raw_avg']
        
        if illum_type == 'electron':
            # Get wavelength
            energy  = self.init_params.get('probe_kv') # kV
            wavelength = get_EM_constants(energy, output_type='wavelength') # wavelength in Ang
            unit_str = 'Ang'
            
            # Run fitRBF routine for electron ptychography
            vprint("Using loaded raw averaged measurement (before crop/pad/resample) to fit RBF as a part of the meas calibration", verbose=self.verbose)
            fitRBF = guess_radius_of_bright_field_disk(meas_raw_avg, thresh=calib_dict.get('thresh', 0.5))
            
            vprint(f"Radius of fitted bright field disk (RBF) = {fitRBF:.2f} px with Npix = {meas_raw_avg.shape[-1]}", verbose=self.verbose)
            vprint(f"Suggested probe_mask_k radius (RBF*2/Npix) > {(fitRBF * 2 / Npix):.4f}", verbose=self.verbose)
            
            vprint("Fitting raw averaged measurement with center, radius, and Gaussian blur std as a sanity check", verbose=self.verbose)
            vprint("Note that the fitted Gaussian blur std (detector blur) would be affected by overlapping Bragg disks", verbose=self.verbose)
            _ = fit_cbed_pattern(meas_raw_avg, verbose=self.verbose)
            
            # Actually calculating dx for each calib_mode
            if calib_mode == 'fitRBF':
                dx = infer_dx_from_params(**{'RBF': fitRBF, 'Npix': Npix, 'wavelength': wavelength, 'conv_angle': conv_angle})
            else:
                dx = infer_dx_from_params(**{calib_mode: calib_value, 'Npix': Npix, 'wavelength': wavelength, 'conv_angle': conv_angle})
                if calib_mode != 'RBF': 
                    inferRBF = conv_angle / 1e3 * Npix * dx / wavelength # We can still infer RBF using the user provided calib value
                    vprint(f"Using init_params, the inferred RBF (conv_angle / 1e3 * Npix * dx / wavelength) = {inferRBF:.2f} px with Npix = {meas_raw_avg.shape[-1]}", verbose=self.verbose)

            if calib_mode in ['fitRBF', 'RBF']:
                vprint("WARNING: The 'fitRBF' and 'RBF' calibration methods are highly dependent on the accuracy of user-provided experimental parameters and acquisition conditions,",verbose=self.verbose)
                vprint("         including convergence angle, kV, dose, specimen thickness, and collection angle for the estimation of RBF.", verbose=self.verbose)
                vprint("         For example, a 5-10% error in convergence angle is fairly common.", verbose=self.verbose)
                vprint("         Users are strongly advised to perform proper microscope calibration to ensure accurate results.", verbose=self.verbose)
                vprint("         These method should only be used as a rough estimate and not as a substitute for proper experimental calibration.", verbose=self.verbose)
                
        elif illum_type == 'xray':
            if calib_mode in ['RBF', 'fitRBF', 'n_alpha']:
                raise ValueError(f"Calibration mode '{calib_mode}' is not supported for xray. Use 'dx', 'dk', 'kMax', 'da', 'angleMax'.")
            # Get wavelength
            energy  = self.init_params.get('beam_kev') # keV
            wavelength = 1.23984193e-9 / energy # wavelength in m, energy in keV
            unit_str = 'm'
            
            # Infer dx calibration from provided values
            dx = infer_dx_from_params(**{calib_mode: calib_value,
                                        'Npix': Npix,
                                        'wavelength': wavelength})
            
        else:
            raise ValueError(f"'probe_illum_type' = {illum_type} not implemented yet, please use either 'electron' or 'xray'!")
        
        # Print the information

        vprint(f"dx (real space pixel size of probe and object) set to {dx:.4f} {unit_str} with Npix = {meas_raw_avg.shape[-1]}", verbose=self.verbose)
        
        Npix_is_modified = False
        
        # Handle additional changes to dx if there's meas_crop
        crop_ranges = self.init_params.get('meas_crop')
        if crop_ranges is not None and len(crop_ranges) == 4:
            if crop_ranges[-1] is not None and len(crop_ranges[-1]) == 2:
                kx_i, kx_f = crop_ranges[-1]
                Npix_new = kx_f - kx_i
                dx = dx * Npix / Npix_new
                Npix_is_modified = True
                Npix_modified = Npix_new
                vprint(f"Update dx to {dx:.4f} {unit_str} due to meas_crop, Npix = {Npix_modified}", verbose=self.verbose)
                if illum_type == 'electron':
                    vprint(f"Suggested probe_mask_k radius (RBF*2/Npix) changes to > {(fitRBF * 2 / Npix_modified):.4f}", verbose=self.verbose)
        
        # Handle additional changes to dx if there's meas_pad
        pad_cfg = self.init_params.get('meas_pad')
        if pad_cfg is not None and pad_cfg.get('mode') is not None:
            mode = pad_cfg['mode']  # 'precompute' or 'on_the_fly'
            padding_type = pad_cfg['padding_type']
            target_Npix = pad_cfg['target_Npix']
            if Npix_is_modified:
                Npix = Npix_modified
            dx = dx * Npix / target_Npix
            vprint(f"Update dx to {dx:.4f} {unit_str} due to meas_pad (mode = {mode}, padding_type = {padding_type}), Npix = {target_Npix}", verbose=self.verbose)
            if illum_type == 'electron':
                vprint(f"Suggested probe_mask_k radius (RBF*2/Npix) changes to > {(fitRBF * 2 / target_Npix):.4f}", verbose=self.verbose)

        # Handle additional change to fitRBF if there's meas_resample
        resample_cfg = self.init_params.get('meas_resample')
        if resample_cfg is not None and resample_cfg.get('mode') is not None:
            mode = resample_cfg['mode'] # 'precompute' or 'on_the_fly'
            scale_factors = resample_cfg['scale_factors']
            fitRBF_modified = fitRBF * scale_factors[0] # Currently the 2 values need to be the same
            final_Npix = self.init_params['meas_Npix']
            vprint(f"Update fitRBF to {fitRBF_modified:.4f} due to meas_resample (mode = {mode}, scale_factors = {scale_factors}), Npix = {final_Npix}", verbose=self.verbose)
            if illum_type == 'electron':
                vprint(f"Suggested probe_mask_k radius (RBF*2/Npix) changes to > {(fitRBF_modified * 2 / final_Npix):.4f}", verbose=self.verbose)
                    
        # Set the final dx for internal calibration, this dx would be used for probe, pos, object_extent, H
        self.init_params['probe_dx'] = dx
        vprint(" ", verbose=self.verbose)

    def set_variables_dict(self):
        vprint("### Setting init_variables dict ###", verbose=self.verbose)
        
        # Note that the self.init_params can be modified by _meas_crop and other methods
        # So this method is called after the entire init_measurements is done
        # Keep in mind that crop could modify dx, Npix, scans
        # pad could modify dx, Npix
        # resample would only modify Npix
        
        probe_illum_type = self.init_params.get('probe_illum_type') or 'electron'
        if  probe_illum_type == 'electron':
            voltage     = self.init_params['probe_kv']
            wavelength  = get_EM_constants(voltage, output_type='wavelength')
            unit_str    = 'Ang'
            conv_angle  = self.init_params['probe_conv_angle']
            Npix        = self.init_params['meas_Npix']
            N_scan_slow = self.init_params['pos_N_scan_slow']
            N_scan_fast = self.init_params['pos_N_scan_fast']
            N_scans     = N_scan_slow * N_scan_fast
            dx          = self.init_params['probe_dx']
            dk          = 1 / (dx * Npix)
            kMax        = Npix * dk / 2
            da          = dk * wavelength * 1e3
            angleMax    = Npix * da / 2
            inferRBF    = conv_angle / da 
            n_alpha     = angleMax / conv_angle
            
            # Print some derived values for sanity check
            if self.verbose:
                vprint("Derived values given input init_params:")
                vprint(f'  kv          = {voltage} kV')    
                vprint(f'  wavelength  = {wavelength:.4f} Ang')
                vprint(f'  conv_angle  = {conv_angle} mrad')
                vprint(f'  Npix        = {Npix} px')
                vprint(f'  dk          = {dk:.4f} Ang^-1')
                vprint(f'  kMax        = {kMax:.4f} Ang^-1')
                vprint(f'  da          = {da:.4f} mrad')
                vprint(f'  angleMax    = {angleMax:.4f} mrad')
                vprint(f'  RBF         = {inferRBF:.4f} px (Inferred from the given calibration, NOT necessarily from the loaded measurement data)')
                vprint(f'  n_alpha     = {n_alpha:.4f} (# conv_angle)')
                vprint(f'  dx          = {dx:.4f} Ang, Nyquist-limited dmin = 2*dx = {2*dx:.4f} Ang')
                vprint(f'  Rayleigh-limited resolution  = {(0.61*wavelength/conv_angle*1e3):.4f} Ang (0.61*lambda/alpha for focused probe )')
                vprint(f'  Real space probe extent = {dx*Npix:.4f} Ang')

        elif probe_illum_type == 'xray':
            energy      = self.init_params['beam_kev']
            wavelength  = 1.23984193e-9 / energy
            unit_str    = 'm'
            dx          = self.init_params['probe_dx']
            N_scan_slow = self.init_params['probe_N_scan_slow']
            N_scan_fast = self.init_params['probe_N_scan_fast']
            N_scans     = N_scan_slow * N_scan_fast
            Npix        = self.init_params['meas_Npix']
            dRn         = self.init_params['probe_dRn']
            Rn          = self.init_params['probe_Rn']
            D_H         = self.init_params['probe_D_H']
            D_FZP       = self.init_params['probe_D_FZP']
            Ls          = self.init_params['probe_Ls']
            dk          = 1/(dx*Npix)
            
            if self.verbose:
                vprint("Derived values given input init_params:")
                vprint(f'  x-ray beam energy  = {energy} keV')    
                vprint(f'  wavelength         = {wavelength} m')
                vprint(f'  outmost zone width = {dRn} m')
                vprint(f'  Rn                 = {Rn} m')
                vprint(f'  D_H                = {D_H} m')
                vprint(f'  D_FZP              = {D_FZP} m')
                vprint(f'  Ls                 = {Ls} m')
                vprint(f'  Npix               = {Npix} px')
                vprint(f'  dx                 = {dx} m')
        
        else:
            raise ValueError(f"init_params['probe_illum_type'] = {probe_illum_type} not implemented yet, please use either 'electron' or 'xray'!")
        
        # Save general values into init_variables
        # While they aren't necessarily "critical" for all initialization scenarios (like some variables aren't needed when we load things),
        # But it's better to request these experimental parameters from users, since most of them should come with the measurements.
        # We should collect all available experimental parameteres needed for minimal reconstruction from scratch
        # And keep them with useful derived values in self.init_variables dict
        
        # TODO: May consider use here to centralize the initalizaiton of all useful/derived variables
        self.init_variables['probe_illum_type'] = probe_illum_type
        self.init_variables['lambd']            = wavelength # Ang for electron, m for x-ray
        self.init_variables['length_unit']      = unit_str
        self.init_variables['Npix']             = Npix
        self.init_variables['probe_shape']      = np.array([Npix, Npix]).astype(float) # Keep this at float for later init_pos
        self.init_variables['N_scan_slow']      = N_scan_slow
        self.init_variables['N_scan_fast']      = N_scan_fast
        self.init_variables['N_scans']          = N_scans
        self.init_variables['scan_step_size']   = self.init_params['pos_scan_step_size']
        self.init_variables['dx']               = dx #   Ang
        self.init_variables['dk']               = dk # 1/Ang
        self.init_variables['slice_thickness']  = self.init_params['obj_slice_thickness']
        vprint(" ", verbose=self.verbose)

    def init_probe(self):
        """
        Initialize the probe by loading or simulating and then processing it.
        """
        vprint("### Initializing probe ###", verbose=self.verbose)

        probe = self._load_probe()
        probe = self._process_probe(probe)
        
        self.init_variables['probe'] = probe

        # Print summary
        vprint(f"probe                         (pmode, Ny, Nx) = {probe.dtype}, {probe.shape}", verbose=self.verbose)
        vprint(" ", verbose=self.verbose)

    def init_pos(self):
        """
        Initialize the probe positions by loading and processing them.
        """
        vprint("### Initializing probe positions ###", verbose=self.verbose)
    
        pos = self._load_pos()
        pos = self._process_pos(pos)

        probe_shape = self.init_variables['probe_shape']
        obj_lateral_extent = (1.2 * np.ceil(pos.max(0) - pos.min(0) + probe_shape)).astype(int)
        crop_pos = np.round(pos).astype('int16')
        probe_pos_shifts = (pos - crop_pos).astype('float32')
        
        # Save the processed positions
        self.init_variables['obj_lateral_extent'] = obj_lateral_extent
        self.init_variables['crop_pos'] = crop_pos
        self.init_variables['probe_pos_shifts'] = probe_pos_shifts
        self.init_variables['scan_affine'] = self.init_params['pos_scan_affine']
    
        # Print summary
        vprint(f"crop_pos                                (N,2) = {crop_pos.dtype}, {crop_pos.shape}", verbose=self.verbose)
        vprint(f"crop_pos 1st and last px coords (y,x)         = {crop_pos[0].tolist(), crop_pos[-1].tolist()}", verbose=self.verbose)
        vprint(f"crop_pos extent (Ang)                         = {(crop_pos.max(0) - crop_pos.min(0))*self.init_variables['dx']}", verbose=self.verbose)
        vprint(f"probe_pos_shifts                        (N,2) = {probe_pos_shifts.dtype}, {probe_pos_shifts.shape}", verbose=self.verbose)
        vprint(" ", verbose=self.verbose)

    def init_obj(self):
        """
        Initialize the object by loading and processing it.
        """
        vprint("### Initializing object ###", verbose=self.verbose)

        obj = self._load_obj()
        obj = self._process_obj(obj)
        obj = obj.astype('complex64')
        
        self.init_variables['obj'] = obj

        # Print summary
        dz = self.init_variables['slice_thickness']
        dx = self.init_variables['dx']
        vprint(f"object                    (omode, Nz, Ny, Nx) = {obj.dtype}, {obj.shape}", verbose=self.verbose)
        vprint(f"object extent                 (Z, Y, X) (Ang) = {np.round((obj.shape[1]*dz, obj.shape[2]*dx, obj.shape[3]*dx),4)}", verbose=self.verbose)
        vprint(" ", verbose=self.verbose)

    def init_omode_occu(self):
        """
        Initialize the mixed-state object mode occupancy so each mode has a fixed weight
        """
        # Note: Initially I tried to make it optimizable, but then I noticed the AD algorithm
        # tended to entirely shut off the mode by reducing the omode_occu rather than improving the mode
        # So I decided to keep it as fixed values for now
        
        omode_occu_params = self.init_params.get('obj_omode_init_occu') or {}
        occu_type = omode_occu_params.get('occu_type', 'uniform')
        init_occu = omode_occu_params.get('init_occu')
        vprint(f"### Initializing omode_occu from '{occu_type}' ###", verbose=self.verbose)

        if occu_type   == 'custom':
            omode_occu = np.array(init_occu)
        elif occu_type == 'uniform':
            omode = self.init_params['obj_omode_max']
            omode_occu = np.ones(omode)/omode
        else:
            raise ValueError(f"Initialization method {occu_type} not implemented yet, please use 'custom' or 'uniform'!")
        
        omode_occu = omode_occu.astype('float32')
        vprint(f"omode_occu                            (omode) = {omode_occu.dtype}, {omode_occu.shape}", verbose=self.verbose)
        self.init_variables['omode_occu'] = omode_occu
        vprint(" ", verbose=self.verbose)

    def init_H(self):
        """
        Initialize the near-field Fresnel propagator for multislice ptychography
        """
        
        vprint("### Initializing H (Fresnel propagator) ###", verbose=self.verbose)
        probe_shape = self.init_variables['probe_shape']
        dx = self.init_variables['dx']
        slice_thickness = self.init_variables['slice_thickness']
        lambd = self.init_variables['lambd']
        unit_str = self.init_variables['length_unit']
        
        vprint(f"Calculating H with probe_shape = {probe_shape} px, dx = {dx:.4f} {unit_str}, slice_thickness = {slice_thickness:.4f} {unit_str}, lambd = {lambd:.4f} {unit_str}", verbose=self.verbose)
        
        H = near_field_evolution(probe_shape, dx, slice_thickness, lambd)
        H = H.astype('complex64')
        self.init_variables['H'] = H
        
        vprint(f"H                                    (Ky, Kx) = {H.dtype}, {H.shape}", verbose=self.verbose)
        vprint(" ", verbose=self.verbose)
    
    def init_obj_tilts(self):
        """
        Initialize the object crystal tilts. Tilts can be global tilt (1,2) or pos-dependent tilt (N,2)
        """
        try:
            tilt_source     = self.init_params['tilt_source']
            tilt_params     = self.init_params['tilt_params']
        except KeyError as e:
            raise KeyError(f"Missing required configuration field: {e}")

        vprint(f"### Initializing obj tilts from = '{tilt_source}' ###", verbose=self.verbose)
        
        if tilt_source == 'custom':
            obj_tilts = tilt_params # (1,2) or (N,2) array in unit of mrad

        elif tilt_source == 'file':
            # Infer file type from extension
            file_path = tilt_params.get('path')
            key = tilt_params.get('key')
            _, ext = os.path.splitext(file_path)
            ext = ext.lower()
            vprint(f"Detected tilt file type = '{ext}'")
            
            # Warning when there's no key specified
            if ext in ('.mat', '.h5', '.hdf5') and key is None:
                vprint(f"WARNING: Couldn't find the 'key' in 'tilt_params' with file type = '{ext}'.")
                vprint("It is strongly recommended to provide an explicit key to better find the desired dataset, which is often much faster as well.")
                vprint("PtyRAD will still try to find the dataset, but you may consider setting 'key': <DATASET_KEY> inside your 'tilt_params' dict.")
            
            if ext == '.raw':
                raise ValueError("PtyRAD doesn't support loading object tilt from .raw file yet, please use other tilt_source.")
            obj_tilts = np.float32(load_array_from_file(**tilt_params,  ndims=[2]))
            vprint(f"Initialized obj_tilts with loaded obj_tilts from file, mean obj_tilts = {obj_tilts.mean(0).round(2)} (theta_y, theta_x) mrad", verbose=self.verbose)
            
        elif tilt_source == 'PtyRAD':
            pt_path = tilt_params
            ckpt = self.cache_contents if pt_path == self.cache_path else load_ptyrad(pt_path)            
            obj_tilts = np.float32(ckpt['optimizable_tensors']['obj_tilts'])
            vprint(f"Initialized obj_tilts with loaded obj_tilts from PtyRAD, mean obj_tilts = {obj_tilts.mean(0).round(2)} (theta_y, theta_x) mrad", verbose=self.verbose)

        elif tilt_source == 'simu':
            N_scans    = self.init_variables['N_scans']
            tilt_type  = tilt_params.get('tilt_type') or 'all' # Use the specified tilt_type when specified, fall back to 'all' for unspecified or None
            init_tilts = tilt_params.get('init_tilts') or [[0,0]] # (1,2) array in unit of mrad

            if tilt_type == 'each':
                obj_tilts = np.broadcast_to(np.float32(init_tilts), shape=(N_scans,2))
                vprint(f"Initialized obj_tilts with init_tilts = {init_tilts} (theta_y, theta_x) mrad", verbose=self.verbose)
            elif tilt_type == 'all':
                obj_tilts = np.broadcast_to(np.float32(init_tilts), shape=(1,2))
                vprint(f"Initialized obj_tilts with init_tilts = {init_tilts} (theta_y, theta_x) mrad", verbose=self.verbose)
            else:
                raise ValueError(f"Tilt type {tilt_type} not implemented yet, please use either 'each', or 'all' when initializing obj_tilts with 'simu'!")

        else:
            raise ValueError(f"File type {tilt_source} not implemented yet, please use 'custom', 'PtyRAD', 'file', or 'simu'!")
        
        # Print summary
        self.init_variables['obj_tilts'] = obj_tilts
        vprint(f"obj_tilts                              (N, 2) = {obj_tilts.dtype}, {obj_tilts.shape}", verbose=self.verbose)
        vprint(" ", verbose=self.verbose)
    
    def init_check(self):
        # Although some of the input experimental parameters might not be used directly by the package
        # I think it's a good practice to check for overall consistency and remind the user to check carefully
        # While these check could be performed within the init methods and achieve early return
        # It's more readable to separate the initializaiton logic with the checking logic in this way
        
        vprint("### Checking consistency between input params with the initialized variables ###", verbose=self.verbose)
        
        # Check the consistency of init params with the initialized variables
        init_params  = self.init_params
        Npix        = init_params['meas_Npix']
        Nlayer      = init_params['obj_Nlayer']
        N_scans     = init_params['pos_N_scans']
        N_scan_slow = init_params['pos_N_scan_slow']
        N_scan_fast = init_params['pos_N_scan_fast']
        
        # Initialized variables
        meas             = self.init_variables['measurements']
        probe            = self.init_variables['probe']
        crop_pos         = self.init_variables['crop_pos']
        probe_pos_shifts = self.init_variables['probe_pos_shifts']
        obj              = self.init_variables['obj']
        omode_occu       = self.init_variables['omode_occu'] 
        H                = self.init_variables['H']
        obj_tilts        = self.init_variables['obj_tilts']
        if self.init_variables.get('on_the_fly_meas_padded') is not None:
            target_Npix  = self.init_variables['on_the_fly_meas_padded'].shape[-1]
        else:
            target_Npix  = meas.shape[-1]
        if self.init_variables.get('on_the_fly_meas_scale_factors') is not None:
            scale_factors = self.init_variables['on_the_fly_meas_scale_factors']
        else:
            scale_factors = [1,1]   
        
        # TODO These checks should probably be refactored a bit with clearer message
        # We could duplicate some of them at the specific section to catch them early
        
        # Check DP shape
        if Npix == meas.shape[-2] == meas.shape[-1] == probe.shape[-2] == probe.shape[-1] == H.shape[-2] == H.shape[-1]:
            vprint(f"Npix, DP measurements, probe, and H shapes are consistent as '{Npix}'", verbose=self.verbose)
        elif Npix == target_Npix == probe.shape[-2] == probe.shape[-1] == H.shape[-2] == H.shape[-1]:
            vprint(f"Npix, DP measurements, probe, and H shapes will be consistent as '{Npix}' during on-the-fly measurement padding", verbose=self.verbose)
        elif Npix == floor(meas.shape[-2]*scale_factors[-2]) == floor(meas.shape[-1]*scale_factors[-1]) == probe.shape[-2] == probe.shape[-1] == H.shape[-2] == H.shape[-1]:
            vprint(f"Npix, DP measurements, probe, and H shapes will be consistent as '{Npix}' during on-the-fly measurement resampling", verbose=self.verbose)
        elif Npix == floor(target_Npix*scale_factors[-2]) == floor(target_Npix*scale_factors[-1]) == probe.shape[-2] == probe.shape[-1] == H.shape[-2] == H.shape[-1]:
            vprint(f"Npix, DP measurements, probe, and H shapes will be consistent as '{Npix}' during on-the-fly measurement padding and then resampling", verbose=self.verbose)
        else:
            raise ValueError(f"Found inconsistency between Npix({Npix}), DP measurements({meas.shape[-2:]}), probe({probe.shape[-2:]}), and H({H.shape[-2:]}) shape")

        # Check scan pattern
        if N_scans == len(meas) == N_scan_slow*N_scan_fast == len(crop_pos) == len(probe_pos_shifts):
            vprint(f"N_scans, len(meas), N_scan_slow*N_scan_fast, len(crop_pos), and len(probe_pos_shifts) are consistent as '{N_scans}'", verbose=self.verbose)
        else:
            raise ValueError(f"Found inconstency between N_scans({N_scans}), len(meas)({len(meas)}), N_scan_slow({N_scan_slow})*N_scan_fast({N_scan_fast}), len(crop_pos)({len(crop_pos)}), and len(probe_pos_shifts)({len(probe_pos_shifts)})")
        
        # Check object shape
        if obj.shape[0] == len(omode_occu):
            vprint(f"obj.shape[0] is consistent with len(omode_occu) as '{obj.shape[0]}'", verbose=self.verbose)
        else:
            raise ValueError(f"Found inconsistency between obj.shape[0]({obj.shape[0]}) and len(omode_occu)({len(omode_occu)})")
        
        if obj.shape[1] == Nlayer:        
            vprint(f"obj.shape[1] is consistent with Nlayer as '{Nlayer}'", verbose=self.verbose)
        else:
            raise ValueError(f"Found inconsistency between obj.shape[1]({obj.shape[1]}) and Nlayer({Nlayer})")

        # Check object extent and probe positions
        if (crop_pos.min(0) < 0).any():
            raise ValueError(f"Found invalid crop position. crop_pos.min(0) {crop_pos.min(0)} must be equal or larger than 0. Please check your position and object initialization.")
        
        if (crop_pos.max(0) + Npix - obj.shape[-2:] > 0).any():
            raise ValueError(f"Found invalid crop position. crop_pos.max(0) {crop_pos.max(0)} + Npix ({Npix}) = {crop_pos.max(0) + Npix} must be equal or smaller than object canvas lateral size (Ny, Nx) = {obj.shape[-2:]}. Please check your position and object initialization.")
        vprint(f"crop positions (yx_min={crop_pos.min(0)}, yx_max={crop_pos.max(0)+Npix}) are well contained inside object canvas (Ny,Nx) = {obj.shape[-2:]}.", verbose=self.verbose)
        
        # Check obj tilts
        if len(obj_tilts) in [1, N_scans]:
            vprint("obj_tilts is consistent with either 1 or N_scans", verbose=self.verbose)
        else:
            raise ValueError(f"Found inconsistency between len(obj_tilts) ({len(obj_tilts)}), 1, and N_scans({N_scans})")
        
        vprint("Pass the consistency check of initialized variables, initialization is done!", verbose=self.verbose)
    
    def init_all(self):
        # Run this method to initialize all
        
        self.init_cache()
        self.init_measurements()
        self.init_calibration()
        self.set_variables_dict()
        self.init_probe()
        self.init_pos()
        self.init_obj()
        self.init_omode_occu()
        self.init_H()
        self.init_obj_tilts()
        self.init_check()
        
        return self
    
    ###### Private methods for setting the cache ######

    def _set_use_cached_flags(self, source):
        """ Set the flags for each field whether we can cache or not """
        # Validate required fields
        try:
            obj_source   = self.init_params['obj_source']
            obj_params   = self.init_params['obj_params']
            probe_source = self.init_params['probe_source']
            probe_params = self.init_params['probe_params']
            pos_source   = self.init_params['pos_source']
            pos_params   = self.init_params['pos_params']
        except KeyError as e:
            raise KeyError(f"Missing required configuration field: {e}")
        
        triplets = [
        ('obj', obj_source, obj_params),
        ('probe', probe_source, probe_params),
        ('pos', pos_source, pos_params)]
        
        # Helper for comparison
        def same_source_and_params(a, b):
            return a[1] == b[1] == source and a[2] == b[2]
        
        # Check if obj, probe, and pos sources are the same
        if same_source_and_params(triplets[0], triplets[1]) and same_source_and_params(triplets[1], triplets[2]):
            self.use_cached_obj = self.use_cached_probe = self.use_cached_pos = True
            self.cache_path = obj_params
            self.cache_source = obj_source
            return

        if same_source_and_params(triplets[0], triplets[1]):
            self.use_cached_obj = self.use_cached_probe = True
            self.cache_path = obj_params
            self.cache_source = obj_source
            return

        if same_source_and_params(triplets[0], triplets[2]):
            self.use_cached_obj = self.use_cached_pos = True
            self.cache_path = obj_params
            self.cache_source = obj_source
            return

        if same_source_and_params(triplets[1], triplets[2]):
            self.use_cached_probe = self.use_cached_pos = True
            self.cache_path = probe_params
            self.cache_source = probe_source
            return

    ###### Private methods for initializing measurements ######
    
    def _load_meas(self):
        """Load diffraction data from file or memory according to init_params['meas']."""
        
        # Validate required fields
        try:
            meas_source = self.init_params['meas_source']
            meas_params = self.init_params['meas_params']
        except KeyError as e:
            raise KeyError(f"Missing required configuration field: {e}")
        
        # Check for 'path' key for all sources
        if meas_source != 'custom' and 'path' not in meas_params:
            raise KeyError(f"'path' is required in 'meas_params' for source '{meas_source}'. Set 'path': <PATH_TO_YOUR_DATASET> inside your 'meas_params' dict.")

        vprint(f"Loading measurements from source = '{meas_source}'", verbose=self.verbose)

        if meas_source == 'custom':
            if not isinstance(meas_params, np.ndarray): # assume to be a numpy array
                raise TypeError(f"'custom' source requires 'meas_params' to be a NumPy array. Got {type(meas_params)}.")
            meas = meas_params
            
        elif meas_source in ['file', 'tif', 'tiff', 'mat', 'h5', 'hdf5', 'npy', 'raw']: # Keep the file types for backward compatibility
            # Infer file type from extension
            file_path = meas_params.get('path')
            key = meas_params.get('key')
            _, ext = os.path.splitext(file_path)
            ext = ext.lower()
            vprint(f"Detected measurement file type = '{ext}'")
            
            # Warning when there's no key specified
            if ext in ('.mat', '.h5', '.hdf5') and key is None:
                vprint(f"WARNING: Couldn't find the 'key' in 'meas_params' with file type = '{ext}'.")
                vprint("It is strongly recommended to provide an explicit key to better find the desired dataset, which is often much faster as well.")
                vprint("PtyRAD will still try to find the dataset, but you may consider setting 'key': <DATASET_KEY> inside your 'meas_params' dict.")
            
            # Provide default shape for .raw files if it's not specified
            if ext == '.raw' and meas_params.get('shape') is None:
                vprint(f"WARNING: Couldn't find the 'shape' in 'meas_params' with file type = '{ext}'.")
                vprint("It is strongly recommended to provide an explicit shape to better load from .raw files")
                vprint("PtyRAD will still try to load the dataset based on the provided 'init_params', but you may consider setting 'shape': (N_scans, Npix, Npix) inside your 'meas_params' dict.")
                meas_params['shape'] = (self.init_params['pos_N_scans'],
                                        self.init_params['meas_Npix'],
                                        self.init_params['meas_Npix'])
            meas = load_array_from_file(**meas_params)

        else:
            raise ValueError(f"Unsupported measurement source '{meas_source}'. Use 'custom' or 'file'.")

        vprint(f"Original measurements dtype is {meas.dtype}, casting to float32 (single precision) for computational efficiency.")
        meas = meas.astype('float32', copy=False)
        
        vprint(f"Imported meausrements shape / dtype = {meas.shape}, dtype = {meas.dtype}", verbose=self.verbose)
        vprint(f"Imported meausrements int. statistics (min, mean, max) = ({meas.min():.4f}, {meas.mean():.4f}, {meas.max():.4f})", verbose=self.verbose)
        return meas
    
    def _process_meas(self, meas):
        """
        Applies all processing steps to raw loaded measurements.
        
        """
        
        # If the processing config is None, the methods will skip it internally
        # Note that _meas_remove_neg_values and _meas_normalization will always be executed
        # If you really want to nullify them, explictly set 
        # 'meas_remove_neg_values': {'mode': 'subtract_value', 'value': 0}
        # 'meas_normalization': {'mode': 'divide_const', 'value': 1}
        
        # Simple geometric operations
        meas = self._meas_permute(meas, self.init_params.get('meas_permute'))
        meas = self._meas_reshape(meas, self.init_params.get('meas_reshape'))
        meas = self._meas_flipT(meas, self.init_params.get('meas_flipT'))
        self.init_variables['meas_raw_avg'] = meas.mean(0) # Save this for initial dx calibration. The crop/pad/resample effect would be accounted accordingly in `init_calibration`
        
        # Shape check after flipT (`meas` corresponds to the freshly loaded dataset before anything that could change its shape)
        N_scans = self.init_params_original['pos_N_scans']
        Npix = self.init_params_original['meas_Npix']
        if meas.ndim != 3 or meas.shape[0] != N_scans or meas.shape[1:] != (Npix, Npix):
            raise ValueError(
                f"Shape mismatch after loading and processing the measurements: expected measurements shape = (N_scans={N_scans}, Npix={Npix}, Npix={Npix}), "
                f"but got {meas.shape}. PtyRAD allows you to directly preprocess your loaded measurements with `meas_permute` and `meas_reshape` specified in params files to make it (N_scans, Npix(ky), Npix(kx)). "
                f"Please read the comments in demo YAML params files or the documentation for more information about how to set `meas_permute` and `meas_reshape`."
            )
        
        # Operations that may change the shape of the measurements
        meas = self._meas_crop(meas, self.init_params.get('meas_crop'))
        meas = self._meas_remove_neg_values(meas, self.init_params.get('meas_remove_neg_values')) # meas need to be positive before the padding with background fitting mode
        meas = self._meas_normalization(meas, self.init_params.get('meas_normalization')) # The normalization is needed because the background is calculated now and it needs to match the level of the final normalized meas
        meas = self._meas_pad(meas, self.init_params.get('meas_pad'))
        meas = self._meas_resample(meas, self.init_params.get('meas_resample'))
        
        # Operations that add realistic factors to (simulated perfect) measurements
        meas = self._meas_add_source_size(meas, self.init_params.get('meas_add_source_size'))
        meas = self._meas_add_detector_blur(meas, self.init_params.get('meas_add_detector_blur'))
        meas = self._meas_add_poisson_noise(meas, self.init_params.get('meas_add_poisson_noise'))

        # Final guard on negative values
        meas = self._meas_remove_neg_values(meas, {'mode': 'clip_neg'})
        
        # Final guard on meas dtype
        meas = meas.astype('float32', copy=False) # Skip if dtype = 'float32', otherwise astype will make a copy
        
        return meas
    
    def _meas_permute(self, meas, order):
        if order is not None:
            vprint(f"Permuting measurements with order = {order}", verbose=self.verbose)
            return meas.transpose(order)
        return meas
    
    def _meas_reshape(self, meas, target_shape):
        if target_shape is not None:
            vprint(f"Reshaping measurements to shape = {target_shape}", verbose=self.verbose)
            return meas.reshape(target_shape)
        return meas

    def _meas_flipT(self, meas, flipT_axes):
        """
        Flip and transpose measurement array.
        flipT_axes: list of 3 binary/int values [flipud, fliplr, transpose]
        """
        if flipT_axes is None:
            return meas

        # Validate length
        if not isinstance(flipT_axes, (list, tuple)) or len(flipT_axes) != 3:
            raise ValueError(f"Expected flipT_axes to be a list of 3 values, got: {flipT_axes}")

        # Safely cast all entries to int
        try:
            flipT_axes = [int(v) for v in flipT_axes]
        except Exception as e:
            raise ValueError(f"flipT_axes must contain values convertible to int (0 or 1). Got: {flipT_axes}") from e

        vprint(f"Flipping measurements with [flipud, fliplr, transpose] = {flipT_axes}", verbose=self.verbose)

        if flipT_axes[0]:
            meas = np.flip(meas, axis=1)
        if flipT_axes[1]:
            meas = np.flip(meas, axis=2)
        if flipT_axes[2]:
            meas = np.transpose(meas, (0, 2, 1))

        return meas

    def _meas_crop(self, meas, crop_ranges):
        """
        Crop measurements across 4 dimensions:
        [[slow_i, slow_f], [fast_i, fast_f], [ky_i, ky_f], [kx_i, kx_f]]
        Allows any entry to be `None` to skip cropping that axis.
        Note that this method would also update the `self.init_params`
        """
        if crop_ranges is None:
            return meas

        if len(crop_ranges) != 4:
            raise ValueError(f"Expected 4 crop ranges [N_slow, N_fast, ky, kx], got {crop_ranges}")

        # Reshape (N, ky, kx) -> (N_slow, N_fast, ky, kx)
        Nslow, Nfast = self.init_params['pos_N_scan_slow'], self.init_params['pos_N_scan_fast']
        meas = meas.reshape(Nslow, Nfast, *meas.shape[-2:])
        vprint(f"Reshaping measurements into {meas.shape} for cropping", verbose=self.verbose)

        axes_names = ['N_slow', 'N_fast', 'ky', 'kx']
        slices = []

        for i, bounds in enumerate(crop_ranges):
            if bounds is None:
                slices.append(slice(None))
            else:
                try:
                    start, stop = bounds
                    slices.append(slice(start, stop))
                    vprint(f"Cropping axis {axes_names[i]} from {start} to {stop}", verbose=self.verbose)
                except Exception as e:
                    raise ValueError(f"Invalid crop bounds for axis {axes_names[i]}: {bounds}") from e

        meas = meas[slices[0], slices[1], slices[2], slices[3]]
        vprint(f"Cropped measurements have shape (N_slow, N_fast, ky, kx) = {meas.shape}", verbose=self.verbose)

        # Update self.init_params
        vprint("Update (Npix, N_scans, N_scan_slow, N_scan_fast) after the measurements cropping", verbose=self.verbose)
        self.init_params['meas_Npix'] = meas.shape[-1]
        self.init_params['pos_N_scans'] = meas.shape[0] * meas.shape[1]
        self.init_params['pos_N_scan_slow'] = meas.shape[0]
        self.init_params['pos_N_scan_fast'] = meas.shape[1]
        meas = meas.reshape(-1, meas.shape[-2], meas.shape[-1])
        vprint(f"Reshape measurements back to (N, ky, kx) = {meas.shape}", verbose=self.verbose)

        return meas  
    
    def _meas_remove_neg_values(self, meas, neg_cfg):
        """
        Correct negative values in the measurement array based on the specified configuration.

        Args:
            meas (numpy.ndarray): The measurement array to process.
            neg_cfg (dict): Configuration for handling negative values. Expected keys:
                - mode (str): Method to handle negative values. Options are 'clip_neg', 'subtract_min',
                'clip_value', or 'subtract_value'. Default is 'clip_neg'.
                - value (float or None): Value used for 'clip_value' or 'subtract_value' modes. Default is None.

        Returns:
            numpy.ndarray: The processed measurement array with negative values handled.
        """

        # This correction is enforced even the neg_cfg is None (not provided by user)
        if neg_cfg is None:
            neg_cfg = {}

        # Extract configuration with defaults
        mode = neg_cfg.get('mode', 'clip_neg')  # Default to 'clip_neg'
        value = neg_cfg.get('value', None)  # Default to None
        force = neg_cfg.get('force', False) # Default to False so it'll skip if there's no negative values

        # Check if there are negative values
        if not (meas < 0).any():
            if not force:
                vprint("No negative values found in measurements. Skipping non-neg correction.", verbose=self.verbose)
                return meas
            else:
                vprint(f"No negative values found in measurements, but force = '{force}' so continuing measurement negative value correction", verbose=self.verbose)

        vprint(f"Removing negative values in measurement with method = {mode} and value = {value}", verbose=self.verbose)

        if mode == 'subtract_min':
            min_value = meas.min()
            meas -= min_value
            value = None  # Not relevant for this mode
            vprint(f"Minimum value of {min_value:.4f} subtracted due to the positive px value constraint of measurements", verbose=self.verbose)

        elif mode == 'clip_value':
            if value is None:
                raise KeyError("Mode 'clip_value' requires a non-None 'value'.")
            vprint(f"Minimum value = {meas.min():.4f}, measurements below {value} are clipped to 0 due to the positive px value constraint of measurements", verbose=self.verbose)
            meas[meas < value] = 0

        elif mode == 'subtract_value':
            if value is None:
                raise KeyError("Mode 'subtract_value' requires a non-None 'value'.")
            vprint(f"Minimum value = {meas.min():.4f}, measurements subtracted by {value} due to the positive px value constraint of measurements", verbose=self.verbose)
            meas -= value

        elif mode == 'clip_neg': # Default mode
            vprint(f"Minimum value = {meas.min():.4f}, negative values are clipped to 0 due to the positive px value constraint of measurements", verbose=self.verbose)
            meas[meas < 0] = 0
            value = None  # Not relevant for clipping

        else:
            raise ValueError(f"Unsupported mode '{mode}' for handling negative values. Use 'clip_neg', 'subtract_min', 'clip_value', or 'subtract_value'.")

        # Final check in case the user specified value is not enough to remove all neg values
        if (meas < 0).any():
            vprint(f"User specified value = {value} is not enough to remove negative values, applying 0 clipping")
            vprint(f"Minimum value of {meas.min():.4f} is clipped to 0 due to the positive px value constraint of measurements", verbose=self.verbose)
            meas[meas<0] = 0

        return meas
    
    def _meas_normalization(self, meas, norm_cfg):
        """
        Normalize the measurement array based on the specified normalization mode.

        Args:
            meas (numpy.ndarray): The measurement array to normalize, shape (N, ky, kx).
            Returns:
            numpy.ndarray: The normalized measurement array.
        """
        
        # This correction is enforced even the norm_cfg is None (not provided by user)
        if norm_cfg is None:
            norm_cfg = {}
        
        norm_mode = norm_cfg.get('mode', 'max_at_one')  # Default to 'max_at_one'
        norm_const = norm_cfg.get('value', None)  # Used for 'divide_const' mode

        vprint(f"Normalizing measurements with mode = '{norm_mode}' and value = '{norm_const}'", verbose=self.verbose)

        if norm_mode == 'max_at_one':
            normalization_const = np.mean(meas, axis=0, dtype=np.float32).max()
            vprint(f"Normalizing by max of the 2D mean pattern intensity: {normalization_const:.8g}", verbose=self.verbose)

        elif norm_mode == 'mean_at_one':
            normalization_const = np.mean(meas, axis=0, dtype=np.float32).mean()
            vprint(f"Normalizing by mean of the 2D mean pattern intensity: {normalization_const:.8g}", verbose=self.verbose)

        elif norm_mode == 'sum_to_one':
            normalization_const = np.mean(meas, axis=0, dtype=np.float32).sum()
            vprint(f"Normalizing by sum of the 2D mean pattern intensity: {normalization_const:.8g}", verbose=self.verbose)

        elif norm_mode == 'divide_const':
            if norm_const is None:
                raise KeyError("Mode 'divide_const' requires a non-None 'norm_const'.")
            normalization_const = norm_const
            vprint(f"Normalizing by user-defined constant: {normalization_const:.8g}", verbose=self.verbose)

        else:
            raise ValueError(f"Unsupported normalization mode '{norm_mode}'. Use 'max_at_one', 'mean_at_one', 'sum_to_one', or 'divide_const'.")

        # Normalize the measurements
        meas /= normalization_const 
        meas = meas.astype('float32', copy=False) # Skip if dtype = 'float32', otherwise astype will make a copy
        vprint(f"meausrements shape / dtype = {meas.shape}, dtype = {meas.dtype}", verbose=self.verbose)
        vprint(f"meausrements int. statistics (min, mean, max) = ({meas.min():.4f}, {meas.mean():.4f}, {meas.max():.4f})", verbose=self.verbose)

        return meas
    
    def _meas_pad(self, meas, pad_cfg):
        """
        _meas_pad Padd the 3D measurements array to a target size using the specified padding mode and type.

        Args:
            meas (numpy.ndarray): The measurement array to normalize, shape (N, ky, kx).
            pad_cfg (dict): A dictionary containing the padding configuration. Expected keys:
            pad_cfg = {'mode': 'on_the_fly', 'padding_type': 'power', 'target_Npix': 256, 'value': 0}

        Raises:
            KeyError: _description_
            KeyError: _description_

        Returns:
           numpy.ndarray: The padded measurement array.
        """
        
        if pad_cfg is None or pad_cfg.get('mode') is None:
            self.init_variables['on_the_fly_meas_padded'] = None
            self.init_variables['on_the_fly_meas_padded_idx'] = None
            return meas

        mode = pad_cfg['mode']  # 'precompute' or 'on_the_fly'. Use `on_the_fly` to save GPU memory
        padding_type = pad_cfg['padding_type']
        target_Npix = pad_cfg['target_Npix']
        value = pad_cfg.get('value', 10) # For constant and linear_ramp padding
        threshold = pad_cfg.get('threshold', 70) # For exp and power padding that requires fitting a thresholded mask

        vprint(f"Padding measurements with mode='{mode}', padding_type='{padding_type}', target_Npix={target_Npix}", verbose=self.verbose)

        # Get amplitude from average DP
        meas_avg = meas.mean(axis=0)
        meas_int_sum = meas_avg.sum()
        amp_avg = np.sqrt(meas_avg)
        H, W = amp_avg.shape
        
        # Calculate padding for each dimension
        pad_y = max(0, target_Npix - H)
        pad_x = max(0, target_Npix - W)
        pad_y1, pad_y2 = pad_y // 2, pad_y - pad_y // 2
        pad_x1, pad_x2 = pad_x // 2, pad_x - pad_x // 2
        pad_h1, pad_h2 = pad_y1, pad_y1 + H
        pad_w1, pad_w2 = pad_x1, pad_x1 + W

        # Create coordinate grid for radial background fitting
        y, x = np.ogrid[:target_Npix, :target_Npix]
        center = (H // 2 + pad_y1, W // 2 + pad_x1)
        r = np.sqrt((y - center[0])**2 + (x - center[1])**2) + 1e-10 # so r is never 0

        # Compute background
        if padding_type == 'constant':
            amp_padded = np.pad(amp_avg, ((pad_y1, pad_y2), (pad_x1, pad_x2)), mode='constant', constant_values=value)
        elif padding_type == 'edge':
            amp_padded = np.pad(amp_avg, ((pad_y1, pad_y2), (pad_x1, pad_x2)), mode='edge')
        elif padding_type == 'linear_ramp':
            amp_padded = np.pad(amp_avg, ((pad_y1, pad_y2), (pad_x1, pad_x2)), mode='linear_ramp', end_values=value)
        elif padding_type == 'exp':
            mask = create_one_hot_mask(amp_avg, percentile=threshold) # It feels like we probably don't need to normalize meas before padding because the mask is calculated by percentile
            popt = fit_background(amp_avg, mask, fit_type='exp')
            amp_padded = exponential_decay(r, *popt)
        elif padding_type == 'power':
            mask = create_one_hot_mask(amp_avg, percentile=threshold)
            popt = fit_background(amp_avg, mask, fit_type='power')
            amp_padded = power_law(r, *popt)
        else:
            raise ValueError(f"Unsupported padding_type = '{padding_type}'")
        
        # Square the padded amplitude back to intensity
        meas_padded = np.square(amp_padded)[None,] # (1, ky, kx)
        meas_padded[..., pad_h1:pad_h2, pad_w1:pad_w2] = 0
        padded_int_sum = meas_padded.sum()
        vprint(f"Original meas int sum = {meas_int_sum:.4f}, padded region int sum = {padded_int_sum:.4f}, or {padded_int_sum/meas_int_sum:.2%} more intensity after padding.", verbose=self.verbose) 
        vprint("This percentage should be ideally less than 5%, or you should set a lower threshold to exclude more central region.", verbose=self.verbose)

        if mode == 'precompute':
            canvas = np.zeros((meas.shape[0], *meas_padded.shape[1:]))
            canvas += meas_padded
            canvas[..., pad_h1:pad_h2, pad_w1:pad_w2] = meas
            meas = canvas
            self.init_variables['on_the_fly_meas_padded'] = None
            self.init_variables['on_the_fly_meas_padded_idx'] = None
        elif mode == 'on_the_fly':
            # For on_the_fly padding, we pass the padded 2D pattern (extra background) and padding indices to the model
            self.init_variables['on_the_fly_meas_padded'] = meas_padded
            self.init_variables['on_the_fly_meas_padded_idx'] = [pad_h1, pad_h2, pad_w1, pad_w2]
        else:
            raise ValueError(f"meas_pad does not support mode = '{mode}', please choose from 'on_the_fly', 'precompute', or null")

        # Update iself.init_params similar to _meas_crop
        vprint("Update Npix after the measurements padding", verbose=self.verbose)
        self.init_params['meas_Npix'] = meas_padded.shape[-1] # This will update Npix to target_Npix no matter what mode is used

        return meas

    def _meas_resample(self, meas, resample_cfg):
        """
        _meas_resample Resample measurements along the ky, kx dimension

        """
        if resample_cfg is None or resample_cfg.get('mode') is None:
            self.init_variables['on_the_fly_meas_scale_factors'] = None
            return meas

        # Validate required fields
        try:
            mode = resample_cfg['mode']
            Npix = self.init_params['meas_Npix']
            scale_factors = resample_cfg['scale_factors']
        except KeyError as e:
            raise KeyError(f"Missing required configuration field: {e}")

        # Ensure scale_factors is a list or tuple of length 2
        if len(scale_factors) != 2:
            raise ValueError("scale_factors for resample must be a list or tuple of two elements.")

        if scale_factors[0] != scale_factors[1]:
            min_scale = min(scale_factors)
            vprint(f"Non-uniform scale_factors {scale_factors} detected. Using uniform scale factor: {min_scale}")
            scale_factors = [min_scale, min_scale]
        
        # If on-the-fly padding is set, force resample to be on-the-fly as well
        if self.init_variables.get('on_the_fly_meas_padded', None) is not None:
            mode = 'on_the_fly'
            vprint("'meas_resample' is set to 'on_the_fly' mode because 'meas_pad' is also set to 'on_the_fly' mode", verbose=self.verbose)

        vprint(f"Resampling measurements with mode = '{mode}', scale_factors = {scale_factors}", verbose=self.verbose)

        if mode == 'precompute':
            zoom_factors = np.array([1.0, *scale_factors]) # scipy.ndimage.zoom applies to all axes.
            meas = zoom(meas, zoom_factors, order=1) # bilinear (order=1) could prevent overshooting. Resampling would change the meas.sum(), but we have normalization at the end of the process.
            Npix = meas.shape[-1] # Update Npix
            self.init_variables['on_the_fly_meas_scale_factors'] = None

        elif mode == 'on_the_fly':
            # Don't change `meas`, just update Npix
            Npix = floor(Npix * scale_factors[-1]) # To match the rounding logic with torch.nn.functional.interpolate()
            self.init_variables['on_the_fly_meas_scale_factors'] = scale_factors

        else:
            raise ValueError(f"meas_resample does not support mode = '{mode}', please choose from 'on_the_fly', 'precompute', or null")

        # Update self.init_params similar to _meas_crop
        self.init_params['meas_Npix'] = Npix
        vprint(f"Update Npix into '{Npix}' after the measurements resampling", verbose=self.verbose)
        vprint(f"Resampled measurements have shape (N_scans, ky, kx) = {meas.shape}", verbose=self.verbose)

        return meas

    def _meas_add_source_size(self, meas, source_size_std_ang):
        if source_size_std_ang is None or source_size_std_ang == 0:
            return meas

        Nslow, Nfast = self.init_params['pos_N_scan_slow'], self.init_params['pos_N_scan_fast']
        meas = meas.reshape(Nslow, Nfast, *meas.shape[-2:])
        vprint(f"Reshaping measurements into {meas.shape} for adding partial spatial coherence (source size) induced blurring on measurements", verbose=self.verbose)

        # Convert real-space blur in Angstroms to Gaussian std in scan units (px)
        source_size_std_px = source_size_std_ang / self.init_params['pos_scan_step_size']
        vprint(f"Adding source size (partial spatial coherence) of Gaussian blur std = {source_size_std_px:.4f} scan_step sizes or {source_size_std_ang:.4f} Ang to measurements along the scan directions", verbose=self.verbose)

        # Apply blur over scan dimensions (0,1)
        meas = gaussian_filter(meas, sigma=source_size_std_px, axes=(0,1)) # Partial spatial coherence is approximated by mixing DPs at nearby probe positions
        meas = meas.reshape(-1, meas.shape[-2], meas.shape[-1])
        vprint(f"Reshape measurements back to (N, ky, kx) = {meas.shape}", verbose=self.verbose)
        
        return meas

    def _meas_add_detector_blur(self, meas, detector_blur_std_px):
        """
        Add detector blur (point-spread function of the detector)

        """
        if detector_blur_std_px is None or detector_blur_std_px == 0:
            return meas
        
        meas = gaussian_filter(meas, sigma=detector_blur_std_px, axes=(-2,-1)) # Detector blur is essentially the Gaussian blur along ky, kx
        vprint(f"Adding detector blur (point-spread function of the detector) of Gaussian blur std = {detector_blur_std_px:.4f} px to measurements along the ky, kx directions", verbose=self.verbose)
        
        return meas
    
    def _meas_add_poisson_noise(self, meas, poisson_cfg):
        if poisson_cfg is None:
            return meas

        # Validate required fields
        try:
            unit = poisson_cfg['unit']
            value = poisson_cfg['value']
            scan_step_size = self.init_params['pos_scan_step_size']
        except KeyError as e:
            raise KeyError(f"Missing required configuration field: {e}")

        # Check negative values before applying Poisson noise
        eps = meas.min() / np.abs(meas.mean() + 1e-12)
        if meas.min() < 0:
            vprint(f"Found negative values in meas, meas.min() = {meas.min():.4g}.", verbose=self.verbose)
            if eps > -1e-5:
                vprint(f"Negative values ({meas[meas < 0].mean():.4g}) are within relative numerical tolerance (min/mean) 1e-5 , clipping negative values to 0.", verbose=self.verbose)
                meas[meas < 0] = 0
            else:
                raise ValueError(f"meas needs to be positive before applying poisson noise, got meas.min = {meas.min():.4g}. Check your 'meas_remove_neg_values'.")
            
        # Convert units to total electrons per pattern
        if unit == 'total_e_per_pattern':
            total_electron = value
            dose = total_electron / scan_step_size**2
        elif unit == 'e_per_Ang2':
            dose = value
            total_electron = dose * scan_step_size**2
        else:
            raise ValueError(f"Unsupported unit for Poisson noise: '{unit}'. Use 'total_e_per_pattern' or 'e_per_Ang2'.")

        vprint(f"total electron per measurement = dose x scan_step_size^2 = {dose:.3f}(e-/Ang^2) x {scan_step_size:.3f}(Ang)^2 = {total_electron:.3f}", verbose=self.verbose)

        # Normalize meas to sum to ~ 1 before applying Poisson noise
        vprint(f"Before applying Poisson noise: meausrements int. statistics (min, mean, max) = ({meas.min():.4f}, {meas.mean():.5f}, {meas.max():.4f})", verbose=self.verbose)
        
        normalization_const = meas.sum() / meas.shape[0]
        vprint(f"Normalization constant = {normalization_const:.4f}, this makes each measurement sum to ~ 1.", verbose=self.verbose)
        
        meas /= normalization_const # Make each slice of the meas to sum to ~ 1. A global normalization constant keeps the relative intensity.
        vprint(f"After applying normalization: meausrements int. statistics (min, mean, max) = ({meas.min():.4f}, {meas.mean():.5f}, {meas.max():.4f})", verbose=self.verbose)
        vprint(f"Mean total electron per pattern = meas.sum((-2,-1)).mean(0) = ({meas.sum((-2,-1)).mean(0):.5f})", verbose=self.verbose)

        set_random_seed(seed=self.random_seed)
        meas = np.random.poisson(meas * total_electron) # poisson returns int32
        vprint(f"Adding Poisson noise with a total electron per diffraction pattern of {int(total_electron)}", verbose=self.verbose)
        vprint(f"After applying Poisson noise: meausrements int. statistics (min, mean, max) = ({meas.min():.4f}, {meas.mean():.5f}, {meas.max():.4f})", verbose=self.verbose)

        meas = (meas * normalization_const / total_electron).astype('float32', copy=False) # Un-normalize meas back to the original scale
        vprint(f"After un-normalizing back to original scale: meausrements int. statistics (min, mean, max) = ({meas.min():.4f}, {meas.mean():.5f}, {meas.max():.4f})", verbose=self.verbose)
        
        return meas

    def _export_meas(self, export_params={}):
        meas = self.init_variables['measurements']
        file_dir = export_params.get("file_dir")
        
        # Handle the case where file_dir is None
        if file_dir in (None, ''):
            meas_path = get_nested(self.init_params, key=['meas_params', 'path'], safe=True, default='')
            export_params["file_dir"] = os.path.dirname(meas_path)
            
        # Ensure the directory exists if it's not empty
        if file_dir and not os.path.exists(file_dir):
            vprint(f"User specified 'file_dir' = '{file_dir}' doesn't exist, creating the directory now.", verbose=self.verbose)
            os.makedirs(file_dir, exist_ok=True)
            
        save_array(meas, **export_params)
        return
            
    ###### Private methods for initializing probe ######
        
    def _load_probe(self):
        """
        Load the probe from the specified source.
        
        returns:
            probe (numpy.ndarray): The loaded probe array would always be casted to (pmode, Ny, Nx).
        """
        
        # Validate required fields
        try:
            probe_source = self.init_params['probe_source']
            probe_params = self.init_params['probe_params']
        except KeyError as e:
            raise KeyError(f"Missing required configuration field: {e}")
        
        vprint(f"Loading probe from source = '{probe_source}'", verbose=self.verbose)

        if probe_source == 'custom':
            probe = probe_params
        elif probe_source == 'PtyRAD':
            probe = self._load_probe_from_ptyrad(probe_params)
        elif probe_source == 'PtyShv':
            probe = self._load_probe_from_ptyshv(probe_params)
        elif probe_source == 'py4DSTEM':
            probe = self._load_probe_from_py4dstem(probe_params)
        elif probe_source == 'simu':
            probe = self._simulate_probe() # Simulation params are all defined in self.init_params
        else:
            raise ValueError(f"Unsupported probe source '{probe_source}'. Use 'custom', 'PtyRAD', 'PtyShv', 'py4DSTEM', or 'simu'.")

        vprint(f"Loaded probe shape = {probe.shape}, dtype = {probe.dtype}", verbose=self.verbose)
        return probe

    def _load_probe_from_ptyrad(self, params: str):
        pt_path = params
        ckpt = self.cache_contents if self.use_cached_probe else load_ptyrad(pt_path)
        probe = ckpt['optimizable_tensors']['probe']
        return probe
    
    def _load_probe_from_ptyshv(self, params: str):
        mat_path = params
        mat_version = get_matfile_version(mat_path) #https://docs.scipy.org/doc/scipy-1.11.3/reference/generated/scipy.io.matlab.matfile_version.html
        use_h5py = (mat_version[0] == 2)
        probe = self.cache_contents['probe'] if self.use_cached_probe else load_mat(mat_path, key='probe')
        vprint(f"Input PtyShv probe has original shape {probe.shape}, while default PtyShv order is (Ny, Nx, pmode, vp)", verbose=self.verbose)

        # First unify the axes order induced by loading with scipy / h5py, now it should be (Ny, Nx, pmode, vp)
        if use_h5py:
            probe = probe.transpose(range(probe.ndim)[::-1])
            vprint(f"Reverse array axes order of probe to {probe.shape} because use_h5py = {use_h5py}, which automatically reverse the order", verbose=self.verbose)
        else:
            vprint(f"Keep array axes order of probe at {probe.shape} because use_h5py = {use_h5py}", verbose=self.verbose)
        
        # Correct the probe dimension to 3 dimensions, now it should be (Ny, Nx, pmode)
        if probe.ndim == 4:
            vprint("Import only the 1st variable probe mode to make a final probe with (pmode, Ny, Nx)", verbose=self.verbose) # I don't find variable probe modes are particularly useful for electon ptychography
            probe = probe[..., 0]
        elif probe.ndim == 2:
            vprint("Expanding PtyShv probe dimension to make a final probe with (pmode, Ny, Nx)", verbose=self.verbose)
            probe = probe[..., None]
        
        # Final permutation to make it (pmode, Ny, Nx)
        probe = probe.transpose(2,0,1)
        vprint(f"Permute the array axes order of probe to {probe.shape} make it (pmode, Ny, Nx)", verbose=self.verbose)
        
        return probe
    
    def _load_probe_from_py4dstem(self, params: str):
        """
        Load the probe from a py4DSTEM hdf5 file.
        
        Note that the ouput file is expected to be generated by my modified py4DSTEM fork.
        https://github.com/chiahao3/py4DSTEM/tree/benchmark
        """
        hdf5_path = params
        probe = self.cache_contents['probe'] if self.use_cached_probe else load_hdf5(hdf5_path, key='probe')

        vprint(f"Input py4DSTEM probe has original shape {probe.shape}", verbose=self.verbose)

        if probe.ndim == 2:
            vprint("Expanding py4DSTEM probe dimension to make a final probe with (pmode, Ny, Nx)", verbose=self.verbose)
            probe = probe[None, ...]

        return probe

    def _simulate_probe(self):
        """
        Simulate the probe based on the specified parameters.
        """
        init_params      = self.init_params
        probe_illum_type = init_params.get('probe_illum_type', 'electron')
        pmodes           = init_params.get('pmodes', 1)
        pmode_init_pows  = init_params.get('pmode_init_pows', [0.02])
        
        vprint("Using experimental parameters specified by 'init_params' for initial probe simulation.", verbose=self.verbose)

        if probe_illum_type == 'electron':
            probe = make_stem_probe(kv=init_params['probe_kv'], 
                                    conv_angle=init_params['probe_conv_angle'], 
                                    Npix=init_params['meas_Npix'], 
                                    dx=init_params['probe_dx'], # dx = 1/(dk*Npix). Unit in angstrom. This entry is automatically generated inside Initializer.init_calibration().
                                    aberrations=init_params['probe_aberrations'], 
                                    verbose=self.verbose)[None, ...]

        elif probe_illum_type == 'xray':
            probe = make_fzp_probe(beam_kev=init_params['beam_kev'],
                                   Npix=init_params['meas_Npix'],
                                   dx=init_params['probe_dx'],
                                   Ls=init_params['probe_Ls'],
                                   Rn=init_params['probe_Rn'],
                                   dRn=init_params['probe_dRn'],
                                   D_FZP=init_params['probe_D_FZP'],
                                   D_H=init_params['probe_D_H'],
                                   verbose=self.verbose)[None, ...]

        else:
            raise ValueError(f"Unsupported illumination type '{probe_illum_type}'. Use 'electron' or 'xray'.")

        # probe is (1, Ny, Nx) after simulation, expand it to (pmode, Ny, Nx) if needed
        if pmodes > 1:
            probe = make_mixed_probe(probe[0], pmodes=pmodes, pmode_init_pows=pmode_init_pows, verbose=self.verbose)

        return probe

    def _process_probe(self, probe):
        """
        Process the loaded probe, including permutation, setting pmode, and normalization
        """
        # If the processing config is None, the methods will skip it internally
        
        pmode_max = self.init_params.get('probe_pmode_max')
        pmode_init_pows = self.init_params.get('probe_pmode_init_pows')
        
        probe = self._probe_permute(probe, self.init_params.get('probe_permute'))
        probe = self._probe_set_pmode_max(probe, pmode_max, pmode_init_pows, orthogonalize=True, sort=True)
        probe = self._probe_z_shift(probe, self.init_params.get('probe_z_shift'))
        probe = self._probe_normalization(probe, self.init_params.get('probe_normalization'))
        return probe

    def _probe_permute(self, probe, order):
        """
        Permute the probe dimensions if specified in the parameters.
        """
        if order is not None:
            vprint(f"Permuting probe with order = {order}", verbose=self.verbose)
            probe = probe.transpose(order)
        return probe
    
    def _probe_set_pmode_max(self, probe, pmode_max, pmode_init_pows, orthogonalize=True, sort=True):
        """
        Either cap or pad the pmode for mixed state probe with optional orthogonalization and sorting
        
        """
        pmode_now = probe.shape[0]
        probe_int_sum = np.sum(np.abs(probe)**2)
        pmode_init_pow = [min(pmode_init_pows)] # pmode_init_pows is a list of float(s), so we convert it into a list of float
        
        if pmode_now > pmode_max:
            vprint(f"pmode_now: {pmode_now} and pmode_max: {pmode_max}, capping the pmode.", verbose=self.verbose)
            probe_final = probe[:pmode_max]
        
        elif pmode_now == pmode_max:
            vprint(f"pmode_now: {pmode_now} and pmode_max: {pmode_max}, leaving the pmode unchanged.", verbose=self.verbose)
            probe_final = probe
        
        else: # pmode_now <= pmode_max: # Need to pad new probe modes
            vprint(f"pmode_now: {pmode_now} and pmode_max: {pmode_max}, padding the pmode.", verbose=self.verbose)
            num_new_modes = pmode_max - pmode_now
            vprint(f"Creating {num_new_modes} new probe modes from the major mode", verbose=self.verbose)
            mixed_probe_temp = make_mixed_probe(probe[0], pmode_max, pmode_init_pow, verbose=False) # Take the strongest probe mode and make a temporary new mixed probe (int sum at 1)
            new_modes = mixed_probe_temp[-num_new_modes:] * probe_int_sum ** 0.5 # Normalize the new mode intensity with original intensity
            probe_final = np.concatenate((probe, new_modes), axis=0) # Total int = 1 + num_new_modes * pmode_init_pow, will normalize it later
            
        # Normalize back to original intensity
        normalization_factor = (np.sum(np.abs(probe_final) ** 2) / probe_int_sum) ** 0.5
        probe_final = probe_final / normalization_factor
        
        # Optional orthogonalization and sorting
        if orthogonalize:
            vprint(f"Orthogonalizing {len(probe_final)} pmodes", verbose=self.verbose)
            probe_final = orthogonalize_modes_vec_np(probe_final)
            
        if sort:
            vprint(f"Sorting {len(probe_final)} pmodes by their intensities", verbose=self.verbose)
            probe_final = sort_by_mode_int_np(probe_final)
            
        return probe_final
    
    def _probe_z_shift(self, probe, prop_distance):
        """
        Applying user-specified additional axial propagation to the initialized probe. This is used for shifting the reconstructed probe along depth.
        
        Note that prop_distance is defined with propagation direction, so positive value means forward propagation (i.e., increasing depth/z).
        """
        
        if prop_distance is None or prop_distance == 0:
            return probe

        else:
            dx = self.init_variables['dx']
            lambd = self.init_variables['lambd']
            unit_str = self.init_variables['length_unit']
            
            vprint(f"Applying additional axial propagation (z) = {prop_distance} {unit_str} to the probe. Positive value means forward propagation (i.e., increasing depth/z).", verbose=self.verbose)
            H = near_field_evolution(probe.shape[-2:], dx, prop_distance, lambd)
            probe_shifted = np.fft.ifft2(H[None,] * np.fft.fft2(probe))
            return probe_shifted
    
    def _probe_normalization(self, probe, norm_cfg):
        """
        Normalize the probe intensity based on the measurements.
        """
        
        # This correction is enforced even the norm_cfg is None (not provided by user)
        if norm_cfg is None:
            norm_cfg = {}
        
        norm_mode = norm_cfg.get('mode', 'mean_total_ints') # Default to 'mean_total_ints'
        norm_const = norm_cfg.get('value', None)  # Used for 'target_intensity' mode
        
        # Grab meas_total_ints
        try:
            # Using the pre-calculated meas_total_ints for probe intensity normalization
            # becasue on-the-fly padding could increase the total meas intensity
            meas_total_ints = self.init_variables['meas_total_ints']
        except KeyError:
            vprint("WARNING: Measurement total intensities ('meas_total_ints') not found in init.init_variables. Initializing measurements first for probe normalization...", verbose=self.verbose)
            vprint(" ", verbose=self.verbose)
            self.init_measurements()
            meas_total_ints = self.init_variables['meas_total_ints']

        vprint(f"Normalizing probe intensity with mode = '{norm_mode}' and value = '{norm_const}'", verbose=self.verbose)

        # Dispatch the normalizing methods
        if norm_mode == 'mean_total_ints':
            target_int = meas_total_ints.mean()
        elif norm_mode == 'max_total_ints':
            target_int = meas_total_ints.max()
        elif norm_mode == 'target_intensity':
            target_int = norm_const
        
        normalization_factor = (np.sum(np.abs(probe) ** 2) / target_int) ** 0.5
        probe = probe / normalization_factor
        vprint(f"sum(|probe_data|**2) = {np.sum(np.abs(probe)**2):.2f}, while meas_total_ints (min, mean, max) = ({meas_total_ints.min():.4f}, {meas_total_ints.mean():.4f}, {meas_total_ints.max():.4f})", verbose=self.verbose)
        return probe.astype('complex64')
   
    ###### Private methods for initializing positions ######
    
    def _load_pos(self):
        """
        Load the probe positions from the specified source.
        """
        
        # Validate required fields
        try:
            pos_source = self.init_params['pos_source']
            pos_params = self.init_params['pos_params']
        except KeyError as e:
            raise KeyError(f"Missing required configuration field: {e}")
    
        vprint(f"Loading probe positions from source = '{pos_source}'", verbose=self.verbose)
    
        if pos_source == 'custom':
            pos = pos_params
        elif pos_source == 'PtyRAD':
            pos = self._load_pos_from_ptyrad(pos_params)
        elif pos_source == 'PtyShv':
            pos = self._load_pos_from_ptyshv(pos_params)
        elif pos_source == 'py4DSTEM':
            pos = self._load_pos_from_py4dstem(pos_params)
        elif pos_source == 'simu':
            pos = self._simulate_pos(pos_params)
        elif pos_source == 'foldslice_hdf5':
            pos = self._load_pos_from_foldslice(pos_params)
        else:
            raise ValueError(f"Unsupported position source '{pos_source}'. Use 'custom', 'PtyRAD', 'PtyShv', 'py4DSTEM', 'simu', or 'foldslice_hdf5'.")
    
        return pos
    
    def _load_pos_from_ptyrad(self, params: str):
        pt_path = params
        ckpt = self.cache_contents if self.use_cached_pos else load_ptyrad(pt_path)
        crop_pos = ckpt['model_attributes']['crop_pos']
        probe_pos_shifts = ckpt['optimizable_tensors']['probe_pos_shifts']
        pos = crop_pos + probe_pos_shifts
        return pos
    
    def _load_pos_from_ptyshv(self, params: str):
        mat_path = params
        mat_version = get_matfile_version(mat_path) # https://docs.scipy.org/doc/scipy-1.11.3/reference/generated/scipy.io.matlab.matfile_version.html
        use_h5py = (mat_version[0] == 2)
        mat_contents = self.cache_contents if self.use_cached_pos else load_mat(mat_path, key=['object', 'probe', 'outputs.probe_positions'], delimiter='.')
        vprint(f"Input PtyShv probe positions has original shape {mat_contents['outputs.probe_positions'].shape}, while default PtyShv order is (N, 2)", verbose=self.verbose)

        # First unify the axes order induced by loading with scipy / h5py, now it should be (N, 2)
        if use_h5py:
            mat_contents = {key: arr.transpose(range(arr.ndim)[::-1]) for key, arr in mat_contents.items()}
            vprint(f"Reverse array axes order because use_h5py = {use_h5py}, which automatically reverse the order", verbose=self.verbose)

        probe_positions = mat_contents['outputs.probe_positions']
        probe_shape = mat_contents['probe'].shape[:2]   # Matlab probe is (Ny,Nx,pmode,vp) or (Ny,Nx,pmode)
        obj_shape   = mat_contents['object'].shape[:2]   # Matlab object is (Ny, Nx, Nz) or (Ny,Nx)
        pos_offset = np.ceil((np.array(obj_shape)/2) - (np.array(probe_shape)/2)) - 1 # For Matlab - Python index shift
        probe_positions_yx   = probe_positions[:, [1,0]] # The first index after shifting is the row index (along vertical axis)
        pos                  = probe_positions_yx + pos_offset 
        return pos
    
    def _load_pos_from_py4dstem(self, params: str):
        hdf5_path       = params
        hdf5_contents   = self.cache_contents if self.use_cached_pos else load_hdf5(hdf5_path)
        probe_positions = hdf5_contents['positions_px']
        probe_shape     = hdf5_contents['probe'].shape[-2:] # py4DSTEM probe is (pmode,Ny,Nx)
        pos             = probe_positions - np.array(probe_shape)/2 
        return pos
    
    def _load_pos_from_foldslice(self, params: str):
        # This preprocessing routine is equivalent to `p.src_positions='hdf5_pos';` in `fold_slice`
        # which was used for many APS instruments
        
        dx = self.init_variables['dx']
        probe_shape = self.init_variables['probe_shape']
        
        hdf5_path = params
        ppY = load_hdf5(hdf5_path, key='ppY')
        ppX = load_hdf5(hdf5_path, key='ppX')
        pos = np.stack((-ppY, -ppX), axis=1) / dx 
        pos = np.flipud(pos) # (N,2) in (pos_y_px, pos_x_px)
        obj_shape = 1.2 * np.ceil(pos.max(0) - pos.min(0) + probe_shape)
        pos = pos + np.ceil((np.array(obj_shape)/2) - (np.array(probe_shape)/2)) # Shift to obj coordinate      
        return pos
    
    def _simulate_pos(self, simu_params: dict):

        if simu_params is not None:
            vprint("Using user-specified parameters in 'init_params['pos_params']' for initial position simulation.", verbose=self.verbose)
        else:
            simu_params = {}
            vprint("Using experimental parameters specified by 'init_params' (dx, scan_step size, N_scan_slow, N_scan_fast) for initial position simulation.", verbose=self.verbose)

        # The unspecified parameters will be set to the values specified in self.init_variables
        dx        = simu_params.get('dx', self.init_variables['dx'])
        scan_step_size = simu_params.get('scan_step_size', self.init_variables['scan_step_size'])
        N_scan_slow    = simu_params.get('N_scan_slow', self.init_variables['N_scan_slow'])
        N_scan_fast    = simu_params.get('N_scan_fast', self.init_variables['N_scan_fast'])
        probe_shape    = simu_params.get('probe_shape', self.init_variables['probe_shape'])
        
        vprint(f"Simulating probe positions with dx = {dx:.4f}, scan_step_size = {scan_step_size:.4f}, N_scan_fast = {N_scan_fast}, N_scan_slow = {N_scan_slow}", verbose=self.verbose)
        pos = scan_step_size / dx * np.array([(y, x) for y in range(N_scan_slow) for x in range(N_scan_fast)]) # (N,2), each row is (y,x)
        pos = pos - pos.mean(0) # Center scan around origin
        obj_shape = 1.2 * np.ceil(pos.max(0) - pos.min(0) + probe_shape)
        pos = pos + np.ceil((np.array(obj_shape)/2) - (np.array(probe_shape)/2)) # Shift to obj coordinate
        return pos

    def _process_pos(self, pos):
        """
        Process the loaded probe positions, including flipping, affine transformations, and random displacements.
        """
        # If the processing config is None, the methods will skip it internally
        
        pos = self._pos_scan_flipT(pos, self.init_params.get('pos_scan_flipT'))
        pos = self._pos_scan_affine_transform(pos, self.init_params.get('pos_scan_affine'))
        pos = self._pos_scan_add_random_displacement(pos, self.init_params.get('pos_scan_rand_std'))
        return pos
    
    def _pos_scan_flipT(self, pos, flipT_axes):
        """
        Flip and transpose scan positions.
        flipT_axes: list of 3 binary/int values [flipud, fliplr, transpose]
        """
        
        if flipT_axes is None:
            return pos

        # Validate length
        if not isinstance(flipT_axes, (list, tuple)) or len(flipT_axes) != 3:
            raise ValueError(f"Expected flipT_axes to be a list of 3 values, got: {flipT_axes}")

        # Safely cast all entries to int
        try:
            flipT_axes = [int(v) for v in flipT_axes]
        except Exception as e:
            raise ValueError(f"flipT_axes must contain values convertible to int (0 or 1). Got: {flipT_axes}") from e
        
        vprint(f"Flipping scan pattern with [flipup, fliplr, transpose] = {flipT_axes}", verbose=self.verbose)
        
        # Convert the binary code to the indices of non-zero axis. E.g. scan_flipT = [0,1,1] => flip the axes = [1,2]
        flipT_axes = np.nonzero(flipT_axes)[0] 
        if len(flipT_axes) > 0:
            pos = pos.reshape(self.init_variables['N_scan_slow'], self.init_variables['N_scan_fast'], 2)
            pos = np.flip(pos, flipT_axes)
            pos = pos.reshape(-1, 2)
        return pos
    
    def _pos_scan_affine_transform(self, pos, scan_affine):
        if scan_affine is not None:
            (scale, asymmetry, rotation, shear) = scan_affine
            vprint(f"Applying affine transformation to scan pattern with (scale, asymmetry, rotation, shear) = {(scale, asymmetry, rotation, shear)}", verbose=self.verbose)
            pos = pos - pos.mean(0)
            pos = pos @ compose_affine_matrix(scale, asymmetry, rotation, shear)
            probe_shape = self.init_variables['probe_shape']
            obj_shape = 1.2 * np.ceil(pos.max(0) - pos.min(0) + probe_shape)
            pos = pos + np.ceil((np.array(obj_shape) / 2) - (np.array(probe_shape) / 2))
        return pos
    
    def _pos_scan_add_random_displacement(self, pos, scan_rand_std):
        if scan_rand_std is not None:
            vprint(f"Applying Gaussian distributed random displacement with std = {scan_rand_std} px to scan positions", verbose=self.verbose)
            set_random_seed(seed=self.random_seed)
            pos = pos + scan_rand_std * np.random.randn(*pos.shape)
        return pos
    
    ###### Private methods for initializing object ######

    def _load_obj(self):
        """
        Load the object from the specified source.
        """
        
        # Validate required fields
        try:
            obj_source = self.init_params['obj_source']
            obj_params = self.init_params['obj_params']
        except KeyError as e:
            raise KeyError(f"Missing required configuration field: {e}")

        vprint(f"Loading object from source = '{obj_source}'", verbose=self.verbose)

        if obj_source == 'custom':
            obj = obj_params
        elif obj_source == 'PtyRAD':
            obj = self._load_obj_from_ptyrad(obj_params)
        elif obj_source == 'PtyShv':
            obj = self._load_obj_from_ptyshv(obj_params)
        elif obj_source == 'py4DSTEM':
            obj = self._load_obj_from_py4dstem(obj_params)
        elif obj_source == 'simu':
            obj = self._simulate_obj(obj_params)
        else:
            raise ValueError(f"Unsupported object source '{obj_source}'. Use 'custom', 'PtyRAD', 'PtyShv', 'py4DSTEM', or 'simu'.")

        return obj
    
    def _load_obj_from_ptyrad(self, params: str):
        pt_path = params
        ckpt = self.cache_contents if self.use_cached_obj else load_ptyrad(pt_path)
        obja = ckpt['optimizable_tensors']['obja']
        objp = ckpt['optimizable_tensors']['objp']
        obj = obja * np.exp(1j * objp)
        return obj
    
    def _load_obj_from_ptyshv(self, params: str):
        mat_path = params
        mat_version = get_matfile_version(mat_path)
        use_h5py = (mat_version[0] == 2)
        obj = self.cache_contents['object'] if self.use_cached_obj else load_mat(mat_path, key='object')
        vprint(f"Input PtyShv object has original shape {obj.shape}, while default PtyShv order is (Ny, Nx, Nz)", verbose=self.verbose)

        # First unify the axes order induced by loading with scipy / h5py, now it should be (Ny, Nx, Nz)
        if use_h5py:
            obj = obj.transpose(range(obj.ndim)[::-1])
            vprint(f"Reverse array axes order because use_h5py = {use_h5py}, which automatically reverse the order", verbose=self.verbose)
    
        vprint("Expanding and permuting PtyShv object dimension to make a final object shape with (omode, Nz, Ny, Nx)", verbose=self.verbose)
        if len(obj.shape) == 2:  # Single-slice ptycho
            obj = obj[None, None, :, :]
        elif len(obj.shape) == 3:  # Multi-slice ptycho
            obj = obj[None,].transpose(0, 3, 1, 2)
    
        return obj
    
    def _load_obj_from_py4dstem(self, params: str):
        hdf5_path = params
        obj = self.cache_contents['object'] if self.use_cached_obj else load_hdf5(hdf5_path, key='object')
    
        vprint(f"Input py4DSTEM object has original shape {obj.shape}", verbose=self.verbose)
        vprint("Expanding py4DSTEM object dimension to (omode, Nz, Ny, Nx)", verbose=self.verbose)

        if len(obj.shape) == 2:  # Single-slice ptycho
            obj = obj[None, None, :, :]
        elif len(obj.shape) == 3:  # Multi-slice ptycho
            obj = obj[None,]
    
        return obj
    
    def _simulate_obj(self, simu_params):
        
        if simu_params is not None:
            vprint("Using user-specified parameters in 'init_params['obj_params']' for initial object simulation.", verbose=self.verbose)
            obj_shape = simu_params
            if len(obj_shape) != 4:
                raise ValueError(f"Input `obj_shape` = {obj_shape}, please provide a total dimension of 4 with (omode, Nz, Ny, Nx).")
            
        else:
            vprint("Using experimental parameters specified by 'init_params' for initial object simulation.", verbose=self.verbose)
            omode = self.init_params['obj_omode_max']
            Nz = self.init_params['obj_Nlayer']
            
            try:
                Ny, Nx = self.init_variables['obj_lateral_extent']
            except KeyError:
                vprint("WARNING: 'obj_lateral_extent' not found. Initializing positions first for obj_shape estimation...", verbose=self.verbose)
                vprint(" ", verbose=self.verbose)
                self.init_pos()
                Ny, Nx = self.init_variables['obj_lateral_extent']
                
        obj_shape = (omode, Nz, Ny, Nx)
        set_random_seed(seed=self.random_seed)
        obj = np.exp(1j * 1e-8 * np.random.rand(*obj_shape))
        return obj
    
    def _process_obj(self, obj):
        """
        Process the loaded object, including z cropping, padding, resampling, and setting omode
        """
        omode_max = self.init_params.get('obj_omode_max')

        obj = self._obj_z_crop(obj, self.init_params.get('obj_z_crop'))
        obj = self._obj_z_pad(obj, self.init_params.get('obj_z_pad'))
        obj = self._obj_z_resample(obj, self.init_params.get('obj_z_resample'))
        obj = self._object_set_omode_max(obj, omode_max)
        
        return obj
    
    def _obj_z_crop(self, obj, crop_range):
        """
        Crop 4D complex object (omode, Nz, Ny, Nx) across depth (Nz) dimension:
        [z_start, z_end]
        Note that this method would also update the `self.init_params['obj_Nlayer']`
        """
        
        if crop_range is None:
            return obj
        
        if len(crop_range) != 2:
            raise ValueError(f"Expected crop range [z_start, z_end], got {crop_range}")
        
        try:
            z_start, z_end = crop_range
            selected_slices = slice(z_start, z_end)
            vprint(f"Cropping object depth from z_start: {z_start} to z_end: {z_end}", verbose=self.verbose)
        except Exception as e:
            raise ValueError(f"Invalid crop range for object depth: {crop_range}, object shape is {obj.shape}") from e

        vprint(f"Current object has shape (omode, Nz, Ny, Nx) = {obj.shape}", verbose=self.verbose)
        
        obj = obj[:,selected_slices,:,:]
        vprint(f"Cropped object has shape (omode, Nz, Ny, Nx) = {obj.shape}", verbose=self.verbose)
        
        # Update init_params['obj_Nlayer]
        self.init_params['obj_Nlayer'] = obj.shape[1]
        
        return obj
    
    def _obj_z_pad(self, obj, pad_cfg):
        """
        Pad 4D complex object (omode, Nz, Ny, Nx) along the depth (Nz) dimension.
        Note that this method would also update the `self.init_params['obj_Nlayer']`
        """
        
        if pad_cfg is None:
            return obj
        
        pad_layers = pad_cfg['pad_layers']
        pad_types  = pad_cfg['pad_types']

        vprint(f"Current object has shape (omode, Nz, Ny, Nx) = {obj.shape}", verbose=self.verbose)
        vprint(f"Padding object along depth with pad_layers = {pad_layers}, pad_types = {pad_types}", verbose=self.verbose)

        # Assign variables
        pad_layer_top, pad_layer_bottom = pad_layers
        pad_type_top, pad_type_bottom = pad_types
        
        # Helper function
        def _create_z_pad(obj, num_layers, pad_type, top_or_bottom):
            obja = np.abs(obj)
            objp = np.angle(obj)
            omode, nz, ny, nx = obj.shape

            # Return an empty array with the same shape as obj but with nz = 0
            if num_layers is None or num_layers == 0:
                return np.empty((omode, 0, ny, nx), dtype=obj.dtype)
            
            # Create new layers
            else:
                new_shape = (omode, num_layers, ny, nx)
                
                if pad_type == 'vacuum':
                    new_layers_a = np.ones(new_shape)
                    new_layers_p = np.zeros(new_shape)
                elif pad_type == 'mean':
                    new_layers_a = np.mean(obja, axis=1, keepdims=True) # The variance of amplitude along depth is usually quite small so geometric mean ~= arithmetric mean, although it might feel more natural to use geometric mean.
                    new_layers_p = np.mean(objp, axis=1, keepdims=True) # Note that this could be a bit biased if phase is not positively constrained. The shape is (omode, 1, Ny, Nx).
                elif pad_type == 'edge':
                    if top_or_bottom == 'top':
                        edge_idx = [0] # Wrap it with [] to preserve the dimension 
                    elif top_or_bottom == 'bottom':
                        edge_idx = [-1]
                    else:
                        raise ValueError(f"top_or_bottom expects 'top', or 'bottom', got {top_or_bottom}.")
                    new_layers_a = obja[:, edge_idx] # (omode, 1, Ny, Nx)
                    new_layers_p = objp[:, edge_idx]
                else:
                     raise ValueError(f"Unsupported pad_type: {pad_type}, please use 'vacuum', 'mean', or 'edge'.")
                
                new_layers = new_layers_a * np.exp(1j * new_layers_p)
                return np.broadcast_to(new_layers, new_shape).copy().astype(obj.dtype)
        
        top_layers    = _create_z_pad(obj, num_layers=pad_layer_top,    pad_type=pad_type_top,    top_or_bottom='top')
        bottom_layers = _create_z_pad(obj, num_layers=pad_layer_bottom, pad_type=pad_type_bottom, top_or_bottom='bottom')
        obj = np.concatenate((top_layers, obj, bottom_layers), axis=1)
        vprint(f"Padded object has shape (omode, Nz, Ny, Nx) = {obj.shape}", verbose=self.verbose)
        
        # Update init_params['obj_Nlayer]
        self.init_params['obj_Nlayer'] = obj.shape[1]
        
        return obj

    def _obj_z_resample(self, obj, resample_cfg):
        """
        Resample 4D complex object (omode, Nz, Ny, Nx) along the depth (Nz) dimension.
        Note that this method would also update the `self.init_params['obj_Nlayer']`, 
        `self.init_params['obj_slice_thickness']`, and `self.init_variables['slice_thickness']`
        This is currently (v0.1.0b11) the only function in Initializer that uses PyTorch because the scipy.ndimage.zoom is just too slow...
        """
        
        if resample_cfg is None or resample_cfg['mode'] is None:
            return obj
        
        # Assign variables
        resample_mode = resample_cfg['mode']
        resample_value = resample_cfg['value']
        Nz_now = obj.shape[1]
        dz_now = self.init_variables['slice_thickness'] # This was set by `set_variables_dict` using values in `init_params['obj_slice_thickness]`
        length_unit = self.init_variables['length_unit']
        
        # Print current status
        vprint(f"Current object has shape (omode, Nz, Ny, Nx) = {(obj.shape)}", verbose=self.verbose)
        vprint(f"Current object has slice thickness = {dz_now:.3f} {length_unit}", verbose=self.verbose)
        vprint(f"Current object has mean(prod(amp, axis='depth')) = {np.mean(np.prod(np.abs(obj), axis=1)):.3f}, mean(sum(phase, axis='depth')) = {np.mean(np.sum(np.angle(obj), axis=1)):.3g}", verbose=self.verbose)
        vprint(f"Resampling object along depth with resampling mode = '{resample_mode}', value = {resample_value}", verbose=self.verbose)
        
        # Get resampled object and infer new slice thickness
        obja_resample, objp_resample = complex_object_z_resample_torch(obj, dz_now, resample_mode, resample_value, output_type='amp_phase', return_np=True) # Output amplitude and phase separately so we can check the phase value directly
        obj_resample = obja_resample * np.exp(1j * objp_resample) # (omode, Nz, Ny, Nx)
        Nz_new = obj_resample.shape[1]
        dz_new = dz_now * Nz_now / Nz_new
        
        # Print warning if there's phase wrapping
        if objp_resample.max() > 2*np.pi:
            vprint(f"Warning: Resampled object phase has a maximum value = {objp_resample.max():.3f} > 2pi, this would cause phase wrapping, try using thinner slices.")
        
        # Update Nlayer and slice thickness
        self.init_params['obj_Nlayer'] = obj_resample.shape[1]
        self.init_params['obj_slice_thickness'] = dz_new
        self.init_variables['slice_thickness'] = dz_new
        
        # Print final status
        vprint(f"Resampled object has shape (omode, Nz, Ny, Nx) = {(obj_resample.shape)}", verbose=self.verbose)
        vprint(f"Resampled object has slice thickness = {dz_new:.3f} {length_unit}", verbose=self.verbose)
        vprint(f"Resampled object has mean(prod(amp, axis='depth')) = {np.mean(np.prod(np.abs(obj_resample), axis=1)):.3f}, mean(sum(phase, axis='depth')) = {np.mean(np.sum(np.angle(obj_resample), axis=1)):.3g}", verbose=self.verbose)
        
        return obj_resample
    
    def _object_set_omode_max(self, obj, omode_max):
        """
        Either cap or pad the omode for mixed state object
        
        """
        omode_now = obj.shape[0]
        
        if omode_now > omode_max:
            vprint(f"omode_now: {omode_now} and omode_max: {omode_max}, capping the omode.", verbose=self.verbose)
            obj_final = obj[:omode_max]
        
        elif omode_now == omode_max:
            vprint(f"omode_now: {omode_now} and omode_max: {omode_max}, leaving the omode unchanged.", verbose=self.verbose)
            obj_final = obj
        
        else: # omode_now <= omode_max: # Need to pad new probe modes
            vprint(f"omode_now: {omode_now} and omode_max: {omode_max}, padding the omode.", verbose=self.verbose)
            num_new_modes = omode_max - omode_now
            vprint(f"Creating {num_new_modes} new object modes from the mean and std of original object modes", verbose=self.verbose)
            
            # Assign variables
            obja = np.abs(obj)
            objp = np.angle(obj)
            spatial_dims = obj[0].shape # (z,y,x)
            obja_mean = np.mean(obja, axis=0, keepdims=True)
            objp_mean = np.mean(objp, axis=0, keepdims=True)
            if omode_now == 1: # There's no std when omode=1. This is rather rudimentary and we'll introduced some spatially structured noise in future release
                obja_std = 5e-4 * obja_mean
                objp_std = 0.20 * objp_mean
            else:
                obja_std = np.std(obja, axis=0, keepdims=True)
                objp_std = np.std(objp, axis=0, keepdims=True)
            
            # Create new modes from random variable eps, note that amplitude and phase are perfectly correlated here
            set_random_seed(seed=self.random_seed)
            eps = np.random.randn(num_new_modes,*spatial_dims) # (num_new_modes, z, y, x)
            obja_new = obja_mean + eps * obja_std
            objp_new = objp_mean + eps * objp_std
            
            # Check min and max
            obja_new = np.clip(obja_new, a_min=np.min(obja), a_max=np.max(obja))
            objp_new = np.clip(objp_new, a_min=np.min(objp), a_max=np.max(objp))
            
            # Recombine amplitude and phase back to complex-valued obj
            new_modes = obja_new * np.exp(1j * objp_new)
            obj_final = np.concatenate((obj, new_modes), axis=0)
            
        return obj_final