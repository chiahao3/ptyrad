"""
Saving functions for PtyRAD outputs including model, arrays, params files, etc.

"""

import os
from typing import Any, Dict

import h5py
import numpy as np
import torch
from tifffile import imwrite

from ptyrad.utils import get_time, normalize_by_bit_depth, safe_filename, vprint, expand_presets

###### These are data saving functions ######

def write_tif(file_path, data):
    """
    Save an array as a TIFF file.
    """
    imwrite(file_path, data, imagej=True)
    vprint(f"Success! Saved data to .tif file: {file_path}")

def write_npy(file_path, data):
    """
    Save an array as a NumPy .npy file.
    """
    np.save(file_path, data)
    vprint(f"Success! Saved data to .npy file: {file_path}")

def write_hdf5(file_path, data, dataset_name="meas", **kwargs):
    """
    Save an array as an HDF5 file.
    """
    with h5py.File(file_path, "w") as hf: # 'w' will override if the file already exists
        hf.create_dataset(dataset_name, data=data, compression="gzip", **kwargs)
    vprint(f"Success! Saved data as '{dataset_name}' to .hdf5 file: {file_path}")

def save_array(data, file_dir='', file_name='ptyrad_init_meas', file_format="hdf5", output_shape=None, append_shape=True, **kwargs):
    """
    Save an ND array to the specified file format.

    Args:
        data (numpy.ndarray): ND array to save.
        file_dir (str): Directory to save the file.
        file_name (str): Base name of the file (without extension).
        file_format (str): File format to save as ("tif", "npy", "hdf5", "mat").
        output_shape (tuple, optional): Desired shape for the output array.
        append_shape (bool): Whether to append the array shape to the filename.
        **kwargs: Additional arguments for specific file formats.
    """
    # Reshape data if output_ndim is specified
    if output_shape is not None:
        try:
            data = data.reshape(output_shape)
        except ValueError as e:
            vprint(f"WARNING: {e}, the data shape is preserved as {data.shape}")
            
    # Append shape to the filename if enabled
    if append_shape:
        shape_str = "_".join(map(str, data.shape))
        file_name = f"{file_name}_{shape_str}"

    # Construct the full file path
    file_format = file_format.lower()
    file_path = os.path.join(file_dir, f"{file_name}.{file_format}")
    vprint(f"Saving array with shape = {data.shape} and dtype = {data.dtype}")
    
    if os.path.isfile(file_path):
        vprint(f"file path = '{file_path}' already exists, the file will be overwritten.")
    
    if file_format in ["tif", "tiff"]:
        write_tif(file_path, data)
    elif file_format == "npy":
        write_npy(file_path, data)
    elif file_format in ["hdf5", "h5", "mat"]:
        # Saving .mat into hdf5 as if it were .mat v7.3. This ensures compatibility with py4DGUI.
        write_hdf5(file_path, data, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {file_format}")
    
###### These are results saving functions ######

def make_save_dict(output_path, model, params, optimizer, niter, indices, batch_losses):
    ''' Make a dict to save relevant paramerers '''
    
    avg_losses = {name: np.mean(values) for name, values in batch_losses.items()}
    avg_iter_t = np.mean(model.iter_times)
    
    # While it might seem redundant to save bothe `params` and lots of `model_attributes`,    
    # one should note that `params` only stores the initial value from params files,
    # the actual values used for reconstuction such as N_scan_slow, N_scan_fast, dx, dk, Npix, N_scans could be different from initial value due to the meas_crop, meas_resample
    # the model behavior and learning rates could also be different from the initial params dict if the user
    # run the reconstuction with manually modified `model_params` in the detailed walkthrough notebook
    
    # Postprocess the opt_probe back to complex view
    optimizable_tensors = {}
    for name, tensor in model.optimizable_tensors.items():
        optimizable_tensors[name] = tensor.detach().clone()
        if name == 'probe':
            optimizable_tensors['probe'] = model.get_complex_probe_view().detach().clone()
    
    from ptyrad import __version__ as ptyrad_version
        
    save_dict = {
                'ptyrad_version'        : ptyrad_version,
                'output_path'           : output_path,
                'optimizable_tensors'   : optimizable_tensors,
                'optim_state_dict'      : optimizer.state_dict() if 'optim_state' in params['recon_params']['save_result'] else None,
                'params'                : params, 
                'model_attributes': # Have to do this explicit saving because I want specific fields but don't want the enitre model with grids and other redundant info
                    {'detector_blur_std': model.detector_blur_std,
                     'obj_preblur_std'  : model.obj_preblur_std,
                     'start_iter'       : model.start_iter,
                     'lr_params'        : model.lr_params,
                     'omode_occu'       : model.omode_occu,
                     'H'                : model.H,
                     'N_scan_slow'      : model.N_scan_slow,
                     'N_scan_fast'      : model.N_scan_fast,
                     'crop_pos'         : model.crop_pos,
                     'slice_thickness'  : model.slice_thickness,
                     'dx'               : model.dx,
                     'dk'               : model.dk,
                     'scan_affine'      : model.scan_affine,
                     'tilt_obj'         : model.tilt_obj,
                     'shift_probes'     : model.shift_probes,
                     'probe_int_sum'    : model.probe_int_sum
                     },
                'loss_iters'            : model.loss_iters,
                'iter_times'            : model.iter_times,
                'dz_iters'              : model.dz_iters,
                'avg_iter_t'            : avg_iter_t,
                'niter'                 : niter,
                'indices'               : indices,
                'batch_losses'          : batch_losses,
                'avg_losses'            : avg_losses
                }
    
    return save_dict

def save_dict_to_hdf5(
    d: Dict[str, Any], output_path: str, none_sentinel: str = "__NONE__", **kwargs
) -> None:
    """
    Save a nested Python dictionary to an HDF5 file.

    Supports common Python, NumPy, and PyTorch types. Non-HDF5-compatible types
    (e.g., list of tuples, None, etc.) are automatically converted to HDF5-friendly formats.

    Note that integer key (e.g. like in optimizer state dict) are coerced to string for HDF5 format.
    
    Args:
        d (Dict[str, Any]): The nested dictionary to save.
        output_path (str): The file path to save the HDF5 output to.
        none_sentinel (str, optional): String used to represent `None` in HDF5. Defaults to "__NONE__".
        **kwargs: Additional keyword arguments to pass to `h5py.File()` or `create_dataset()`.
                  This can include compression settings like `compression="gzip"`, etc.

    Returns:
        None
    """

    def _recursively_save_dict_to_hdf5(d: Dict[str, Any], h5group: h5py.Group, path="") -> None:
        for key, value in d.items():
            full_key = f"{path}/{key}" if path else str(key)
            key = str(key)  # convert to string for HDF5, especially important for optimizer state dict with integer as key
            
            try:
                # Delete existing group/dataset if it exists
                if key in h5group:
                    del h5group[key]
                
                if value is None:
                    h5group.create_dataset(key, data=none_sentinel, **kwargs)
                
                elif isinstance(value, dict):
                    subgroup = h5group.create_group(key)
                    _recursively_save_dict_to_hdf5(value, subgroup)
                
                elif isinstance(value, list):
                    if all(isinstance(i, (int, float, np.number)) for i in value):
                        h5group.create_dataset(key, data=np.array(value), **kwargs)
                    
                    elif all(isinstance(i, str) for i in value):
                        dt = h5py.special_dtype(vlen=str)
                        h5group.create_dataset(key, data=np.array(value, dtype=dt), **kwargs)
                    
                    elif all(isinstance(i, tuple) for i in value):
                        try:
                            arr = np.array([list(t) for t in value])
                            h5group.create_dataset(key, data=arr, **kwargs)
                        except Exception:
                            h5group.create_dataset(key, data=str(value), **kwargs)
                    
                    elif all(isinstance(i, dict) for i in value):
                        subgroup = h5group.create_group(key)
                        for idx, item in enumerate(value):
                            item_group = subgroup.create_group(str(idx))
                            _recursively_save_dict_to_hdf5(item, item_group)
                    
                    elif all(isinstance(i, (np.ndarray, torch.Tensor)) for i in value):
                        try:
                            arr = np.stack([i.detach().cpu().numpy() if isinstance(i, torch.Tensor) else i for i in value])
                            h5group.create_dataset(key, data=arr, **kwargs)
                        except Exception:
                            h5group.create_dataset(key, data=str(value), **kwargs)
                    
                    else:
                        # fallback to storing list as strings (warn if needed)
                        h5group.create_dataset(key, data=str(value), **kwargs)
                
                elif isinstance(value, tuple):
                    h5group.create_dataset(key, data=np.array(value), **kwargs)
                
                elif isinstance(value, (int, float, str, np.number)):
                    h5group.create_dataset(key, data=value, **kwargs)
                
                elif isinstance(value, torch.Tensor):
                    h5group.create_dataset(key, data=value.detach().cpu().numpy(), **kwargs)
                
                elif isinstance(value, np.ndarray):
                    h5group.create_dataset(key, data=value, **kwargs)
                
                # Fallback option
                else:
                    h5group.create_dataset(key, data=str(value), **kwargs)
            
            except Exception as e:
                raise RuntimeError(f"Failed to save key '{key}' (full path: '{full_key}') of type {type(value)}") from e
            
    with h5py.File(output_path, "w") as hf:
        _recursively_save_dict_to_hdf5(d, hf)

def make_output_folder(
    output_dir,
    indices,
    init_params,
    recon_params,
    model,
    constraint_params,
    loss_params,
    recon_dir_affixes=["default"],
    verbose=True,
):
    """
    Generate the output folder name based on reconstruction parameters, model attributes, constraints, and loss settings.

    Args:
        output_dir (str): Base directory where the output folder will be created.
        indices (list): List of indices used in the reconstruction.
        init_params (dict): Initialization parameters for the reconstruction.
        recon_params (dict): Reconstruction parameters, including settings like prefix, postfix, and batch size.
        model (object): Model object containing attributes such as probe, object, and optimizer parameters.
        constraint_params (dict): Constraints applied during reconstruction, such as filters and smoothing.
        loss_params (dict): Loss parameters used in the reconstruction process.
        recon_dir_affixes (list): List of tags or presets to include in the folder name. Defaults to ["default"].
        verbose (bool): Whether to print verbose messages during folder creation. Defaults to True.

    Returns:
        str: Path to the generated output folder.
    """

    prefix_time = recon_params.get("prefix_time", False)
    prefix = recon_params.get("prefix", "")
    postfix = recon_params.get("postfix", "")
    parts = []

    recon_dir_presets = {
        "minimal": ['indices', 'meas', 'batch', 'pmode', 'omode', 'nlayer'],
        
        "default": ['indices', 'meas', 'batch', 'pmode', 'omode', 'nlayer',
                    'lr', 'model', 'constraint',
                    'loss', 'affine', 'tilt'],
        
        "all":     ['indices', 'meas', 'batch', 'pmode', 'omode', 'nlayer',
                    'optimizer', 'start_iter', 'lr', 'model', 'constraint',
                    'loss', 'illumination', 'dx', 'affine', 'tilt']        
        }
    
    # Process recon_dir_affixes to expand presets
    if any(tag in recon_dir_presets for tag in recon_dir_affixes):
        vprint(f"Original recon_dir_affixes = {recon_dir_affixes}", verbose=verbose)
        recon_dir_affixes = expand_presets(recon_dir_affixes, recon_dir_presets)
        vprint(f"Expanded recon_dir_affixes = {recon_dir_affixes}", verbose=verbose)
    
    # Attach time string if prefix_time is true or non-empty str
    if prefix_time is True or (isinstance(prefix_time, str) and prefix_time):
        time_str = get_time(prefix_time)  # e.g. '20250606'
        parts.append(time_str)

    # Attach prefix (only if prefix is non-empty str)
    if isinstance(prefix, str) and prefix:
        parts.append(prefix)

    # Attach indices mode (optional)
    if "indices" in recon_dir_affixes:
        indices_mode = recon_params["INDICES_MODE"].get("mode")
        parts.append(f"{indices_mode}_N{len(indices)}")

    # Attach DP size and meas flip (optional)
    if "meas" in recon_dir_affixes:
        dp_size = model.get_complex_probe_view().size(-1)
        parts.append(f"dp{dp_size}")

        meas_flipT = init_params["meas_flipT"]
        # Attach meas flipping
        if meas_flipT is not None:  # Note that [0,0,0] will be attached is specified for clarity
            flipT_str = "flipT" + "".join(str(x) for x in meas_flipT)
            parts.append(flipT_str)

    # Attach group mode and batch size (optional)
    if "batch" in recon_dir_affixes:
        group_mode = recon_params["GROUP_MODE"]
        batch_size = recon_params["BATCH_SIZE"].get("size")
        grad_accum = recon_params["BATCH_SIZE"].get("grad_accumulation", 1)
        batch_size *= grad_accum  # Affix the effective batch size
        parts.append(f"{group_mode}{batch_size}")

    # Attach pmode (optional)
    if "pmode" in recon_dir_affixes:
        pmode = model.get_complex_probe_view().size(0)
        parts.append(f"p{pmode}")

    # Attach omode (optional)
    if "omode" in recon_dir_affixes:
        omode = model.opt_objp.size(0)
        parts.append(f"{omode}obj")

    # Attach obj Nlayer and dz (optional)
    if "nlayer" in recon_dir_affixes:
        nlayer = model.opt_objp.size(1)
        parts.append(f"{nlayer}slice")

        if nlayer != 1:
            slice_thickness = (
                model.slice_thickness.detach().cpu().numpy()
            )  # This is the initialized slice thickness
            parts.append(f"dz{slice_thickness:.3g}")

    # Attach optimizer name (optional)
    if "optimizer" in recon_dir_affixes:
        optimizer_str = model.optimizer_params["name"]
        parts.append(f"{optimizer_str}")

    # Attach start_iter (optional)
    if "start_iter" in recon_dir_affixes:
        start_iter_map = {
            "probe": "ps",
            "obja": "oas",
            "objp": "ops",
            "probe_pos_shifts": "ss",
            "obj_tilts": "ts",
            "slice_thickness": "dzs",
        }

        for key, tag in start_iter_map.items():
            start_val = model.start_iter.get(key)
            if start_val is not None and start_val > 1:
                parts.append(f"{tag}{start_val}")

    # Attach learning rate (optional)
    if "lr" in recon_dir_affixes:
        lr_map = {
            "probe": "plr",
            "obja": "oalr",
            "objp": "oplr",
            "probe_pos_shifts": "slr",
            "obj_tilts": "tlr",
            "slice_thickness": "dzlr",
        }

        for key, tag in lr_map.items():
            lr_val = model.lr_params[key]
            if lr_val != 0:
                lr_str = format(lr_val, ".0e").replace("e-0", "e-")
                parts.append(f"{tag}{lr_str}")

    # Attach model params (optional)
    if "model" in recon_dir_affixes:
        if model.obj_preblur_std is not None and model.obj_preblur_std != 0:
            parts.append(f"opreb{model.obj_preblur_std}")

        if model.detector_blur_std is not None and model.detector_blur_std != 0:
            parts.append(f"dpblur{model.detector_blur_std}")

    # Attach constraint params (optional)
    if "constraint" in recon_dir_affixes:
        if constraint_params["kr_filter"]["freq"] is not None:
            obj_type = constraint_params["kr_filter"]["obj_type"]
            kr_str = {"both": "kr", "amplitude": "kra", "phase": "krp"}.get(obj_type)
            radius = constraint_params["kr_filter"]["radius"]
            parts.append(f"{kr_str}f{radius}")

        if constraint_params["kz_filter"]["freq"] is not None:
            obj_type = constraint_params["kz_filter"]["obj_type"]
            kz_str = {"both": "kz", "amplitude": "kza", "phase": "kzp"}.get(obj_type)
            beta = constraint_params["kz_filter"]["beta"]
            parts.append(f"{kz_str}f{beta}")

        if (
            constraint_params["obj_rblur"]["freq"] is not None
            and constraint_params["obj_rblur"]["std"] != 0
        ):
            obj_type = constraint_params["obj_rblur"]["obj_type"]
            obj_str = {"both": "o", "amplitude": "oa", "phase": "op"}.get(obj_type)
            parts.append(f"{obj_str}rblur{constraint_params['obj_rblur']['std']}")

        if (
            constraint_params["obj_zblur"]["freq"] is not None
            and constraint_params["obj_zblur"]["std"] != 0
        ):
            obj_type = constraint_params["obj_zblur"]["obj_type"]
            obj_str = {"both": "o", "amplitude": "oa", "phase": "op"}.get(obj_type)
            parts.append(f"{obj_str}zblur{constraint_params['obj_zblur']['std']}")

        if constraint_params["complex_ratio"]["freq"] is not None:
            obj_type = constraint_params["complex_ratio"]["obj_type"]
            obj_str = {"both": "o", "amplitude": "oa", "phase": "op"}.get(obj_type)
            alpha1 = round(constraint_params["complex_ratio"]["alpha1"], 2)
            alpha2 = round(constraint_params["complex_ratio"]["alpha2"], 2)
            parts.append(f"{obj_str}cplx{alpha1}_{alpha2}")

        if constraint_params["mirrored_amp"]["freq"] is not None:
            scale = round(constraint_params["mirrored_amp"]["scale"], 2)
            power = round(constraint_params["mirrored_amp"]["power"], 2)
            parts.append(f"mamp{scale}_{power}")

        if constraint_params["obja_thresh"]["freq"] is not None:
            parts.append(f"oathr{round(constraint_params['obja_thresh']['thresh'][0], 2)}")

        if constraint_params["objp_postiv"]["freq"] is not None:
            mode = constraint_params["objp_postiv"].get("mode", "clip_neg")
            mode_str = "s" if mode == "subtract_min" else "c"
            relax = constraint_params["objp_postiv"]["relax"]
            relax_str = "" if relax == 0 else f"{round(relax, 2)}"
            parts.append(f"opos{mode_str}{relax_str}")

        if constraint_params["tilt_smooth"]["freq"] is not None:
            parts.append(f"tsm{round(constraint_params['tilt_smooth']['std'], 2)}")

        if constraint_params["probe_mask_k"]["freq"] is not None:
            parts.append(f"pmk{round(constraint_params['probe_mask_k']['radius'], 2)}")

    # Attach loss params (optional)
    if "loss" in recon_dir_affixes:
        loss_map = {
            "loss_single": ("sng", 2),
            "loss_poissn": ("psn", 2),
            "loss_pacbed": ("pcb", 2),
            "loss_sparse": ("spr", 2),
            "loss_simlar": ("sml", 2),
        }

        for key, (tag, digits) in loss_map.items():
            loss = loss_params.get(key, {})
            if loss.get("state"):
                parts.append(f"{tag}{round(loss.get('weight', 0), digits)}")

    # Attach illumination params (optional)
    if "illumination" in recon_dir_affixes:
        illumination = init_params["probe_illum_type"]
        if illumination == "electron":
            init_conv_angle = init_params["probe_conv_angle"]
            init_defocus = init_params["probe_defocus"]
            init_c3 = init_params["probe_c3"]
            init_c5 = init_params["probe_c5"]
            parts.append(f"ca{init_conv_angle:.3g}")
            if init_defocus != 0:
                parts.append(f"df{init_defocus:.3g}")
            if init_c3 != 0:
                parts.append(f"c3{format(init_c3, '.0e')}")
            if init_c5 != 0:
                parts.append(f"c5{format(init_c5, '.0e')}")
        elif illumination == "xray":
            init_Ls = init_params["Ls"]
            parts.append(f"Ls{init_Ls * 1e9:.0f}")
        else:
            raise ValueError(
                f"init_params['probe_illum_type'] = {illumination} not implemented yet, please use either 'electron' or 'xray'!"
            )

    # Attach dx (optional)
    if "dx" in recon_dir_affixes:
        dx = model.dx.detach().cpu().numpy()
        parts.append(f"dx{dx:.4g}")

    # Attach scan_affine (optional)
    if "affine" in recon_dir_affixes:
        scan_affine = model.scan_affine  # Note that scan_affine could be None
        if scan_affine is not None and not np.allclose(scan_affine, [1, 0, 0, 0]):
            affine_str = "aff" + "_".join(f"{x:.2g}" for x in scan_affine)  # (4,)
            parts.append(f"{affine_str}")

    # Attach init tilts (optional)
    if "tilt" in recon_dir_affixes:
        init_tilts = (
            model.opt_obj_tilts.mean(0).detach().cpu().numpy()
        )  # (2,) regardless tilt_type = 'all' or 'each'
        if np.any(init_tilts):
            parts.append(f"tilt{init_tilts[0]:.2g}_{init_tilts[1]:.2g}")

    # Attach postfix (only if postfix is non-empty str)
    if isinstance(postfix, str) and postfix:
        parts.append(postfix)

    # Make output folder
    output_path = os.path.join(output_dir, "_".join(parts)) if parts else output_dir
    output_path = safe_filename(output_path)
    os.makedirs(output_path, exist_ok=True)
    vprint(f"output_path = '{output_path}' is generated!", verbose=verbose)
    return output_path

def copy_params_to_dir(params_path, output_dir, params=None, verbose=True):
    """
    Copies the params file to the output directory if it exists. If the params file does not exist,
    it dumps the provided params dictionary to a YAML file in the output directory.

    Args:
        params_path (str): Path to the params file (can be None if params are programmatically generated).
        output_dir (str): Directory where the params file or YAML dump will be saved.
        params (dict, optional): The programmatically generated params dictionary to save if no file exists.
        verbose (bool): Whether to print verbose messages.
    """
    import os
    import shutil

    import yaml

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    if params_path and os.path.isfile(params_path):
        # If the params file exists, copy it to the output directory
        file_name = os.path.basename(params_path)
        output_path = os.path.join(output_dir, file_name)
        shutil.copy2(params_path, output_path)
        vprint(" ")
        vprint(f"### Successfully copy '{file_name}' to '{output_dir}' ###", verbose=verbose)

    elif params is not None:
        # If no file exists, dump the params dictionary to a YAML file
        output_path = os.path.join(output_dir, "params_dumped.yml")
        with open(output_path, "w") as f:
            yaml.safe_dump(params, f, sort_keys=False)
        vprint(" ")
        vprint(f"### No params file found. Dumped params dictionary to '{output_path}' ###")

    else:
        # If neither a file nor params are provided, skip with a warning
        vprint(" ")
        vprint("### Warning: No params file found and no params dictionary provided. Skipping. ###", verbose=verbose)

@torch.compiler.disable
def save_results(output_path, model, params, optimizer, niter, indices, batch_losses, collate_str=''):
    
    save_result_list = params['recon_params'].get('save_result', ['model', 'obj', 'probe'])
    result_modes = params['recon_params'].get('result_modes')
    iter_str = '_iter' + str(niter).zfill(4)
    
    if 'model' in save_result_list:
        save_dict = make_save_dict(output_path, model, params, optimizer, niter, indices, batch_losses)
        save_dict_to_hdf5(save_dict, safe_filename(os.path.join(output_path, f"model{collate_str}{iter_str}.hdf5")))
    probe      = model.get_complex_probe_view() 
    probe_amp  = probe.reshape(-1, probe.size(-1)).t().abs().detach().cpu().numpy()
    probe_prop = model.get_propagated_probe([0]).permute(0,2,1,3) # (Z, pmode, Y, X) -> (Z, Y, pmode, X)
    shape      = probe_prop.shape
    prop_p_amp = probe_prop.reshape(shape[0], shape[1], shape[2]*shape[3]).abs().detach().cpu().numpy()
    objp       = model.opt_objp.detach().cpu().numpy()
    obja       = model.opt_obja.detach().cpu().numpy()
    # omode_occu = model.omode_occu # Currently not used but we'll need it when omode_occu != 'uniform'
    omode      = model.opt_objp.size(0)
    zslice     = model.opt_objp.size(1)
    crop_pos   = model.crop_pos[indices].detach().cpu().numpy() + np.array(probe.shape[-2:])//2
    y_min, y_max = crop_pos[:,0].min(), crop_pos[:,0].max()
    x_min, x_max = crop_pos[:,1].min(), crop_pos[:,1].max()
    
    for bit in result_modes['bit']:
        if bit == '8':
            bit_str = '_08bit'
        elif bit == '16':
            bit_str = '_16bit'
        elif bit == '32':
            bit_str = '_32bit'
        elif bit == 'raw':
            bit_str = ''
        else:
            bit_str = ''
        if 'probe' in save_result_list:
            imwrite(safe_filename(os.path.join(output_path, f"probe_amp{bit_str}{collate_str}{iter_str}.tif")), normalize_by_bit_depth(probe_amp, bit))
        if 'probe_prop' in save_result_list:
            imwrite(safe_filename(os.path.join(output_path, f"probe_prop_amp{bit_str}{collate_str}{iter_str}.tif")), normalize_by_bit_depth(prop_p_amp, bit))
        for fov in result_modes['FOV']:
            if fov == 'crop':
                fov_str = '_crop'
                objp_crop = objp[:, :, y_min-1:y_max, x_min-1:x_max]
                obja_crop = obja[:, :, y_min-1:y_max, x_min-1:x_max]
            elif fov == 'full':
                fov_str = ''
                objp_crop = objp
                obja_crop = obja
            else:
                fov_str = ''
                objp_crop = objp
                obja_crop = obja
                
            postfix_str = fov_str + bit_str + collate_str + iter_str
                
            if any(keyword in save_result_list for keyword in ['obj', 'objp', 'object']):
                # TODO: For omode_occu != 'uniform', we should do a weighted sum across omode instead
                
                for dim in result_modes['obj_dim']:
                    
                    if omode == 1 and zslice == 1:
                        if dim == 2: 
                            imwrite(safe_filename(os.path.join(output_path, f"objp{postfix_str}.tif")),              normalize_by_bit_depth(objp_crop[0,0], bit))
                    elif omode == 1 and zslice > 1:
                        if dim == 3:
                            imwrite(safe_filename(os.path.join(output_path, f"objp_zstack{postfix_str}.tif")),       normalize_by_bit_depth(objp_crop[0,:], bit))
                        if dim == 2:
                            imwrite(safe_filename(os.path.join(output_path, f"objp_zsum{postfix_str}.tif")),         normalize_by_bit_depth(objp_crop[0,:].sum(0), bit))
                    elif omode > 1 and zslice == 1:
                        if dim == 3:
                            imwrite(safe_filename(os.path.join(output_path, f"objp_ostack{postfix_str}.tif")),       normalize_by_bit_depth(objp_crop[:,0], bit))
                        if dim == 2:
                            imwrite(safe_filename(os.path.join(output_path, f"objp_omean{postfix_str}.tif")),        normalize_by_bit_depth(objp_crop[:,0].mean(0), bit))
                            imwrite(safe_filename(os.path.join(output_path, f"objp_ostd{postfix_str}.tif")),         normalize_by_bit_depth(objp_crop[:,0].std(0), bit))
                    else:
                        if dim == 4:
                            imwrite(safe_filename(os.path.join(output_path, f"objp_4D{postfix_str}.tif")),           normalize_by_bit_depth(objp_crop[:,:], bit))
                        if dim == 3:
                            imwrite(safe_filename(os.path.join(output_path, f"objp_ostack_zsum{postfix_str}.tif")),  normalize_by_bit_depth(objp_crop[:,:].sum(1), bit))
                            imwrite(safe_filename(os.path.join(output_path, f"objp_omean_zstack{postfix_str}.tif")), normalize_by_bit_depth(objp_crop[:,:].mean(0), bit))
                        if dim == 2:
                            imwrite(safe_filename(os.path.join(output_path, f"objp_omean_zsum{postfix_str}.tif")),   normalize_by_bit_depth(objp_crop[:,:].mean(0).sum(0), bit))
                            
            if any(keyword in save_result_list for keyword in ['obja']):
                # TODO: For omode_occu != 'uniform', we should do a weighted sum across omode instead
                
                for dim in result_modes['obj_dim']:
                    
                    if omode == 1 and zslice == 1:
                        if dim == 2: 
                            imwrite(safe_filename(os.path.join(output_path, f"obja{postfix_str}.tif")),              normalize_by_bit_depth(obja_crop[0,0], bit))
                    elif omode == 1 and zslice > 1:
                        if dim == 3:
                            imwrite(safe_filename(os.path.join(output_path, f"obja_zstack{postfix_str}.tif")),       normalize_by_bit_depth(obja_crop[0,:], bit))
                        if dim == 2:
                            imwrite(safe_filename(os.path.join(output_path, f"obja_zmean{postfix_str}.tif")),         normalize_by_bit_depth(obja_crop[0,:].mean(0), bit))
                            imwrite(safe_filename(os.path.join(output_path, f"obja_zprod{postfix_str}.tif")),         normalize_by_bit_depth(obja_crop[0,:].prod(0), bit))
                    elif omode > 1 and zslice == 1:
                        if dim == 3:
                            imwrite(safe_filename(os.path.join(output_path, f"obja_ostack{postfix_str}.tif")),       normalize_by_bit_depth(obja_crop[:,0], bit))
                        if dim == 2:
                            imwrite(safe_filename(os.path.join(output_path, f"obja_omean{postfix_str}.tif")),        normalize_by_bit_depth(obja_crop[:,0].mean(0), bit))
                            imwrite(safe_filename(os.path.join(output_path, f"obja_ostd{postfix_str}.tif")),         normalize_by_bit_depth(obja_crop[:,0].std(0), bit))
                    else:
                        if dim == 4:
                            imwrite(safe_filename(os.path.join(output_path, f"obja_4D{postfix_str}.tif")),           normalize_by_bit_depth(obja_crop[:,:], bit))
                        if dim == 3:
                            imwrite(safe_filename(os.path.join(output_path, f"obja_ostack_zmean{postfix_str}.tif")),  normalize_by_bit_depth(obja_crop[:,:].mean(1), bit))
                            imwrite(safe_filename(os.path.join(output_path, f"obja_ostack_zprod{postfix_str}.tif")),  normalize_by_bit_depth(obja_crop[:,:].prod(1), bit))
                            imwrite(safe_filename(os.path.join(output_path, f"obja_omean_zstack{postfix_str}.tif")), normalize_by_bit_depth(obja_crop[:,:].mean(0), bit))
                        if dim == 2:
                            imwrite(safe_filename(os.path.join(output_path, f"obja_omean_zmean{postfix_str}.tif")),   normalize_by_bit_depth(obja_crop[:,:].mean(0).mean(0), bit))
                            imwrite(safe_filename(os.path.join(output_path, f"obja_omean_zprod{postfix_str}.tif")),   normalize_by_bit_depth(obja_crop[:,:].mean(0).prod(0), bit))