"""
Reconstruction workflow related functions

"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch

from ptyrad.io.adapter import ndarrays_to_tensors
from ptyrad.io.save import make_output_folder, safe_filename, save_results
from ptyrad.params.parser import copy_params_to_dir
from ptyrad.plotting.basic import plot_pos_grouping
from ptyrad.plotting.model import plot_summary
from ptyrad.runtime.logging import vprint
from ptyrad.runtime.seed import set_random_seed
from ptyrad.solver.grouping import (
    remap_batches_to_global,
    sparse_sampler_fps,
    sparse_sampler_hilbert,
)
from ptyrad.utils.image_proc import get_blob_size
from ptyrad.utils.time import parse_sec_to_time_str

# This suppresses the '..._inductor/compile_fx.py:236: UserWarning: TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled. 
# Consider setting `torch.set_float32_matmul_precision('high')` for better performance.'
# Although I didn't see much effect on performance because there's very little matrix multiplication in PtyRAD.
torch.set_float32_matmul_precision('high') 

# The actual performance is significantly better than 'eager' so I supress this for clarity
warnings.filterwarnings(
    "ignore",
    message="Torchinductor does not support code generation for complex operators. Performance may be worse than eager."
)

# This will show up torch.compile but it's harmless
warnings.filterwarnings("ignore", message=".*Profiler function.*will be ignored.*")

# This will show up with DDP via accelerate but this doesn't affect multi GPU
warnings.filterwarnings("ignore", message=".*No device id is provided.*")

# This will show up when multiGPU + compile but has no affect
warnings.filterwarnings("ignore", message=".*Dynamo does not know how to trace.*")

# ==============================================================================
# SECTION 1: SETUP & INITIALIZATION
# ==============================================================================

def create_optimizer(optimizer_params, optimizable_params, verbose=True):
    
    def _fix_optimizer_state_dict_format(optim_state_dict: dict) -> dict:
        """
        Fix HDF5-loaded optimizer state dict by:
        - Recovering integer keys (HDF5 forces strings as keys).
        - Converting param_groups from dicts back to list format, if needed.
        - Converting any remaining param indices to lists.

        Args:
            op_state_dict (dict): Loaded optimizer state dict (e.g. from HDF5).

        Returns:
            dict: Fixed optimizer state dict.
        """
        fixed = {}

        for key, val in optim_state_dict.items():
            # If the value is a dict (like 'state'), fix its integer keys
            if isinstance(val, dict):
                fixed_val = {}
                for nested_key, nested_val in val.items():
                    try:
                        fixed_nested_key = int(nested_key)  # Convert '0', '1' etc. to 0, 1
                    except (ValueError, TypeError):
                        fixed_nested_key = nested_key  # Keep string keys as-is
                    fixed_val[fixed_nested_key] = nested_val
                fixed[key] = fixed_val
            else:
                fixed[key] = val

        # Fix param_groups format if it was accidentally stored as a dict
        if isinstance(fixed.get("param_groups"), dict):
            param_groups_dict = fixed["param_groups"]
            # Convert {0: {...}, 1: {...}} -> [{...}, {...}]
            fixed["param_groups"] = [
                param_groups_dict[k] for k in sorted(param_groups_dict, key=lambda x: int(x))
            ]

        # Ensure 'params' field is a list of ints, not tensors or ndarrays
        for group in fixed.get("param_groups", []):
            if isinstance(group.get("params"), torch.Tensor):
                group["params"] = group["params"].tolist()
            elif isinstance(group.get("params"), np.ndarray):
                group["params"] = group["params"].tolist()

        return fixed
    
    # Extract the optimizer name and configs
    optimizer_name = optimizer_params['name']
    optimizer_configs = optimizer_params.get('configs') or {} # if "None" is provided or missing, it'll default an empty dict {}
    ptyrad_path = optimizer_params.get('load_state')
    
    vprint(f"### Creating PyTorch '{optimizer_name}' optimizer with configs = {optimizer_configs} ###", verbose=verbose)
    
    # Get the optimizer class from torch.optim
    optimizer_class = getattr(torch.optim, optimizer_name, None)
    
    if optimizer_class is None:
        raise ValueError(f"Optimizer '{optimizer_name}' is not supported.")
    if optimizer_name == 'LBFGS':
        vprint("Note: LBFGS optimizer is a quasi-Newton 2nd order optimizer that will run multiple forward passes (default: 20) for 1 update step")
        vprint("Note: LBFGS usually converges faster for convex problem with full-batch non-noisy gradients, but each update step is computationally slower")
        non_zero_lr = [p['lr'] for p in optimizable_params if p['lr'] != 0]
        optimizer_configs['lr'] = min(non_zero_lr)
        vprint(f"Note: LBFGS optimizer does not support per parameter learning rate so it'll be set to the minimal non-zero learning rate = {min(non_zero_lr)}")
        optimizable_params = [p['params'][0] for p in optimizable_params if p['params'][0].requires_grad] # LBFGS only takes 1 params group as an iterable

    optimizer = optimizer_class(optimizable_params, **optimizer_configs)
    device = optimizable_params[0]['params'][0].device
    
    if ptyrad_path is not None and isinstance(ptyrad_path, str):
        try:
            from ptyrad.io.load import load_ptyrad
            optim_state_dict = load_ptyrad(ptyrad_path)['optim_state_dict']
            optim_state_dict = _fix_optimizer_state_dict_format(optim_state_dict)
            # Convert 'state' to tensors on the right device, while 'param_groups' are kept as generic scalars/arrays/boolean/None/list of int
            optim_state_dict['state'] = ndarrays_to_tensors(optim_state_dict['state'], device=device) 
            optimizer.load_state_dict(optim_state_dict)
            vprint(f"Loaded optimizer state from '{ptyrad_path}'", verbose=verbose)
        except Exception as e:
            vprint(f"Failed to load optimizer state from '{ptyrad_path}': {e}. Using fresh optimizer.", verbose=verbose)
    vprint(" ", verbose=verbose)
    return optimizer

def prepare_recon(model, init, params):
    """
    Prepares the indices, batches, and output path for ptychographic reconstruction.

    This function parses the necessary parameters and generates the indices for scanning, 
    creates batches based on the probe positions, and sets up the output directory for 
    saving results. It also plots and saves a figure illustrating the grouping of probe 
    positions.

    Args:
        model (PtychoModel): The ptychographic model containing the object, probe, 
            probe positions, and other relevant parameters.
        init (Initializer): The initializer object containing the initialized variables 
            needed for reconstruction.
        params (dict): A dictionary containing various parameters needed for the 
            reconstruction process, including experimental parameters, loss parameters, 
            constraint parameters, and reconstruction settings.

    Returns:
        tuple: A tuple containing the following:
            - indices (numpy.ndarray): Array of indices for scanning positions.
            - batches (list of numpy.ndarray): List of batches where each batch contains 
              indices grouped according to the selected grouping mode.
            - output_path (str): The path to the directory where reconstruction results 
              and figures will be saved.
    """
    verbose = not params['recon_params']['if_quiet']
    vprint("### Generating indices, batches, and output_path ###", verbose=verbose)
    # Parse the variables
    init_variables = init.init_variables
    init_params = init.init_params # These could be modified by Optuna, hence can be different from params['init_params]
    params_path = params.get('params_path')
    loss_params = params.get('loss_params')
    constraint_params = params.get('constraint_params')
    recon_params = params.get('recon_params')
    INDICES_MODE = recon_params['INDICES_MODE'].get("mode")
    subscan_slow = recon_params['INDICES_MODE'].get("subscan_slow")
    subscan_fast = recon_params['INDICES_MODE'].get("subscan_fast")
    GROUP_MODE = recon_params['GROUP_MODE']
    SAVE_ITERS = recon_params['SAVE_ITERS']
    batch_size = recon_params['BATCH_SIZE'].get("size")
    grad_accumulation = recon_params['BATCH_SIZE'].get("grad_accumulation")
    output_dir = recon_params['output_dir']
    recon_dir_affixes = recon_params['recon_dir_affixes']
    copy_params = recon_params['copy_params']
    if_hypertune = params.get('hypertune_params', {}).get('if_hypertune', False)
    
    # Generate the indices, batches, and fig_grouping
    pos          = (model.crop_pos + model.opt_probe_pos_shifts).detach().cpu().numpy()
    probe_int    = model.get_complex_probe_view().abs().pow(2).sum(0).detach().cpu().numpy()
    dx           = init_variables['dx']
    d_out        = get_blob_size(dx, probe_int, output='d90', verbose=verbose) # d_out unit is in Ang
    indices      = select_scan_indices(init_variables['N_scan_slow'], init_variables['N_scan_fast'], subscan_slow=subscan_slow, subscan_fast=subscan_fast, mode=INDICES_MODE, verbose=verbose)
    batches      = make_batches(indices, pos, batch_size, mode=GROUP_MODE, seed=init_variables['random_seed'], verbose=verbose)
    fig_grouping = plot_pos_grouping(pos, batches, circle_diameter=d_out/dx, diameter_type='90%', dot_scale=1, show_fig=False, pass_fig=True)
    vprint(f"The effective batch size (i.e., how many probe positions are simultaneously used for 1 update of ptychographic parameters) is batch_size * grad_accumulation = {batch_size} * {grad_accumulation} = {batch_size*grad_accumulation}", verbose=verbose)

    # Create the output path, save fig_grouping, and copy params file
    if SAVE_ITERS is not None:
        output_path = make_output_folder(output_dir, indices, init_params, recon_params, model, constraint_params, loss_params, recon_dir_affixes, verbose=verbose)
        fig_grouping.savefig(safe_filename(output_path + "/summary_pos_grouping.png"))
        if copy_params and not if_hypertune:
            # Save params.yml to separate reconstruction folder for normal mode. Hypertune mode params copying is handled at hypertune()
            copy_params_to_dir(params_path, output_path, params)
    else:
        output_path = None
    
    plt.close(fig_grouping)
    vprint(" ", verbose=verbose)
    return indices, batches, output_path

def select_scan_indices(N_scan_slow, N_scan_fast, subscan_slow=None, subscan_fast=None, mode='full', verbose=True):
    
    N_scans = N_scan_slow * N_scan_fast
    vprint(f"Selecting indices with the '{mode}' mode ", verbose=verbose)
    # Generate flattened indices for the entire FOV
    if mode == 'full':
        indices = np.arange(N_scans)
        return indices

    # Set default values for subscan params
    if subscan_slow is None and subscan_fast is None:
        vprint("Subscan params are not provided, setting subscans to default as half of the total scan for both directions", verbose=verbose)
        subscan_slow = N_scan_slow//2
        subscan_fast = N_scan_fast//2
        
    # Generate flattened indices for the center rectangular region
    if mode == 'center':
        vprint(f"Choosing subscan with {(subscan_slow, subscan_fast)}", verbose=verbose) 
        start_row = (N_scan_slow - subscan_slow) // 2
        end_row = start_row + subscan_slow
        start_col = (N_scan_fast - subscan_fast) // 2
        end_col = start_col + subscan_fast
        indices = np.array([row * N_scan_fast + col for row in range(start_row, end_row) for col in range(start_col, end_col)])

    # Generate flattened indices for the entire FOV with sub-sampled indices
    elif mode == 'sub':
        vprint(f"Choosing subscan with {(subscan_slow, subscan_fast)}", verbose=verbose) 
        full_indices = np.arange(N_scans).reshape(N_scan_slow, N_scan_fast)
        subscan_slow_id = np.linspace(0, N_scan_slow-1, num=subscan_slow, dtype=int)
        subscan_fast_id = np.linspace(0, N_scan_fast-1, num=subscan_fast, dtype=int)
        slow_grid, fast_grid = np.meshgrid(subscan_slow_id, subscan_fast_id, indexing='ij')
        indices = full_indices[slow_grid, fast_grid].reshape(-1)

    else:
        raise ValueError(f"Indices selection mode {mode} not implemented, please use either 'full', 'center', or 'sub'")   
        
    return indices

def make_batches(indices, pos, batch_size, mode='random', seed=None, verbose=True):
    ''' Make batches from input indices '''
    # Input:
    #   indices: int, (Ns,) array. indices could be a subset of all indices.
    #   pos: int/float (N,2) array. Always pass in the full positions.
    #   batch_size: int. The number of indices of each mini-batch
    #   mode: str. Choose between 'random', 'compact', or 'sparse' grouping. Explicit sparse sampling methods 'hilbert' and 'fps' can be passed in as well.
    # Output:
    #   batches: A list of `num_batch` arrays, or [batch0, batch1, ...]
    # Note:
    #   The actual batch size would only be "close" if it's not divisible by len(indices) for 'random' grouping
    #   For 'compact' or 'sparse', it's generally fluctuating around the specified batch size
    #   'sparse' grouping can be relatively slow for large scan positions, hence 2 methods are provided in PtyRAD.
    #     - fps: Farthest Point Sampling gives highest quality hyperuniform points with O(N^2) complexity, 128x128 scan takes around 8 sec, 256x256 takes ~ 110 sec with batch size = 32.
    #     - hilbert: Hilbert curve sorting gives good quality hyperuniform points with O(N) complexity, 128x128 scan takes around 0.14 sec, 256x256 takes ~ 0.5 sec with batch size = 32.
    #   If 'sparse' is chosen, it will automatically select suitable methods based on len(indices)
    #   In PtychoShelves, MLs (sparse grouping) automatically switches to 'random' for len(pos) > 1e3 to reduce the processing time 

    from time import time
    
    try:
        from sklearn.cluster import MiniBatchKMeans
    except ImportError as e:
        missing_package = str(e).split()[-1]
        vprint(f"### {missing_package} is not available, group mode set to 'random'. 'scikit-learn' is needed for 'compact' grouping ###")
        mode = 'random'
        
    if len(indices) > len(pos):
        raise ValueError(f"len(indices) = '{len(indices)}' is larger than total number of probe positions ({len(pos)}), check your indices generation params")
    
    if indices.max() > len(pos):
        raise ValueError(f"Maximum index '{indices.max()}' is larger than total number of probe positions ({len(pos)}), check your indices generation params")

    num_batch = len(indices) // batch_size   
    t_start = time()

    # Choose grouping methods
    if mode == 'compact':
        pos_s = pos[indices] # Choose the selected pos from indices
        kmeans = MiniBatchKMeans(init="k-means++", n_init=10, n_clusters=num_batch, max_iter=10, batch_size=3072, random_state=seed) # Kmeans for clustering
        kmeans.fit(pos_s)
        labels = kmeans.labels_
        
        # Separate data points into groups
        output_batches = []
        for batch_idx in range(num_batch):
            batch_indices_s = np.where(labels == batch_idx)[0]
            output_batches.append(indices[batch_indices_s])
    
    elif mode in ['sparse', 'fps', 'hilbert']:
        pos_s = pos[indices] # Choose the selected pos from indices

        if mode == 'sparse':
            if len(indices) <= 65536:
                method = 'fps' # fps with 256x256 scan and batch size 32 takes ~ 120 sec on CPU, and scales as O(N^2)
            else:
                method = 'hilbert' # hilbert with 256x256 scan and batch size 32 takes ~ 0.50 sec on CPU, and scales as O(N)
            vprint(f"len(indices) = {len(indices)}, '{method}' is automatically selected for sparse grouping.")
        else:
            # explicit user request
            method = mode

        if method == 'fps':
            output_batches = sparse_sampler_fps(pos_s, num_batch, seed=seed)
        elif method == 'hilbert':
            output_batches = sparse_sampler_hilbert(pos_s, num_batch, resolution=16)
            
        if len(indices) < len(pos): # If a subset of indices was used (i.e., INDICES_MODE != 'full'), map the output indices back
            output_batches = remap_batches_to_global(output_batches, global_lookup=indices)
    
    else: # random
        rng = np.random.default_rng(seed=seed)
        shuffled_indices = rng.permutation(indices) # This will make a shuffled copy    
        random_batches = np.array_split(shuffled_indices, num_batch)
        output_batches = random_batches
    
    # Final check
    flatten_indices = np.concatenate(output_batches)
    flatten_indices.sort()
    indices.sort()
    assert all(flatten_indices == indices), f"Sorry, something went wrong with the '{mode}' grouping, please try 'random' instead"
    vprint(f"Generated {num_batch} '{mode}' groups of ~{batch_size} scan positions in {time() - t_start:.3f} sec", verbose=verbose)

    return output_batches

# ==============================================================================
#  SECTION 2: RECONSTRUCTION LOOP
# ==============================================================================

def recon_loop(model, init, params, optimizer, loss_fn, constraint_fn, indices, batches, output_path, acc=None):
    """
    Executes the iterative optimization loop for ptychographic reconstruction.

    This function performs the iterative reconstruction process by optimizing the model 
    parameters over a specified number of iterations. During each iteration, it applies 
    the loss and constraint functions, updates the model, and logs the loss values. 
    Intermediate results are saved at specified intervals, and a summary is plotted.

    Args:
        model (PtychoModel): The ptychographic model containing the parameters and variables 
            to be optimized.
        init (Initializer): The initializer object containing the initialized variables 
            needed for reconstruction.
        params (dict): A dictionary containing various parameters for the reconstruction 
            process, including experimental parameters, source parameters, loss parameters, 
            constraint parameters, and reconstruction settings.
        optimizer (torch.optim.Optimizer): The optimizer used to update the model parameters.
        loss_fn (CombinedLoss): The loss function object used to compute the loss during 
            each iteration.
        constraint_fn (CombinedConstraint): The constraint function object applied during 
            each iteration to enforce specific constraints on the model.
        indices (numpy.ndarray): Array of indices for scanning positions.
        batches (list of numpy.ndarray): List of batches where each batch contains indices 
            grouped according to the selected grouping mode.
        output_path (str): The path to the directory where reconstruction results and 
            figures will be saved.

    Returns:
        list: A list of tuples, where each tuple contains the iteration number, the loss 
            value for that iteration, and the time taken for that iteration.
    """
    
    # Parse the variables
    init_variables    = init.init_variables
    recon_params      = params.get('recon_params')
    NITER             = recon_params['NITER']
    SAVE_ITERS        = recon_params['SAVE_ITERS']
    grad_accumulation = recon_params['BATCH_SIZE'].get("grad_accumulation", 1)
    selected_figs     = recon_params['selected_figs']
    compiler_configs  = parse_torch_compile_configs(recon_params['compiler_configs'])
    verbose           = not recon_params['if_quiet']
    
    # Use the method on the wrapped model (DDP) if it exists
    model_instance = model.module if hasattr(model, "module") else model
    
    vprint("### Start the PtyRAD iterative ptycho reconstruction ###", verbose=verbose)
    
    # Optimization loop
    for niter in range(1,NITER+1):
        
        # Toggle the grad calculation to enable or disable AD update on tensors at certain iterations
        toggle_grad_requires(model_instance, niter, verbose)
        
        # Apply torch.compile to `recon_step``
        if niter in model_instance.compilation_iters: # compilation_iters always contain niter=1
            vprint(f"Setting up PyTorch compiler with {compiler_configs}", verbose=verbose)
            torch._dynamo.reset()
            recon_step_compiled = torch.compile(recon_step, **compiler_configs)
        
        batch_losses = recon_step_compiled(batches, grad_accumulation, model, optimizer, loss_fn, constraint_fn, niter, verbose=verbose, acc=acc)
        
        # Only log the main process
        if acc is None or acc.is_main_process:
            
            ## Saving intermediate results
            if SAVE_ITERS is not None and niter % SAVE_ITERS == 0:
                with torch.no_grad():
                    # Note that `params` stores the original params from the configuration file, 
                    # while `model` contains the actual params that could be updated by meas_crop, meas_pad, or meas_resample
                    save_results(output_path, model_instance, params, optimizer, niter, indices, batch_losses)
                    
                    ## Saving summary
                    plot_summary(output_path, model_instance, niter, indices, init_variables, selected_figs=selected_figs, show_fig=False, save_fig=True, verbose=verbose)
    
    vprint(f"### Finished {NITER} iterations, averaged iter_t = {np.mean(model_instance.iter_times):.5g} with std = {np.std(model_instance.iter_times):.3f} ###", verbose=verbose)
    vprint(" ", verbose=verbose)

def recon_step(batches, grad_accumulation, model, optimizer, loss_fn, constraint_fn, niter, verbose=True, acc=None):
    """
    Performs one iteration (or step) of the ptychographic reconstruction in the optimization loop.

    This function executes a single iteration of the reconstruction process, including:
    - Computing the forward model to generate diffraction patterns.
    - Calculating the loss by comparing the modeled and measured diffraction patterns.
    - Performing a backward pass to compute gradients and update the model parameters using the optimizer.
    - Applying iteration-wise constraints after all batches are processed.

    Args:
        batches (list of numpy.ndarray): List of batches where each batch contains indices 
            grouped according to the selected grouping mode.
        model (PtychoModel): The ptychographic model containing the parameters and variables 
            to be optimized.
        optimizer (torch.optim.Optimizer): The optimizer used to update the model parameters.
        loss_fn (CombinedLoss): The loss function object used to compute the loss for each batch.
        constraint_fn (CombinedConstraint): The constraint function object applied after each iteration 
            to enforce specific constraints on the model.
        niter (int): The current iteration number in the optimization loop.
        verbose (bool, optional): If True, prints progress information during the batch processing. 
            Defaults to True.

    Returns:
        tuple: A tuple containing:
            - batch_losses (dict): A dictionary where each key corresponds to a loss component name, 
              and the value is a list of loss values computed for each batch in the iteration.
            - iter_t (float): The total time taken to complete the iteration.
    """
    batch_losses = {name: [] for name in loss_fn.loss_params.keys()}
    start_iter_t = time_sync()
    
    # Use the method on the wrapped model (DDP) if it exists
    model_instance = model.module if hasattr(model, "module") else model
    
    # Run the iteration with closure for LBFGS optimizer
    if isinstance(optimizer, torch.optim.LBFGS):

        # Make nested list of batches for the closure with internal grad accumulation over mini-batches
        num_batch = len(batches)
        batch_indices = np.arange(num_batch)
        if model.random_seed is not None:
            set_random_seed(seed=model.random_seed + niter) # This ensures batch_indices is different for each iter in a reproducible way
        np.random.shuffle(batch_indices)
        accu_batch_indices = np.array_split(batch_indices,num_batch//grad_accumulation)
        
        def closure():
            optimizer.zero_grad()
            total_loss = 0
            # Run grad accumulation inside the closure for LBFGS, note that each closure is ideally 1 full iter with grad_accu
            for batch_idx in accu_batch_idx:
                batch = batches[batch_idx]
                model_DP = model(batch) # Forward pass is handled automatically by DDP, but methods/attributes should use the unwrapped model
                measured_DP = model_instance.get_measurements(batch)
                object_patches = model_instance._current_object_patches
                loss_batch, losses = loss_fn(model_DP, measured_DP, object_patches, model_instance.omode_occu)
                total_loss += loss_batch # LBFGS uses the returned loss to perform the line-search so it's better to return the loss that's associated to all the batches
            total_loss = total_loss / len(accu_batch_idx)
            acc.backward(total_loss) if acc is not None else total_loss.backward()
            return total_loss, losses
        
        # Iterate through all accumulated batches. accu_batches = [[batch1],[batch2],[batch3]...], batches = [[accu_batches1],[accu_batches2],[accu_batches3]...]
        for accu_batch_idx in accu_batch_indices:
            optimizer.step(lambda: closure()[0])
            
        # This extra evaluation on accumulated batches is just to get the `losses` for logging purpose
        _, losses = closure()
        optimizer.zero_grad()
        
        # Clear the model cache after the mini-batch
        model_instance.clear_cache()
        
        # Append losses and log batch progress
        if acc is not None:
            acc.wait_for_everyone()
        for loss_name, loss_value in zip(loss_fn.loss_params.keys(), losses):
            batch_losses[loss_name].append(loss_value.detach().cpu().numpy())
    
    # Start mini-batch optimization for all other optimizers doesn't require a closure
    else:
        optimizer.zero_grad() # Since PyTorch 2.0 the default behavior is set_to_none=True for performance https://github.com/pytorch/pytorch/issues/92656
        
        for batch_idx, batch in enumerate(batches):
            start_batch_t = time_sync()
            
            # Compute forward pass and loss (wrapped in autocast if accelerate is enabled)
            loss_batch, losses = compute_loss(batch, model, model_instance, loss_fn, acc)
            
            # Normalize the `loss_batch`` before populating the gradients
            # We only want to scale the `loss_batch` so the grad/update is scaled accordingly
            # while keeping `losses` to be batch-size-independent for logging purpose 
            loss_batch = loss_batch / grad_accumulation
                        
            # Perform backward pass
            acc.backward(loss_batch) if acc is not None else loss_batch.backward()
                
            # Perform the optimizer step when batch_idx + 1 is divisible by grad_accumulation or it's the last batch
            if (batch_idx + 1) % grad_accumulation == 0 or (batch_idx + 1) == len(batches):
                if acc is not None:
                    acc.wait_for_everyone()
                optimizer.step() 
                optimizer.zero_grad() 
            batch_t = time_sync() - start_batch_t
        
            # Clear the model cache after the mini-batch
            model_instance.clear_cache()
        
            # Append losses and log batch progress
            if acc is not None:
                acc.wait_for_everyone()
            for loss_name, loss_value in zip(loss_fn.loss_params.keys(), losses):
                batch_losses[loss_name].append(loss_value.detach().cpu().numpy())
            if batch_idx in np.linspace(0, len(batches)-1, num=6, dtype=int):
                vprint(f"Done batch {batch_idx+1} with {len(batch)} indices ({batch[:5].tolist()}...) in {batch_t:.3f} sec", verbose=verbose)
    
    constraint_fn(model_instance, niter)
    
    iter_t = time_sync() - start_iter_t
    model_instance.loss_iters.append((niter, loss_logger(batch_losses, niter, iter_t, verbose=verbose)))
    model_instance.iter_times.append(iter_t)
    model_instance.dz_iters.append((niter, model_instance.opt_slice_thickness.detach().cpu().numpy()))
    model_instance.avg_tilt_iters.append((niter, model_instance.opt_obj_tilts.detach().mean(0).cpu().numpy()))
    return batch_losses

# ==============================================================================
#  SECTION 3: HELPERS
# ==============================================================================

@torch.compiler.disable
def time_sync():
    # PyTorch doesn't have a direct exposed API to check the selected default device 
    # so we'll be checking these .is_available() just to prevent error.
    # Luckily these checks won't really affect the performance.
    
    from time import perf_counter
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    # Check if MPS (Metal Performance Shaders) is available (macOS only)
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()
    
    # Measure the time
    t = perf_counter()
    return t

def parse_torch_compile_configs(configs):
    """
    Convert user-facing CompilerConfigs to dict suitable for torch.compile
    
    Note:
        The params.yaml defines as 'enable': bool = False, 
        while torch.compile takes only 'disable': bool, so a conversion is needed.
    """
    if 'enable' in configs:
        configs['disable'] = not configs.pop('enable')
    return configs

def toggle_grad_requires(model, niter, verbose=True):
    """Toggle requires_grad based on start and end iteration for each optimizable tensor."""

    vprint(" ", verbose=verbose) # Empty line for the start of each iteration
    
    optimizable_tensors = model.optimizable_tensors
    for param_name in model.optimizable_tensors.keys():
        start_iter = model.start_iter.get(param_name)
        end_iter = model.end_iter.get(param_name)
        
        # Determine if gradients should be enabled
        grad_started = start_iter is not None and niter >= start_iter
        grad_ended = end_iter is not None and niter + 1 > end_iter # end_iter is exclusive
        requires_grad = grad_started and not grad_ended
        
        optimizable_tensors[param_name].requires_grad = requires_grad
        vprint(f"Iter: {niter}, {param_name}.requires_grad = {requires_grad}", verbose=verbose)

def compute_loss(batch, model, model_instance, loss_fn, acc=None):
    """Compute the model output and loss, with optional support for accelerate's autocast."""
    if acc is not None:
        with acc.autocast():
            model_DP = model(batch)
            measured_DP = model_instance.get_measurements(batch)
            object_patches = model_instance._current_object_patches
            loss_batch, losses = loss_fn(model_DP, measured_DP, object_patches, model_instance.omode_occu)
    else:
        model_DP = model(batch)
        measured_DP = model_instance.get_measurements(batch)
        object_patches = model_instance._current_object_patches
        loss_batch, losses = loss_fn(model_DP, measured_DP, object_patches, model_instance.omode_occu)
    
    return loss_batch, losses

@torch.compiler.disable
def loss_logger(batch_losses, niter, iter_t, verbose=True):
    """
    Logs and summarizes the loss values for an iteration during the ptychographic reconstruction.

    This function computes the average loss for each loss component across all batches in the 
    current iteration. It then logs the total loss, the individual loss components, and the 
    time taken for the iteration. The function also returns the total loss for the iteration.

    Args:
        batch_losses (dict): A dictionary where each key corresponds to a loss component name, 
            and the value is a list of loss values computed for each batch in the iteration.
        niter (int): The current iteration number in the optimization loop.
        iter_t (float): The total time taken to complete the iteration, in seconds.
        verbose (bool, optional): If True, prints the loss summary to the console. Defaults to True.

    Returns:
        float: The total loss for the current iteration, computed as the sum of the average 
        loss values for each component.
    """
    avg_losses = {name: np.mean(values) for name, values in batch_losses.items()}
    loss_str = ', '.join([f"{name}: {value:.4f}" for name, value in avg_losses.items()])
    vprint(f"Iter: {niter}, Total Loss: {sum(avg_losses.values()):.4f}, {loss_str}, in {parse_sec_to_time_str(iter_t)}", verbose=verbose)
    loss_iter = sum(avg_losses.values())
    return loss_iter
