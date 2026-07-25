"""
Reconstruction workflow related functions

"""
import logging
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch

from ptyrad.io.adapter import ndarrays_to_tensors
from ptyrad.io.save import make_output_folder, safe_filename, save_results
from ptyrad.params.parser import copy_params_to_dir
from ptyrad.plotting.basic import plot_pos_grouping
from ptyrad.plotting.model import plot_summary
from ptyrad.runtime.convergence import create_convergence_monitor
from ptyrad.runtime.jit import resolve_jit_enable
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

logger = logging.getLogger(__name__)

# ==============================================================================
# SECTION 1: SETUP & INITIALIZATION
# ==============================================================================

def create_optimizer(optimizer_params, optimizable_params):
    
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
    
    device = optimizable_params[0]['params'][0].device

    logger.info(f"### Creating PyTorch '{optimizer_name}' optimizer with configs = {optimizer_configs} ###")
    
    # Get the optimizer class from torch.optim
    optimizer_class = getattr(torch.optim, optimizer_name, None)
    
    if optimizer_class is None:
        raise ValueError(f"Optimizer '{optimizer_name}' is not supported.")
    if optimizer_name == 'LBFGS':
        logger.info("Note: LBFGS optimizer is a quasi-Newton 2nd order optimizer that will run multiple forward passes (default: 20) for 1 update step")
        logger.info("Note: LBFGS usually converges faster for convex problem with full-batch non-noisy gradients, but each update step is computationally slower")
        non_zero_lr = [p['lr'] for p in optimizable_params if p['lr'] != 0]
        optimizer_configs['lr'] = min(non_zero_lr)
        logger.info(f"Note: LBFGS optimizer does not support per parameter learning rate so it'll be set to the minimal non-zero learning rate = {min(non_zero_lr)}")
        optimizable_params = [p['params'][0] for p in optimizable_params if p['params'][0].requires_grad] # LBFGS only takes 1 params group as an iterable

    optimizer = optimizer_class(optimizable_params, **optimizer_configs)
    
    if ptyrad_path is not None and isinstance(ptyrad_path, str):
        try:
            from ptyrad.io.load import load_ptyrad
            optim_state_dict = load_ptyrad(ptyrad_path)['optim_state_dict']
            optim_state_dict = _fix_optimizer_state_dict_format(optim_state_dict)
            # Convert 'state' to tensors on the right device, while 'param_groups' are kept as generic scalars/arrays/boolean/None/list of int
            optim_state_dict['state'] = ndarrays_to_tensors(optim_state_dict['state'], device=device) 
            optimizer.load_state_dict(optim_state_dict)
            logger.info(f"Loaded optimizer state from '{ptyrad_path}'")
        except Exception as e:
            logger.info(f"Failed to load optimizer state from '{ptyrad_path}': {e}. Using fresh optimizer.")
    logger.info(" ")
    return optimizer

def create_scheduler(scheduler_params, optimizer):
    """
    Creates a PyTorch LR scheduler from the given params and binds it to the optimizer.

    Converts param-group LRs (and betas[0] for OneCycleLR/CyclicLR) to CPU tensors so that
    torch.compile guards on tensor identity rather than scalar value, preventing recompilation
    when LR or momentum changes each step. Patches scheduler.step() to restore tensor identity
    in-place after each call. Optionally loads a previously saved scheduler state.

    Args:
        scheduler_params (dict or None): Scheduler configuration dict with keys 'name',
            'configs', and 'load_state'. If None, returns None.
        optimizer (torch.optim.Optimizer): The optimizer whose param-group LRs will be managed.

    Returns:
        torch.optim.lr_scheduler.LRScheduler or None: The configured scheduler, or None.
    """
    if scheduler_params is None:
        return None

    scheduler_name = scheduler_params['name']
    scheduler_configs = scheduler_params.get('configs') or {}
    ptyrad_path = scheduler_params.get('load_state')

    scheduler_class = getattr(torch.optim.lr_scheduler, scheduler_name, None)
    if scheduler_class is None:
        raise ValueError(f"LR Scheduler '{scheduler_name}' is not found in torch.optim.lr_scheduler.")


    step_unit = scheduler_params.get('step_unit', 'iter')
    if step_unit not in ('iter', 'batch'):
        raise ValueError(f"scheduler_params.step_unit must be 'iter' or 'batch', got '{step_unit!r}'")

    logger.info(f"### Creating LR scheduler '{scheduler_name}' with step_unit = {step_unit}, configs = {scheduler_configs} ###")

    # Warn at creation time (once) for known step_unit mismatches, rather than per iteration.
    if scheduler_name in ('CyclicLR', 'OneCycleLR') and step_unit == 'iter':
        logger.warning(
            f"WARNING: '{scheduler_name}' is designed to step after every optimizer update, but "
            f"step_unit='iter' will step it only once per outer iteration. "
            f"Set step_unit: batch in scheduler_params for correct behaviour."
        )
    elif scheduler_name == 'ReduceLROnPlateau' and step_unit == 'batch':
        logger.warning(
            "WARNING: ReduceLROnPlateau requires a full-iteration loss metric and does not support "
            "step_unit='batch'. It will step per iteration regardless."
        )

    # Convert param_group LRs to CPU tensors before creating the scheduler so that:
    # (1) scheduler.base_lrs captures tensors (arithmetic stays in tensor-land), and
    # (2) torch.compile guards on tensor identity rather than scalar value → no recompilation
    #     when the LR changes each iteration (see PyTorch lr_scheduler + compile tutorial).
    for group in optimizer.param_groups:
        group['lr'] = torch.tensor(float(group['lr']))

    # OneCycleLR and CyclicLR also cycle momentum (betas[0]) alongside LR. Without the same
    # tensor-identity treatment, Dynamo guards on the changing scalar value of betas[0] and
    # exhausts recompile_limit within the first iteration.
    _cycle_momentum = (
        scheduler_name in ('OneCycleLR', 'CyclicLR') and
        scheduler_configs.get('cycle_momentum', True) and
        all('betas' in group for group in optimizer.param_groups)
    )

    scheduler = scheduler_class(optimizer, **scheduler_configs)

    if ptyrad_path is not None and isinstance(ptyrad_path, str):
        try:
            from ptyrad.io.load import load_ptyrad
            scheduler_state_dict = load_ptyrad(ptyrad_path)['scheduler_state_dict']
            scheduler.load_state_dict(scheduler_state_dict)
            logger.info(f"Loaded scheduler state from '{ptyrad_path}'")
        except Exception as e:
            logger.info(f"Failed to load scheduler state from '{ptyrad_path}': {e}. Using fresh scheduler.")

    # Patch scheduler.step to always restore the original tensor objects in-place.
    # Standard schedulers do `param_group['lr'] = new_float` (and OneCycleLR/CyclicLR also do
    # `param_group['betas'] = (new_float, beta2)`), which replaces the tensors with floats and
    # breaks Dynamo's tensor-identity guard on the compiled optimizer step. This wrapper
    # re-fills the original tensors and puts them back after each step.
    #
    # betas[0] is converted AFTER scheduler creation because OneCycleLR/CyclicLR reset
    # group['betas'][0] in __init__ to the schedule's starting value, overwriting any tensor
    # set before construction. Converting after ensures _beta1_tensors captures a real tensor.
    _lr_tensors = [group['lr'] for group in optimizer.param_groups]
    if _cycle_momentum:
        for group in optimizer.param_groups:
            b1, b2 = group['betas']
            group['betas'] = (torch.tensor(float(b1)), b2)
    _beta1_tensors = [group['betas'][0] for group in optimizer.param_groups] if _cycle_momentum else []
    _original_step = scheduler.step

    def _inplace_step(*args, **kwargs):
        _original_step(*args, **kwargs)
        for tensor, group in zip(_lr_tensors, optimizer.param_groups):
            new_lr = group['lr']
            if new_lr is not tensor:
                tensor.fill_(new_lr.item() if isinstance(new_lr, torch.Tensor) else float(new_lr))
                group['lr'] = tensor
        for beta1_tensor, group in zip(_beta1_tensors, optimizer.param_groups):
            new_beta1 = group['betas'][0]
            if new_beta1 is not beta1_tensor:
                beta1_tensor.fill_(new_beta1.item() if isinstance(new_beta1, torch.Tensor) else float(new_beta1))
                group['betas'] = (beta1_tensor, group['betas'][1])

    scheduler.step = _inplace_step

    logger.info(" ")
    return scheduler

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
    logger.info("### Generating indices, batches, and output_path ###")
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
    d_out        = get_blob_size(dx, probe_int, output='d90') # d_out unit is in Ang
    indices      = select_scan_indices(init_variables['N_scan_slow'], init_variables['N_scan_fast'], subscan_slow=subscan_slow, subscan_fast=subscan_fast, mode=INDICES_MODE)
    batches      = make_batches(indices, pos, batch_size, mode=GROUP_MODE, seed=init_variables['random_seed'])
    fig_grouping = plot_pos_grouping(pos, batches, circle_diameter=d_out/dx, diameter_type='90%', dot_scale=1, show_fig=False, pass_fig=True)
    logger.info(f"The effective batch size (i.e., how many probe positions are simultaneously used for 1 update of ptychographic parameters) is batch_size * grad_accumulation = {batch_size} * {grad_accumulation} = {batch_size*grad_accumulation}")

    # Create the output path, save fig_grouping, and copy params file
    if SAVE_ITERS is not None:
        output_path = make_output_folder(output_dir, indices, init_params, recon_params, model, constraint_params, loss_params, recon_dir_affixes)
        fig_grouping.savefig(safe_filename(output_path + "/summary_pos_grouping.png"))
        if copy_params and not if_hypertune:
            # Save params.yaml to separate reconstruction folder for normal mode. Hypertune mode params copying is handled at hypertune()
            copy_params_to_dir(params_path, output_path, params)
    else:
        output_path = None
    
    plt.close(fig_grouping)
    logger.info(" ")
    return indices, batches, output_path

def select_scan_indices(N_scan_slow, N_scan_fast, subscan_slow=None, subscan_fast=None, mode='full'):
    
    N_scans = N_scan_slow * N_scan_fast
    logger.info(f"Selecting indices with the '{mode}' mode ")
    # Generate flattened indices for the entire FOV
    if mode == 'full':
        indices = np.arange(N_scans)
        return indices

    # Set default values for subscan params
    if subscan_slow is None and subscan_fast is None:
        logger.info("Subscan params are not provided, setting subscans to default as half of the total scan for both directions")
        subscan_slow = N_scan_slow//2
        subscan_fast = N_scan_fast//2
        
    # Generate flattened indices for the center rectangular region
    if mode == 'center':
        logger.info(f"Choosing subscan with {(subscan_slow, subscan_fast)}") 
        start_row = (N_scan_slow - subscan_slow) // 2
        end_row = start_row + subscan_slow
        start_col = (N_scan_fast - subscan_fast) // 2
        end_col = start_col + subscan_fast
        indices = np.array([row * N_scan_fast + col for row in range(start_row, end_row) for col in range(start_col, end_col)])

    # Generate flattened indices for the entire FOV with sub-sampled indices
    elif mode == 'sub':
        logger.info(f"Choosing subscan with {(subscan_slow, subscan_fast)}") 
        full_indices = np.arange(N_scans).reshape(N_scan_slow, N_scan_fast)
        subscan_slow_id = np.linspace(0, N_scan_slow-1, num=subscan_slow, dtype=int)
        subscan_fast_id = np.linspace(0, N_scan_fast-1, num=subscan_fast, dtype=int)
        slow_grid, fast_grid = np.meshgrid(subscan_slow_id, subscan_fast_id, indexing='ij')
        indices = full_indices[slow_grid, fast_grid].reshape(-1)

    else:
        raise ValueError(f"Indices selection mode {mode} not implemented, please use either 'full', 'center', or 'sub'")   
        
    return indices

def make_batches(indices, pos, batch_size, mode='random', seed=None):
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
        logger.info(f"### {missing_package} is not available, group mode set to 'random'. 'scikit-learn' is needed for 'compact' grouping ###")
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
            logger.info(f"len(indices) = {len(indices)}, '{method}' is automatically selected for sparse grouping.")
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
    logger.info(f"Generated {num_batch} '{mode}' groups of ~{batch_size} scan positions in {time() - t_start:.3f} sec")

    return output_batches

# ==============================================================================
#  SECTION 2: RECONSTRUCTION LOOP
# ==============================================================================

def recon_loop(model, init, params, optimizer, scheduler, loss_fn, constraint_fn, indices, batches, output_path, acc=None):
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
        scheduler (torch.optim.lr_scheduler.LRScheduler or None): The LR scheduler that
            adjusts the optimizer's learning rate each iteration. Pass None to disable.
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

    # Use the method on the wrapped model (DDP) if it exists
    model_instance = model.module if hasattr(model, "module") else model

    # Resolve the JIT (torch.compile) configs right before the reconstruction loop.
    # With the default 'enable': 'auto', this detects whether JIT compilation is achievable
    # on this machine and falls back to eager mode if it isn't.
    compiler_configs  = parse_torch_compile_configs(recon_params['compiler_configs'], device=getattr(model_instance, 'device', None))
    use_jit_compile   = not compiler_configs.get('disable', False)

    # Check on DDP and compiler_configs['fullgraph']
    is_distributed = (acc.num_processes > 1) if acc else isinstance(model, torch.nn.parallel.DistributedDataParallel)
    if is_distributed and compiler_configs.get('fullgraph', False):
        logger.warning("WARNING: DDP Multi-GPU detected, forcefully setting 'fullgraph=False' because DDP requires intentional graph breaks.")
        compiler_configs['fullgraph'] = False
    
    # Duplicate a compiler configs for optimizer because optimizer.step can't be compiled with fullgraph=True by design
    optim_compiler_configs = compiler_configs.copy()
    optim_compiler_configs['fullgraph'] = False

    scheduler_step_unit = (params.get('model_params', {}).get('scheduler_params') or {}).get('step_unit', 'iter')

    convergence_monitor_params = recon_params.get('convergence_monitor')
    monitor = create_convergence_monitor(convergence_monitor_params, model_instance)

    logger.info("### Start the PtyRAD iterative ptycho reconstruction ###")

    # Initialize the compute_loss_fn and optimizer.step
    compute_loss_fn = compute_loss
    optim_step_fn = optimizer.step

    # Optimization loop
    for niter in range(1,NITER+1):
        
        # Toggle the grad calculation to enable or disable AD update on tensors at certain iterations
        toggle_grad_requires(model_instance, niter)

        # Apply torch.compile
        if use_jit_compile and niter in model_instance.compilation_iters: # compilation_iters always contain niter=1
            logger.info(f"Setting up PyTorch compiler with {compiler_configs}")
            torch._dynamo.reset()
            compute_loss_fn = torch.compile(compute_loss, **compiler_configs)
            
            if not isinstance(optimizer, torch.optim.LBFGS): # Only compile first-order optimizers (like Adam), L-BFGS relies on dynamic closures that cannot be safely traced.
                optimizer.step = torch.compile(optim_step_fn, **optim_compiler_configs)
        
        batch_losses = recon_step(batches, grad_accumulation, model, optimizer, scheduler, loss_fn, constraint_fn, niter, acc=acc, compute_loss_fn=compute_loss_fn, scheduler_step_unit=scheduler_step_unit)

        # Only log the main process
        if acc is None or acc.is_main_process:

            ## Convergence monitoring snapshot
            if monitor is not None and monitor.should_step(niter, SAVE_ITERS):
                with torch.no_grad():
                    monitor.step(model_instance, niter)

            ## Saving intermediate results
            if SAVE_ITERS is not None and niter % SAVE_ITERS == 0:
                with torch.no_grad():
                    # Note that `params` stores the original params from the configuration file, 
                    # while `model` contains the actual params that could be updated by meas_crop, meas_pad, or meas_resample
                    save_results(output_path, model_instance, params, optimizer, scheduler, niter, indices, batch_losses)
                    
                    ## Saving summary
                    plot_summary(output_path, model_instance, niter, indices, init_variables, selected_figs=selected_figs, show_fig=False, save_fig=True)
    
    logger.info(f"### Finished {NITER} iterations, averaged iter_t = {np.mean(model_instance.iter_times):.5g} with std = {np.std(model_instance.iter_times):.3f} ###")
    logger.info(" ")

def recon_step(batches, grad_accumulation, model, optimizer, scheduler, loss_fn, constraint_fn, niter, acc=None, compute_loss_fn=None, scheduler_step_unit: str = "iter"):
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
        grad_accumulation (int): Number of mini-batches to accumulate gradients over before stepping.
        model (PtychoModel): The ptychographic model containing the parameters and variables
            to be optimized.
        optimizer (torch.optim.Optimizer): The optimizer used to update the model parameters.
        scheduler (torch.optim.lr_scheduler.LRScheduler or None): LR scheduler stepped according to
            scheduler_step_unit after the optimizer step. Pass None to disable.
        loss_fn (CombinedLoss): The loss function object used to compute the loss for each batch.
        constraint_fn (CombinedConstraint): The constraint function object applied after each iteration
            to enforce specific constraints on the model.
        niter (int): The current iteration number in the optimization loop.
        acc (Accelerator or None): HuggingFace Accelerator for multi-GPU support. None for single-GPU.
        compute_loss_fn (callable or None): Compiled or uncompiled loss function; defaults to compute_loss.
        scheduler_step_unit (str): 'iter' to step the scheduler once per outer iteration (default),
            or 'batch' to step after every optimizer.step() call (grad-accumulation boundary).
            ReduceLROnPlateau always steps per iteration regardless of this setting.

    Returns:
        dict: A dictionary where each key corresponds to a loss component name and the value is
            a list of loss values computed for each batch in the iteration.
    """
    loss_names = list(loss_fn.loss_params.keys())

    # Use the method on the wrapped model (DDP) if it exists
    model_instance = model.module if hasattr(model, "module") else model

    # Unwrap accelerate-prepared optimizer to access param_groups for LR tracking
    optimizer_instance = optimizer.optimizer if hasattr(optimizer, "optimizer") else optimizer

    # Build param group name list for LR logging (non-zero-lr params, same order as optimizer.param_groups)
    param_group_names = [name for name, lr in model_instance.lr_params.items() if lr != 0]

    # Pre-allocate GPU tensors for loss accumulation to avoid per-batch CPU syncs
    num_batches = len(batches)
    loss_accum = {name: torch.zeros(num_batches, device=model_instance.device) for name in loss_names}
    batch_count = 0  # Track actual number of recorded batches

    # Record the LR at the start of this iteration, before any scheduler.step() calls, so that
    # lr_iters[N] always means "LR used during iteration N" regardless of step_unit.
    current_lrs = [group['lr'] for group in optimizer_instance.param_groups]
    model_instance.lr_iters['niter'].append(niter)
    for name, lr in zip(param_group_names, current_lrs):
        model_instance.lr_iters[name].append(lr.item() if isinstance(lr, torch.Tensor) else lr)
    if scheduler is not None:
        lr_str = ', '.join([f"{name}: {float(lr):.3e}" for name, lr in zip(param_group_names, current_lrs)])
        logger.debug(f"Iter: {niter}, LR: {lr_str}")

    start_iter_t = time_sync(model_instance.device)
    
    # Run the iteration with closure for LBFGS optimizer
    if isinstance(optimizer, torch.optim.LBFGS):

        # Make nested list of batches for the closure with internal grad accumulation over mini-batches
        num_batch = len(batches)
        batch_indices = np.arange(num_batch)
        if model.random_seed is not None:
            set_random_seed(seed=model.random_seed + niter) # This ensures batch_indices is different for each iter in a reproducible way
        np.random.shuffle(batch_indices)
        accu_batch_indices = np.array_split(batch_indices, max(1, num_batch//grad_accumulation))
        
        # Iterate through all accumulated batches. accu_batches = [[batch1],[batch2],[batch3]...], batches = [[accu_batches1],[accu_batches2],[accu_batches3]...]
        for accu_batch_idx in accu_batch_indices:
            
            # Define the closure INSIDE the loop to safely capture the current accu_batch_idx
            def closure():
                optimizer.zero_grad()
                total_loss = 0
                
                # Run grad accumulation inside the closure for LBFGS, note that each closure is ideally 1 full iter with grad_accu
                for batch_idx in accu_batch_idx:
                    batch = batches[batch_idx]
                    
                    measured_DP = model_instance.get_measurements(batch)
                    loss_batch, _ = compute_loss_fn(batch, model, model_instance, measured_DP, loss_fn) # Forward pass is handled automatically by DDP, but methods/attributes should use the unwrapped model

                    total_loss += loss_batch # LBFGS uses the returned loss to perform the line-search so it's better to return the loss that's associated to all the batches
                    
                total_loss = total_loss / len(accu_batch_idx)
                acc.backward(total_loss) if acc is not None else total_loss.backward()
                
                return total_loss
            
            # Execute the L-BFGS step (which will call the closure multiple times for line-searches)
            optimizer.step(closure)
        
        # This extra evaluation on accumulated batches is just to get the `losses` for logging purpose
        with torch.no_grad():
            total_loss = 0.0
            losses = None

            for batch_idx in accu_batch_idx:
                batch = batches[batch_idx]

                measured_DP = model_instance.get_measurements(batch)
                loss_batch, losses = compute_loss_fn(batch, model, model_instance, measured_DP, loss_fn)

                total_loss += loss_batch

            total_loss = total_loss / len(accu_batch_idx)

        # Clear any stale gradients
        optimizer.zero_grad()
        
        # Clear the model cache after the mini-batch
        model_instance.clear_cache()
        
        # Record losses on GPU without CPU sync
        if acc is not None:
            acc.wait_for_everyone()
        for loss_name, loss_value in zip(loss_names, losses):
            loss_accum[loss_name][batch_count] = loss_value.detach()
        batch_count += 1

    # Start mini-batch optimization for all other optimizers doesn't require a closure
    else:
        optimizer.zero_grad() # Since PyTorch 2.0 the default behavior is set_to_none=True for performance https://github.com/pytorch/pytorch/issues/92656
        
        for batch_idx, batch in enumerate(batches):
            
            torch.compiler.cudagraph_mark_step_begin() # Marks the start of a new compiled step to prevent CUDA Graph memory overwrite errors.
            
            measured_DP = model_instance.get_measurements(batch)
            
            # Compute forward pass and loss
            loss_batch, losses = compute_loss_fn(batch, model, model_instance, measured_DP, loss_fn)
            
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
                if scheduler is not None and scheduler_step_unit == "batch" \
                        and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step()

            # Clear the model cache after the mini-batch
            model_instance.clear_cache()
        
            # Record losses on GPU without CPU sync
            if acc is not None:
                acc.wait_for_everyone()
            for loss_name, loss_value in zip(loss_names, losses):
                loss_accum[loss_name][batch_count] = loss_value.detach()
            batch_count += 1

    # Transfer all losses to CPU once at end of iteration
    batch_losses = {name: tensor[:batch_count].cpu().numpy().tolist() for name, tensor in loss_accum.items()}

    # Apply iter-wise constraint
    constraint_fn(model_instance, niter)

    # Step the LR scheduler
    if scheduler is not None:
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            loss_iter = float(sum([np.mean(v) for v in batch_losses.values()]))
            scheduler.step(loss_iter)
        elif scheduler_step_unit == "iter":
            scheduler.step()
        # else: step_unit == "batch" — already stepped inside the mini-batch loop above
    
    iter_t = time_sync(model_instance.device) - start_iter_t
    model_instance.loss_iters.append((niter, loss_logger(batch_losses, niter, iter_t)))
    model_instance.iter_times.append(iter_t)
    model_instance.dz_iters.append((niter, model_instance.opt_slice_thickness.detach().cpu().numpy()))
    avg_tilts = model_instance.opt_obj_tilts.detach().mean(0).cpu().numpy()
    model_instance.avg_tilt_iters['niter'].append(niter)
    model_instance.avg_tilt_iters['tilt_y'].append(float(avg_tilts[0]))
    model_instance.avg_tilt_iters['tilt_x'].append(float(avg_tilts[1]))
    return batch_losses

# ==============================================================================
#  SECTION 3: HELPERS
# ==============================================================================

def time_sync(device=None):
    # PyTorch doesn't have a direct exposed API to check the selected default device 
    # so we'll be checking these .is_available() just to prevent error.
    # Luckily these checks won't really affect the performance.
    
    from time import perf_counter

    device_type = None
    if device is not None:
        try:
            device_type = torch.device(device).type
        except (TypeError, RuntimeError):
            device_type = str(device).split(":", 1)[0]
    
    # Check if CUDA is available
    if torch.cuda.is_available() and device_type in (None, "cuda"):
        torch.cuda.synchronize(device)
    # Check if MPS (Metal Performance Shaders) is available (macOS only)
    elif torch.backends.mps.is_available() and device_type in (None, "mps"):
        torch.mps.synchronize() # As of pytorch 2.10, torch.mps.synchronize doesn't take any arg
    # TODO: Refactor with torch.accelerator.synchronize() once min pytorch is bumped to >=2.7
    #       torch.accelerator provides a unified API across CUDA, MPS, and future accelerators.
    
    # Measure the time
    t = perf_counter()
    return t

def parse_torch_compile_configs(configs, device=None):
    """
    Convert user-facing CompilerConfigs to dict suitable for torch.compile

    Note:
        The params.yaml defines 'enable' as either 'auto' (default), true, or false,
        while torch.compile takes only 'disable': bool, so a conversion is needed.
        When 'enable' is 'auto', the JIT capability of the current machine is detected
        (see `ptyrad.runtime.jit`) and JIT is silently skipped if it's not achievable.

    Args:
        configs (dict or None): The user-facing `compiler_configs` dict.
        device (torch.device or str or None, optional): The device the reconstruction runs on,
            used for the JIT capability detection. None infers it from available accelerators.

    Returns:
        dict: A copy of `configs` with 'enable' replaced by the resolved 'disable' flag.
    """
    configs = dict(configs or {}) # Copy so the user-facing params dict is not mutated
    configs['disable'] = not resolve_jit_enable(configs, device=device)
    for key in ('enable', 'auto_probe'): # PtyRAD-only keys that torch.compile doesn't take
        configs.pop(key, None)
    return configs

def toggle_grad_requires(model, niter):
    """Toggle requires_grad based on start and end iteration for each optimizable tensor."""

    logger.debug(" ") # Empty line for the start of each iteration
    
    optimizable_tensors = model.optimizable_tensors
    for param_name in model.optimizable_tensors.keys():
        start_iter = model.start_iter.get(param_name)
        end_iter = model.end_iter.get(param_name)
        
        # Determine if gradients should be enabled
        grad_started = start_iter is not None and niter >= start_iter
        grad_ended = end_iter is not None and niter + 1 > end_iter # end_iter is exclusive
        requires_grad = grad_started and not grad_ended
        
        optimizable_tensors[param_name].requires_grad = requires_grad
        logger.debug(f"Iter: {niter}, {param_name}.requires_grad = {requires_grad}")

def compute_loss(batch, model, model_instance, measured_DP, loss_fn):
    """Compute the model output and loss."""
    
    model_DP = model(batch)
    object_patches = model_instance._current_object_patches
    loss_batch, losses = loss_fn(model_DP, measured_DP, object_patches[0], object_patches[1], model_instance.omode_occu)
   
    return loss_batch, losses

def loss_logger(batch_losses, niter, iter_t):
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

    Returns:
        float: The total loss for the current iteration, computed as the sum of the average 
        loss values for each component.
    """
    avg_losses = {name: np.mean(values) for name, values in batch_losses.items()}
    loss_str = ', '.join([f"{name}: {value:.4f}" for name, value in avg_losses.items()])
    logger.info(f"Iter: {niter}, Total Loss: {sum(avg_losses.values()):.4f}, {loss_str}, in {parse_sec_to_time_str(iter_t)}")
    loss_iter = sum(avg_losses.values())
    return loss_iter
