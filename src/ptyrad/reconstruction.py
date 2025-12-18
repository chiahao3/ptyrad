"""
Reconstruction and hypertune workflows for ptychographic reconstructions

"""

from copy import deepcopy
import logging
from random import shuffle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import numba
import torch
import torch.distributed as dist
from torch.utils.data import Dataset

from ptyrad.constraints import CombinedConstraint
from ptyrad.initialization import Initializer
from ptyrad.losses import CombinedLoss, get_objp_contrast
from ptyrad.models import PtychoAD
from ptyrad.save import copy_params_to_dir, make_output_folder, save_results
from ptyrad.utils import (
    get_blob_size,
    get_time,
    ndarrays_to_tensors,
    parse_hypertune_params_to_str,
    parse_sec_to_time_str,
    safe_filename,
    set_random_seed,
    time_sync,
    vprint,
)
from ptyrad.visualization import plot_pos_grouping, plot_summary

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


class PtyRADSolver(object):
    """
    A wrapper class to perform ptychographic reconstruction or hyperparameter tuning.

    The PtyRADSolver class initializes the necessary components for ptychographic 
    reconstruction and provides methods to execute the reconstruction or perform 
    hyperparameter tuning using Optuna.

    Attributes:
        params (dict): Dictionary containing all the parameters required for 
            initialization, loss functions, constraints, model, and optional 
            hyperparameter tuning.
        if_hypertune (bool): A flag to indicate whether hyperparameter tuning should 
            be performed instead of regular reconstruction. Defaults to False.
        verbose (bool): A flag to control the verbosity of the output. Defaults to True unless
            if_quiet is set to True.
        device (str): The device to run the computations on (e.g., 'cuda' for GPU, 'cpu' for CPU). 
            Defaults to None to let `accelerate` automatically decide.
    """
    def __init__(self, params, device=None, seed=None, acc=None, logger=None):
        self.params          = deepcopy(params)
        self.if_hypertune    = self.params.get('hypertune_params', {}).get('if_hypertune', False)
        self.verbose         = not self.params['recon_params']['if_quiet']
        self.accelerator     = acc
        self.use_acc_device  = device is None and acc is not None
        self.device          = self.accelerator.device if self.use_acc_device else device
        self.random_seed     = seed
        self.logger          = logger
        
        # model and optimizer are instantiate inside reconstruct() and hypertune()
        self.init_initializer()
        self.init_loss()
        self.init_constraint()
        vprint("### Done initializing PtyRADSolver ###")
        vprint(" ")
    
    def init_initializer(self):
        """Initializes the variables and objects needed for the reconstruction process."""
        # These components are organized into individual methods so we can re-initialize some of them if needed 
        vprint("### Initializing Initializer ###")
        self.init          = Initializer(self.params['init_params'], seed=self.random_seed).init_all()
        vprint(" ")

    def init_loss(self):
        """Initializes the loss function using the provided parameters."""
        vprint("### Initializing loss function ###")
        loss_params = self.params['loss_params']
        
        # Print loss params
        vprint("Active loss types:")
        for key, value in loss_params.items():
            if value.get('state', False):
                vprint(f"  {key.ljust(12)}: {value}")
                
        self.loss_fn       = CombinedLoss(loss_params, device=self.device)
        vprint(" ")

    def init_constraint(self):
        """Initializes the constraint function using the provided parameters."""
        vprint("### Initializing constraint function ###")
        constraint_params = self.params['constraint_params']

        # Print constraint params
        vprint("Active constraint types:")
        for key, value in constraint_params.items():
            if value.get('start_iter', None) is not None:
                vprint(f"  {key.ljust(14)}: {value}")
                
        self.constraint_fn = CombinedConstraint(constraint_params, device=self.device, verbose=self.verbose)
        vprint(" ")
        
    def reconstruct(self):
        """Executes the ptychographic reconstruction process by creating the model, 
            optimizer, and running the reconstruction loop."""
        params = self.params
        device = self.device
        logger = self.logger
        
        # Create the model and optimizer, prepare indices, batches, and output_path
        model         = PtychoAD(self.init.init_variables, params['model_params'], device=device, verbose=self.verbose)
        optimizer     = create_optimizer(model.optimizer_params, model.optimizable_params)
        
        if not self.use_acc_device:
            indices, batches, output_path = prepare_recon(model, self.init, params)
        else:
            if params['model_params']['optimizer_params']['name'] == 'LBFGS' and self.accelerator.num_processes >1:
                vprint(f"WARNING: Optimizer 'LBFGS' is not supported for multiGPU mode (accelerator.num_processes = {self.accelerator.num_processes}), switch to default optimizer 'Adam'")
                params['model_params']['optimizer_params']['name'] = 'Adam'
                model.optimizer_params['name'] = 'Adam'
                optimizer     = create_optimizer(model.optimizer_params, model.optimizable_params)
            
            vprint(f"params['recon_params']['GROUP_MODE'] is set to 'random' because `use_acc_device` = {self.use_acc_device}", verbose=self.verbose)
            params['recon_params']['GROUP_MODE'] = 'random'
            # `batches` would be replaced by a random DataLoader if we use_acc_device because I haven't figured out how to do specified indices in DataLoader
            # In other words, only `random` grouping is available for accelerate-powered multiGPU and mixed-precision
            indices, batches, output_path = prepare_recon(model, self.init, params)
            ds = IndicesDataset(indices)
            dl = torch.utils.data.DataLoader(ds, batch_size = params['recon_params']['BATCH_SIZE']['size'], shuffle = True) # This will do the batching
            batches = self.accelerator.prepare(dl) # Note that `batches` is replaced by a DataLoader (accelerate mode) that is also an iterable object
            model, optimizer = self.accelerator.prepare(model, optimizer)
            
            vprint(f"len(DataLoader) = num_batches = {len(dl)}, DataLoader.batch_size = {len(indices)//len(dl)}", verbose=self.verbose)
            vprint("Note that the DataLoader will be duplicated for each process, while DataLoader.batch_size is the effective batch size (batch_size_per_process * num_process)", verbose=self.verbose) 
            vprint("The actual batch_size_per_process will be printed below for the reported batches from the main process", verbose=self.verbose) 
            vprint("For example, batch size = 512 with 2 GPUs (2 processes), the reported/observed batch size per GPU will be 512/2=256.", verbose=self.verbose) 

        if logger is not None and logger.flush_file:
            logger.flush_to_file(log_dir=output_path) # Note that output_path can be None, and there's an internal flag of self.flush_file controls the actual file creation
        recon_loop(model, self.init, params, optimizer, self.loss_fn, self.constraint_fn, indices, batches, output_path, acc=self.accelerator)
        self.reconstruct_results = model
        self.optimizer = optimizer
    
    def hypertune(self):
        """Performs hyperparameter tuning using Optuna."""
        import optuna
        hypertune_params = self.params['hypertune_params']
        params_path      = self.params.get('params_path')
        n_trials         = hypertune_params.get('n_trials')
        timeout          = hypertune_params.get('timeout')
        study_name       = hypertune_params.get('study_name')
        storage_path     = hypertune_params.get('storage_path')
        sampler_params   = hypertune_params['sampler_params']
        pruner_params    = hypertune_params['pruner_params']
        error_metric     = hypertune_params['error_metric']
        sampler          = create_optuna_sampler(sampler_params)
        pruner           = create_optuna_pruner(pruner_params)
        logger           = self.logger
        
        # Print hypertune params
        vprint("### Hypertune params ###")
        for key, value in hypertune_params.items():

            if key == 'tune_params':  # Check if 'tune_params' exists
                vprint("Active tune_params:")
                for param, param_config in value.items():
                    if param_config.get('state', False):  # Print only if 'state' is True
                        vprint(f"    {param.ljust(12)}: {param_config}")
            else:
                vprint(f"{key.ljust(16)}: {value}")
        vprint(" ")
        
        # Check error metric validity
        valid_metrics = {"contrast", "loss"}
        if error_metric not in valid_metrics:
            raise ValueError(f"Invalid error metric: '{error_metric}'. Expected one of {valid_metrics}.")
        
        copy_params = self.params['recon_params']['copy_params']
        output_dir  = self.params['recon_params']['output_dir'] # This will be later modified     
        prefix_time = self.params['recon_params']['prefix_time']
        prefix      = self.params['recon_params']['prefix']
        postfix     = self.params['recon_params']['postfix']

        # Retrieve Optuna's logger
        optuna_logger = logging.getLogger("optuna")
        optuna_logger.setLevel(logging.INFO)
        # Remove any existing console handlers from Optuna's logger to avoid duplicate logs
        for handler in optuna_logger.handlers:
            if isinstance(handler, logging.StreamHandler):  # StreamHandler is the console handler
                optuna_logger.removeHandler(handler)
        # Redirect Optuna's logger to custom logger
        optuna_logger.addHandler(logger.buffer_handler)
        optuna_logger.addHandler(logger.console_handler)
                
        # Create a study object and optimize the objective function
        study = optuna.create_study(
                    direction='minimize',
                    sampler=sampler,
                    pruner=pruner, # In Optuna default, setting pruner=None will change to a MedianPruner which is a bit odd. In PtyRAD optuna_objective we will skip the pruning if pruner=None.
                    storage=storage_path,  # Specify the storage URL here.
                    study_name=study_name,
                    load_if_exists=True)
        
        # Modify the 'output_dir' and reset the params dict specifically for hypertune mode
        # Note this will change the params saved with model.pt, but has no effect to the 'copy_params'
        prefix  = prefix + '_' if prefix  != '' else ''
        postfix = '_'+ postfix if postfix != '' else ''
        
        # Attach time string if prefix_time is true or non-empty str
        if prefix_time is True or (isinstance(prefix_time, str) and prefix_time):
            time_str = get_time(prefix_time)  # e.g. '20250606'
            prefix = f"{time_str}_{prefix}" 
        sampler_str = sampler_params['name']
        pruner_str = '_' + pruner_params['name'] if pruner_params is not None else ''
        
        output_dir += f"/{prefix}hypertune_{sampler_str}{pruner_str}_{error_metric}{postfix}"
        self.params['recon_params']['output_dir'] = output_dir 
        self.params['recon_params']['prefix_time'] = ''
        self.params['recon_params']['prefix'] = ''
        self.params['recon_params']['postfix'] = ''
        
        if copy_params:
            copy_params_to_dir(params_path, output_dir, self.params)

        # Set output_dir to None if the user doesn't want to create the output_dir at all
        if not copy_params and self.params['recon_params']['SAVE_ITERS'] is None and not hypertune_params['collate_results']:
            output_dir = None
            
        if logger is not None and logger.flush_file:
            logger.flush_to_file(log_dir=output_dir) # Note that there's an internal flag of self.flush_file controls the actual file creation
            optuna_logger.addHandler(logger.file_handler)
        
        study.optimize(lambda trial: optuna_objective(trial, self.params, self.init, self.loss_fn, self.constraint_fn, self.device, self.verbose), 
                       n_trials=n_trials,
                       timeout=timeout)
        vprint(f"Hypertune study is finished due to either (1) n_trials = {n_trials} or (2) study timeout = {timeout} sec has reached")
        vprint("Best hypertune params:")
        for key, value in study.best_params.items():
            vprint(f"\t{key}: {value}")
    
    # Wrapper function to run either "reconstruction" or "hypertune" modes    
    def run(self):
        """A wrapper method to run the solver in either reconstruction or hyperparameter 
            tuning mode based on the if_hypertune flag"""
        start_t = time_sync()
        solver_mode = 'hypertune' if self.if_hypertune else 'reconstruct'
        
        vprint(f"### Starting the PtyRADSolver in {solver_mode} mode ###")
        vprint(" ")
        
        if self.if_hypertune:
            self.hypertune()
        else:
            self.reconstruct()
        end_t = time_sync()
        solver_t = end_t - start_t
        time_str = "" if solver_t < 60 else f", or {parse_sec_to_time_str(solver_t)}"
        
        vprint(f"### The PtyRADSolver is finished in {solver_t:.3f} sec{time_str} ###")
        vprint(" ")
        if self.logger is not None and self.logger.flush_file:
            self.logger.close()
        
        # End the process properly when in DDP mode
        if dist.is_initialized():
            dist.destroy_process_group()
        
class IndicesDataset(Dataset):
    """
    The Dataset class used specifically for the multiGPU mode for DDP
    """
    def __init__(self, indices):
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.indices[idx]

###### Reconstruction workflow related functions ######
# These are called within PtyRADSolver, and the detailed walkthrough notebook
 
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
            from ptyrad.load import load_ptyrad
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
        model (PtychoAD): The ptychographic model containing the object, probe, 
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
            copy_params_to_dir(params_path, output_path, params, verbose=verbose)
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
    #   mode: str. Choose between 'random', 'compact', or 'sparse' grouping.
    # Output:
    #   batches: A list of `num_batch` arrays, or [batch0, batch1, ...]
    # Note:
    #   The actual batch size would only be "close" if it's not divisible by len(indices) for 'random' grouping
    #   For 'compact' or 'sparse', it's generally fluctuating around the specified batch size
    #   'sparse' can be quite slow for large scan positions (like 256x256 takes more than 10min, and 128x128 takes more than 1min on a CPU)
    #   PtychoShelves automatically switches to 'random' for len(pos) > 1e3 and relying on the random statistics 
        #   
    #   IMPROVEMENT by Zehao Dong: Currently make_batches takes ~ 3000 secs for a 300x300 dataset. 
    #                              Try to speed up by incorporating numba JIT compilation.
    #                              Now it takes ~ 800 secs for a 300x300 dataset, which is a 4x speed up.
    #       
    #   To check the correctness of each grouping, you may visualize the pos
    #   Also we want to make sure we're not missing any indices, so we can do:
    #
    #   flatten_indices = np.concatenate(batches)
    #   flatten_indices.sort()
    #   indices.sort()
    #   all(flatten_indices == indices)
    
    from time import time
    
    try:
        from sklearn.cluster import MiniBatchKMeans
    except ImportError as e:
        missing_package = str(e).split()[-1]
        vprint(f"### {missing_package} is not available, group mode set to 'random'. 'scikit-learn' is needed for 'sparse' and 'compact' ###")
        mode = 'random'
        
    if len(indices) > len(pos):
        raise ValueError(f"len(indices) = '{len(indices)}' is larger than total number of probe positions ({len(pos)}), check your indices generation params")
    
    if indices.max() > len(pos):
        raise ValueError(f"Maximum index '{indices.max()}' is larger than total number of probe positions ({len(pos)}), check your indices generation params")

    num_batch = len(indices) // batch_size   
    t_start = time()
    if mode == 'random':
        rng = np.random.default_rng(seed=seed)
        shuffled_indices = rng.permutation(indices)           # This will make a shuffled copy    
        random_batches = np.array_split(shuffled_indices, num_batch)
        vprint(f"Generated {num_batch} '{mode}' groups of ~{batch_size} scan positions in {time() - t_start:.3f} sec", verbose=verbose)
        return random_batches
        
    else: # Either 'compact' or 'sparse'
        # Choose the selected pos from indices
        pos_s = pos[indices]
        # Kmeans for clustering
        kmeans = MiniBatchKMeans(init="k-means++", n_init=10, n_clusters=num_batch, max_iter=10, batch_size=3072, random_state=seed)
        kmeans.fit(pos_s)
        labels = kmeans.labels_
        
        # Separate data points into groups
        compact_batches = []
        for batch_idx in range(num_batch):
            batch_indices_s = np.where(labels == batch_idx)[0]
            compact_batches.append(indices[batch_indices_s])

        if mode == 'compact':
            vprint(f"Generated {num_batch} '{mode}' groups of ~{batch_size} scan positions in {time() - t_start:.3f} sec", verbose=verbose)
            return compact_batches
        else: # 'sparse' mode - optimized by numba JIT compilation
            # Extract position data for selected indices
            pos_s = pos[indices]
            
            # Calculate centroids for compact batches - same logic as original
            centroids = np.array([np.mean(pos[cbatch], axis=0) for cbatch in compact_batches])
            
            # Use numba-optimized functions
            vprint("Using numba-optimized sparse grouping", verbose=verbose)
            sparse_batches = _sparse_grouping_numba(indices, pos, pos_s, centroids, num_batch)
            
            # Validate the result - same validation as original
            flatten_indices = np.concatenate(sparse_batches)
            flatten_indices.sort()
            indices_sorted = indices.copy()
            indices_sorted.sort()
            assert np.array_equal(flatten_indices, indices_sorted), "Sorry, something went wrong with the sparse grouping, please try 'random' for now"
            
            vprint(f"Generated {num_batch} '{mode}' groups of ~{batch_size} scan positions in {time() - t_start:.3f} sec", verbose=verbose)
            return sparse_batches

        
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

def recon_loop(model, init, params, optimizer, loss_fn, constraint_fn, indices, batches, output_path, acc=None):
    """
    Executes the iterative optimization loop for ptychographic reconstruction.

    This function performs the iterative reconstruction process by optimizing the model 
    parameters over a specified number of iterations. During each iteration, it applies 
    the loss and constraint functions, updates the model, and logs the loss values. 
    Intermediate results are saved at specified intervals, and a summary is plotted.

    Args:
        model (PtychoAD): The ptychographic model containing the parameters and variables 
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
        
        start_iter_t = time_sync()
        batch_losses = recon_step_compiled(batches, grad_accumulation, model, optimizer, loss_fn, constraint_fn, niter, verbose=verbose, acc=acc)
        end_iter_t = time_sync()
        remain_t = (NITER - niter) * (end_iter_t - start_iter_t)
        time_str = parse_sec_to_time_str(remain_t)
        vprint(f"Iter: {niter}, Estimated remaining time: {time_str} ", verbose=verbose)

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
        model (PtychoAD): The ptychographic model containing the parameters and variables 
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

###### Hypertune / Optuna related functions ######
# These are called inside PtyRADSolver.hypertune

def create_optuna_sampler(sampler_params, verbose=True):
    # Note that this function supports all Optuna samplers except "PartialFixedSampler" because it requires a sequential sampler setup
    # Different samplers have different available configurations so please refer to https://optuna.readthedocs.io/en/stable/reference/samplers/index.html for more details
    # For example, GridSampler would need to pass in the 'search_space' so you need to explicitly specify every target variable range in 'sampler_params' : {'name': GridSampler, 'configs': {'search_space': {'optimizer': ['Adam', 'AdamW', 'RMSprop'], 'batch_size': [16,24,32,64,128,256,512], 'oalr': [1.0e-4, 1.0e-3, 1.0e-2], 'oplr': [1.0e-4, 1.0e-3, 1.0e-2]}}}
    # Also the GridSampler would only use the defined search_space and will ignore the range/step setup in 'tune_params'.
    # A handy usage of GridSampler is to exhaust some combination of reconstruction parameters
    # The recommmendation setup for PtyRAD is `sampler_params = {'name': 'TPESampler', 'configs': {'multivariate':True, 'group':True, 'constant_liar':True}}`

    import optuna 
    
    # Extract the sampler name and configs
    sampler_name = sampler_params['name']
    sampler_configs = sampler_params.get('configs') or {} # if "None" is provided or missing, it'll default an empty dict {}
    
    vprint(f"### Creating Optuna '{sampler_name}' sampler with configs = {sampler_configs} ###", verbose=verbose)
    
    # Get the optimizer class from optuna.samplers
    sampler_class = getattr(optuna.samplers, sampler_name, None)
    
    if sampler_class is None or sampler_name == 'ParitalFixedSampler':
        raise ValueError(f"Optuna sampler '{sampler_name}' is not supported.")

    sampler = sampler_class(**sampler_configs)

    vprint(" ", verbose=verbose)
    return sampler

def create_optuna_pruner(pruner_params, verbose=True):
    # Note that this function supports all Optuna pruners except "WilcoxonPruner" because it requires a nested evaluation setup
    # Different pruners have different available configurations so please refer to https://optuna.readthedocs.io/en/stable/reference/pruners.html for more details
    # PatientPruner and PercentilePruner have required fields that need to be passed in with 'configs'
    # For PatientPruner that wraps around a base pruner, you need to specify the base pruner name and configs in a nested way
    # pruner_params = {'name': 'PatientPruner', 
    #              'configs': {'patience': 1, 
    #                          'wrapped_pruner_configs':{'name': 'MedianPruner',
    #                                                    'configs': {}}}}
    # If you're testing pruner with some other objective function, note that the objective function must contain iterative steps for you to prune (early termination)
    # The recommendation setup for PtyRAD is `pruner_params = {'name': 'HyperbandPruner', 'configs': {'min_resource': 5, 'reduction_factor': 2}}`
    
    import optuna 
    
    if pruner_params is None:
        return None
    else:
        # Extract the pruner name and configs
        pruner_name = pruner_params['name']
        pruner_configs = pruner_params.get('configs') or {} # if "None" is provided or missing, it'll default an empty dict {}
        
        vprint(f"### Creating Optuna '{pruner_name}' pruner with configs = {pruner_configs} ###", verbose=verbose)
        
        # Get the pruner class from optuna.pruners
        pruner_class = getattr(optuna.pruners, pruner_name, None)
        
        if pruner_class is None or pruner_name == 'WilcoxonPruner':
            raise ValueError(f"Optuna pruner '{pruner_name}' is not supported.")
        elif pruner_name == 'NopPruner':
            raise ValueError("Optuna NopPruner is an empty pruner, please set pruner_params = None if you don't want to prune.")
        elif pruner_name == 'PatientPruner':
            wrapped_pruner = create_optuna_pruner(pruner_configs['wrapped_pruner_configs'], verbose=verbose)
            pruner_configs.pop('wrapped_pruner_configs', None) # Delete the wrapped_pruner_configs
            pruner = pruner_class(wrapped_pruner, **pruner_configs)
        else:
            pruner = pruner_class(**pruner_configs)

        vprint(" ", verbose=verbose)
        return pruner

# Major Optuna routine
def optuna_objective(trial, params, init, loss_fn, constraint_fn, device='cuda', verbose=False):
    """
    Objective function for Optuna hyperparameter tuning in ptychographic reconstruction.

    This function is used by Optuna to optimize the hyperparameters of the ptychographic reconstruction
    process. The function updates the reconstruction parameters based on the trial's suggestions and 
    runs the reconstruction loop to evaluate the performance. The function also implements Optuna's 
    pruning mechanism to stop unpromising trials early.

    Args:
        trial (optuna.trial.Trial): A trial object that suggests hyperparameter values and handles 
            pruning.
        params (dict): A dictionary containing all the parameters for the reconstruction, including 
            experimental parameters, model parameters, and hyperparameter tuning configurations.
        init (Initializer): An instance of the Initializer class that holds initialized variables 
            and methods for updating them based on the trial's suggestions.
        loss_fn (CombinedLoss): The loss function object that calculates the reconstruction loss.
        constraint_fn (CombinedConstraint): The constraint function object that applies constraints 
            during optimization.
        device (str, optional): The device to run the reconstruction on, e.g., 'cuda'. Defaults to 'cuda'.
        verbose (bool, optional): If True, enables verbose output. Defaults to False.

    Returns:
        float: The total loss for the final iteration of the reconstruction process, used by Optuna 
        to evaluate the trial's performance.

    Raises:
        optuna.exceptions.TrialPruned: Raised when the trial should be pruned based on the 
        intermediate results.
    """
    import optuna
    
    init.verbose = verbose # This would affect the initialization printing for each hypertune trials
    params = deepcopy(params)
        
    # Parse the recon_params
    recon_params      = params.get('recon_params')
    NITER             = recon_params['NITER']
    SAVE_ITERS        = recon_params['SAVE_ITERS']
    grad_accumulation = recon_params['BATCH_SIZE'].get("grad_accumulation", 1)
    output_dir        = recon_params['output_dir']
    selected_figs     = recon_params['selected_figs']
    compiler_configs  = parse_torch_compile_configs(recon_params['compiler_configs'])
    
    # Parse the hypertune_params
    hypertune_params  = params['hypertune_params']
    collate_results   = hypertune_params['collate_results']
    append_params     = hypertune_params['append_params']
    error_metric      = hypertune_params['error_metric']
    tune_params       = hypertune_params['tune_params']
    trial_id = 't' + str(trial.number).zfill(4)
    params['recon_params']['prefix'] += trial_id
    
    ## Currently only re-initialize the required parts for performance, but once there're too many correlated params need to be re-initialized,
    ## we might put the entire initialization inside optuna_objective for readability, although init_measurements for every trial would be a large overhead.

    ## TODO After the refactoring of `init_calibration` and better dx setting logic, it's possible to include more optimizable params without exploding the logic here
            
    # Batch size
    if tune_params['batch_size']['state']:
        vname = 'batch_size'
        vparams = tune_params[vname]
        params['recon_params']['BATCH_SIZE']['size'] = get_optuna_suggest(trial, vparams['suggest'], vname, vparams['kwargs'])
        
    # Optimizer
    if tune_params['optimizer']['state']:
        vname = 'optimizer'
        vparams = tune_params[vname]
        optim_name = get_optuna_suggest(trial, vparams['suggest'], vname, vparams['kwargs'])
        params['model_params']['optimizer_params']['name'] = optim_name
        params['model_params']['optimizer_params']['configs'] = vparams['kwargs']['optim_configs'].get(optim_name, {}) # Update optimizer_configs if the user has specified them for each optimizer
    
    # learning rates
    lr_to_tensor = {'plr': 'probe', 'oalr': 'obja', 'oplr': 'objp', 'slr': 'probe_pos_shifts', 'tlr': 'obj_tilts', 'dzlr': 'slice_thickness'}
    for vname in ['plr', 'oalr', 'oplr', 'slr', 'tlr', 'dzlr']:
        if tune_params[vname]['state']:
            vparams = tune_params[vname]
            params['model_params']['update_params'][lr_to_tensor[vname]]['lr'] = get_optuna_suggest(trial, vparams['suggest'], vname, vparams['kwargs'])
    
    # dx (calibration)
    if tune_params['dx']['state']:
        vname = 'dx'
        vparams = tune_params[vname]
        init.init_params['meas_calib'] = {'mode': vname, 'value': get_optuna_suggest(trial, vparams['suggest'], vname, vparams['kwargs'])}
        init.init_calibration()
        init.set_variables_dict()
        init.init_probe()
        init.init_pos()
        init.init_obj()
        init.init_H()
        
    # probe_params (pmode_max, conv_angle, defocus, z_shift, c3, c5)
    remake_probe = False
    for vname in ['pmode_max', 'conv_angle', 'defocus', 'z_shift', 'c3', 'c5']:
        if tune_params[vname]['state']:
            vparams = tune_params[vname]
            init.init_params['probe_' + vname] = get_optuna_suggest(trial, vparams['suggest'], vname, vparams['kwargs'])
            remake_probe = True
    if remake_probe:
        init.init_probe()
            
    # Nlayer
    if tune_params['Nlayer']['state']:
        vname = 'Nlayer'
        vparams = tune_params[vname]
        init.init_params['obj_Nlayer'] = get_optuna_suggest(trial, vparams['suggest'], vname, vparams['kwargs'])
        init.init_obj()
    
    # slice_thickness
    if tune_params['dz']['state']:
        vname = 'dz'
        vparams = tune_params[vname]
        init.init_params['obj_slice_thickness'] = get_optuna_suggest(trial, vparams['suggest'], vname, vparams['kwargs'])
        init.set_variables_dict()
        init.init_obj() # Currently the slice_thickness only modifies the printed obj_extent value, but eventually we'll add obj resampling so let's keep it for now
        init.init_H()
    
    # scan_affine
    scan_affine = []
    scan_affine_init = params['init_params']['pos_scan_affine']
    if scan_affine_init is not None:
        default_affine = {'scale':scan_affine_init[0], 'asymmetry':scan_affine_init[1], 'rotation':scan_affine_init[2], 'shear':scan_affine_init[3]}
    else:
        default_affine = {'scale':1, 'asymmetry':0, 'rotation':0, 'shear':0}
    for vname in ['scale', 'asymmetry', 'rotation', 'shear']:
        if tune_params[vname]['state']:
            vparams = tune_params[vname]
            scan_affine.append(get_optuna_suggest(trial, vparams['suggest'], vname, vparams['kwargs']))
        else:
            scan_affine.append(default_affine[vname])
    if scan_affine != [1,0,0,0]:
        init.init_params['pos_scan_affine'] = scan_affine
        init.init_pos()
        init.init_obj() # Update obj initialization because the scan range has changed
    
    # tilt (This will override the current tilts and force it to be a global tilt (2,1))
    obj_tilts = []
    for vname in ['tilt_y', 'tilt_x']:
        if tune_params[vname]['state']:
            vparams = tune_params[vname]
            obj_tilts.append(get_optuna_suggest(trial, vparams['suggest'], vname, vparams['kwargs']))
        else:
            obj_tilts.append(0)
    obj_tilts = [obj_tilts] # Make it into [[tilt_y, tilt_x]]
    if obj_tilts != [[0,0]]:
        init.init_variables['obj_tilts'] = obj_tilts # No need to update init_params['tilt_params'] because the pass-in value is only used when `tilt_params = 'custom'`
   
    # Create the model and optimizer, prepare indices, batches, and output_path
    model         = PtychoAD(init.init_variables, params['model_params'], device=device, verbose=verbose)
    optimizer     = create_optimizer(model.optimizer_params, model.optimizable_params, verbose=verbose)
    indices, batches, output_path = prepare_recon(model, init, params)
      
    # Optimization loop
    for niter in range(1, NITER+1):
        
        # Toggle the grad calculation to enable or disable AD update on tensors at certain iterations
        toggle_grad_requires(model, niter, verbose)
        
        # Apply torch.compile to `recon_step``
        if niter in model.compilation_iters: # compilation_iters always contain niter=1
            vprint(f"Setting up PyTorch compiler with {compiler_configs}", verbose=verbose)
            torch._dynamo.reset()
            recon_step_compiled = torch.compile(recon_step, **compiler_configs)
        
        if model.random_seed is not None:
            set_random_seed(seed=model.random_seed + niter) # This ensures the batches order are different for each iter in a reproducible way
        shuffle(batches)
        batch_losses = recon_step_compiled(batches, grad_accumulation, model, optimizer, loss_fn, constraint_fn, niter, verbose=verbose)

        ## Saving intermediate results
        if SAVE_ITERS is not None and niter % SAVE_ITERS == 0:
            save_results(output_path, model, params, optimizer, niter, indices, batch_losses, collate_str='')
            plot_summary(output_path, model, niter, indices, init.init_variables, selected_figs=selected_figs, collate_str='', show_fig=False, save_fig=True, verbose=verbose)
               
        ## Pruning logic for optuna
        if hypertune_params['pruner_params'] is not None:
            optuna_error = compute_optuna_error(model, indices, error_metric)
            trial.report(optuna_error, niter)
            
            # Handle pruning based on the intermediate value.
            if trial.should_prune():
            
                # Save the current results of the pruned trials
                params_str = parse_hypertune_params_to_str(trial.params) if append_params else ''
                collate_str = f"_error_{optuna_error:.5f}_{trial_id}{params_str}"
                if collate_results:
                    save_results(output_dir, model, params, optimizer, niter, indices, batch_losses, collate_str=collate_str)
                    plot_summary(output_dir, model, niter, indices, init.init_variables, selected_figs=selected_figs, collate_str=collate_str, show_fig=False, save_fig=True, verbose=verbose)
                raise optuna.exceptions.TrialPruned()

    ## Final optuna_error evaluation (only needed if pruner never ran)
    if hypertune_params['pruner_params'] is None:
        optuna_error = compute_optuna_error(model, indices, error_metric)
    
    ## Saving collate results and figs of the finished trials
    params_str = parse_hypertune_params_to_str(trial.params) if append_params else ''
    collate_str = f"_error_{optuna_error:.5f}_{trial_id}{params_str}"
    if collate_results:
        save_results(output_dir, model, params, optimizer, niter, indices, batch_losses, collate_str=collate_str)
        plot_summary(output_dir, model, niter, indices, init.init_variables, selected_figs=selected_figs, collate_str=collate_str, show_fig=False, save_fig=True, verbose=verbose)
    
    vprint(f"### Finished {NITER} iterations, averaged iter_t = {np.mean(model.iter_times):.3g} sec ###", verbose=verbose)
    vprint(" ", verbose=verbose)
    return optuna_error

def get_optuna_suggest(trial, suggest, name, kwargs):
    
    if suggest == 'cat':
        return trial.suggest_categorical(name, **kwargs)
    elif suggest == 'int':
        return trial.suggest_int(name, **kwargs)
    elif suggest == 'float':
        return trial.suggest_float(name, **kwargs)        
    else:
        raise (f"Optuna trail.suggest method '{suggest}' is not supported.")

def compute_optuna_error(model, indices, metric):
    """
    Helper function to compute the current error for Optuna
    """
    if metric == 'contrast':
        return -1*get_objp_contrast(model, indices) # Negative for minimization
    elif metric == 'loss':
        return model.loss_iters[-1][-1]
    else:
        raise ValueError(f"Unsupported hypertune error metric: '{metric}'. Expected 'contrast' or 'loss'.")


###### Numba JIT optimized functions for sparse batch generation ######

@numba.njit
def _sparse_grouping_numba(indices, pos_full, pos_s, centroids, num_batch):
    """
    Numba JIT-compiled core function for sparse grouping.
    This implements the exact same logic as the original sparse mode algorithm
    but with numba optimization for faster execution.
    
    Args:
        indices: Array of selected scan indices
        pos_full: Full position array for all scan points  
        pos_s: Position array for selected indices only
        centroids: Centroid positions for each batch
        num_batch: Number of batches to create
    
    Returns:
        List of arrays containing indices for each sparse batch
    """
    n_indices = len(indices)
    
    # Initialize sparse batches as lists - equivalent to original sparse_batches = []
    sparse_batches = [numba.typed.List.empty_list(numba.int64) for _ in range(num_batch)]
    used_mask = np.zeros(n_indices, dtype=numba.boolean)
    
    # Find closest index to each centroid for initialization - same logic as original
    for batch_idx in range(num_batch):
        min_dist = np.inf
        closest_idx_pos = -1
        
        for i in range(n_indices):
            if used_mask[i]:
                continue
            # Same distance calculation as original: np.linalg.norm(pos_s - centroids[batch_idx], axis=1)
            dist = np.sqrt(np.sum((pos_s[i] - centroids[batch_idx])**2))
            if dist < min_dist:
                min_dist = dist
                closest_idx_pos = i
        
        if closest_idx_pos >= 0:
            sparse_batches[batch_idx].append(indices[closest_idx_pos])
            used_mask[closest_idx_pos] = True
    
    # Assign remaining indices to batches - same logic as original nested loops
    for i in range(n_indices):
        if used_mask[i]:
            continue
            
        current_idx = indices[i]
        
        max_min_dist = -1.0
        best_batch = 0
        
        # For each batch, find the minimum distance to any point in that batch
        # Same logic as original: pairwise_distances[sparse_batches[batch_idx], idx]
        for batch_idx in range(num_batch):
            if len(sparse_batches[batch_idx]) == 0:
                continue
                
            min_dist_to_batch = np.inf
            for j in range(len(sparse_batches[batch_idx])):
                batch_point_idx = sparse_batches[batch_idx][j]
                # Calculate distance between current point and points in this batch
                dist = np.sqrt(np.sum((pos_full[current_idx] - pos_full[batch_point_idx])**2))
                if dist < min_dist_to_batch:
                    min_dist_to_batch = dist
            
            # Choose the batch with the maximum of these minimum distances
            # Same logic as original: max_group_index = np.argmax(min_distances)
            if min_dist_to_batch > max_min_dist:
                max_min_dist = min_dist_to_batch
                best_batch = batch_idx
        
        sparse_batches[best_batch].append(current_idx)
    
    # Convert to numpy arrays - same as original final processing
    result = []
    for batch_list in sparse_batches:
        batch_array = np.empty(len(batch_list), dtype=np.int64)
        for i, val in enumerate(batch_list):
            batch_array[i] = val
        result.append(batch_array)
    
    return result
