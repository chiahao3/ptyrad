""" 
Hypertune / Optuna related functions 

"""

import logging
from copy import deepcopy
from random import shuffle

import numpy as np
import torch

from ptyrad.core import PtychoModel
from ptyrad.core.losses import get_objp_contrast
from ptyrad.io.save import save_results
from ptyrad.optics.aberrations import Aberrations
from ptyrad.plotting.model import plot_summary
from ptyrad.runtime.seed import set_random_seed

from .reconstruction import (
    compute_loss,
    create_optimizer,
    parse_torch_compile_configs,
    prepare_recon,
    recon_step,
    toggle_grad_requires,
)

logger = logging.getLogger(__name__)

# ==============================================================================
# SECTION 1: OPTUNA SETUP
# ==============================================================================
    
def create_optuna_sampler(sampler_params):
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
    
    logger.info(f"### Creating Optuna '{sampler_name}' sampler with configs = {sampler_configs} ###")
    
    # Get the optimizer class from optuna.samplers
    sampler_class = getattr(optuna.samplers, sampler_name, None)
    
    if sampler_class is None or sampler_name == 'ParitalFixedSampler':
        raise ValueError(f"Optuna sampler '{sampler_name}' is not supported.")

    sampler = sampler_class(**sampler_configs)

    logger.info(" ")
    return sampler

def create_optuna_pruner(pruner_params):
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
        
        logger.info(f"### Creating Optuna '{pruner_name}' pruner with configs = {pruner_configs} ###")
        
        # Get the pruner class from optuna.pruners
        pruner_class = getattr(optuna.pruners, pruner_name, None)
        
        if pruner_class is None or pruner_name == 'WilcoxonPruner':
            raise ValueError(f"Optuna pruner '{pruner_name}' is not supported.")
        elif pruner_name == 'NopPruner':
            raise ValueError("Optuna NopPruner is an empty pruner, please set pruner_params = None if you don't want to prune.")
        elif pruner_name == 'PatientPruner':
            wrapped_pruner = create_optuna_pruner(pruner_configs['wrapped_pruner_configs'])
            pruner_configs.pop('wrapped_pruner_configs', None) # Delete the wrapped_pruner_configs
            pruner = pruner_class(wrapped_pruner, **pruner_configs)
        else:
            pruner = pruner_class(**pruner_configs)

        logger.info(" ")
        return pruner

# ==============================================================================
# SECTION 2: OPTIMIZATION OBJECTIVE
# ==============================================================================

def optuna_objective(trial, params, init, loss_fn, constraint_fn, device='cuda'):
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

    Returns:
        float: The total loss for the final iteration of the reconstruction process, used by Optuna 
        to evaluate the trial's performance.

    Raises:
        optuna.exceptions.TrialPruned: Raised when the trial should be pruned based on the 
        intermediate results.
    """
    import optuna
    
    params = deepcopy(params)
    
    # ==============================================================================
    # SECTION 1: PARSE CONFIGS
    # ==============================================================================
        
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
    
    # ==============================================================================
    # SECTION 2: DYNAMIC PARAMETER INJECT 
    # ==============================================================================
    
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
        
    # probe_params (pmode_max, conv_angle, z_shift)
    remake_probe = False
    for vname in ['pmode_max', 'conv_angle', 'z_shift']:
        if tune_params[vname]['state']:
            vparams = tune_params[vname]
            init.init_params['probe_' + vname] = get_optuna_suggest(trial, vparams['suggest'], vname, vparams['kwargs'])
            remake_probe = True
            
    ## probe_aberrations
    valid_aberrations = [
        'C10', 'C12', 
        'C21', 'C23', 
        'C30', 'C32', 'C34', 
        'C41', 'C43', 'C45', 
        'C50', 'C52', 'C54', 'C56']
    init.init_params['probe_aberrations'] = Aberrations(init.init_params['probe_aberrations']).get_krivanek_cartesian() # Sanitize it first
    for vname in valid_aberrations: # Hard code up to 5th order
        if tune_params[vname]['state']:
            vparams = tune_params[vname]
            
            # parse aberration keys
            m = int(vname[-1])
            
            if m==0: # Round-lens
                init.init_params['probe_aberrations'][vname] = get_optuna_suggest(trial, vparams['suggest'], vname, vparams['kwargs'])
            else:
                vparams['kwargs']
                init.init_params['probe_aberrations'][f'{vname}a'] = get_optuna_suggest(trial, vparams['suggest'], f'{vname}a', vparams['kwargs'])
                init.init_params['probe_aberrations'][f'{vname}b'] = get_optuna_suggest(trial, vparams['suggest'], f'{vname}b', vparams['kwargs'])
            
            remake_probe = True
            
    # Re-initialize the probe
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
   
    # ==============================================================================
    # SECTION 3: RECONSTRUCTION LOOP
    # ==============================================================================
   
    # Create the model and optimizer, prepare indices, batches, and output_path
    model         = PtychoModel(init.init_variables, params['model_params'], device=device)
    optimizer     = create_optimizer(model.optimizer_params, model.optimizable_params)
    indices, batches_np, output_path = prepare_recon(model, init, params)

    # Initialize the compute_loss_fn
    compute_loss_fn = compute_loss 
      
    # Optimization loop
    for niter in range(1, NITER+1):
        
        # Toggle the grad calculation to enable or disable AD update on tensors at certain iterations
        toggle_grad_requires(model, niter)
        
        # Apply torch.compile
        if niter in model.compilation_iters: # compilation_iters always contain niter=1
            logger.info(f"Setting up PyTorch compiler with {compiler_configs}")
            torch._dynamo.reset()
            compute_loss_fn = torch.compile(compute_loss, **compiler_configs)
        
            if not isinstance(optimizer, torch.optim.LBFGS): # Only compile first-order optimizers (like Adam), L-BFGS relies on dynamic closures that cannot be safely traced.
                optimizer.step = torch.compile(optimizer.step, **compiler_configs)
        
        if model.random_seed is not None:
            set_random_seed(seed=model.random_seed + niter) # This ensures the batches order are different for each iter in a reproducible way
        shuffle(batches_np)
        batches = [torch.from_numpy(arr).to(device=device) for arr in batches_np]
        
        batch_losses = recon_step(batches, grad_accumulation, model, optimizer, loss_fn, constraint_fn, niter, compute_loss_fn=compute_loss_fn)

        ## Saving intermediate results
        if SAVE_ITERS is not None and niter % SAVE_ITERS == 0:
            save_results(output_path, model, params, optimizer, niter, indices, batch_losses, collate_str='')
            plot_summary(output_path, model, niter, indices, init.init_variables, selected_figs=selected_figs, collate_str='', show_fig=False, save_fig=True)
               
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
                    plot_summary(output_dir, model, niter, indices, init.init_variables, selected_figs=selected_figs, collate_str=collate_str, show_fig=False, save_fig=True)
                raise optuna.exceptions.TrialPruned()

    ## Final optuna_error evaluation (only needed if pruner never ran)
    if hypertune_params['pruner_params'] is None:
        optuna_error = compute_optuna_error(model, indices, error_metric)
    
    ## Saving collate results and figs of the finished trials
    params_str = parse_hypertune_params_to_str(trial.params) if append_params else ''
    collate_str = f"_error_{optuna_error:.5f}_{trial_id}{params_str}"
    if collate_results:
        save_results(output_dir, model, params, optimizer, niter, indices, batch_losses, collate_str=collate_str)
        plot_summary(output_dir, model, niter, indices, init.init_variables, selected_figs=selected_figs, collate_str=collate_str, show_fig=False, save_fig=True)
    
    logger.info(f"### Finished {NITER} iterations, averaged iter_t = {np.mean(model.iter_times):.3g} sec ###")
    logger.info(" ")
    return optuna_error

# ==============================================================================
# SECTION 3: TRIAL HELPERS
# ==============================================================================

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
    
def parse_hypertune_params_to_str(hypertune_params):
    
    hypertune_str = ''
    for key, value in hypertune_params.items():
        if key[-2:].lower() == "lr":
            hypertune_str += f"_{key}_{value:.1e}"
        elif isinstance(value, (int, float)):
            hypertune_str += f"_{key}_{value:.3g}"
        else:
            hypertune_str += f"_{key}_{value}"
    
    return hypertune_str