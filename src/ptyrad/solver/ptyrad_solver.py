"""
High-level solver interface for PtyRAD workflows, including reconstruction and hypertune modes

"""

import logging
from copy import deepcopy

import numpy as np
import torch.distributed as dist
from torch.utils.data import DataLoader

from ptyrad.core import CombinedConstraint, CombinedLoss, PtychoModel
from ptyrad.init import Initializer
from ptyrad.io.dataloader import IndicesDataset
from ptyrad.params.parser import copy_params_to_dir
from ptyrad.utils.time import get_time, parse_sec_to_time_str
from ptyrad.runtime.logging import get_logging_manager

from .hypertune import create_optuna_pruner, create_optuna_sampler, optuna_objective
from .reconstruction import create_optimizer, prepare_recon, recon_loop, time_sync

logger = logging.getLogger(__name__)

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
        device (str): The device to run the computations on (e.g., 'cuda' for GPU, 'cpu' for CPU). 
            Defaults to None to let `accelerate` automatically decide.
    """
    def __init__(self, params, device=None, seed=None, acc=None):
        self.params          = deepcopy(params)
        self.if_hypertune    = self.params.get('hypertune_params', {}).get('if_hypertune', False)
        self.accelerator     = acc
        self.use_acc_device  = device is None and acc is not None
        self.device          = self.accelerator.device if self.use_acc_device else device
        self.random_seed     = seed
        
        # model and optimizer are instantiate inside reconstruct() and hypertune()
        self.init_initializer()
        self.init_loss()
        self.init_constraint()
        logger.info("### Done initializing PtyRADSolver ###")
        logger.info(" ")
    
    def init_initializer(self):
        """Initializes the variables and objects needed for the reconstruction process."""
        # These components are organized into individual methods so we can re-initialize some of them if needed 
        logger.info("### Initializing Initializer ###")
        self.init          = Initializer(self.params['init_params'], seed=self.random_seed).init_all()
        logger.info(" ")

    def init_loss(self):
        """Initializes the loss function using the provided parameters."""
        logger.info("### Initializing loss function ###")
        loss_params = self.params['loss_params']
        
        # Print loss params
        logger.info("Active loss types:")
        for key, value in loss_params.items():
            if value.get('state', False):
                logger.info(f"  {key.ljust(12)}: {value}")
                
        self.loss_fn       = CombinedLoss(loss_params, device=self.device)
        logger.info(" ")

    def init_constraint(self):
        """Initializes the constraint function using the provided parameters."""
        logger.info("### Initializing constraint function ###")
        constraint_params = self.params['constraint_params']

        # Print constraint params
        logger.info("Active constraint types:")
        for key, value in constraint_params.items():
            if value.get('start_iter', None) is not None:
                logger.info(f"  {key.ljust(14)}: {value}")
                
        self.constraint_fn = CombinedConstraint(constraint_params, device=self.device)
        logger.info(" ")
        
    def reconstruct(self):
        """Executes the ptychographic reconstruction process by creating the model, 
            optimizer, and running the reconstruction loop."""
        params = self.params
        device = self.device
        
        # Create the model and optimizer, prepare indices, batches, and output_path
        model         = PtychoModel(self.init.init_variables, params['model_params'], device=device)
        optimizer     = create_optimizer(model.optimizer_params, model.optimizable_params)
        indices, batches, output_path = prepare_recon(model, self.init, params)
        
        # Handle LBFGS incompatibility
        if params['model_params']['optimizer_params']['name'] == 'LBFGS' and self.accelerator.num_processes >1:
            logger.warning(f"Optimizer 'LBFGS' is not supported for multiGPU mode (accelerator.num_processes = {self.accelerator.num_processes}), switch to default optimizer 'Adam'")
            params['model_params']['optimizer_params']['name'] = 'Adam'
            model.optimizer_params['name'] = 'Adam'
            optimizer     = create_optimizer(model.optimizer_params, model.optimizable_params)
        
        # If using multi GPU, prepare the batches, model, optimizer with Accelerator
        if self.use_acc_device:
            ordered_indices = IndicesDataset(np.concatenate(batches)) # Ordered indices would keep the original spatial distribution of each batch
            dataloader = DataLoader(ordered_indices, batch_size = params['recon_params']['BATCH_SIZE']['size'], shuffle=False) # This will do the batching sequentially
            batches = self.accelerator.prepare(dataloader) # Note that `batches` is replaced by a DataLoader (accelerate mode) that is also an iterable object
            model, optimizer = self.accelerator.prepare(model, optimizer)
        
        # Look for the logging manager and trigger the flush
        logging_manager = get_logging_manager()
        if logging_manager:
            logging_manager.flush_to_file(log_dir=output_path) # Note that output_path can be None, and there's an internal flag of self.flush_file controls the actual file creation       

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
        
        # Print hypertune params
        logger.info("### Hypertune params ###")
        for key, value in hypertune_params.items():

            if key == 'tune_params':  # Check if 'tune_params' exists
                logger.info("Active tune_params:")
                for param, param_config in value.items():
                    if param_config.get('state', False):  # Print only if 'state' is True
                        logger.info(f"    {param.ljust(12)}: {param_config}")
            else:
                logger.info(f"{key.ljust(16)}: {value}")
        logger.info(" ")
        
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
        # Redirect Optuna's logger to LoggingManager
        logging_manager = get_logging_manager()
        if logging_manager:
            optuna_logger.addHandler(logging_manager.buffer_handler)
            optuna_logger.addHandler(logging_manager.console_handler)
                
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
            copy_params_to_dir(params_path, output_dir)

        # Set output_dir to None if the user doesn't want to create the output_dir at all
        if not copy_params and self.params['recon_params']['SAVE_ITERS'] is None and not hypertune_params['collate_results']:
            output_dir = None

        # Look for the logging manager and trigger the flush
        if logging_manager:
            logging_manager.flush_to_file(log_dir=output_dir) # Note that there's an internal flag of self.flush_file controls the actual file creation
            optuna_logger.addHandler(logging_manager.file_handler)
        
        # Temporarily slicence the logger 
        ptyrad_logger = logging.getLogger('ptyrad')
        original_level = ptyrad_logger.level # Save the current state (usually DEBUG or INFO)
        ptyrad_logger.setLevel(logging.WARNING)
        
        study.optimize(lambda trial: optuna_objective(trial, self.params, self.init, self.loss_fn, self.constraint_fn, self.device), 
                       n_trials=n_trials,
                       timeout=timeout)
        
        # Turn back on the volume of the logger
        ptyrad_logger.setLevel(original_level)
        
        logger.info(f"Hypertune study is finished due to either (1) n_trials = {n_trials} or (2) study timeout = {timeout} sec has reached")
        logger.info("Best hypertune params:")
        for key, value in study.best_params.items():
            logger.info(f"\t{key}: {value}")
    
    # Wrapper function to run either "reconstruction" or "hypertune" modes    
    def run(self):
        """A wrapper method to run the solver in either reconstruction or hyperparameter 
            tuning mode based on the if_hypertune flag"""
        start_t = time_sync()
        solver_mode = 'hypertune' if self.if_hypertune else 'reconstruct'
        
        logger.info(f"### Starting the PtyRADSolver in {solver_mode} mode ###")
        logger.info(" ")
        
        if self.if_hypertune:
            self.hypertune()
        else:
            self.reconstruct()
        end_t = time_sync()
        solver_t = end_t - start_t
        time_str = "" if solver_t < 60 else f", or {parse_sec_to_time_str(solver_t)}"
        
        logger.info(f"### The PtyRADSolver is finished in {solver_t:.3f} sec{time_str} ###")
        logger.info(" ")
        
        logging_manager = get_logging_manager()
        if logging_manager and logging_manager.flush_file:
            logging_manager.close()

        # End the process properly when in DDP mode
        if dist.is_initialized():
            dist.destroy_process_group()