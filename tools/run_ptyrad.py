"""
Run PtyRAD reconstruction
"""
import argparse
import logging

from ptyrad.params import load_params
from ptyrad.runtime.device import set_accelerator, set_gpu_device
from ptyrad.runtime.diagnostics import print_system_info
from ptyrad.runtime.logging import LoggingManager
from ptyrad.solver import PtyRADSolver

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run PtyRAD", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--params_path", type=str, required=True)
    parser.add_argument("--skip_validate", action="store_true", help="Skip parameter validation and default filling. Use only if your params file is complete and consistent.")
    parser.add_argument("--gpuid", type=str, required=False, default="0", help="GPU ID to use ('acc', 'cpu', or an integer)")
    parser.add_argument("--jobid", type=int, required=False, default=0, help="Unique identifier for hypertune mode with multiple GPU workers")
    parser.add_argument("--n_iter", type=int, required=False, default=1, help="Iteration number")
    parser.add_argument('--compile', type=str2bool, default=False, help="PyTorch JIT compilation")
    parser.add_argument('--preload', type=str2bool, default=False, help="Preload data")
    parser.add_argument("--output_path", type=str, required=False, default="./output/", help="Output path")

    args = parser.parse_args()

    # Setup LoggingManager
    LoggingManager(
        log_file='ptyrad_log.txt',
        log_dir='auto',
        prefix_time='datetime',
        append_to_file=True,
        show_timestamp=True,
        verbosity='INFO'
    )

    accelerator = set_accelerator() 

    print_system_info()
    params = load_params(args.params_path, validate=not args.skip_validate)
    device = set_gpu_device(args.gpuid)
    
    logger = logging.getLogger('ptyrad')
    logger.info(f"Running (n_iter, gpu, compile, preload) = ({args.n_iter, args.gpuid, args.compile, args.preload})")
    logger.info("")
    
    # Update params
    params['recon_params']['NITER'] = args.n_iter
    params['recon_params']['compiler_configs'] = {'enable': args.compile}
    params['model_params']['preload_data'] = args.preload
    params['recon_params']['output_dir'] = args.output_path

    # Run PtyRAD
    ptycho_solver = PtyRADSolver(params, device=device, seed=42, acc=accelerator)
    ptycho_solver.run()