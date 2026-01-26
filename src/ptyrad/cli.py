"""
Command line Interface of PtyRAD, including running reconstructions, checking system information, and others

"""

import argparse


def run(args):
    from ptyrad.load import load_params
    from ptyrad.solver import PtyRADSolver
    from ptyrad.utils import CustomLogger, get_nested, print_system_info, resolve_seed_priority, set_accelerator, set_gpu_device
    
    # Setup CustomLogger
    logger = CustomLogger(
        log_file='ptyrad_log.txt',
        log_dir='auto',
        prefix_time='datetime',
        prefix_jobid=args.jobid,
        append_to_file=True,
        show_timestamp=True
    )

    # Set up accelerator for multiGPU/mixed-precision setting, 
    # note that these we need to call the command as:
    # `accelerate launch --num_processes=2 --mixed_precision='no' -m ptyrad run <PTYRAD_ARGUMENTS> --gpuid 'acc'`
    accelerator = set_accelerator() 

    print_system_info()
    params = load_params(args.params_path, validate=not args.skip_validate)
    device = set_gpu_device(args.gpuid)
    seed = resolve_seed_priority(args_seed=args.seed, params_seed=get_nested(params, "init_params.random_seed", safe=True), acc=accelerator)
    ptycho_solver = PtyRADSolver(params, device=device, seed=seed, acc=accelerator, logger=logger)
    ptycho_solver.run()


def check_gpu(args):
    from ptyrad.utils import print_gpu_info
    print_gpu_info()


def print_info(args):
    from ptyrad.utils import print_system_info
    print_system_info()

def export_meas_init(args):
    from pathlib import Path

    from ptyrad.initialization import Initializer
    from ptyrad.load import load_params
    
    # 1. Load init_params
    init_params = load_params(args.params_path, validate=not args.skip_validate)['init_params']
    
    # 2. Parse and normalize export config from file.
    export_cfg = init_params.get('meas_export') # True, False, None, dict (could be {})
    if export_cfg in [True, False, None]:
        export_cfg = {}  # initialize as empty dict if not enabled
    elif not isinstance(export_cfg, dict):
        raise TypeError("`meas_export` in init_params must be True, False, None, or a dict")
    
    # 3. CLI overrides (highest priority)
    if args.output:
        output_path = Path(args.output)
        export_cfg['file_dir'] = str(output_path.parent)
        export_cfg['file_name'] = output_path.stem
        export_cfg['file_format'] = output_path.suffix.lstrip(".") or "hdf5"
    else:
        # Use defaults if not specified
        export_cfg.setdefault('file_dir', "")
        export_cfg.setdefault('file_name', "ptyrad_init_meas")
        export_cfg.setdefault('file_format', "hdf5")

    if args.reshape:
        export_cfg['output_shape'] = tuple(args.reshape)

    export_cfg['append_shape'] = args.append  # Always override

    # 4. Save modified export config back to init_params
    init_params['meas_export'] = export_cfg

    # 5. Proceed with initialization
    init = Initializer(init_params)
    init.init_measurements()
    

def validate_params(args):
    from ptyrad.load import load_params
    
    try:
        _ = load_params(args.params_path, validate=True)
    except Exception as e:
        print(f"Invalid parameters: {e}")


def gui(args):
    print("[placeholder] GUI not implemented yet.")


def main():
    parser = argparse.ArgumentParser(
        description="PtyRAD Command-Line Interface",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # run
    parser_run = subparsers.add_parser("run", help="Run PtyRAD reconstruction")
    parser_run.add_argument("--params_path", type=str, required=True)
    parser_run.add_argument("--skip_validate", action="store_true", help="Skip parameter validation and default filling. Use only if your params file is complete and consistent.")
    parser_run.add_argument("--gpuid", type=str, required=False, default="0", help="GPU ID to use ('acc', 'cpu', or an integer)")
    parser_run.add_argument("--jobid", type=int, required=False, default=0, help="Unique identifier for hypertune mode with multiple GPU workers")
    parser_run.add_argument("--seed", type=int, required=False, help="Random seed for improved reproducibility")
    parser_run.set_defaults(func=run)

    # check-gpu
    parser_check_gpu = subparsers.add_parser("check-gpu", help="Check GPU availability")
    parser_check_gpu.set_defaults(func=check_gpu)

    # print-system-info
    parser_info = subparsers.add_parser("print-system-info", help="Print system info")
    parser_info.set_defaults(func=print_info)

    # export-meas-init
    parser_export = subparsers.add_parser("export-meas-init", help="Export initialized measurements file to disk")
    parser_export.add_argument("--params_path", type=str, required=True)
    parser_export.add_argument("--skip_validate", action="store_true", help="Skip parameter validation and default filling. Use only if your params file is complete and consistent.")
    parser_export.add_argument("--output", type=str, help="Optional output path / file type (.mat, .hdf5, .tif, .npy) for the exported array")
    parser_export.add_argument("--reshape", type=int, nargs="+", help="Optional new shape for the exported array, e.g. --reshape 128 128 128 128")
    parser_export.add_argument("--append", action="store_true", help="Optionally appending the array shape to file name")
    parser_export.set_defaults(func=export_meas_init)
    
    # validate-params
    parser_validate = subparsers.add_parser("validate-params", help="Validate parameter file")
    parser_validate.add_argument("--params_path", type=str, required=True)
    parser_validate.set_defaults(func=validate_params)

    # gui (placeholder) #TODO 
    parser_gui = subparsers.add_parser("gui", help="Launch GUI (not implemented)")
    parser_gui.set_defaults(func=gui)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
