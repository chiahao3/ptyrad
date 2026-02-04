"""
Commands for the CLI, including initializing a project folder, getting params and examples files, running PtyRAD, etc.

"""

def init_project(args):
    from .templates import create_starter_project
    create_starter_project(
        project_name=args.name, 
        force=args.force)

def get_params(args):
    from .templates import export_params
    export_params(dest_dir=args.dest, force=args.force)

def get_templates(args):
    from .templates import export_templates
    export_templates(dest_dir=args.dest, force=args.force)

def get_examples(args):
    from .templates import export_examples
    export_examples(dest_dir=args.dest, force=args.force)

def run(args):
    import sys
    from ptyrad.load import load_params
    from ptyrad.solver import PtyRADSolver
    from ptyrad.utils import CustomLogger, get_nested, print_system_info, resolve_seed_priority, set_accelerator, set_gpu_device
    
    # Prefer positional, fallback to flag
    params_path = args.config_path or args.params_path
    
    # If neither is provided, we must fail manually because we made both optional
    if params_path is None:
        print("Error: missing params file.")
        print("Usage: ptyrad run <path/to/params.yaml>")
        sys.exit(1)
    
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
    params = load_params(params_path, validate=not args.skip_validate)
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

def export_meas(args):
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