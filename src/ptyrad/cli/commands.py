"""
Commands for the CLI, including initializing a project folder, getting params and examples files, running PtyRAD, etc.

"""

def init_project(args):
    """
    Initialize a new PtyRAD starter project
    """
    from .templates import create_starter_project
    create_starter_project(
        project_name=args.name, 
        force=args.force)

def get_params(args):
    """
    Copy all parameter files (templates + examples)
    """
    from .templates import export_params
    export_params(dest_dir=args.dest, force=args.force)

def get_templates(args):
    """
    Copy clean templates params files
    """
    from .templates import export_templates
    export_templates(dest_dir=args.dest, force=args.force)

def get_examples(args):
    """
    Copy examples params files
    """
    from .templates import export_examples
    export_examples(dest_dir=args.dest, force=args.force)

def run(args):
    """
    Run PtyRAD reconstruction
    """
    import sys

    from ptyrad.io.hierarchy import get_nested
    from ptyrad.params import load_params
    from ptyrad.runtime.device import set_accelerator, set_gpu_device
    from ptyrad.runtime.diagnostics import print_system_info
    from ptyrad.runtime.logging import LoggingManager
    from ptyrad.runtime.seed import resolve_seed_priority
    from ptyrad.solver import PtyRADSolver
    
    # Prefer positional, fallback to flag
    params_path = args.config_path or args.params_path
    
    # If neither is provided, we must fail manually because we made both optional
    if params_path is None:
        print("Error: missing params file.")
        print("Usage: ptyrad run <path/to/params.yaml>")
        sys.exit(1)
    
    # Setup LoggingManager
    LoggingManager(
        log_file='ptyrad_log.txt',
        log_dir='auto',
        prefix_time='datetime',
        prefix_jobid=args.jobid,
        append_to_file=True,
        show_timestamp=True,
        verbosity=args.verbosity
    )

    # Set up accelerator for multiGPU/mixed-precision setting, 
    # note that these we need to call the command as:
    # `accelerate launch --num_processes=2 --mixed_precision='no' -m ptyrad run <PTYRAD_ARGUMENTS> --gpuid 'acc'`
    accelerator = set_accelerator() 

    print_system_info()
    params = load_params(params_path, validate=not args.skip_validate)
    device = set_gpu_device(args.gpuid)
    seed = resolve_seed_priority(args_seed=args.seed, params_seed=get_nested(params, "init_params.random_seed", safe=True), acc=accelerator)
    ptycho_solver = PtyRADSolver(params, device=device, seed=seed, acc=accelerator)
    ptycho_solver.run()

def check_gpu(args):
    """
    Check GPU availability
    """
    from ptyrad.runtime.diagnostics import print_gpu_info
    print_gpu_info()

def print_info(args):
    """
    Print system info
    """
    from ptyrad.runtime.diagnostics import print_system_info
    print_system_info()

def export_meas(args):
    """
    Export initialized measurements file to disk
    """
    from pathlib import Path

    from ptyrad.init import Initializer
    from ptyrad.params import load_params
    
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
    """
    Validate parameter file
    """
    from ptyrad.params import load_params
    
    try:
        _ = load_params(args.params_path, validate=True)
    except Exception as e:
        print(f"Invalid parameters: {e}")

def gui(args):
    """
    Launch GUI (not implemented yet)
    """
    print("[placeholder] GUI not implemented yet.")