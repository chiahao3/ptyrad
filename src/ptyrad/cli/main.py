"""
Main entry point for the PtyRAD Command-Line Interface

"""

import argparse

from .commands import (
    check_gpu,
    export_meas,
    get_examples,
    get_params,
    get_templates,
    get_walkthrough,
    gui,
    init_project,
    print_info,
    run,
    validate_params,
)

from ptyrad.runtime.logging import VERBOSITY_MAPPING

def main():
    """
    Main function for PtyRAD CLI
    """
    parser = argparse.ArgumentParser(
        description="PtyRAD Command-Line Interface",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # init
    parser_init = subparsers.add_parser("init", help="Initialize a new PtyRAD starter project")
    parser_init.add_argument("name", nargs="?", default="ptyrad",help="Name of the project folder (default: 'ptyrad')")
    parser_init.add_argument("--force", "-f", action="store_true", help="Overwrite directory if it exists")
    parser_init.set_defaults(func=init_project)

    # get-params
    parser_params = subparsers.add_parser("get-params", help="Copy all parameter files (templates + examples + walkthrough)")
    parser_params.add_argument("dest", nargs="?", default=".", help="Destination directory")
    parser_params.add_argument("--force", "-f", action="store_true", help="Overwrite existing folder")
    parser_params.set_defaults(func=get_params)

    # get-templates
    parser_temp = subparsers.add_parser("get-templates", help="Copy clean templates params files")
    parser_temp.add_argument("dest", nargs="?", default=".", help="Destination directory")
    parser_temp.add_argument("--force", "-f", action="store_true", help="Overwrite existing folder")
    parser_temp.set_defaults(func=get_templates)

    # get-examples
    parser_ex = subparsers.add_parser("get-examples", help="Copy examples params files")
    parser_ex.add_argument("dest", nargs="?", default=".", help="Destination directory")
    parser_ex.add_argument("--force", "-f", action="store_true", help="Overwrite existing folder")
    parser_ex.set_defaults(func=get_examples)
    
    # get-walkthrough
    parser_walk = subparsers.add_parser("get-walkthrough", help="Copy walkthrough params files")
    parser_walk.add_argument("dest", nargs="?", default=".", help="Destination directory")
    parser_walk.add_argument("--force", "-f", action="store_true", help="Overwrite existing folder")
    parser_walk.set_defaults(func=get_walkthrough)

    # run
    parser_run = subparsers.add_parser("run", help="Run PtyRAD reconstruction")
    parser_run.add_argument("config_path", nargs="?", help="Path to params .yaml file") # Positional arg is preferred
    parser_run.add_argument("--params_path", "-p", type=str, required=False, help="(LEGACY) Path to params .yaml file") # Legacy flag
    parser_run.add_argument("--skip_validate", action="store_true", help="Skip parameter validation and default filling. Use only if your params file is complete and consistent.")
    parser_run.add_argument("--gpuid", type=str, required=False, default="0", help="GPU ID to use ('acc', 'cpu', or an integer)")
    parser_run.add_argument("--jobid", type=int, required=False, default=0, help="Unique identifier for hypertune mode with multiple GPU workers")
    parser_run.add_argument("--seed", type=int, required=False, help="Random seed for improved reproducibility")
    parser_run.add_argument("--verbosity", "-v",type=str, default="INFO", choices=list(VERBOSITY_MAPPING.keys()), help="Set the logging verbosity level (default: INFO)")
    parser_run.set_defaults(func=run)

    # check-gpu
    parser_check_gpu = subparsers.add_parser("check-gpu", help="Check GPU availability")
    parser_check_gpu.set_defaults(func=check_gpu)

    # print-system-info
    parser_info = subparsers.add_parser("print-system-info", help="Print system info")
    parser_info.set_defaults(func=print_info)

    # export-meas
    parser_export = subparsers.add_parser("export-meas", help="Export initialized measurements file to disk")
    parser_export.add_argument("params_path", nargs="?", help="Path to params .yaml file") # Positional arg is preferred
    parser_export.add_argument("--skip_validate", action="store_true", help="Skip parameter validation and default filling. Use only if your params file is complete and consistent.")
    parser_export.add_argument("--output", type=str, help="Optional output path / file type (.mat, .hdf5, .tif, .npy) for the exported array")
    parser_export.add_argument("--reshape", type=int, nargs="+", help="Optional new shape for the exported array, e.g. --reshape 128 128 128 128")
    parser_export.add_argument("--append", action="store_true", help="Optionally appending the array shape to file name")
    parser_export.set_defaults(func=export_meas)
    
    # validate-params
    parser_validate = subparsers.add_parser("validate-params", help="Validate parameter file")
    parser_validate.add_argument("params_path", nargs="?", help="Path to params .yaml file") # Positional arg is preferred
    parser_validate.set_defaults(func=validate_params)

    # gui (placeholder) #TODO 
    parser_gui = subparsers.add_parser("gui", help="Launch GUI (not implemented yet)")
    parser_gui.set_defaults(func=gui)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
