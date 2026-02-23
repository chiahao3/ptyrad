"""
Params files parsing functions
"""

import logging
import os
from typing import Dict

logger = logging.getLogger(__name__)

###### These are params loading functions ######

def load_params(file_path: str, validate: bool = True):
    
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The specified file '{file_path}' does not exist. Please check your file path and working directory.")
    
    logger.info("### Loading params file ###")
    logger.info(f"params_path = {file_path}")
    
    param_path, param_type = os.path.splitext(file_path)
    if param_type in (".yml", ".yaml"):
        params_dict = load_yml_params(file_path)
    elif param_type == ".toml":
        params_dict = load_toml_params(file_path)
    elif param_type == ".json":
        params_dict = load_json_params(file_path)
    elif param_type == ".py":
        params_dict =  load_py_params(param_path)
    else:
        raise ValueError("param_type needs to be either 'yml', 'json', or 'py'")
    
    # Additional correction for constraint_params (temporarily added for smooth transition to v0.1.0b11)
    if params_dict.get('constraint_params') is not None:
        params_dict['constraint_params'] = normalize_constraint_params(params_dict['constraint_params'])
        
    # Additional correction for the probe aberrations in init_params (temporatily added for smooth transition to v0.1.0b13)
    if params_dict.get('init_params') is not None:
        params_dict['init_params'] = normalize_probe_params(params_dict['init_params'])
    
    # Additional correction for model_params.obj_preblur_std (temporatily added for smooth transition to v0.1.0b13)
    if params_dict.get('model_params') is not None:
        if 'obj_preblur_std' in params_dict.get('model_params'):
            logger.warning(
            "WARNING: The 'obj_preblur_std' parameter is deprecated since v0.1.0v13. "
            "This flag will be ignored.") 
            params_dict['model_params'].pop('obj_preblur_std')
    
    # Additional correction for recon_params.if_quiet (temporatily added for smooth transition to v0.1.0b13)
    if params_dict.get('recon_params') is not None:
        if 'if_quiet' in params_dict.get('recon_params'):
            logger.warning(
            "WARNING: The 'if_quiet' parameter is deprecated since v0.1.0v13. PtyRAD now uses a central LoggingManager. "
            "This flag will be ignored.") 
            params_dict['recon_params'].pop('if_quiet')
    
    # Pass into PtyRADParams (pydantic model) for default filling and validation
    if validate:
        from .ptyrad_params import PtyRADParams
        logger.info("validate = True: Filling defaults and validating the params file...")
        params_dict = PtyRADParams(**params_dict).model_dump()
        logger.info("Success! Params file validated and defaults applied.")
    else:
        logger.warning("WARNING: validate = False: Skipping validation and default filling.")
        logger.warning("         Ensure your params file is complete and consistent.")
        logger.warning("         If you encounter issues, consider enabling validation or report the bug.")
    
    
    # Add the file path to the params_dict while we save the params file to output folder
    params_dict['params_path'] = file_path
    
    logger.info(" ")
    
    return params_dict

def load_json_params(file_path):
    import json
    
    with open(file_path, "r", encoding='utf-8') as file:
        params_dict = json.load(file)
    return params_dict

def load_toml_params(file_path):
    """
    Load parameters from a TOML file.
    
    Parameters:
    file_path (str): The path to the TOML file to be loaded.
    
    Returns:
    dict: A dictionary containing the parameters loaded from the TOML file.
    
    Raises:
    FileNotFoundError: If the specified file does not exist.
    ImportError: If the tomli package is not installed for Python < 3.11.
    """

    try:
        # Read the file with utf-8
        # Note that "A TOML file must be a valid UTF-8 encoded Unicode document." per documentation.
        # Therefore, the toml file is read in binary mode ("rb") and the encoding is handled internally.
        # But I've observed some encoding mismatch when people run the script with terminal that has different default encoding.
        # Therefore, it is safer to read it with utf-8 encoding first and pass it to tomllib.
        with open(file_path, "r", encoding='utf-8') as file:
            content = file.read()
        
        try:
            # For Python 3.11+
            import tomllib
            params_dict = tomllib.loads(content)
        except ImportError:
            # For Python < 3.11
            import tomli  # type: ignore
            params_dict = tomli.loads(content)
    except ImportError:
        raise ImportError("TOML support requires 'tomli' package for Python < 3.11 or built-in 'tomllib' for Python 3.11+. ")
    
    return params_dict

def load_yml_params(file_path):
    import yaml

    with open(file_path, "r", encoding='utf-8') as file:
        params_dict = yaml.safe_load(file)
    return params_dict

def load_py_params(file_path):
    import importlib

    params_module = importlib.import_module(file_path)
    params_dict = {
        name: getattr(params_module, name)
        for name in dir(params_module)
        if not name.startswith("__")
    }
    return params_dict

###### These are sanitization functions for backward compatibility #####

def normalize_probe_params(init_params: Dict) -> Dict:
    """ Normalize probe params in `init_params` 
    This includes:
    - Migrate legacy keys (pre v0.1.0b13) like `probe_defocus`, `probe_c3`, `probe_c5` into `probe_aberrations`.
    - Canonicalizes `probe_aberrations` into standard Krivanek polar format {'Cnm': XX, 'phinm': XX}.
    
    Note that the init_params will be normalized before optionally passing into pydantic
    """
    
    from ptyrad.optics.aberrations import Aberrations
    
    # --- STEP 1: Legacy Migration (The "Move" Phase) ---
    
    # Explictly initialize `probe_aberrations` as {} if it's missing or is set to None
    if init_params.get('probe_aberrations') is None:
        init_params['probe_aberrations'] = {}

    aberrations = init_params['probe_aberrations']
    migrated_keys = []
    
    # Define Legacy Mappings AND their Blocking Aliases
    # Format: 'legacy_key': ('canonical_modern_key', [list_of_aliases_to_check])
    legacy_map = {
        'probe_defocus': ('defocus', ['defocus', 'C1', 'C10', (1,0), '(1,0)']),
        'probe_c3':      ('C30',     ['C30', 'C3', 'Cs', (3,0), '(3,0)']),
        'probe_c5':      ('C50',     ['C50', 'C5', (5,0), '(5,0)']),
    }
    
    # Merge Logic with Precedence
    # Only migrate if the modern key is NOT already present in aberrations.
    # This ensures explicit modern config wins over legacy config.
    for legacy_key, (modern_key, blocking_aliases) in legacy_map.items():
        if legacy_key in init_params:
            legacy_val = init_params[legacy_key]

            # Check if ANY of the blocking aliases are already in the new dict.
            conflict_found = any(alias in aberrations for alias in blocking_aliases)
            
            if not conflict_found:
                aberrations[modern_key] = legacy_val
                migrated_keys.append(legacy_key)
                
            else:
                logger.warning(f"WARNING: Ignoring '{legacy_key}' because it is already defined in 'probe_aberrations' as one of {legacy_map[legacy_key][-1]}")
                pass
        
            # Old keys are deleted regardless
            del init_params[legacy_key] 
        
    if migrated_keys:
        logger.warning(f"WARNING: Probe aberrations '{migrated_keys}' in 'init_params' are depracated since PtyRAD v0.1.0b13 and are automatically converted to 'probe_aberrations' dict.")
    
    # --- STEP 2: Canonicalization (The "Clean" Phase) ---
    if aberrations:
        init_params['probe_aberrations'] = Aberrations(aberrations).export(notation='krivanek', style='polar')
    
    return init_params

def normalize_constraint_params(constraint_params):
    """Convert old constraint param format {freq} (pre v0.1.0b11) to {start_iter, step, end_iter}."""
    # Note that the constraint_params will be normalized before optionally passing into pydantic
    # so it may contain either {freq}, or {start_iter, step, end_iter}
    
    normalized_params = {}
    print_freq_warning = False
    
    for name, params in constraint_params.items():
        # Extract legacy and new parameters
        freq       = params.get("freq", None) # Legacy constraint param before PtyRAD v0.1.0b11
        start_iter = params.get("start_iter", 1 if freq is not None else None)
        step       = params.get("step", freq if freq is not None else 1)
        end_iter   = params.get("end_iter", None)
        
        if freq is not None:
            print_freq_warning = True

        # Create normalized parameters
        normalized_params[name] = {
            "start_iter": start_iter,
            "step": step,
            "end_iter": end_iter,
            **{k: v for k, v in params.items() if k not in ("freq", "step", "start_iter", "end_iter")},  # Copy other keys
        }

    if print_freq_warning:
        logger.warning("WARNING: For constraint_params, 'freq' is depracated since PtyRAD v0.1.0b11 and is automatically converted to 'step'.")
    
    return normalized_params

###### Params exporting / copying #####

def copy_params_to_dir(params_path, output_dir, params=None):
    """
    Copies the params file to the output directory if it exists. If the params file does not exist,
    it dumps the provided params dictionary to a YAML file in the output directory.

    Args:
        params_path (str): Path to the params file (can be None if params are programmatically generated).
        output_dir (str): Directory where the params file or YAML dump will be saved.
        params (dict, optional): The programmatically generated params dictionary to save if no file exists.
    """
    import os
    import shutil

    import yaml

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    if params_path and os.path.isfile(params_path):
        # If the params file exists, copy it to the output directory
        file_name = os.path.basename(params_path)
        output_path = os.path.join(output_dir, file_name)
        shutil.copy2(params_path, output_path)

    elif params is not None:
        # If no file exists, dump the params dictionary to a YAML file
        output_path = os.path.join(output_dir, "params_dumped.yml")
        with open(output_path, "w") as f:
            yaml.safe_dump(params, f, sort_keys=False)

    else:
        # If neither a file nor params are provided, skip with a warning
        return
    
def yaml2json(input_filepath, output_filepath):
    import json

    import yaml
    with open(input_filepath, 'r') as file:
        try:
            # Load as YAML
            data = yaml.safe_load(file)
            
            # Save to JSON
            with open(output_filepath, 'w') as json_file:
                json.dump(data, json_file, indent=4)
                
            print(f"YAML {input_filepath} has been successfully converted and saved to JSON {output_filepath}")

        except yaml.YAMLError as e:
            print("Error parsing YAML file:", e)