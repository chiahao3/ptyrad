"""
Reconstruction provenance handling (load/save)

"""

import datetime
import json
import os
import uuid
from pathlib import Path

import h5py
import numpy as np


class SafeJSONEncoder(json.JSONEncoder):
    """Sanitizes scientific types into JSON-safe formats.

    This custom encoder ensures that NumPy numerical types, arrays, 
    pathlib Path objects, and stray PyTorch tensors can be cleanly 
    serialized into the provenance JSON string without throwing 
    TypeErrors.
    """
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            # We don't save arrays in the history log, just their shape/type
            return f"<Array shape={obj.shape} dtype={obj.dtype}>"
        elif isinstance(obj, Path):
            return str(obj)
        # Catch-all for Torch objects if they accidentally slip in
        if hasattr(obj, 'cpu'): 
            return f"<{type(obj).__name__}>"
        return super().default(obj)

def collect_provenance(init_params):
    """Initializes the provenance tracking dictionary for a new run.

    This function scans the initialization parameters to determine the 
    origin of the probe, object, positions, and tilts. It handles file 
    loading (including inheriting provenance from older PtyRAD HDF5 files), 
    simulations, and in-memory custom arrays.

    Args:
        init_params (dict): The initialization parameters for the reconstruction.

    Returns:
        dict: A dictionary mapping component names ('probe', 'pos', 'obj', 'tilt') 
        to a list of provenance entry dictionaries representing their history.
    """
    import json
    import os

    import h5py

    from ptyrad.io.load import load_ptyrad

    # 1. Map internal component names to their config keys
    component_config_map = [
        {'name': 'probe', 'src_key': 'probe_source', 'param_key': 'probe_params'},
        {'name': 'pos',   'src_key': 'pos_source',   'param_key': 'pos_params'},
        {'name': 'obj',   'src_key': 'obj_source',   'param_key': 'obj_params'},
        {'name': 'tilt',  'src_key': 'tilt_source',  'param_key': 'tilt_params'}
    ]

    provenance = {item['name']: [] for item in component_config_map}

    for comp in component_config_map:
        name = comp['name']
        source_type = init_params.get(comp['src_key'])
        param_data  = init_params.get(comp['param_key'])

        # Case A: File-based sources (PtyRAD, PtyShv, py4DSTEM, file)
        if source_type in ['PtyRAD', 'PtyShv', 'py4DSTEM', 'file', 'foldslice_hdf5']:
             
            # 1. Resolve to absolute path
            if isinstance(param_data, str):
                abs_path = os.path.abspath(param_data)
            else:
                # Fallback if param_data is somehow not a string (e.g. None)
                abs_path = "Unknown_Path"

            # 2. Default: Create a "Genesis" entry (Fresh Import)
            # We assume this is a fresh start unless proven otherwise
            entry_list = [_create_genesis_entry(
                action=f'Imported {source_type}',
                details={'original_path': abs_path},
                name=abs_path
            )]

            # 3. PtyRAD Special Handling (Inheritance or Legacy Extraction)
            if source_type == 'PtyRAD' and os.path.exists(abs_path):
                try:
                    with h5py.File(abs_path, 'r') as f:
                        json_str = f.attrs.get('provenance_json', None)
                        
                    if json_str:
                        # Modern File: Inherit the bloodline
                        parent_full_prov = json.loads(json_str)
                        inherited_list = parent_full_prov.get(name, [])
                        if inherited_list:
                            entry_list = inherited_list
                        
                    else:
                        # Legacy PtyRAD File: Enhance the Genesis Entry
                        # Extract version and params
                        ckpt = load_ptyrad(abs_path)
                        ver = ckpt.get('ptyrad_version')
                        legacy_params = ckpt.get('params')

                        # UPDATE the entry
                        entry_list[0]['action'] = f'Imported PtyRAD {ver}'
                        if legacy_params:
                            entry_list[0]['metadata']['legacy_params'] = legacy_params

                except Exception as e:
                    print(f"[Provenance Warning] Failed to inspect {abs_path}: {e}")
        
            provenance[name] = entry_list

        # Case B: Simulation
        elif source_type == 'simu':
             provenance[name] = [_create_genesis_entry(
                 action='Simulated',
                 details={'sim_params': param_data},
                 name='<Simulation>'
             )]

        # Case C: Custom Array
        elif source_type == 'custom':
            shape_info = str(param_data.shape) if hasattr(param_data, 'shape') else 'unknown'
            provenance[name] = [_create_genesis_entry(
                action='Custom Array',
                details={'shape': shape_info},
                name='<In-Memory Array>'
            )]
            
        # Case D: Unknown/Null
        else:
            if source_type is not None:
                provenance[name] = [_create_genesis_entry(
                    action=f'Unknown Source: {source_type}',
                    details={'params': str(param_data)[:100]},
                    name='<Unknown>'
                )]

    return provenance

def _create_genesis_entry(action, details, name="Unknown Source"):
    """Creates a base provenance entry for a new component.

    Args:
        action (str): The action that created the component (e.g., 'Simulated').
        details (dict): Metadata and parameters associated with the action.
        name (str, optional): A descriptive name or path. Defaults to "Unknown Source".

    Returns:
        dict: A structured dictionary representing the genesis event.
    """
    return {
        'uid': str(uuid.uuid4())[:8],
        'timestamp': datetime.datetime.now().isoformat(),
        'run_name': name,
        'action': action,
        'metadata': details
    }

def generate_provenance_json(current_provenance, params, output_filename="current_run"):
    """Appends the current run details to the provenance timeline and serializes it.

    This safely deepcopies the existing provenance history, appends the current 
    run's configuration to every component's timeline, and encodes the entire 
    history into a JSON string using `SafeJSONEncoder`.

    Args:
        current_provenance (dict): The inherited provenance dictionary.
        params (dict): The configuration parameters for the current run.
        output_filename (str or Path, optional): The filename or path identifying 
            this specific run. Defaults to "current_run".

    Returns:
        str: A JSON-formatted string of the complete provenance history.
    """
    
    import copy
    import os

    from ptyrad import __version__
    
    # Ensure we log the FULL path, not just 'output.h5'
    # Check if it looks like a file path (contains separator or extension)
    if '.' in str(output_filename) or os.sep in str(output_filename):
        run_identifier = os.path.abspath(str(output_filename))
    else:
        run_identifier = str(output_filename)

    # 1. Create the Entry
    run_entry = {
        'uid': str(uuid.uuid4())[:8],
        'timestamp': datetime.datetime.now().isoformat(),
        'ptyrad_version': __version__,
        'run_name': run_identifier, # <--- Will now be /home/user/exp/output.h5
        'note': params.get('experiment_note', 'PtyRAD Run'),
        'params': params
    }
    
    # 2. Append to everything
    # We deepcopy so we don't mutate the runtime history (in case you save multiple snapshots)
    final_provenance = copy.deepcopy(current_provenance)
    
    for component_name, timeline in final_provenance.items():
        # Append this run to the end of the list
        # This signifies: "This component was present in this run"
        timeline.append(run_entry)
            
    # 3. Serialize
    return json.dumps(final_provenance, cls=SafeJSONEncoder, indent=2)

def save_provenance_to_hdf5(hdf5_path, provenance_json_str):
    """Saves the serialized provenance JSON string to an HDF5 file attribute.

    Args:
        hdf5_path (str or Path): The path to the target HDF5 file.
        provenance_json_str (str): The JSON string representing the history.
    """
    
    with h5py.File(hdf5_path, 'r+') as f:
        f.attrs['provenance_json'] = provenance_json_str
        
def load_provenance_from_h5(file_path):
    """Reads and parses the provenance history from an HDF5 file.

    Extracts the 'provenance_json' attribute from the root of the specified 
    HDF5 file and parses it back into a Python dictionary.

    Args:
        file_path (str or Path): The path to the HDF5 file.

    Returns:
        dict: The lineage dictionary (e.g., {'probe': [...], 'obj': [...]}). 
        Returns an empty dict `{}` if the attribute is missing or corrupted.

    Raises:
        FileNotFoundError: If the specified HDF5 file does not exist.
    """
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cannot load provenance: File {file_path} not found.")

    with h5py.File(file_path, 'r') as f:
        # 1. safe .get() - prevents KeyError if attribute doesn't exist
        # Default to '{}' (empty JSON object) so json.loads doesn't crash
        json_str = f.attrs.get('provenance_json', '{}')
    
    # 2. Parse the string back into Python objects
    try:
        provenance_data = json.loads(json_str)
        return provenance_data
    except json.JSONDecodeError as e:
        print(f"[Warning] Provenance attribute in {file_path} is corrupted: {e}")
        return {}
    
def export_hdf5_provenance_to_json(h5_path, output_json_path=None):
    """Extracts provenance history from an HDF5 file and saves it as JSON.

    Args:
        h5_path (str or Path): The path to the source .h5 file.
        output_json_path (str, optional): The destination path for the .json file. 
            If None, it defaults to '<h5_filename>_provenance.json' in the same 
            directory. Defaults to None.

    Returns:
        str: The path to the generated JSON output file.
    """
    h5_path = str(h5_path) # Handle Path objects
    
    # 1. Determine output path if not provided
    if output_json_path is None:
        base_name = os.path.splitext(h5_path)[0]
        output_json_path = f"{base_name}_provenance.json"

    # 2. Load the data
    print(f"Reading provenance from: {h5_path}")
    history_data = load_provenance_from_h5(h5_path)
    
    if not history_data:
        print("Warning: No provenance data found (or file is empty). JSON will be empty.")

    # 3. Write to JSON file
    try:
        with open(output_json_path, 'w') as f:
            json.dump(history_data, f, indent=2)
        print(f"Successfully exported to: {output_json_path}")
    except IOError as e:
        print(f"Error writing JSON file: {e}")

    return output_json_path