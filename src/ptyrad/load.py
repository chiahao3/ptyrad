"""
Loading functions for different file types including .mat, .hdf5, .tif, .raw, .yaml, etc.

"""

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import scipy.io as sio

from ptyrad.utils import (
    get_nested,
    handle_hdf5_types,
    list_nested_keys,
    tensors_to_ndarrays,
    vprint,
)
from ptyrad.utils.aberrations import Aberrations

KeyType = Union[str, list[str], None]

###### These are data loading functions ######

def load_raw(file_path, shape, dtype=np.float32, offset=0, gap=1024):
    # shape = (N, height, width)
    # np.fromfile with custom dtype is faster than the np.read and np.frombuffer
    # This implementaiton is also roughly 2x faster (10sec vs 20sec) than load_hdf5 with a 128x128x128x128 (1GB) EMPAD dataset
    # Note that for custom processed empad2 raw there might be no gap between the images
    N, height, width = shape

    # Verify file size first
    expected_size = offset + N * (height * width * dtype().itemsize + gap)
    actual_size = os.path.getsize(file_path)

    if actual_size != expected_size:
        raise ValueError(f"Mismatch in expected ({expected_size} bytes = offset + N * (height * width * 4 + gap)) vs. actual ({actual_size} bytes) file size! Check your loading configurations!")
    
    # Define the custom dtype to include both data and gap
    custom_dtype = np.dtype([
        ('data', dtype, (height, width)),
        ('gap', np.uint8, gap)  # uint8 means 1 byte per gap element
    ])

    # Read the entire file using the custom dtype
    with open(file_path, 'rb') as f:
        f.seek(offset)
        raw_data = np.fromfile(f, dtype=custom_dtype, count=N)

    # Extract just the 'data' part (ignoring the gaps)
    data = raw_data['data']
    vprint("Success! Loaded .raw file path =", file_path)
    vprint("Imported .raw data shape =", data.shape)
    vprint("Imported .raw data type =", data.dtype)
    return data

def load_tif(file_path):
    from tifffile import imread

    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The specified file '{file_path}' does not exist. Please check your file path and working directory.")
    
    data = imread(file_path)
    vprint("Success! Loaded .tif file path =", file_path)
    vprint("Imported .tif data shape =", data.shape)
    return data

def load_npy(file_path):

    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The specified file '{file_path}' does not exist. Please check your file path and working directory.")
    
    data = np.load(file_path)
    vprint("Success! Loaded .npy file path =", file_path)
    vprint("Imported .npy data shape =", data.shape)
    return data

def load_array_from_file(
    path: str,
    key: Optional[str] = None,
    ndims: Optional[List[int]] = None,
    shape: Optional[Tuple[int, ...]] = None,
    offset: Optional[int] = None,
    gap: Optional[int] = None,
) -> np.ndarray:
    """
    Load array from a file. The file type is inferred from the extension.
    Currently supports .tif, .tiff, .npy, .mat, .h5, .hdf5, and .raw.

    Args:
        path (str): Path to the file.
        key (str): Key to specify the dataset (optional).
        ndims (list): List of desired dimensions for filtering datasets.
        shape (tuple): Shape of the data for .raw files (optional).
        offset (int): Offset for .raw files (optional).
        gap (int): Gap for .raw files (optional).

    Returns:
        numpy.ndarray: The loaded array.

    Raises:
        ValueError: If the file type is unsupported or no valid dataset is found.
    """

    file_path = path  # The function signature is simplified for users, although I think file_path is clearer

    # Check file existence
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The specified file '{file_path}' does not exist. Please check your file path and working directory.")

    # Infer file type from extension
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext in [".tif", ".tiff"]:
        return load_tif(file_path)

    elif ext == ".npy":
        return load_npy(file_path)

    elif ext in [".mat", ".h5", ".hdf5"]:
        return load_ND_with_key(file_path, key, ndims)

    elif ext == ".raw":
        if shape is None:
            raise ValueError(
                f"Please at least provide 'shape' of the expected data array to correctly load the .raw file {file_path}."
            )
        raw_args = {"shape": shape, "offset": offset, "gap": gap}
        raw_args = {
            k: v for k, v in raw_args.items() if v is not None
        }  # Remove argument with None
        return load_raw(file_path, **raw_args)

    else:
        raise ValueError(
            f"Unsupported file type: '{ext}'. Supported types are .tif, .tiff, .mat, .h5, .hdf5, .npy, and .raw."
        )

def load_ND_with_key(
    file_path: str,
    key: Optional[str] = None,
    ndims: Optional[List[int]] = None,
    verbose: bool = True,
) -> np.ndarray:
    """
    Load exactly one ND dataset from (possibly nested) files like .mat and .hdf5.

    Args:
        file_path (str): Path to the file.
        key (str, optional): Key to specify the dataset. If not provided, will search for all valid ND datasets.
        ndims (list): List of desired dimensions for filtering datasets.
        verbose (bool): Whether to print information about the datasets.

    Returns:
        numpy.ndarray: The loaded dataset.

    Raises:
        ValueError: If the file type is unsupported, or the key is invalid, or multiple/zero valid datasets are found.
    """

    if ndims is None:
        ndims = [3, 4]

    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"The specified file '{file_path}' does not exist. Please check your file path and working directory."
        )

    # Infer file type from extension
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    # Select loader
    if ext == ".mat":
        load_func = load_mat
    elif ext in [".h5", ".hdf5"]:
        load_func = load_hdf5
    else:
        raise ValueError(
            f"Unsupported file type: '{ext}'. Supported types are .mat, .h5, .hdf5."
        )

    # Load the data using the selected loader.
    if key in (None, ""):
        datasets_dict = load_func(file_path)  # None key would return a dict of the file
        valid_datasets = collect_ND_datasets(
            datasets_dict, ndims=ndims, verbose=verbose
        )  # This will search recursively and return all valid ND datasets
        if len(valid_datasets) == 1:
            return next(iter(valid_datasets.values()))
        elif len(valid_datasets) == 0:
            raise ValueError(
                f"No eligible datasets found in file with ndims = {ndims}. Please check the file and file path."
            )
        else:
            raise ValueError(
                f"Multiple eligible ND datasets found: {list(valid_datasets.keys())}. Please specify the dataset key explicitly."
            )

    elif isinstance(key, str):
        data_or_dict = load_func(
            file_path, key
        )  # String key would normally return ndarray, but incorrectly specified key may point to a group or anything else
        if isinstance(data_or_dict, np.ndarray):
            return data_or_dict
        else:
            raise ValueError(
                f"The returned value at key '{key}' is not an ndarray dataset, got type = {type(data_or_dict).__name__}. "
                "If you don't know the correct dataset key, try 'key=None' to search for eligible ND datasets from the entire file."
            )
    else:
        raise TypeError(f"`key` must be None or a string, but got key = '{key}'")

def collect_ND_datasets(
    data_dict: dict[str, Any],
    ndims: list[int] = None,
    delimiter: str = ".",
    verbose: bool = True,
    _parent_key: Optional[str] = None,
) -> dict[str, np.ndarray]:
    """
    Collect ND numpy arrays from a (possibly nested) dictionary that match desired dimensionalities.

    Automatically traverses nested dictionaries and flattens keys with '//'.

    Args:
        data_dict (dict): Dictionary of datasets (flat or nested).
        ndims (list of int): Desired dimensionalities to match (e.g., [3, 4]).
        delimiter (str): String symbol used to seperate different levels of the full path to the dataset
        verbose (bool): Whether to print matched datasets.
        _parent_key (str, optional): **Internal use only.** Tracks nested keys during recursion. Do not set manually.

    Returns:
        dict[str, np.ndarray]: Matching datasets with flattened hierarchical keys.

    Raises:
        ValueError: If input is not a dict or no datasets match.
    """
    if not isinstance(data_dict, dict):
        raise ValueError("Input must be a dictionary containing datasets.")

    if ndims is None:
        ndims = [3, 4]

    results: dict[str, np.ndarray] = {}

    for key, val in data_dict.items():
        full_key = f"{_parent_key}{delimiter}{key}" if _parent_key else key

        if isinstance(val, np.ndarray):
            if val.ndim in ndims:
                results[full_key] = val

        elif isinstance(val, dict):
            results.update(
                collect_ND_datasets(
                    val, ndims=ndims, verbose=False, _parent_key=full_key
                )
            )

    if verbose and results:
        vprint(f"Found the following ND datasets with ndim in {ndims}:")
        for k, arr in results.items():
            vprint(f"  Key: '{k}', Shape: {arr.shape}, Dtype: {arr.dtype}")

    return results
    
###### These are reconstruction file loading functions ######
# Note that .mat and .hdf5 are also used for normal data

def load_mat(
    file_path: str, key: KeyType = None, delimiter: str = ".",
    squeeze_me=True, simplify_cells=True
) -> Union[np.ndarray, dict[str, np.ndarray]]:
    """
    Load dataset(s) from a MATLAB .mat file, handling both default and v7.3 (HDF5) formats.
    The version is used to switch between scipy.io.loadmat or h5py.

    Parameters:
        file_path (str): Path to the .mat file.
        key (str | list[str] | None): Name(s) of the dataset(s) to load.
            - If None, '', or []: Load all datasets, preserving the original nested structure.
            - If str: Load a single dataset or group. Supports hierarchical keys (e.g., 'group1.dataset1').
            - If list[str]: Load multiple datasets. The returned dictionary will have a flattened structure.
        delimiter (str): Delimiter for hierarchical keys (default: ".").
        squeeze_me (bool): Whether to squeeze unit matrix dimensions (scipy.io.loadmat parameter).
        simplify_cells (bool): Whether to simplify cell arrays (scipy.io.loadmat parameter).

    Returns:
        data (np.ndarray or dict): The loaded dataset(s) with the same structure as load_hdf5.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        KeyError: If provided key(s) are not found in the file.
        TypeError: If the key is not None, a string, or a list of strings.
    """

    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"The specified file '{file_path}' does not exist. Please check your file path or working directory."
        )

    # Check file version
    from scipy.io.matlab import matfile_version as get_matfile_version
    try:
        mat_version = get_matfile_version(file_path)
    except ValueError as e:
        vprint(f"WARNING: {e}. Switching to `load_hdf5` as it's probably not generated by MATLAB.")
        mat_version = (2,0) # Since Scipy can't find the version, it's likely a fake mat file that's actually HDF5
    is_hdf5_format = (mat_version[0] == 2)
    
    # If v7.3 (HDF5), delegate to load_hdf5 directly
    if is_hdf5_format:
        vprint("Detected .mat v7.3 (HDF5 format). Delegating to `load_hdf5`.")
        return load_hdf5(file_path, key=key, delimiter=delimiter)
    
    # Handle normal .mat formats
    vprint("Detected .mat version less than v7.3. Using `scipy.io.loadmat`.")
    
    # Load the entire .mat file first
    mat_contents = sio.loadmat(file_path, squeeze_me=squeeze_me, simplify_cells=simplify_cells) # mat_contents is already a nested dict
    
    # Handle different key scenarios
    if key in (None, "", []):
        vprint(f"Success! Loaded .mat file as a dict from path = '{file_path}'")
        return mat_contents
    
    elif isinstance(key, str):
        data = get_nested(mat_contents, key=key, delimiter=delimiter)
        vprint(
            f"Success! Loaded .mat file with key = '{key}' from path = '{file_path}'"
        )
        if isinstance(data, np.ndarray):
            vprint(f"Imported .mat data shape = {data.shape}")
            vprint(f"Imported .mat data type = {data.dtype}")
        return data
    
    elif isinstance(key, list):
        if not all(isinstance(k, str) for k in key):
            raise TypeError(
                f"All elements in 'key' list must be strings, got {[type(k).__name__ for k in key]}"
            )
        missing = []
        datasets_dict = {}
        
        for k in key:
            try:
                datasets_dict[k] = get_nested(mat_contents, key=k, delimiter=delimiter)
            except KeyError:
                missing.append(k)
                
        if missing:
            raise KeyError(
                f"Key(s) = {missing} not found. "
                f"Available key(s) in this mat file are {list_nested_keys(mat_contents)}. "
                "Tip: If you don't know the correct key, try 'key=None' to load the entire file as a dict."
            )

        vprint(
            f"Success! Loaded .hdf5 file as a dict with keys = {key} from path = '{file_path}'"
        )
        return datasets_dict

    else:
        raise TypeError(
            f"`key` must be None, a string, or a list of strings but got key = '{key}'"
        )    

def load_hdf5(
    file_path: str, key: KeyType = None, delimiter: str = "."
) -> Union[np.ndarray, dict[str, np.ndarray]]:
    """
    Load dataset(s) from an HDF5 file, recursively if groups are encountered.

    Parameters:
        file_path (str): Path to the HDF5 file.
        key (str | list[str] | None): Name(s) of the dataset(s) to load.
            - If None, '', or []: Load all datasets recursively, preserving the original nested structure.
            - If str: Load a single dataset or group. Supports hierarchical keys (e.g., 'group1.dataset1').
            - If list[str]: Load multiple datasets. The returned dictionary will have a flattened structure with the hierarchical key strings as keys.
        delimiter (str): Delimiter for hierarchical keys (default: ".").

    Returns:
        data (np.ndarray or dict): The loaded dataset(s).
            - If `key` is a string, returns a single `np.ndarray` or a nested dictionary if the key points to a group.
            - If `key` is a list of strings, returns a dictionary with the hierarchical key strings as keys and the corresponding datasets as values.
            - If `key` is None, returns a nested dictionary preserving the original structure of the HDF5 file.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        KeyError: If provided key(s) are not found in the file.
        TypeError: If the key is not None, a string, or a list of strings.

    Notes:
        - Hierarchical Keys:
            - The function supports hierarchical keys (e.g., 'group1.dataset1') to directly access nested datasets or groups.
            - When a list of hierarchical keys is provided, the returned dictionary will have a flattened structure with the hierarchical key strings as keys.
        - Preserving Original Structure:
            - If `key=None`, the function recursively loads all datasets and groups, preserving the original nested structure of the HDF5 file.
        - Performance Considerations:
            - Providing an exact key (e.g., `key="group1/dataset1"`) is significantly faster than recursively loading the entire file or traversing the hierarchy.

    """

    def _recursively_load(hobj, key=None, delimiter="."):
        """Recursively load h5py Group or Dataset into dict or array."""

        # Traverse hierarchically with a user-specified key
        if key is not None:
            parts = key.split(delimiter)
            for part in parts:
                if not isinstance(hobj, (h5py.Group, h5py.File)) or part not in hobj:
                    raise KeyError(
                        f"Key '{key}' not found. Failed at '{part}'. "
                        f"Available key(s) in this HDF5 file are {list_nested_keys(hf)}. "
                        "Tip: If you don't know the correct key, try 'key=None' to load the entire file as a dict."
                    )
                hobj = hobj[part]

        # Load the object without user-specified key
        if isinstance(hobj, h5py.Dataset):
            return handle_hdf5_types(hobj[()])
        elif isinstance(hobj, h5py.Group):
            return {k: _recursively_load(hobj[k]) for k in hobj}
        else:
            raise TypeError(f"Unsupported HDF5 object type: {type(hobj)}")

    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"The specified file '{file_path}' does not exist. Please check your file path or working directory."
        )

    with h5py.File(file_path, "r") as hf:
        if key in (None, "", []):
            file_dict = {k: _recursively_load(hf[k]) for k in hf.keys()}
            vprint(f"Success! Loaded .hdf5 file as a dict from path = '{file_path}'")
            return file_dict

        elif isinstance(key, str):
            data = _recursively_load(hf, key=key, delimiter=delimiter)
            vprint(
                f"Success! Loaded .hdf5 file with key = '{key}' from path = '{file_path}'"
            )
            if isinstance(data, np.ndarray):
                vprint(f"Imported .hdf5 data shape = {data.shape}")
                vprint(f"Imported .hdf5 data type = {data.dtype}")
            return data

        elif isinstance(key, list):
            if not all(isinstance(k, str) for k in key):
                raise TypeError(
                    f"All elements in 'key' list must be strings, got {[type(k).__name__ for k in key]}"
                )
            datasets_dict = {}
            missing = []
            
            for k in key:
                try:
                    datasets_dict[k] = _recursively_load(hf, key=k, delimiter=delimiter)
                except KeyError:
                    missing.append(k)
                
            if missing:
                raise KeyError(
                    f"Key(s) = {missing} not found. Available key(s) in this HDF5 file are {list_nested_keys(hf)}. "
                    "Tip: If you don't know the correct key, try 'key=None' to load the entire file as a dict."
                )

            vprint(
                f"Success! Loaded .hdf5 file as a dict with keys = {key} from path = '{file_path}'"
            )
            return datasets_dict

        else:
            raise TypeError(
                f"`key` must be None, a string, or a list of strings but got key = '{key}'"
            )

def load_pt(file_path, weights_only=False):
    import torch

    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The specified file '{file_path}' does not exist. Please check your file path and working directory.")

    data = torch.load(file_path, weights_only=weights_only) 
    # The default behavior of torch.load is `weights_only=True` since PyTorch 2.6 (2025.01.29)
    # https://dev-discuss.pytorch.org/t/bc-breaking-change-torch-load-is-being-flipped-to-use-weights-only-true-by-default-in-the-nightlies-after-137602/2573
    # Because PtyRAD .pt isn't a true PyTorch model, so `weights_only=True` would break this critical loading function.
    # However, `weights_only=False` has potential risk if the .pt file contains malicious code, so please only use this `load_pt` for PtyRAD-generated .pt file.
    
    vprint("Success! Loaded .pt file path =", file_path)
    return data

def load_ptyrad(file_path: str) -> Dict[str, Any]:
    """
    Load PtyRAD reconstruction files based on their file extension.

    This function supports loading files with extensions `.h5`, `.hdf5`, and `.pt`.
    The file type is inferred from the extension, and the appropriate loader function is called.
    The suggested model output file type has changed to HDF5 since PtyRAD v0.1.0b7 for cross-platform interoperability.

    Args:
        file_path (str): Path to the file to be loaded.

    Returns:
        Any: The loaded data, typically as a numpy array or dictionary, depending on the file type.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file type is unsupported.

    Notes:
        - `.h5` and `.hdf5` files are loaded using the `load_hdf5` function.
        - `.pt` files are loaded using the `load_pt` function and converted to numpy arrays for backward compatibility.
        - Unsupported file types will raise a `ValueError`.

    Example:
        ```python
        data = load_ptyrad("example.h5")
        ```
    """
    
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The specified file '{file_path}' does not exist. Please check your file path and working directory.")
    
    # Infer file type from extension
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext in [".h5", ".hdf5"]:
        return load_hdf5(file_path)

    elif ext == ".pt":
        vprint("WARNING: Loading PtyRAD reconstruction from .pt file is deprecated and will likely be removed by 2025 Aug.")
        vprint("INFO: PtyRAD reconstruction output has been using .hdf5 format since v0.1.0b7.")
        return tensors_to_ndarrays(load_pt(file_path)) # .pt is supported for backward compatibility before 0.1.0b7. (e.g. PtyRAD reconstructions used for the paper)
    
    else:
        raise ValueError(
            f"Unsupported file type: '{ext}'. Supported types are .h5, .hdf5, and .pt."
        )

###### These are params loading functions ######

def load_params(file_path: str, validate: bool = True):
    
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The specified file '{file_path}' does not exist. Please check your file path and working directory.")
    
    vprint("### Loading params file ###")
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
    
    # Pass into PtyRADParams (pydantic model) for default filling and validation
    if validate:
        from ptyrad.params import PtyRADParams
        vprint("validate = True: Filling defaults and validating the params file...")
        params_dict = PtyRADParams(**params_dict).model_dump()
        vprint("Success! Params file validated and defaults applied.")
    else:
            vprint("WARNING: validate = False: Skipping validation and default filling.")
            vprint("         Ensure your params file is complete and consistent.")
            vprint("         If you encounter issues, consider enabling validation or report the bug.")
    
    # Add the file path to the params_dict while we save the params file to output folder
    params_dict['params_path'] = file_path
    
    vprint(" ")
    return params_dict

def load_json_params(file_path):
    import json
    
    with open(file_path, "r", encoding='utf-8') as file:
        params_dict = json.load(file)
    vprint("Success! Loaded .json file path =", file_path)
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
    
    vprint("Success! Loaded .toml file path =", file_path)
    return params_dict

def load_yml_params(file_path):
    import yaml

    with open(file_path, "r", encoding='utf-8') as file:
        params_dict = yaml.safe_load(file)
    vprint("Success! Loaded .yml file path =", file_path)
    return params_dict

def load_py_params(file_path):
    import importlib

    params_module = importlib.import_module(file_path)
    params_dict = {
        name: getattr(params_module, name)
        for name in dir(params_module)
        if not name.startswith("__")
    }
    vprint("Success! Loaded .py file path =", file_path)
    return params_dict

###### These are sanitization functions for backward compatibility #####

def normalize_probe_params(init_params: Dict) -> Dict:
    """ Normalize probe params in `init_params` 
    This includes:
    - Migrate legacy keys (pre v0.1.0b13) like `probe_defocus`, `probe_c3`, `probe_c5` into `probe_aberrations`.
    - Canonicalizes `probe_aberrations` into standard Krivanek polar format {'Cnm': XX, 'phinm': XX}.
    
    Note that the init_params will be normalized before optionally passing into pydantic
    """
    
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
                vprint(f"Ignoring '{legacy_key}' because it is already defined in 'probe_aberrations' as one of {legacy_map[legacy_key][-1]}")
                pass
        
            # Old keys are deleted regardless
            del init_params[legacy_key] 
        
    if migrated_keys:
        vprint(f"WARNING: Probe aberrations '{migrated_keys}' in 'init_params' are depracated since PtyRAD v0.1.0b13 and are automatically converted to 'probe_aberrations' dict.")
    
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
        vprint("WARNING: For constraint_params, 'freq' is depracated since PtyRAD v0.1.0b11 and is automatically converted to 'step'.")
    
    return normalized_params