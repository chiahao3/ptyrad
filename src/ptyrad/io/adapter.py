import numpy as np
import torch


def tensors_to_ndarrays(data):
    """
    Recursively convert all torch.Tensor instances in any nested structure 
    (including lists, dicts, and tuples) into numpy.ndarray.

    This function supports both single objects and arbitrarily nested 
    container types. Non-tensor types are returned unchanged.

    Args:
        data: Input data which can be nested dict, list, tuple, torch.Tensor, or other.

    Returns:
        The same structure with torch.Tensor replaced by numpy.ndarray.
    """
    
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    elif isinstance(data, dict):
        return {k: tensors_to_ndarrays(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [tensors_to_ndarrays(x) for x in data]
    elif isinstance(data, tuple):
        return tuple(tensors_to_ndarrays(x) for x in data)
    else:
        return data

def ndarrays_to_tensors(data, device='cuda'):
    """
    Recursively convert all numpy.ndarray instances in any nested structure 
    (including lists, dicts, and tuples) into torch.Tensor on the specified device.

    This function supports both single objects and arbitrarily nested 
    container types. Non-array types are returned unchanged.

    Args:
        data: Input data which can be nested dict, list, tuple, np.ndarray, or other.
        device (str): Device on which to place the tensors. Default is 'cuda'.

    Returns:
        The same structure with numpy.ndarray replaced by torch.Tensor.
    """
    
    if isinstance(data, np.ndarray):
        return torch.tensor(data).to(device)
    elif isinstance(data, dict):
        return {k: ndarrays_to_tensors(v, device=device) for k, v in data.items()}
    elif isinstance(data, list):
        return [ndarrays_to_tensors(x, device=device) for x in data]
    elif isinstance(data, tuple):
        return tuple(ndarrays_to_tensors(x, device=device) for x in data)
    else:
        return data

def handle_hdf5_types(x):
    """
    Convert data to native Python or NumPy types. Especially when loaded by h5py.

    Handles special cases like MATLAB v7.3 complex128 data types and ensures
    that data is converted to a format compatible with native Python or NumPy.

    Also handles sentinel string "__NONE__" as a substitute for None in HDF5.

    Args:
        x: The input data to be converted.

    Returns:
        The converted data into native Python or NumPy types.
    """
    # Handle scalar Numpy types
    if isinstance(x, np.generic):
        x = x.item()

    # Handle 0-dimensional Numpy arrays (convert to Python scalars) as they were probably forced by HDF5
    if isinstance(x, np.ndarray) and x.ndim == 0:
        x = x.item()

    # Handle bytes (e.g., HDF5 strings or sentinel)
    if isinstance(x, bytes):
        try:
            x = x.decode('utf-8')
        except UnicodeDecodeError:
            return x  # Leave undecodable bytes unchanged

    # Convert sentinel string to None — only safe for scalar strings
    if isinstance(x, str) and x == "__NONE__":
        return None

    # Handle MATLAB-style complex128 compound dtype
    if isinstance(x, np.ndarray) and x.dtype == [('real', '<f8'), ('imag', '<f8')]:
        print(f"Detected data.shape = {x.shape} with data.dtype = {x.dtype}. Casting back to 'complex128'.")
        return x.view(np.complex128)

    # Convert 1D array of strings (or object-dtype strings) to Python list of str
    if isinstance(x, np.ndarray) and x.ndim == 1:
        if np.issubdtype(x.dtype, np.str_) or np.issubdtype(x.dtype, np.object_):
            try:
                return [i.decode('utf-8') if isinstance(i, bytes) else str(i) for i in x]
            except Exception:
                pass  # fallback to returning as-is
            
    # Try parsing stringified literals
    if isinstance(x, str):
        import ast
        try:
            parsed = ast.literal_eval(x)
            return parsed
        except (ValueError, SyntaxError):
            pass
    
    return x