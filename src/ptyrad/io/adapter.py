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