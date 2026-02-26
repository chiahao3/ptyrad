"""
PtyRAD-specific loading function

"""

import os
from typing import Any, Dict

from .adapter import tensors_to_ndarrays
import warnings

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
        from ptyrad.io.hierarchy import load_hdf5
        return load_hdf5(file_path)

    elif ext == ".pt":
        from ptyrad.io.hierarchy import load_pt
        warnings.warn(
            "Loading PtyRAD reconstruction from .pt file is deprecated and will likely be removed by 2025 Aug."
            "PtyRAD reconstruction output has been using .hdf5 format since v0.1.0b7.",
            DeprecationWarning,
            stacklevel=2,
        )
        return tensors_to_ndarrays(load_pt(file_path)) # .pt is supported for backward compatibility before 0.1.0b7. (e.g. PtyRAD reconstructions used for the paper)
    
    else:
        raise ValueError(
            f"Unsupported file type: '{ext}'. Supported types are .h5, .hdf5, and .pt."
        )