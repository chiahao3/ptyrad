"""
High-level file handlers (load/save) for generic and hierarchical file formats

"""

import logging
import os
from typing import List, Optional, Tuple

import numpy as np

from .generic import load_npy, load_raw, load_tif, write_npy, write_tif
from .hierarchy import load_ND_with_key, write_hdf5

logger = logging.getLogger(__name__)

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
        
def save_array(data, file_dir='', file_name='ptyrad_init_meas', file_format="hdf5", output_shape=None, append_shape=True, **kwargs):
    """
    Save an ND array to the specified file format.

    Args:
        data (numpy.ndarray): ND array to save.
        file_dir (str): Directory to save the file.
        file_name (str): Base name of the file (without extension).
        file_format (str): File format to save as ("tif", "npy", "hdf5", "mat").
        output_shape (tuple, optional): Desired shape for the output array.
        append_shape (bool): Whether to append the array shape to the filename.
        **kwargs: Additional arguments for specific file formats.
    """
    # Reshape data if output_ndim is specified
    if output_shape is not None:
        try:
            data = data.reshape(output_shape)
        except ValueError as e:
            logger.warning(f"WARNING: {e}, the data shape is preserved as {data.shape}")
            
    # Append shape to the filename if enabled
    if append_shape:
        shape_str = "_".join(map(str, data.shape))
        file_name = f"{file_name}_{shape_str}"

    # Construct the full file path
    file_format = file_format.lower()
    file_path = os.path.join(file_dir, f"{file_name}.{file_format}")
    logger.info(f"Saving array with shape = {data.shape} and dtype = {data.dtype}")
    
    if os.path.isfile(file_path):
        logger.info(f"file path = '{file_path}' already exists, the file will be overwritten.")
    
    if file_format in ["tif", "tiff"]:
        write_tif(file_path, data)
    elif file_format == "npy":
        write_npy(file_path, data)
    elif file_format in ["hdf5", "h5", "mat"]:
        # Saving .mat into hdf5 as if it were .mat v7.3. This ensures compatibility with py4DGUI.
        write_hdf5(file_path, data, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {file_format}")