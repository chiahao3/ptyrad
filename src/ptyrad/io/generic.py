"""
Generic file handling (load/save) for raw, npy, tif formats

"""

import os

import numpy as np
from tifffile import imread, imwrite


def load_raw(file_path, shape, dtype=np.float32, offset=0, gap=1024):
    """Loads a raw binary file containing interleaved image data and gaps.

    This implementation uses a custom `numpy.dtype` with `np.fromfile` for 
    fast I/O performance, extracting only the valid data regions and skipping 
    the specified byte gaps between frames. Note that custom processed raw 
    data might have a gap of 0.

    Args:
        file_path (str): The path to the raw binary file.
        shape (tuple of int): The expected shape of the data in the format 
            (N, height, width), where N is the number of frames.
        dtype (data-type, optional): The NumPy data type of the image pixels. 
            Defaults to np.float32.
        offset (int, optional): The number of bytes to skip at the beginning 
            of the file. Defaults to 0.
        gap (int, optional): The number of gap bytes to skip between each 
            image frame. Defaults to 1024.

    Returns:
        numpy.ndarray: An array of the extracted data with the specified shape 
        and dtype.

    Raises:
        ValueError: If the actual file size does not match the expected size 
            calculated from the inputs.
    """
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
    return data

def load_tif(file_path):
    """Loads an image array from a TIFF file.

    Args:
        file_path (str): The path to the TIFF file.

    Returns:
        numpy.ndarray: The loaded image data.

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The specified file '{file_path}' does not exist. Please check your file path and working directory.")
    
    data = imread(file_path)

    return data

def load_npy(file_path):
    """Loads an array from a binary NumPy .npy file.

    Args:
        file_path (str): The path to the .npy file.

    Returns:
        numpy.ndarray: The loaded array data.

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The specified file '{file_path}' does not exist. Please check your file path and working directory.")
    
    data = np.load(file_path)

    return data

def write_tif(file_path, data):
    """Saves a NumPy array as a TIFF file.

    The file is saved with ImageJ compatibility enabled to ensure proper 
    handling of hyperstacks and metadata in common microscopy viewers.

    Args:
        file_path (str): The destination path for the TIFF file.
        data (numpy.ndarray): The array data to save.
    """
    imwrite(file_path, data, imagej=True)

def write_npy(file_path, data):
    """Saves a NumPy array to a binary .npy file.

    Args:
        file_path (str): The destination path for the .npy file.
        data (numpy.ndarray): The array data to save.
    """
    np.save(file_path, data)