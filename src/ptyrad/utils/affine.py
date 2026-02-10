"""
Math operations of FFTs, masks, affine transformation matrices, etc.

"""

import numpy as np

# Affine
def compose_affine_matrix(scale, asymmetry, rotation, shear):
    # Adapted from PtychoShelves +math/compose_affine_matrix.m
    # The input rotation and shear is in unit of degree
    rotation_rad = np.radians(rotation)
    shear_rad = np.radians(shear)
    
    A1 = np.array([[scale, 0], [0, scale]])
    A2 = np.array([[1 + asymmetry/2, 0], [0, 1 - asymmetry/2]])
    A3 = np.array([[np.cos(rotation_rad), np.sin(rotation_rad)], [-np.sin(rotation_rad), np.cos(rotation_rad)]])
    A4 = np.array([[1, 0], [np.tan(shear_rad), 1]])
    
    affine_mat = A1 @ A2 @ A3 @ A4

    return affine_mat

def decompose_affine_matrix(input_affine_mat):
    from scipy.optimize import least_squares
    def err_fun(x):
        scale, asymmetry, rotation, shear = x
        fit_affine_mat = compose_affine_matrix(scale, asymmetry, rotation, shear)
        return (input_affine_mat - fit_affine_mat).ravel()

    # Initial guess
    initial_guess = np.array([1, 0, 0, 0])
    result = least_squares(err_fun, initial_guess)
    scale, asymmetry, rotation, shear = result.x

    return scale, asymmetry, rotation, shear

def get_decomposed_affine_matrix_from_bases(input, output):
    """ Fit the affine matrix components from input and output matrices A and B """
    # This util function is used to quickly estimate the needed affine transformation for scan positions
    # If we know the lattice constant and angle between lattice vectors, then we can easily correct the scale, asymmetry, and shear
    # The global rotation of the object is NOT defined by lattice constant/angle so we still need to compare with the actual CBED
    # Typical usage of this function is to first construct A by measuring the lattice vectors of a reconstructed object suffers from affine transformation
    # Then estimate ideal lattice vectors with prior knowledge (lattice constant and angle)
    # Lastly we use this function to estimate the needed F such that B = F @ A
    
    from scipy.optimize import minimize

    def objective(params, A, B):
        scale, asymmetry, rotation, shear = params
        F = compose_affine_matrix(scale, asymmetry, rotation, shear)
        return np.linalg.norm(B - F @ A)

    initial_guess = [1, 0, 0, 0]  # Initial guess for scale, asymmetry, rotation, shear
    result = minimize(objective, initial_guess, args=(input, output), method='L-BFGS-B')
    
    if result.success:
        (scale, asymmetry, rotation, shear) = result.x
        return (scale, asymmetry, rotation, shear)
    else:
        raise ValueError("Optimization failed")