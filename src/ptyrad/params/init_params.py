import pathlib
from typing import Any, Dict, List, Literal, Optional, Union, get_args

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_serializer, model_validator


class FilePathWithKey(BaseModel):
    path: pathlib.Path = Field(description="File path")
    key: Optional[str] = Field(default=None, description="key to the dataset")
    shape: Optional[List[int]] = Field(default=None, description="Shape of the dataset for loading from .raw")
    offset: Optional[int] = Field(default=None, description="Offset of the dataset for loading from .raw")
    gap: Optional[int] = Field(default=None, description="Gap of the dataset for loading from .raw")


class MeasCalibration(BaseModel):
    model_config = {"extra": "forbid"}
    
    mode: Literal['dx', 'dk', 'kMax', 'da', 'angleMax', 'n_alpha', 'RBF', 'fitRBF'] = Field(default='fitRBF', description="Mode for measurements calibration")
    value: Optional[float] = Field(default=None, gt=0.0, description="Value for measurements calibration. Unit: Ang, Ang-1, mrad, # of alpha, px depends on modes")
    """ Value is required for all mode except 'fitRBF """

    @model_validator(mode='before')
    def check_calibration_value(cls, values: dict) -> dict:
        mode = values.get('mode', 'fitRBF')
        value = values.get('value')
        if mode == 'fitRBF' and 'value' not in values:
            values['value'] = None
        if mode != 'fitRBF' and value is None:
            raise KeyError("'value' is required in meas_calibration if mode is not 'fitRBF'.")
        return values


class ObjOmodeInitOccu(BaseModel):
    model_config = {"extra": "forbid"}
    
    occu_type: Literal['uniform', 'custom'] = Field(default='uniform', description="Mode for object mode occupancy initialization")
    init_occu: Optional[List[float]] = Field(description="Value for object mode occupancy initialization")
    
    @model_validator(mode='before')
    def set_default_init_occu(cls, values: dict) -> dict:
        occu_type = values.get('occu_type', 'uniform')
        if occu_type == 'uniform' and 'init_occu' not in values:
            values['init_occu'] = None
        return values


class MeasPad(BaseModel):
    model_config = {"extra": "forbid"}
    
    mode: Optional[Literal['on_the_fly', 'precompute']] = Field(default='on_the_fly', description="Padding mode for measurements. Choose between 'on_the_fly' or 'precompute', or None.")
    padding_type: Literal['constant', 'edge', 'linear_ramp', 'exp', 'power'] = Field(default='power', description="Padding type for measurements. Suggested type is 'power'.")
    target_Npix: int = Field(default=256, description="Target measurement number of pixels")
    value: Optional[float] = Field(default=0, description="Value used for padding background if mode='constant' or 'linear_ramp'.")
    threshold: Optional[float] = Field(default=70, description="Threshold value used for fitting background if mode='power' or 'exp'.")


class MeasResample(BaseModel):
    model_config = {"extra": "forbid"}
    
    mode: Optional[Literal['on_the_fly', 'precompute']] = Field(default='on_the_fly', description="Resampling mode for measurements. Choose between 'on_the_fly' or 'precompute', or None.")
    scale_factors: List[float] = Field(default=[2, 2], min_items=2, max_items=2, description="Resampling scale factors (2,) for measurements")


class MeasRemoveNegValues(BaseModel):
    model_config = {"extra": "forbid"}

    mode: Literal["subtract_min", "subtract_value", "clip_neg", "clip_value"] = Field(default="clip_neg", description="Mode to remove negative values in measurements")
    value: Optional[float] = Field(default=None, description="Value used for removing negative values in measurements if mode='subtract_value' or 'clip_value'")
    force: bool = Field(default=False, description="Boolean flag to force execute the operation no matter whether measurements contain negative values or not")

    @model_validator(mode='before')
    def check_calibration_value(cls, values: dict) -> dict:
        mode = values.get('mode', 'clip_neg')
        value = values.get('value')
        if mode in ['subtract_value','clip_value'] and value is None:
            raise KeyError("'value' is required in meas_remove_neg_values for mode='subtract_value' or 'clip_value'.")
        return values


class MeasNormalization(BaseModel):
    model_config = {"extra": "forbid"}
    
    mode: Literal["max_at_one", "mean_at_one", "sum_to_one", "divide_const"] = Field(default="max_at_one", description="Mode to normalize measurements intensities")
    value: Optional[float] = Field(default=None, description="Value used for normalizing measurements intensities if mode='divide_const'")
    
    @model_validator(mode='before')
    def check_normalization_value(cls, values: dict) -> dict:
        mode = values.get('mode', 'max_at_one')
        value = values.get('value')
        if mode == 'divide_const' and value is None:
            raise KeyError("'value' is required in meas_normalization for mode='divide_const'.")
        return values
   

class MeasAddPoissonNoise(BaseModel):
    model_config = {"extra": "forbid"}
    
    unit: Literal["total_e_per_pattern", "e_per_Ang2"] = Field(description="Unit of dose. Choose between 'total_e_per_pattern' or 'e-per_Ang2'.")
    value: Union[int, float] = Field(gt=0.0, description="Dose to be added to measurements")


class MeasExport(BaseModel):
    model_config = {"extra": "forbid"}

    file_dir: Optional[str] = Field(default=None, description="Output directory for exported measurements")
    file_name: str = Field(default='ptyrad_init_meas', description="Output filename for exported measurements")
    file_format: Literal["hdf5", "tif", "npy", "mat"] = Field(default='hdf5', description="File format for exported measurements")
    output_shape: Optional[List[int]] = Field(default=None, description="Output shape for exported measurements")
    append_shape: bool = Field(default=True, description="Whether to append the shape at the end of the exported file name")


class ObjZPad(BaseModel):
    model_config = {"extra": "forbid"}
    
    pad_layers: List[Optional[int]] = Field(default=[0,0], min_length=2, max_length=2, description="Number of layers to pad on object's top and bottom surfaces. List of 2 ints.")
    pad_types: List[Literal['vacuum', 'mean', 'edge']] = Field(default=['vacuum', 'vacuum'], min_length=2, max_length=2, description="Type of layers to pad on object's top and bottom surfaces. List of 2 strings of 'vacuum', 'mean', 'edge")
    
    @field_validator("pad_layers")
    @classmethod
    def validate_pad_layers(cls, v):
        if not all(x is None or x >= 0 for x in v):
            raise ValueError("pad_layers must contain non-negative integers or None")
        return v


class ObjZResample(BaseModel):
    model_config = {"extra": "forbid"}
    
    mode: Literal['scale_Nlayer', 'scale_slice_thickness', 'target_Nlayer', 'target_slice_thickness'] = Field(description="Resampling mode for object depth. Available options are 'scale_Nlayer', 'scale_slice_thickness', 'target_Nlayer', 'target_slice_thickness'.")
    value: Union[int, float] = Field(description="Corresponding values for the specified resampling mode for object depth.")

    @model_validator(mode="after")
    def validate_value(cls, values):
        mode = values.mode
        value = values.value

        if mode in ("scale_Nlayer", "scale_slice_thickness"):
            if not isinstance(value, (int, float)):
                raise ValueError(f"For mode '{mode}', value must be a float > 0.")
            if value <= 0:
                raise ValueError(f"For mode '{mode}', value must be > 0.")
            # force float if int was passed
            values.value = float(value)

        elif mode == "target_Nlayer":
            if not isinstance(value, int):
                raise ValueError("For mode 'target_Nlayer', value must be an integer >= 1.")
            if value < 1:
                raise ValueError("For mode 'target_Nlayer', value must be >= 1.")

        elif mode == "target_slice_thickness":
            if not isinstance(value, (int, float)):
                raise ValueError("For mode 'target_slice_thickness', value must be a float > 0.")
            if value <= 0:
                raise ValueError("For mode 'target_slice_thickness', value must be > 0.")
            values.value = float(value)

        return values

class TiltParams(BaseModel):
    model_config = {"extra": "forbid"}
    
    tilt_type: Literal['all', 'each'] = Field(default='all', description="Type of initial titls, can be either 'all' (1,2), or 'each' (N,2)")
    init_tilts: List[List[float]] = Field(default=[[0, 0]], description="Initial value for (N,2) object tilts")


SOURCE_PARAMS_MAPPING = {
    'meas': {
        'file': FilePathWithKey,
        'custom': np.ndarray,
    },
    'obj': {
        'simu': Union[list, None],
        'PtyRAD': pathlib.Path,
        'PtyShv': pathlib.Path, 
        'py4DSTEM': pathlib.Path,
        'custom': np.ndarray,
    },
    'probe': {
        'simu': Union[dict, None],
        'PtyRAD': pathlib.Path,
        'PtyShv': pathlib.Path, 
        'py4DSTEM': pathlib.Path,
        'custom': np.ndarray,
    },
    'pos': {
        'simu': type(None),
        'PtyRAD': pathlib.Path,
        'PtyShv': pathlib.Path, 
        'py4DSTEM': pathlib.Path,
        'foldslice_hdf5': pathlib.Path,
        'custom': np.ndarray,
    },
    'tilt': {
        'simu': TiltParams,
        'PtyRAD': pathlib.Path,
        'file': FilePathWithKey,
        'custom': np.ndarray,
    },
}

def validate_source_params_pair(
    source_name: str,
    source_value: str,
    params_value: Any,
):
    mapping = SOURCE_PARAMS_MAPPING[source_name]
    expected_type = mapping.get(source_value)

    if expected_type is None:
        raise ValueError(
            f"Invalid source '{source_value}' for {source_name}_source. "
            f"Allowed: {list(mapping.keys())}"
        )

    # Handle Union[...] properly
    if getattr(expected_type, '__origin__', None) is Union:
        if not isinstance(params_value, get_args(expected_type)):
            raise TypeError(
                f"For {source_name}_source='{source_value}', "
                f"{source_name}_params must be of type {expected_type}, "
                f"but got {type(params_value).__name__}."
            )
    else:
        if not isinstance(params_value, expected_type):
            raise TypeError(
                f"For {source_name}_source='{source_value}', "
                f"{source_name}_params must be of type {expected_type.__name__}, "
                f"but got {type(params_value).__name__}."
            )
    
class InitParams(BaseModel):
    model_config = {"extra": "forbid", 
                    "arbitrary_types_allowed": True} # This is needed to validate np.ndarray type

    random_seed: Optional[int] = Field(default=None, description="Random seed for improved reproducibility")
    
    # Experimental params
    probe_illum_type: Literal['electron', 'xray'] = Field(default="electron", description="Probe illumination type")
    """
    Choose between 'electron' or 'xray' 
    """
    
    ## Electron probe params (used if probe_illum_type == 'electron')
    probe_kv: Optional[float] = Field(default=None, description="Electron acceleration voltage in kV") # Required for electron
    """ 
    Acceleration voltage for relativistic electron wavelength calculation 
    """
    
    probe_conv_angle: Optional[float] = Field(default=None, gt=0.0, description="Semi-convergence angle in mrad") # Required for electron
    """ 
    Semi-convergence angle in mrad for probe-forming aperture 
    """
    
    probe_defocus: Optional[float] = Field(default=0.0, description="Defocus (-C1) in Angstrom")
    """ 
    Defocus (-C1) aberration coefficient for the probe. 
    Positive defocus here refers to actual underfocus or weaker lens strength following Kirkland/abtem/ptychoshelves convention 
    """
    
    probe_c3: Optional[float] = Field(default=0.0, description="3rd-order spherical aberration (C3) in Angstrom")
    """
    3rd-order spherical aberration coefficient (C3) in Ang for the simulated probe 
    """
    
    probe_c5: Optional[float] = Field(default=0.0, description="5th-order spherical aberration (C5) in Angstrom")
    """
    5th-order spherical aberration coefficient (C5) in Ang for the simulated probe 
    """
    
    ## Xray probe params (used if probe_illum_type == 'xray')
    beam_kev: Optional[float] = Field(default=None, description="Xray beam energy in keV")

    probe_dRn: Optional[float] = Field(default=None, description="Xray probe param: Width of outermost zone (in meters)")
    
    probe_Rn: Optional[float] = Field(default=None, description="Xray probe param: Radius of outermost zone (in meters)")
    
    probe_D_H: Optional[float] = Field(default=None, description="Xray probe param: Diameter of the central beamstop (in meters)")
    
    probe_D_FZP: Optional[float] = Field(default=None, description="Xray probe param: Diameter of pinhole in meters")
    
    probe_Ls: Optional[float] = Field(default=None, description="Xray probe param: Distance (in meters) from the focal plane to the sample")

    
    meas_Npix: int = Field(ge=1, description="Detector pixel number (square detector)") # Required
    """
    Detector pixel number, EMPAD is 128. Only supports square detector for simplicity
    """
    
    # Many features actually assume raster scan with fast/slow so we may always infer N_scans
    # Specifying N_scans is useful for checking while loading meas from .raw, or custom pos source like spiral scan
    # Inferred from N_scan_slow/fast
    pos_N_scans: int = Field(ge=1, description="Number of probe positions")
    """
    Number of probe positions (or equivalently diffraction patterns since 1 DP / position)
    """
    
    pos_N_scan_slow: int = Field(ge=1, description="Number of scan positions along slow direction") # Required
    """
    Number of scan position along slow scan direction. Usually it's the vertical direction of acquisition GUI
    """
    
    pos_N_scan_fast: int = Field(ge=1, description="Number of scan positions along fast direction") # Required
    """
    Number of scan position along fast scan direction. Usually it's the horizontal direction of acquisition GUI
    """
    
    pos_scan_step_size: float = Field(gt=0.0, description="Scan step size in Angstrom") # Required
    """
    Step size between probe positions in a rectangular raster scan pattern
    """
    
    meas_calibration: MeasCalibration = Field(default_factory=MeasCalibration, description="Calibration mode and value")
    """
    Calibration for the measurements.
    'mode' can be 'dx', 'dk', 'kMax', 'da', 'angleMax', 'n_alpha', 'RBF', and 'fitRBF'. 
    All modes requires a scalar 'value' input except 'fitRBF'. 
    The units are Ang, 1/Ang and mrad for electron.
    """

    # Model complexity
    probe_pmode_max: int = Field(ge=1, description="Maximum number of mixed probe modes") # Required
    """
    Maximum number of mixed probe modes. 
    Set to pmode_max = 1 for single probe state, pmode_max > 1 for mixed-state probe during initialization. 
    For simulated initial probe, it'll be generated with the specified number of probe modes. 
    For loaded probe, the pmode dimension would be capped at this number
    """
    
    probe_pmode_init_pows: List[float] = Field(default=[0.02], description="Initial power weights for probe modes")
    """
    list of 1 or a few (pmode_max) floats. 
    Initial power for each additional probe modes. 
    If set at [0.02], all additional probe modes would contain 2% of the total intensity. 
    sum(pmode_init_pows) must be 1 if more than len(pmode_init_pows) > 1. 
    See 'utils.make_mixed_probe' for more details
    """
    
    obj_omode_max: int = Field(default=1, ge=1, description="Maximum number of mixed object modes")
    """
    Maximum number of mixed object modes. 
    Set to omode_max = 1 for single object state, omode_max > 1 for mixed-state object during initialization. 
    For simulated initial object, it'll be generated with the specified number of object modes. 
    For loaded object, the omode dimension would be capped at this number
    """
    
    obj_omode_init_occu: ObjOmodeInitOccu = Field(default_factory=ObjOmodeInitOccu, description="Occupancy type and value for mixed-object modes")
    """
    Occupancy type and value for mixed-object modes. 
    Typically we do 'uniform' for frozen phonon like configurations as {'occu_type': 'uniform', 'init_occu': null}. 
    'occu_type' can be either 'uniform' or 'custom', if 'custom', pass in the desired occupancy as an array to 'init_occu'
    """
    
    obj_Nlayer: int = Field(ge=1, description="Number of slices for multislice object") # Required
    """
    Number of slices for multislice object
    """
    
    obj_slice_thickness: float = Field(gt=0.0, description="Slice thickness in Angstrom") # Required
    """
    Slice thickness (propagation distance) for multislice ptychography. 
    Typical values are between 1 to 20 Ang
    """

    # Preprocessing
    meas_permute: Optional[List[int]] = Field(default=None, description="Permutation for diffraction patterns")
    """
    type: null or list of ints.
    This applies additional permutation (reorder axes) for the initialized diffraction patterns. 
    The syntax is the same as np.transpose()
    """

    meas_reshape: Optional[List[int]] = Field(default=None, min_items=3, max_items=3, description="Reshape for diffraction patterns")
    """
    type: null or list of 3 ints. 
    This applies additional reshaping (rearrange elements) for the initialized diffraction patterns. 
    The syntax is the same as np.reshape(). 
    This is commonly needed to convert the 4D diffraction dataset (Ry,Rx,ky,kx) into 3D (N_scans,ky,kx)
    """

    meas_flipT: Optional[List[int]] = Field(default=None, min_items=3, max_items=3, description="Flip and transpose for diffraction patterns")
    """
    type: null or list of 3 binary booleans (0 or 1) as [flipup, fliplr, transpose] just like PtychoShleves. 
    Default is null or [0,0,0] but you may need to find the correct flip and transpose to match your dataset configuration. 
    This applies additional flip and transpose to initialized diffraction patterns. 
    It's suggested to use 'meas_flipT' to correct the dataset orientation and this is the only orientaiton-related value attached to output reconstruction folder name
    """
    
    meas_crop: Optional[List[Optional[List[int]]]] = Field(default=None, description="Crop for 4D diffraction patterns")
    """
    type: null or (4,2) nested list of ints as [[scan_slow_start, scan_slow_end], [scan_fast_start, scan_fast_end], [ky_start, ky_end], [kx_start, kx_end]]. 
    If you want to keep some of the dimensions, for example, do [[0,64],[0,64], null, null] to crop in real space but leaves the k-space untouched. 
    This applies additional cropping to the 4D dataset in both real and k-space. 
    This is useful for reconstrucing a subset of real-space probe positions, or to crop the kMax of diffraction patterns. 
    The syntax follows conventional numpy indexing so the upper bound is not included
    """

    meas_pad: Optional[MeasPad] = Field(default=None, description="Padding configuration for CBED")
    """
    type: dict. 
    'mode' can be 'on_the_fly', 'precompute', or null (will disable padding). 
    'padding_type' can be 'constant', 'edge', 'linear_ramp', 'exp', or 'power'. 
    This will pad the CBED to side length = 'target_Npix' based on the padding_type. 
    If using 'exp' or 'power', the mean diffraction pattern amplitude is used to fit the functional coefficients. 
    'precompute' will pad the measurements during initialization, while 'on_the_fly' will only pad the measurements during optimization, 
    so it's more efficient for GPU memory. 
    'on_the_fly' padding doesn't really affect the reconstruction time so it's suggested to always use 'on_the_fly' if you're padding to save the GPU memory.
    """
    
    meas_resample: Optional[MeasResample] = Field(default=None, description="Resampling configuration for diffraction patterns")
    """
    'mode' can be 'on_the_fly', 'precompute', or null. 
    'scale_factor' takes a list of 2 floats as [ky_zoom, kx_zoom] that must be the same. 
    This applies additional resampling of initialized diffraction patterns along ky and kx directions. 
    This is useful for changing the k-space sampling of diffraction patterns. 
    See scipy.ndimage.zoom for more details. 
    'precompute' will resample the measurements during initialization, while 'on_the_fly' will only resample the measurements during optimization, 
    so it's more efficient for GPU memory if you're upsampling (i.e., scale_factor > 1). 
    For downsampling, it's much better to do 'precompute' to save GPU memory. 
    'on_the_fly' resampling doesn't really affect the reconstruction time so it's suggested to always use 'on_the_fly' if you're upsampling.
    """
    
    meas_add_source_size: Optional[float] = Field(default=None, gt=0.0, description="Gaussian blur std for spatial partial coherence in Angstrom")
    """
    type: null or float, unit: Ang. 
    This adds additional spatial partial coherence to diffraction patterns by applying Gaussian blur along scan directions. 
    The provided value is used as the std (sigma) for the Gaussian blurring kernel in real space. 
    Note that FWHM ~ 2.355 std, so a std of 0.34 Ang is equivalent to a source size (FWHM) of 0.8 Ang
    """

    meas_add_detector_blur: Optional[float] = Field(default=None, gt=0.0, description="Gaussian blur std for detector in pixels")
    """
    type: null or float, unit: px (k-space). 
    This adds additional detector blur to diffraction patterns to emulate the PSF on detector. 
    The provided value is used as the std (sigma) for the Gaussian blurring kernel in k-space. 
    Note that this is applied to the "measured", or "ground truth" diffraction pattern and is different from 'model_params['detector_blur_std']'
    that applies to the forward simulated diffraction pattern
    """
    
    meas_remove_neg_values: MeasRemoveNegValues = Field(default_factory=MeasRemoveNegValues, description="Preprocessing for negative values in measurements")
    """
    Choose the preprocessing method for handling negative values in the measurements. 
    Available options are 'subtract_min', 'subtract_value', 'clip_neg', and 'clip_value'. 
    Previously (before beta3.1) the PtyRAD default is 'subtract_min', while for low dose data is recommended to use 'clip_neg' or 'clip_value' 
    i.e. {'mode': 'clip_neg', 'value': 20}. Current default is 'clip_neg'. 
    'value' is only needed for 'subtract_value' and 'clip_value'. 
    This correction is skipped if there's no negative values in measurements unless 'force'=True. 
    If you want to enforce some offsetting when there's no negative values, set 'force': true to enforce the correction.
    """
    
    meas_normalization: MeasNormalization = Field(default_factory=MeasNormalization, description="Normalization method for measurements")
    """
    type: null or dict. 
    Choose the normalization method for measurements. 
    Available options are 'max_at_one' (default), 'mean_at_one', 'sum_to_one', and 'divide_const'. 
    For 'divide_const', you need to provide another dict entry 'value': <VALUE>.
    """
    
    meas_add_poisson_noise: Optional[MeasAddPoissonNoise] = Field(default=None, description="Poisson noise configuration")
    """
    type: null or dict, 
    i.e., {'unit': 'total_e_per_pattern', 'value': 10000} or {'unit': 'e_per_Ang2', 'value': 10000}. 
    This applies additional Poisson noise to diffraction patterns to emulate the Poisson statistics of electron dose based on the given unit, value, and scan step size. 
    This is useful when you have a noise-free simulated dataset and want to try ptychographic reconstruciton at different dose conditions
    """
    
    meas_export: Optional[Union[bool, MeasExport]] = Field(default=None, description="Export configuration for measurements")
    """
    type: null, boolean, or dict, 
    i.e., {'file_dir': null, 'file_name': 'ptyrad_init_meas', 'file_format': 'hdf5', 'output_shape': null, 'append_shape': true}. 
    Set this to True or a dict to enable exporting the final initialized measurements array to disk for further processing, analysis, or visualization. 
    The exported data layout has the same Python convention with py4DGUI so there's no need to worry about orientation mismatch. 
    This can be used to interactively check whether the meas_flipT is correct. 
    By default the output layout is (N_scans, Ky, Kx), and dropping it to py4DGUI then reshape it into (Ny, Nx, Ky, Kx) keeps the correct orientation. 
    'file_dir' sets the output directory, if None, it'll export to the same folder as 'meas_params['path']'. 
    'file_format' supports 'hdf5', 'tif', 'npy', and 'mat'. 
    'output_shape' takes a list of integers like [Ny, Nx, Ky, Kx] or [N_scans, Ky, Kx]. 
    'append_shape' is a boolean, if True, it'll append the shape of the array to the output file name.
    """

    probe_permute: Optional[List[int]] = Field(default=None, description="Permutation for probe")
    """
    type: null or list of int. 
    This applies additional permutation (reorder axes) for the initialized probe. 
    The syntax is the same as np.transpose()
    """

    probe_z_shift: Optional[float] = Field(default=None, description="Axially (z) shift the initialized probe")
    """
    type: null or float. 
    unit: Ang for electron and m for x-ray. 
    This shifts the initialized probe axially along the depth (z) dimension. 
    The sign convention follows the propagation direction, 
    so positive value means to propagate the probe further into the object (i.e., forward direction), 
    and negative value refers to rewind the probe backward. 
    This is used to shift the probe focus vertically along the object depth dimension in accordance with the object z-recentering (i.e., obj_z_crop, obj_z_pad). 
    For example, if you pad the reconstructed object with additional 50 Ang vacuum layer ON TOP via 'obj_z_pad', 
    you'll want to apply a 'probe_z_shift: -50' Ang to the originally reconstructed probe so the relative geometry of probe and object is conserved. 
    """
    
    pos_scan_flipT: Optional[List[int]] = Field(default=None, description="Flip and transpose for scan patterns")
    """
    type: null or list of 3 binary booleans (0 or 1) as [flipup, fliplr, transpose] just like PtychoShleves. 
    Default value is null or equivalently [0,0,0]. 
    This applies additional flip and transpose to initialized scan patterns. 
    Note that modifying 'scan_flipT' would change the image orientation, so it's recommended to set this to null, and only use 'meas_flipT' to get the orientation correct
    """

    pos_scan_affine: Optional[List[float]] = Field(default=None, description="Affine transformation for scan patterns")
    """
    type: null or list of 4 floats as [scale, asymmetry, rotation, shear] just like PtychoShleves. 
    Default is null or equivalently [1,0,0,0], rotation and shear are in unit of degree. 
    This applies additional affine transformation to initialized scan patterns to correct sample drift and imperfect scan coils
    """
    
    pos_scan_rand_std: Optional[float] = Field(default=0.15, ge=0.0, description="Random displacement std for scan positions in pixels")
    """
    type: null or float, unit: px (real space). 
    Randomize the initial guess of scan positions with Gaussian distributed displacement (std in px) to reduce raster grid pathology
    """

    obj_z_crop: Optional[List[int]] = Field(default=None, description="Cropping object along depth dimension")
    """
    type: null or list of 2 ints as [z_start, z_end]. 
    This applies additional cropping to the 4D object (omode, Nz, Ny, Nx) along depth dimension and removes the unselected layers. 
    This is useful for removing the unwanted vacuum layers above and below the specimen. 
    The syntax follows conventional numpy indexing so the upper bound is not included.
    """
    
    obj_z_pad: Optional[ObjZPad] = Field(default=None, description="Padding object along depth dimension")
    """
    type: null or dict, i.e., {'pad_layers': [2,2], 'pad_types': ['vacuum', 'vacuum']}. 
    This applies additional padding to the 4D object (omode, Nz, Ny, Nx) along depth dimension (Nz) and expands the object thickness. 
    'pad_layers' is a list of 2 ints that determines how many layers are appended to the top and bottom surfaces. 
    [2,2] means 2 layers on top and 2 layers on the bottom. 
    Set the value to 0 or None to achieve asymmetric padding on one of the surface. 
    For example 'pad_layers': [0,5] means to only pad 5 layers to the bottom. 
    'pad_types' is a list of 2 strings that specifies whether we are padding vacuum layers, mean layers, or edge layers. 
    The available options are 'vacuum', 'mean', and 'edge'. 
    This is useful for expanding reconstructed object to test object positioning or total thickness. 
    Note that one should also adjust the 'probe_z_shift' (loaded probe) or 'probe_defocus' (simulated probe) accordingly, 
    because padding object on the top surface will effectively change the entrance plane of the probe. 
    For example, if one padded 10 nm of vacuum on the top surface, one should also add a 10 nm underfocus (for simulated probe via 'probe_defocus'), 
    or apply a z-shift of -10 nm to the loaded probe via 'probe_z_shift' to maintain the relative position of probe and object.
    """

    obj_z_resample: Optional[ObjZResample] = Field(default=None, description="Resampling object along depth dimension")
    """
    type: null or dict, i.e., {'mode': 'scale_Nlayer', 'value': 2}. 
    This applies additional resampling to the 4D object (omode, Nz, Ny, Nx) along depth dimension (Nz), 
    and modifies the slice thickness altogether to preserve prod(amp), sum(phase), and total thickness. 
    'mode' can take string options including 'scale_Nlayer', 'scale_slice_thickness', 'target_Nlayer', 'target_slice_thickness', or null to disable it. 
    'value' takes the corresponding value with the specified mode. 
    The values for 'scale_Nlayer', 'scale_slice_thickness', and 'target_slice_thickness' can be any positive float, while 'target_Nlayer' must take a positive integer. 
    This is useful when we need to reslice the reconstructed object into different slice thicknesses including converting between single and multislice objects.
    """
    
    # Input source and params
    meas_source: Literal['file', 'custom'] = Field(default="file", description="Data source for measurements")
    """
    Data type of the measurements (diffraction patterns). 
    Currently supporting 'file' or 'custom'. 
    """

    meas_params: Union[FilePathWithKey, np.ndarray] = Field(description="Parameters for measurement loading") # Required
    """
    type: dict, or numpy array. 
    For file type of 'mat', or 'hdf5', it's preferred to provide both the 'path' and 'key' in a dict
    {'path': <PATH_TO_DATA>, 'key': <DATA_KEY>} to retrieve the data matrix, although PtyRAD would try to retrive the data even without a key. 
    'tif' would only need the {'path':<PATH_TO_DATA>}. 
    For 'raw' you can optionally pass in 'shape':(N,height,width), 'offset':int, 'gap':int to load the .raw files from EMPAD1 and pre-processed EMPAD datasets. 
    For example, {'path': <PATH_TO_DATA>, 'offset':0, 'gap':0} can be used to load pre-processed EMPAD2 raw dataset with no gap between binary diffraction patterns. 
    The 'shape' will be automatically filled in from 'exp_params', while 'offset':0, and 'gap':1024 are default values for EMPAD1 datasets. 
    For py4dstem processed diffraction patterns (hdf5), use '/datacube_root/datacube/data' for your 'key'. 
    For 'custom' source, pass the numpy array to the 'measurements_params' entry after you load this .yml as a dict
    """
    
    probe_source: Literal['simu', 'PtyRAD', 'PtyShv', 'py4DSTEM','custom'] = Field(default="simu",description="Data source for probe")
    """
    Data source of the probe. Currently supporting 'simu', 'PtyRAD', 'PtyShv', 'py4DSTEM', and 'custom'
    """
    
    probe_params: Optional[Union[Dict[str, Any], pathlib.Path, np.ndarray]] = Field(default=None, description="Parameters for probe loading/initialization")
    """
    type: null, dict, str, or numpy array. 
    Parameters of the probe loading/initialization. 
    For 'simu' (simulating probe), provide a dict of 'probe_simu_params' to specify the simulation parameters 
    (see 'utils/make_stem_probe' for more details) or null to use only basic paramaters like kV, conv_angle, defocus and c3. 
    For loading probe from 'PtyRAD' or 'PtyShv', provide a str of <PATH_TO_RECONSTRUCTION_FILE>. 
    For 'custom' probe source, pass the 3D numpy array to the 'probe_params' entry after you load this .yml as a dict
    """
    
    pos_source: Literal['simu', 'PtyRAD', 'PtyShv', 'py4DSTEM', 'foldslice_hdf5', 'custom'] = Field(default="simu", description="Data source for probe positions")
    """
    Data source of the probe positions. 
    Currently supporting 'simu', 'PtyRAD', 'PtyShv', 'py4DSTEM', 'foldslice_hdf5', and 'custom'
    """
    
    pos_params: Optional[Union[pathlib.Path, np.ndarray]] = Field(default=None, description="Parameters for probe positions loading/initialization")
    """
    type: null, str, or numpy array. 
    Parameters of the probe positions loading/initialization. 
    For 'simu' (simulating probe positions), provide null and check whether you need 'scan_flipT'. 
    The positions would be simulated based on 'N_scan_slow', 'N_scan_fast', 'scan_step_size', and 'scan_affine'. 
    For loading probe positions from 'PtyRAD' or 'PtyShv', provide a str of <PATH_TO_RECONSTRUCTION_FILE>. 
    For loading probe positions from 'foldslice_hdf5', provide a str of <PATH_TO_POSITION_FILE>. 
    These hdf5 files are generated from many APS instruments that were previously handled in `fold_slice` using 'p.src_positions='hdf5_pos'. 
    For 'custom' probe position source, pass the (N_scans,2) numpy array to the 'pos_params' entry after you load this .yml as a dict
    """
    
    obj_source: Literal['simu', 'PtyRAD', 'PtyShv', 'py4DSTEM','custom'] = Field(default="simu", description="Data source for object")
    """
    Data source of the object. 
    Currently supporting 'simu', 'PtyRAD', 'PtyShv', 'py4DSTEM', and 'custom'
    """
    
    obj_params: Optional[Union[List[int], pathlib.Path, np.ndarray]] = Field(default=None, description="Parameters for object loading/initialization")
    """
    type: null, list of 4 ints, str, or numpy array. 
    Parameters of the object loading/initialization. 
    For 'simu' (simulating object), provide a list of 4 ints (omode, Nz, Ny, Nx) to specify the object shape or null to let PtyRAD determine it 
    (null is suggested and is consistent with how PtyShv creates their initial object). 
    For loading object from 'PtyRAD' or 'PtyShv', provide a str of <PATH_TO_RECONSTRUCTION_FILE>. 
    For 'custom' object source, pass the 4D numpy array to the 'obj_params' entry after you load this .yml as a dict
    """

    tilt_source: Literal['simu', 'PtyRAD', 'file','custom'] = Field(default="simu", description="Data source for object tilts")
    """
    Data source of the object tilts. Currently supporting 'simu', 'PtyRAD', 'file', and 'custom'
    """

    tilt_params: Union[TiltParams, FilePathWithKey, pathlib.Path, np.ndarray] = Field(default_factory=TiltParams, description="Parameters for object tilt loading/initialization")
    """
    type: dict, str, or numpy array. 
    Parameters of the object tilt loading/initialization. 
    The object tilt is implemeted by tilted Fresnel propagator, which should be fairly accurate within 1 degree (17 mrad). 
    For 'simu' (simulating object tilts), provide a dict as {'tilt_type':'all', 'init_tilts':[[tilt_y, tilt_x]]}. 
    tilt_y and tilt_x are floats in unit of mrad, defaults are [[0,0]]. 
    'tilt_type' can be either 'all' or 'each'. 
    'tilt_type': 'all' will create a (1,2) tilt array specifying all positions has the same tilt as 'initial_tilts', 
    this is a globally uniform object tilt that can be either fixed, hypertune optimized, or AD-optimized (if learning rate of 'obj_tilts' != 0). 
    'tilt_type': 'each' will create a (N_scans,2) tilt array that all position starts at the same 'init_tilts', 
    but it can be later individually optimized through AD by setting learning rate of 'obj_tilts' != 0, which allows pos-dependent local object tilt correction. 
    For loading object tilts from 'PtyRAD', provide a str of <PATH_TO_RECONSTRUCTION_FILE>. 
    If loading object tilt from 'file', provide a dict as {'path': <PATH_TO_DATA>, 'key': <DATA_KEY>} similar to 'meas_params'. 
    The supported file types are 'tif', 'mat', 'hdf5', and 'npy'. 
    Note that the object tilt array must be 2D with shape equals to (1,2) or (N,2). 
    For 'custom' object tilts source, pass the (N_scans,2) or (1,2) numpy array to the 'tilt_params'. 
    You should always provide an initial tilt guess for tilt correction either through hypertune or estimate with '/scripts/get_local_obj_tilts.ipynb',
    because optimizing tilts with AD from scratch would be too slow and most likely arrive at barely corrected, slice-shifted object
    """

################################################################################################################################
################################################################################################################################
################################################################################################################################


    @field_validator("probe_pmode_init_pows")
    @classmethod
    def validate_probe_pmode_init_pows(cls, v: List[float], info) -> List[float]:
        """Ensure probe_pmode_init_pows matches probe_pmode_max (if >1), is non-negative, and sums to 1 if length > 1."""
        pmode_max = info.data.get("probe_pmode_max", 1)
        if len(v) > 1 and len(v) != pmode_max:
            raise ValueError(
                f"probe_pmode_init_pows must have length 1 or equal to probe_pmode_max ({pmode_max})"
            )
        if not all(x >= 0.0 for x in v):
            raise ValueError("probe_pmode_init_pows must contain non-negative values")
        if len(v) > 1 and not np.isclose(sum(v), 1.0, rtol=1e-5):
            raise ValueError("probe_pmode_init_pows must sum to 1 when length > 1")
        return v


    @field_validator("meas_crop")
    @classmethod
    def validate_meas_crop(cls, v: Optional[List[Optional[List[int]]]]) -> Optional[List[Optional[List[int]]]]:
        """Ensure meas_crop is None or a (4,2) nested list of integers, allowing None in sublists."""
        if v is not None:
            if not isinstance(v, list) or len(v) != 4:
                raise ValueError("meas_crop must be a list of length 4 or None.")

            for sublist in v:
                if sublist is not None:
                    if not isinstance(sublist, list) or len(sublist) != 2:
                        raise ValueError("Each sublist in meas_crop must be a list of length 2 or None.")
                    if not all(isinstance(x, int) for x in sublist):
                        raise ValueError("Each element in the meas_crop sublists must be an integer.")
        return v

    @field_validator("meas_flipT")
    @classmethod
    def validate_meas_flipT(cls, v: List[int]) -> List[int]:
        """Ensure meas_flipT is None or contains 3 binary integers (0 or 1)."""
        if v is not None and (len(v) != 3 or not all(x in {0, 1} for x in v)):
            raise ValueError("meas_flipT must None or a list of 3 binary integers (0 or 1)")
        return v

    @field_validator("pos_scan_flipT")
    @classmethod
    def validate_pos_scan_flipT(cls, v: Optional[List[int]]) -> Optional[List[int]]:
        """Ensure pos_scan_flipT is None or contains 3 binary integers."""
        if v is not None and (len(v) != 3 or not all(x in {0, 1} for x in v)):
            raise ValueError("pos_scan_flipT must be None or a list of 3 binary integers (0 or 1)")
        return v
    
    @field_validator("pos_scan_affine")
    @classmethod
    def validate_pos_scan_affine(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        """Ensure pos_scan_affine is None or a list of 4 floats."""
        if v is not None and (len(v) != 4 or not all(isinstance(x, (int, float)) for x in v)):
            raise ValueError("pos_scan_affine must be None or a list of 4 floats")
        return v
    
    @field_validator("obj_z_crop")
    @classmethod
    def validate_obj_z_crop(cls, v: Optional[List[int]]) -> Optional[List[int]]:
        """Ensure obj_z_crop is None or a list of 2 ints."""
        if v is not None:
            if not isinstance(v, list):
                raise TypeError("obj_z_crop must be a list of two integers or None")
            if len(v) != 2:
                raise ValueError("obj_z_crop must have length 2")
            if not all(isinstance(x, int) for x in v):
                raise TypeError(f"obj_z_crop elements must be integers, got {[type(x) for x in v]}")
            if v[0] >= v[1]:
                raise ValueError(f"obj_z_crop must have z_start < z_end, got {v}")
        return v
        
    # 2025.07.02 CHL    
    # pydantic.FilePath would check the file existence during field instantiation along with type check.
    # So it will raise ValidationError if the path is invalid.
    # However, since XXX_params all have Union type, 
    # pydantic would continue type check for all other types and print an individual ValidationError for each type.
    # This makes the error message much less useful and confusing.
    # The solution is to loosen up the type check by switching pydantic.FilePath with pathlib.Path,
    # which only check if it's a string and path-like during field instantiation.
    # Once we pass the field type check, we then use @field_validator to check if the path actually exist and raise FoundNotFoundError if needed.
    # This produces much cleaner error message if the path is invalid.
    # Note that the `validate_all_source_params` is a @model_validator)mode='after') that happens after the model instantiation,
    # so if the source and params are not correctly matching along with an invalid path,
    # the error message would be the FileNotFoundError coming from @field_validator.
    # The @model_validator)mode='after') is more like a final consistency check.
        
    @field_validator('meas_params')    
    @classmethod
    def validate_meas_params(cls, v: Union[FilePathWithKey, np.ndarray], info) -> Union[FilePathWithKey, np.ndarray]:
        if isinstance(v, FilePathWithKey):
            if not v.__dict__['path'].is_file():
                raise FileNotFoundError(f"{info.field_name}: Path '{v}' does not point to a valid file")
            return v
        if isinstance(v, np.ndarray):
            return v
        else:
            raise ValueError(f"{info.field_name} must be a dict, a valid file path, or a NumPy array, got {type(v).__name__}")  

    @field_validator('probe_params')    
    @classmethod
    def validate_probe_params(cls, v: Optional[Union[Dict[str, Any], pathlib.Path, np.ndarray]], info) -> Optional[Union[Dict[str, Any], pathlib.Path, np.ndarray]]:
        if v is None:
            return None
        if isinstance(v, pathlib.Path):
            if not v.is_file():
                raise FileNotFoundError(f"{info.field_name}: Path '{v}' does not point to a valid file")
            return v
        if isinstance(v, (dict, np.ndarray)):
            return v
        else:
            raise ValueError(f"{info.field_name} must be a dict, a valid file path, or a NumPy array, got {type(v).__name__}")
        
    @field_validator('pos_params')    
    @classmethod
    def validate_pos_params(cls, v: Optional[Union[pathlib.Path, np.ndarray]], info) -> Optional[Union[Dict[str, Any], pathlib.Path, np.ndarray]]:
        if v is None:
            return None
        if isinstance(v, pathlib.Path):
            if not v.is_file():
                raise FileNotFoundError(f"{info.field_name}: Path '{v}' does not point to a valid file")
            return v
        if isinstance(v, (np.ndarray)):
            return v
        else:
            raise ValueError(f"{info.field_name} must be either None, a valid file path, or a NumPy array, got {type(v).__name__}")

    @field_validator('obj_params')    
    @classmethod
    def validate_obj_params(cls, v: Optional[Union[List[int], pathlib.Path, np.ndarray]], info) -> Optional[Union[List[int], pathlib.Path, np.ndarray]]:
        if v is None:
            return None
        if isinstance(v, list):
            if len(v) != 4 or not all(isinstance(x, int) for x in v):
                raise ValueError(f"{info.field_name} must be a List of 4 ints")
            return v
        if isinstance(v, pathlib.Path):
            if not v.is_file():
                raise FileNotFoundError(f"{info.field_name}: Path '{v}' does not point to a valid file")
            return v
        if isinstance(v, np.ndarray):
            return v
        else:
            raise ValueError(f"{info.field_name} must be either None, a List of 4 ints, a valid file path, or a NumPy array, got {type(v).__name__}")

    @field_validator('tilt_params')    
    @classmethod
    def validate_tilt_params(cls, v: Union[TiltParams, FilePathWithKey, pathlib.Path, np.ndarray], info) -> Union[TiltParams, FilePathWithKey, pathlib.Path, np.ndarray]:
        if v is None:
            return None
        if isinstance(v, (TiltParams, np.ndarray)):
            return v
        if isinstance(v, FilePathWithKey):
            if not v.__dict__['path'].is_file():
                raise FileNotFoundError(f"{info.field_name}: Path '{v}' does not point to a valid file")
            return v
        if isinstance(v, pathlib.Path):
            if not v.is_file():
                raise FileNotFoundError(f"{info.field_name}: Path '{v}' does not point to a valid file")
            return v
        else:
            raise ValueError(f"{info.field_name} must be a dict, a valid file path, or a NumPy array, got {type(v).__name__}")   

    @model_validator(mode='before')
    def infer_pos_N_scans(cls, values: dict) -> dict:
        pos_N_scans     = values.get('pos_N_scans')
        pos_N_scan_slow = values.get('pos_N_scan_slow')
        pos_N_scan_fast = values.get('pos_N_scan_fast')
        
        if pos_N_scans is None:
            if pos_N_scan_slow is not None and pos_N_scan_fast is not None:
                values['pos_N_scans'] = pos_N_scan_slow * pos_N_scan_fast
        return values
        
    @model_validator(mode="after")
    def validate_mode_specific_fields(self):
        """
        Enforce mode-dependent required fields while allowing a unified init_params YAML structure.
        """
        if self.probe_illum_type == 'electron':
            required_fields = ['probe_kv', 'probe_conv_angle']
            for field in required_fields:
                if getattr(self, field) is None:
                    raise ValueError(
                        f"'{field}' must be provided when probe_illum_type='electron'."
                    )
            # Clear irrelevant fields for clarity
            for field in ['beam_kev', 'probe_dRn', 'probe_Rn', 'probe_D_H', 'probe_D_FZP', 'probe_Ls']:
                setattr(self, field, None)

        elif self.probe_illum_type == 'xray':
            required_fields = [
                'beam_kev',
                'probe_dRn',
                'probe_Rn',
                'probe_D_H',
                'probe_D_FZP',
                'probe_Ls'
            ]
            for field in required_fields:
                if getattr(self, field) is None:
                    raise ValueError(
                        f"'{field}' must be provided when probe_illum_type='xray'."
                    )
            # Clear irrelevant fields for clarity
            for field in ['probe_kv', 'probe_conv_angle', 'probe_defocus', 'probe_c3', 'probe_c5']:
                setattr(self, field, None)

        return self
    
    @model_validator(mode="after")
    def validate_all_source_params(self):
        validate_source_params_pair('meas', self.meas_source, self.meas_params)
        validate_source_params_pair('obj', self.obj_source, self.obj_params)
        validate_source_params_pair('probe', self.probe_source, self.probe_params)
        validate_source_params_pair('pos', self.pos_source, self.pos_params)
        validate_source_params_pair('tilt', self.tilt_source, self.tilt_params)
        return self
    
    @model_serializer
    def serialize_model(self):
        """Custom serializer to convert pathlib.Path back to str."""
        data = self.__dict__.copy()
        fields = ['meas_params', 'probe_params', 'pos_params', 'obj_params', 'tilt_params']
        for field in fields:
            if isinstance(data[field], pathlib.Path):
                data[field] = str(data[field])
            if isinstance(data[field], FilePathWithKey):
                data[field].__dict__['path'] = str(data[field].__dict__['path'])
        return data