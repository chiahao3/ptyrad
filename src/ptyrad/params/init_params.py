"""
Defines available options and validation rules for the ``init_params`` dictionary.

Note that ``init_params`` contains **dataset-dependent required fields**.
"""

from __future__ import annotations

import pathlib
from typing import Any, Dict, List, Literal, Optional, Union, get_args

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_serializer, model_validator

from ptyrad.optics.aberrations import Aberrations


class FilePathWithKey(BaseModel):
    path: pathlib.Path = Field(description="File path")
    """
    Absolute or relative path to the data file.
    """

    key: Optional[str] = Field(default=None, description="key to the dataset")
    """
    Internal key path (e.g., for HDF5 or MAT files) to access the specific dataset.
    """

    shape: Optional[List[int]] = Field(default=None, description="Shape of the dataset for loading from .raw")
    """
    Explicit shape of the dataset ``[N, height, width]``. Required when loading binary ``.raw`` files (e.g., EMPAD).
    """

    offset: Optional[int] = Field(default=None, description="Offset of the dataset for loading from .raw")
    """
    **[bytes]** Number of bytes to skip at the beginning of the file. Used for raw binary loading.
    """

    gap: Optional[int] = Field(default=None, description="Gap of the dataset for loading from .raw")
    """
    **[bytes]** Number of bytes to skip between each diffraction pattern. Used for raw binary loading.
    """


class MeasCalibration(BaseModel):
    model_config = {"extra": "forbid"}
    
    mode: Literal['dx', 'dk', 'kMax', 'da', 'angleMax', 'n_alpha', 'RBF', 'fitRBF'] = Field(default='fitRBF', description="Mode for measurements calibration")
    """
    Available options for calibration ``mode``, and corresponding units of their ``value`` :
    
    - ``dx`` : Ang
    - ``dk`` : 1/Ang
    - ``kMax`` : 1/Ang
    - ``da`` : mrad
    - ``angleMax`` : mrad
    - ``n_alpha`` : unitless factor (kMax = n_alpha * conv_angle)
    - ``RBF`` : px (radius of bright field disk in px)
    - ``fitRBF`` : None (default)
    
    .. note::
        real space pixel size and k-space pixel size are directly related via dx = 1/(2*kMax).
    """
    
    value: Optional[float] = Field(default=None, gt=0.0, description="Value for measurements calibration. Unit: Ang, Ang-1, mrad, # of alpha, px depends on modes")
    """
    Value used for specific calibration mode. All mode requires a ``value`` except for ``fitRBF``, 
    as it fits the radius of bright field disk (RBF) in px, and calibrate it with user-provided ``conv_angle``.
    """

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
    """
    Available options are ``uniform`` (default) or ``custom``
    
    - ``uniform`` : Equally split the occupancy to each object mode, meaning each omode would have 1/omode occupancy.
    - ``custom`` : Pass in the desired occupancy as an array to ``init_occu``. Note that length(arr) = omode, and sum(arr) = 1.
    
    """
    
    init_occu: Optional[List[float]] = Field(description="Value for object mode occupancy initialization")
    """
    List of floats that are used to initialize the object mode occupancy if 'occu_type' = 'custom'.
    """
    
    @model_validator(mode='before')
    def set_default_init_occu(cls, values: dict) -> dict:
        occu_type = values.get('occu_type', 'uniform')
        if occu_type == 'uniform' and 'init_occu' not in values:
            values['init_occu'] = None
        return values


class MeasPad(BaseModel):
    model_config = {"extra": "forbid"}
    
    mode: Optional[Literal['on_the_fly', 'precompute']] = Field(default='on_the_fly', description="Padding mode for measurements. Choose between 'on_the_fly' or 'precompute', or None.")
    """
    Available options are ``on_the_fly`` (default), ``precompute``, or null (will disable padding).

    - ``precompute`` : Pad the measurements during initialization, and keep the padded array in memory.
    - ``on_the_fly`` : Pad the measurements during optimization, so it's more efficient for memory. 
    
    ``on_the_fly`` padding doesn't really affect the reconstruction time, so it's suggested to always use ``on_the_fly`` to save the memory.
    """
    
    padding_type: Literal['constant', 'edge', 'linear_ramp', 'exp', 'power'] = Field(default='power', description="Padding type for measurements. Suggested type is 'power'.")
    """
    Available options are ``constant``, ``edge``, ``linear_ramp``, ``exp``, or ``power`` (default).
    
    - ``constant`` : Pad with constant value specified by ``value``
    - ``edge`` : Pad with the edge value of the mean diffraction pattern amplitude
    - ``linear_ramp`` : Pad with a linear ramp function to go from edge value to the specified ``value``
    - ``exp``: Pad with a background fitted with exponential decay function ``a * np.exp(-b * r)``
    - ``power``: Pad with a background fitted with a power law ``a * r**-b``

    If choosing ``constant`` or ``linear_ramp``, you'll need to supply an addition field of ``value``.
    If using ``exp`` or ``power``, the mean diffraction pattern amplitude is used to fit the functional coefficients. 
        
    ``power`` seems to fit the high angle scattering intensities the best, although padding with ``constant`` 0 is also a popular choice.
    """
    target_Npix: int = Field(default=256, description="Target measurement number of pixels")
    """
    Final targeted diffraction pattern size, and it doesn't need to be power of 2.
    """
    
    value: Optional[float] = Field(default=0, description="Value used for padding background if mode='constant' or 'linear_ramp'.")
    """
    Amplitude value used for padding background if ``'mode'='constant'`` or ``'linear_ramp'``.
    
    Since ``meas_normalization`` is done before padding, so the supplied ``value`` must take the normalization into account.
    """
    
    threshold: Optional[float] = Field(default=70, description="Threshold value used for fitting background if mode='power' or 'exp'.")
    """
    Amplitude threshold percentile used for fitting background if ``'mode'='power'`` or ``'exp'``.
    
    This creates an "exclusion mask" by thresholding the diffraction pattern with values above such percentile threshold.
    The exclusion mask is used to exclude the contribution from BF disk and strong Bragg disks, so we can fit the background more accurately.
    Lower ``threshold`` value would exclude more values from diffraction pattern, hence less area is left for background fitting.
    """


class MeasResample(BaseModel):
    model_config = {"extra": "forbid"}
    
    mode: Optional[Literal['on_the_fly', 'precompute']] = Field(default='on_the_fly', description="Resampling mode for measurements. Choose between 'on_the_fly' or 'precompute', or None.")
    """
    Available options are ``on_the_fly`` (default), ``precompute``, or null (will disable resampling).

    - ``precompute`` : Resample the measurements during initialization, and keep the padded array in memory. This is more efficient if you're "downsampling".
    - ``on_the_fly`` : Resample the measurements during optimization, so it's more efficient for memory if you're "upsampling". 
    """
    
    scale_factors: List[float] = Field(default=[2, 2], min_items=2, max_items=2, description="Resampling scale factors (2,) for measurements")
    """
    List of 2 floats as [ky_zoom, kx_zoom], currently only square detectors are supported, so ky_zoom must be the same as kx_zoom. 
    """

class MeasRemoveNegValues(BaseModel):
    model_config = {"extra": "forbid"}

    mode: Literal["subtract_min", "subtract_value", "clip_neg", "clip_value"] = Field(default="clip_neg", description="Mode to remove negative values in measurements")
    """
    Available options are ``clip_neg`` (default), ``clip_value``, ``subtract_min``, ``subtract_value``. 
    
    - ``clip_neg`` : Clips the negative value (i.e., setting negative values to 0)
    - ``clip_value`` : Clips the pixel intensity below specified ``value``
    - ``subtract_min`` : Subtracts the entire dataset by ``dataset.min`` if ``dataset.min < 0``
    - ``subtract_value`` : Subtracts the entire dataset by specified ``value``

    If the dataset array still contains negative values for any reason, a ``clip_neg`` correction would be enforced again to guarantee no negative values.
    
    """
    
    value: Optional[float] = Field(default=None, description="Value used for removing negative values in measurements if mode='subtract_value' or 'clip_value'")
    """
    User-specified intensity value that is used for ``subtract_value`` and ``clip_value`` modes.
    """

    force: bool = Field(default=False, description="Boolean flag to force execute the operation no matter whether measurements contain negative values or not")
    """
    Boolean flag to force executing the ``MeasRemoveNegValues`` operation specified in ``mode``.
    
    By default, this non-negative correction is skipped if there's no negative values in measurements. 
    
    Use ``'meas_remove_neg_values': {'mode': 'subtract_value', 'value': 20, 'force': true}`` if you want to force subtract 20 from the entire dataset when there was no negative values.
    """
    
    
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
    """
    Available options are ``max_at_one`` (default), ``mean_at_one``, ``sum_to_one``, and ``divide_const``.
    
    - ``max_at_one`` : Normalize the dataset such that the mean diffraction pattern intensity has a maximum value at 1
    - ``mean_at_one`` : Normalize the dataset such that the mean diffraction pattern intensity has a mean value at 1
    - ``sum_to_one`` : Normalize the dataset such that the mean diffraction pattern intensity sum to 1
    - ``divide_const`` : Normalize the dataset by dividing with the specified ``value``
    """

    # For 'divide_const', you need to provide another dict entry 'value': <VALUE>.
    
    value: Optional[float] = Field(default=None, description="Value used for normalizing measurements intensities if mode='divide_const'")
    """
    User-specified intensity value that is used for ``divide_const`` mode.
    """
    
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
    """
    Available options are ``total_e_per_pattern``, ``e-PerAng2``.
    
    - ``total_e_per_pattern`` : Applies Poisson noise based on total electron per diffraction pattern
    - ``e-PerAng2`` : Applies Poisson noise based on electron per Angstrom^2
    
    """
    value: Union[int, float] = Field(gt=0.0, description="Dose to be added to measurements")
    """
    User-specified dose value that is used for ``total_e_per_pattern`` or ``e-PerAng2`` modes.
    """


class MeasExport(BaseModel):
    model_config = {"extra": "forbid"}

    file_dir: Optional[str] = Field(default=None, description="Output directory for exported measurements")
    """
    Sets the output directory. 
    
    If ``{'file_dir': null}``, it will export to the same folder as ``meas_params['path']``.
    """

    file_name: str = Field(default='ptyrad_init_meas', description="Output filename for exported measurements")
    """
    The output filenamefor the exported measurement array.
    """
    
    file_format: Literal["hdf5", "tif", "npy", "mat"] = Field(default='hdf5', description="File format for exported measurements")
    """
    Available options are ``hdf5``, ``tif``, ``npy``, and ``mat``.
    
    Note that the ``mat`` is actually HDF5, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.savemat.html
    """
    output_shape: Optional[List[int]] = Field(default=None, description="Output shape for exported measurements")
    """
    List of integers to reshape the output array like [Ny, Nx, Ky, Kx] or [N_scans, Ky, Kx].
    
    **Example:**

    .. code-block:: yaml
    
        'meas_export': {'output_shape': [128, 128, 128, 128]}
    
    By default (if ``None``), the output layout is the PtyRAD internal convention (N_scans, Ky, Kx).
    """
    append_shape: bool = Field(default=True, description="Whether to append the shape at the end of the exported file name")
    """
    If ``True``, it will append the shape of the array to the output file name (e.g., ``filename_100_100_128_128.h5``).
    """


class ProbeNormalization(BaseModel):
    model_config = {"extra": "forbid"}
    
    mode: Literal["mean_total_ints", "max_total_ints", "total_intensity"] = Field(default="mean_total_ints", description="Mode to normalize probe intensity")
    """
    Available options are ``mean_total_ints`` (default), ``max_total_ints``, and ``target_intensity``.

    - ``mean_total_ints``: Normalizes based on the mean total intensities of measurements.

      .. code-block:: python
          
        probe_int = np.mean(dataset.sum((1,2))) # Sum along (ky, kx)
        
    - ``max_total_ints``: Normalizes by the strongest diffraction pattern intensity (ideally the vacuum region).

      .. code-block:: python
    
        probe_int = np.max(dataset.sum((1,2))) # Sum along (ky, kx)

    - ``target_intensity``: Normalizes to a specific value provided in ``value``.

      .. code-block:: python
    
        probe_int = target_intensity
    """
    
    value: Optional[float] = Field(default=None, description="Value used for normalizing probe intensity if mode='target_intensity'")
    """
    User-specified intensity value that is used for ``target_intensity`` mode.
    """
    
    @model_validator(mode='before')
    def check_normalization_value(cls, values: dict) -> dict:
        mode = values.get('mode', 'mean_total_ints')
        value = values.get('value')
        if mode == 'target_intensity' and value is None:
            raise KeyError("'value' is required in probe_normalization for mode='target_intensity'.")
        return values


class ObjZPad(BaseModel):
    model_config = {"extra": "forbid"}
    
    pad_layers: List[Optional[int]] = Field(default=[0,0], min_length=2, max_length=2, description="Number of layers to pad on object's top and bottom surfaces. List of 2 ints.")
    """
    A list of 2 integers determining how many layers are appended to the top and bottom surfaces.
    
    **Example:**
    
    - ``[2, 2]``: Adds 2 layers on top and 2 layers on the bottom.
    - ``[0, 5]``: Asymmetric padding (only pads 5 layers to the bottom).
    """
    
    pad_types: List[Literal['vacuum', 'mean', 'edge']] = Field(default=['vacuum', 'vacuum'], min_length=2, max_length=2, description="Type of layers to pad on object's top and bottom surfaces. List of 2 strings of 'vacuum', 'mean', 'edge")
    """
    A list of 2 strings specifying the content of the padded layers.
    
    Available options are ``vacuum``, ``mean``, ``edge``.
    
    Default is ``['vacuum', 'vacuum']``
    
    - ``vacuum``: Fills with empty (vacuum) space
    - ``mean``: Fills with the mean layer of the object
    - ``edge``: Repeats the edge layer values
    """
    
    @field_validator("pad_layers")
    @classmethod
    def validate_pad_layers(cls, v):
        if not all(x is None or x >= 0 for x in v):
            raise ValueError("pad_layers must contain non-negative integers or None")
        return v


class ObjZResample(BaseModel):
    model_config = {"extra": "forbid"}
    
    mode: Literal['scale_Nlayer', 'scale_slice_thickness', 'target_Nlayer', 'target_slice_thickness'] = Field(description="Resampling mode for object depth. Available options are 'scale_Nlayer', 'scale_slice_thickness', 'target_Nlayer', 'target_slice_thickness'.")
    """
    Available options:
    
    - ``scale_Nlayer`` : Scales the number of layers
    - ``scale_slice_thickness`` : Scales the thickness of each slice
    - ``target_Nlayer`` : Resamples to a specific total number of layers
    - ``target_slice_thickness`` : Resamples to a specific slice thickness
    """

    value: Union[int, float] = Field(description="Corresponding values for the specified resampling mode for object depth.")
    """
    The numerical value for the operation.
    
    - Must be a **positive integer** for ``target_Nlayer``.
    - Must be a **positive float** for all other modes.
    """
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
    """
    Determines how the tilt array is initialized.

    - ``all``: Creates a ``(1, 2)`` array. All positions share the same global tilt.
    - ``each``: Creates a ``(N_scans, 2)`` array. Starts uniform but allows position-dependent optimization (if ``obj_tilts`` learning rate > 0).
    """

    init_tilts: List[List[float]] = Field(default=[[0, 0]], description="Initial value for (N,2) object tilts")
    """
    **[mrad]** Initial tilt values ``[[tilt_y, tilt_x]]``.
    
    Default is ``[[0, 0]]``.
    """


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
        'simu': type(None),
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

def _validate_source_params_pair(
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
    """
    The ``init_params`` can be roughly categorized into 4 parts:
    
    1. Experimental params (kv, convergence angle, aberrations)
    2. Model complexity (number of probe / object modes, number of object slices and slice thickness)
    3. Preprocessing (permutation, reshaping, flipping, cropping, padding, and normalization)
    4. Input source and params (where and how to load / initialize the diffraction pattern, probe, object)
    
    Most of the fields in ``init_params`` are **dataset-dependent**, so users must manually check and input values.
    While the preprocessing steps are not required fields, incorrect data orientation (``meas_flipT``) would lead to 
    incorrect reconstructions, so it's practically required. 
    
    """
    model_config = {"extra": "forbid", 
                    "arbitrary_types_allowed": True} # This is needed to validate np.ndarray type

    random_seed: Optional[int] = Field(default=None, description="Random seed for improved reproducibility")
    """
    Random seed is used to initialize the random number generator for improved reproducibility.
    
    **Example:**
    
    .. code-block:: yaml
    
        'random_seed': 42
    
    Or direcly passed in from CLI with:
    
    .. code-block:: bash
    
        ptyrad run --params_path <PATH> --seed 42
        
    .. note::
        This random seed is used for both initialization and reconstruction.
        For multiGPU, a fixed seed is automatically generated if not provided to ensure both GPUs started from the exact same object, probe, and positions.
    """
    
    # Experimental params
    probe_illum_type: Literal['electron', 'xray'] = Field(default="electron", description="Probe illumination type")
    """
    Type of probe illumination, choose between ``electron`` or ``xray``.
    
    **Example:**
    
    .. code-block:: yaml
        
        'probe_illum_type': 'electron' 
    
    This affects the required fields for probe generation, as electron and x-ray ptychography use different parameters.
    
    """
    
    ## Electron probe params (used if probe_illum_type == 'electron')
    probe_kv: Optional[float] = Field(default=None, description="Electron acceleration voltage in kV") # Required for electron
    """ 
    **[kV]** Acceleration voltage for relativistic electron wavelength calculation

    **Example:**
    
    .. code-block:: yaml
        
        'probe_conv_angle': 24.9
    
    Required for ``'probe_illum_type': 'electron'``
    
    """
    
    probe_conv_angle: Optional[float] = Field(default=None, gt=0.0, description="Semi-convergence angle in mrad") # Required for electron
    """ 
    **[mrad]** Semi-convergence angle in mrad for probe-forming aperture 

    **Example:**
    
    .. code-block:: yaml

        'probe_conv_angle': 24.9
    
    Required for ``'probe_illum_type': 'electron'``
    
    .. warning::
        This value should be experimentally calibrated as close as possible, 
        as it could affect the real and k-space px size calibration if useing ``'meas_calibration': {'mode': 'fitRBF'}``.
    """

    # The validation of dict entries are deferred to Aberrations class
    probe_aberrations: Dict[str, Any] = Field(default_factory=dict,
                                              description="Dictionary of aberration coefficients (Krivanek or Haider). e.g. {'defocus': -100, 'A1': 5, 'C21': 50}")
    """
    **[Ang, degree]** Aberration coefficients for electron probe. 

    **Example:** 

    .. code-block:: yaml
    
        'probe_aberrations': {'C10': 200, 'A1': 50, 'A1phi': 25, 'C21a': 300, 'C23': 400, 'phi23': 25, 'Cs': 5000}

    Supports Krivanek (polar and cartesian), Haider notations, common aliases, or a mix of them.
    
    Aberrations will be internally normalized to Krivanek polar while generating the probe.
    
    - Krivanek polar: ``C10``, ``C21``, ``phi21``, etc. 
    
        :math:`C_{nm} = \\sqrt{C_{nma}^2 + C_{nmb}^2}`
        
        :math:`\\phi_{nm} = \\arctan(\\frac{C_{nmb}}{C_{nma}}) / m`
        
    - Krivanek cartesian: ``C21a``, ``C21b``, etc.
    
        :math:`C_{nma} = C_{nm} * cos(m * \\phi_{nm})`
        
        :math:`C_{nmb} = C_{nm} * sin(m * \\phi_{nm})`
        
        :math:`C_{nm, complex} = C_{nma} + iC_{nmb}`
            
    - Haider notation: ``A1``, ``A1phi``, ``B2``, ``B2phi``, etc. Only supported up to A5 (equivalently C56).
     
    - Common aliases: ``defocus``, ``Cs`` are also supported. 

    .. note::
        **C10 = -df, and positive C10 refers to overfocus** (stronger lens) following Kirkland/abtem convention.
        See https://ptyrad.readthedocs.io/en/latest/_autosummary/ptyrad.utils.aberrations.html#module-ptyrad.utils.aberrations for more details.
    """
    
    
    ## Xray probe params (used if probe_illum_type == 'xray')
    beam_kev: Optional[float] = Field(default=None, description="Xray beam energy in keV")
    """ 
    **[keV]** X-ray beam energy for photon wavelength calculation
    
    Required for ``'probe_illum_type': 'xray'``
    """

    probe_dRn: Optional[float] = Field(default=None, description="Xray probe param: Width of outermost zone (in meters)")
    """ 
    **[meter]** Width of the outermost zone for X-ray probes using Fresnel zone plates (FZP)
    
    Required for ``'probe_illum_type': 'xray'``
    """
    
    probe_Rn: Optional[float] = Field(default=None, description="Xray probe param: Radius of outermost zone (in meters)")
    """ 
    **[meter]** Radius of the outermost zone for X-ray probes using Fresnel zone plates (FZP)
    
    Required for ``'probe_illum_type': 'xray'``
    """    

    probe_D_H: Optional[float] = Field(default=None, description="Xray probe param: Diameter of the central beamstop (in meters)")
    """ 
    **[meter]** Diameter of the central beamstop for X-ray probes using Fresnel zone plates (FZP)
    
    Required for ``'probe_illum_type': 'xray'``
    """    

    probe_D_FZP: Optional[float] = Field(default=None, description="Xray probe param: Diameter of pinhole in meters")
    """ 
    **[meter]** Diameter of pinhole for X-ray probes using Fresnel zone plates (FZP)
    
    Required for ``'probe_illum_type': 'xray'``
    """    

    probe_Ls: Optional[float] = Field(default=None, description="Xray probe param: Distance (in meters) from the focal plane to the sample")
    """ 
    **[meter]** Distance from the focal plane to the sample for X-ray probes using Fresnel zone plates (FZP)
    
    Required for ``'probe_illum_type': 'xray'``
    """
    
    meas_Npix: int = Field(ge=1, description="Detector pixel number (square detector)") # Required
    """
    Detector pixel number, EMPAD is 128. 
    
    Only supports square detector (ky = kx) for simplicity.
    
    **Example:**

    .. code-block:: yaml
        
        'meas_Npix': 128
    
    .. note::
        This value always refer to the raw data pixel number, and is used to load and validate the raw data while loading.
        If a user loaded a dataset with Npix=128, and later upsample it with ``meas_resample`` by a factor of 2,
        PtyRAD will automatically update ``meas_Npix`` internally to 256. Same logic for other dimension-modifying preprocessing.  
    """
    
    # Many features actually assume raster scan with fast/slow so we may always infer N_scans
    # Specifying N_scans is useful for checking while loading meas from .raw, or custom pos source like spiral scan
    # Inferred from N_scan_slow/fast
    pos_N_scans: int = Field(ge=1, description="Number of probe positions")
    """
    Number of probe positions (or equivalently diffraction patterns since 1 DP / position)

    **Example:**
    
    .. code-block:: yaml
    
        'pos_N_scans': 16384
    
    .. note::
        ``pos_N_scans`` is optional for raster scan, since it can be internally inferred from ``pos_N_scan_slow * pos_N_scan_fast``.
        However, ``pos_N_scans`` is required when using custom positions (``'pos_source': 'custom'``) like spiral or other scan patterns.
        In those cases, ``pos_N_scans`` is required, 
        while ``pos_N_scan_slow`` and ``pos_N_scan_fast`` would need to satisfy ``pos_N_scan_slow * pos_N_scan_fast = pos_N_scans``
        in order to pass the consistency test by the end of initialization.
    """
    
    pos_N_scan_slow: int = Field(ge=1, description="Number of scan positions along slow direction") # Required
    """
    Number of scan position along slow scan direction. Usually it's the vertical direction of acquisition GUI
    
    **Example:**
    
    .. code-block:: yaml
    
        'pos_N_scan_slow': 128
    
    .. note::
        This value always refer to the raw data scan dimension, and is used to load and validate the raw data while loading.
        If a user loaded a dataset with ``N_scan_slow`` = 128, and later crop it with ``meas_crop`` to 64,
        PtyRAD will automatically update ``pos_N_scan_slow`` internally to 64. Same logic for other dimension-modifying preprocessing.

    .. note::
        For custom scan positions (``'pos_source': 'custom'``) that doesn't have so-called fast/slow scan directions, 
        put in values such that ``pos_N_scan_slow`` * ``pos_N_scan_fast`` = ``pos_N_scans`` to pass the consistency test by the end of initialization.  
    """
    
    pos_N_scan_fast: int = Field(ge=1, description="Number of scan positions along fast direction") # Required
    """
    Number of scan position along fast scan direction. Usually it's the horizontal direction of acquisition GUI
    
    **Example:**

    .. code-block:: yaml
    
        'pos_N_scan_fast': 128
    
    .. note::
        This value always refer to the raw data scan dimension, and is used to load and validate the raw data while loading.
        If a user loaded a dataset with ``N_scan_fast`` = 128, and later crop it with ``meas_crop`` to 64,
        PtyRAD will automatically update `pos_N_scan_fast` internally to 64. Same logic for other dimension-modifying preprocessing.

    .. note::
        For custom scan positions (``'pos_source': 'custom'``) that doesn't have so-called fast/slow scan directions, 
        put in values such that ``pos_N_scan_slow`` * ``pos_N_scan_fast`` = ``pos_N_scans`` to pass the consistency test by the end of initialization.    
    """
    
    pos_scan_step_size: float = Field(gt=0.0, description="Scan step size in Angstrom") # Required
    """
    **[Ang]** Step size between probe positions in a rectangular raster scan pattern.
    
    **Example:**

    .. code-block:: yaml
    
        'pos_scan_step_size': 0.415
    
    .. warning::
        This value should be experimentally calibrated as close as possible.
        Incorrect scan step size is equivalent to a global scaling of the scan pattern.
        Even a 2-5% error can take more than 30K iterations to properly converge, 
        and larger error can cause significant gridding artifacts. 
        
        If possible, always acquire data when the stage is stable, 
        or pre-calibrate a dataset-dependent scan step size to accelerate convergence of ptychographic reconstructions.
        See tutorial notebook: https://github.com/chiahao3/ptyrad/tree/main/tutorials/get_affine_from_image.ipynb
    """
    
    meas_calibration: MeasCalibration = Field(default_factory=MeasCalibration, description="Calibration mode and value")
    """
    Calibration mode for the measurements (i.e., diffraction patterns) pixel size.

    **Example:**

    .. code-block:: yaml

        'meas_calibration': {'mode': 'fitRBF'}

    See :class:`MeasCalibration` for more details.
    
    .. warning::
        Ptychography is very sensitive to microscope calibration, and mis-calibration can lead to slow, incorrect, or even failed reconstructions.
        Factory or institutional calibration can often be 5-10% off (convergence angle, scan step size) and the accuracy is also often kV-dependent.
        Users are strongly advised to perform proper microscope calibration to ensure accurate results.
    """

    # Model complexity
    probe_pmode_max: int = Field(default=4, ge=1, description="Maximum number of mixed probe modes")
    """
    Maximum number of mixed probe modes. 
    
    **Example:**
    
    .. code-block:: yaml

        'probe_pmode_max': 4
    
    Set to ``probe_pmode_max`` = 1 for single probe state, ``probe_pmode_max`` > 1 for mixed-state probe during initialization.
    
    Typical values of suggested probe modes are 4 to 12, depends on the coherence of the probe. 
    
    For simulated initial probe, it'll be generated with the specified number of probe modes. 
    
    For loaded probe, the pmode dimension would be capped or padded to this number.
    
    .. note::
        The required reconstruction time often scale linearly with number of probe modes, along with a non-zero intercept.
    """
    
    probe_pmode_init_pows: List[float] = Field(default=[0.02], description="Initial power weights for probe modes")
    """
    List of 1 or a few (up to pmode_max) floats.
    
    This specifies initial power(s) for each additional probe modes.

    **Example:**
    
    .. code-block:: yaml
    
        'probe_pmode_init_pows': [0.02]
         
    If set at [0.02], all additional probe modes would contain 2% of the total intensity. 
    
    ``sum(pmode_init_pows)`` must be 1 if ``len(pmode_init_pows) > 1``. 

    See ``ptyrad.optics.probe.make_mixed_probe`` for more details
    """
    
    obj_omode_max: int = Field(default=1, ge=1, description="Maximum number of mixed object modes")
    """
    Maximum number of mixed object modes.
    
    **Example:**

    .. code-block:: yaml
    
        'obj_omode_max': 1
     
    Set to ``obj_omode_max`` = 1 for single object state, ``obj_omode_max`` > 1 for mixed-state object during initialization. 
    
    For simulated initial object, it'll be generated with the specified number of object modes. 
    
    For loaded object, the omode dimension would be capped or padded to at this number.
    
    .. note::
        The required reconstruction time often scale linearly with number of object modes, along with a non-zero intercept.
    """
    
    obj_omode_init_occu: ObjOmodeInitOccu = Field(default_factory=ObjOmodeInitOccu, description="Occupancy type and value for mixed-object modes")
    """
    Occupancy type and value for mixed-object modes. 
    
    Typically we do 'uniform' for frozen phonon like configurations.
    
    **Example:**

    .. code-block:: yaml
    
        'obj_omode_init_occu': {'occu_type': 'uniform', 'init_occu': null}

    See :class:`ObjOmodeInitOccu` for more details.    
    """
    
    obj_Nlayer: int = Field(ge=1, description="Number of slices for multislice object")
    """
    Number of slices for multislice object.

    **Example:**

    .. code-block:: yaml
    
        'obj_Nlayer': 1
    
    Set to ``obj_Nlayer`` = 1 for single slice object, ``obj_Nlayer`` > 1 for multislice object during initialization.
    
    For quick params check, single slice ptychography can often give fair results for 10-15 nm thick specimens with light/medium atoms.
    
    However, most electron microscopy specimens (except monolayer graphene) exhibit noticeable multiple scatterings,
    
    so multislice ptychography is often helpful / required for improved reconstruction quality and quantitativeness.
        
    .. note::
        The required reconstruction time often scale linearly with number of layers, along with a non-zero intercept.
        Reconstruction with more than 40 slices can be computationally expensive and less stable.
    """
    
    obj_slice_thickness: float = Field(gt=0.0, description="Slice thickness in Angstrom")
    """
    **[Ang]** Slice thickness (propagation distance) for multislice ptychography. 
    
    **Example:**
    
    .. code-block:: yaml
    
        'obj_slice_thickness': 10
    
    Typical values are between 1 to 20 Ang, for most samples 10 Ang (1 nm) is a good starting point.
    
    Slice thickness has no effect for single-slice ptychography.
    
    .. note::
        Comparing to conventional multislice formalism (slice thickness ~ 1 Ang), 
        multislice electron ptychography (MEP) is making a wild approximation to trade for speed.
        Although using 1-nm-thick slices seems to give qualitativly correct result with MEP,
        it will not match with high-precision quantitative electron diffraction simulation with 1-Ang-slice.
    """

    # Preprocessing
    meas_permute: Optional[List[int]] = Field(default=None, description="Permutation for diffraction patterns")
    """
    Applies additional permutation (reorder axes) to the loaded diffraction patterns with a list of ints. 

    The syntax is the same as ``np.transpose()``.
    
    **Example:**

    .. code-block:: yaml
    
        'meas_permute': [2, 3, 0, 1]

    .. note::
        This is offen needed if the original dataset is arranged as (ky, kx, Ry, Rx), where Ry and Rx are real-space scan dimensions.
        For such cases, use ``'meas_permute': [2, 3, 0, 1]`` to convert the dataset into (Ry, Rx, ky, kx) first, 
        and then use ``meas_reshape`` to make it into (N_scans, ky, kx).
    """

    meas_reshape: Optional[List[int]] = Field(default=None, min_items=3, max_items=3, description="Reshape for diffraction patterns")
    """
    Applies additional reshaping (rearrange elements) to the loaded diffraction patterns with a list of 3 ints. 
    
    The syntax is the same as ``np.reshape()``.
    
    **Example:**

    .. code-block:: yaml
    
        'meas_reshape': [-1, 128, 128]
     
    .. note::
        This is often needed to convert the 4D diffraction dataset (Ry,Rx,ky,kx) into 3D (N_scans,ky,kx).
        We can put ``'meas_reshape': [-1, <Npix>, <Npix>]`` for convenient representation, make sure to replace <Npix> with your actual value in params files.
    """

    meas_flipT: Optional[List[int]] = Field(default=None, min_items=3, max_items=3, description="Flip and transpose for diffraction patterns")
    """
    Applies additional flipping and transposing to the loaded diffraction patterns with a list of 3 binary booleans (0 or 1)
    
    The operation syntax is [flipud, fliplr, transpose], which is the same as PtychoShleves / fold_slice.
     
    **Example:**

    .. code-block:: yaml
    
        'meas_flipT': [1,0,0] 
    
    [1,0,0] means to flip diffraction patterns vertically.
    
    .. note::
        Although the programmic default is null or equivalently [0,0,0], practically we often need to find the correct dataset orientation for each microscope. 

    .. note::
        Despite ``pos_scan_flipT`` can also be used to compensate relative orientation between scan coordinates and diffraction patterns,
        it's strongly suggested to use ``meas_flipT`` to correct for the dataset orientation, and keep ``pos_scan_flipT`` as null.
        Because ``pos_scan_flipT`` would alter the visual layout of "fast/slow" directions, and can cause discrepancy with other simultaneously acquired datasets (like ADF-STEM). 
        Therefore, ``meas_flipT`` is the recommended approach, and is the only orientaiton-related value that can be optionally attached to output reconstruction folder name.
    """
    
    meas_crop: Optional[List[Optional[List[int]]]] = Field(default=None, description="Crop for 4D diffraction patterns")
    """
    Applies additional cropping to the 4D dataset in both real and k-space with a (4,2) nested list of ints as: 
    
    ``[[scan_slow_start, scan_slow_end], [scan_fast_start, scan_fast_end], [ky_start, ky_end], [kx_start, kx_end]]``. 
    
    **Example:**
    
    .. code-block:: yaml

        'meas_crop': [[0, 64], [0, 64], null, null]
        
    ``[[0,64], [0,64], null, null]`` means cropping scan positions to 64 x 64 in real space, but leaves the k-space untouched. 

    This is useful for reconstrucing only a subset of real-space probe positions, or to crop the kMax of diffraction patterns. 

    The slicing syntax follows conventional numpy indexing so the upper bound is not included.
    
    .. note::
        **This does NOT affect the file on disk, and is only operating on the loaded array.** 
        This feature allows flexible change of reconstruction FOV (if cropping in real space) without processing and saving multiple versions of the same dataset.
        If cropping in k-space, it will modify kMax and hence the real space pixel size, which can be useful if one want a faster reconstruction (due to smaller Npix).
        There is no need to modify ``meas_Npix``, ``pos_N_scans``, ``pos_N_fast_scan``, or ``pos_N_slow_scan`` with this feature, as the value will be automatically updated internally.
    """

    meas_pad: Optional[MeasPad] = Field(default=None, description="Padding configuration for CBED")
    """
    Applies additional padding to the diffraction pattern to side length = ``'target_Npix'`` based on the ``padding_type``. 
    
    **Example:**
    
    .. code-block:: yaml
    
        'meas_pad': {'mode': 'on_the_fly', 'padding_type': 'power', 'target_Npix': 256, 'value': null, 'threshold': 70}
    
    There are 5 configurational fields, see :class:`MeasPad` for more details.

    .. note::
        Padding the diffraction pattern will increase kMax, and hence reducing the real space pixel size (dx). 
        While the nominal kMax increases, it doesn't necessarily enhance the ultimate spatial resolution as there's no actual information about the specimen being added.
    """
    
    meas_resample: Optional[MeasResample] = Field(default=None, description="Resampling configuration for diffraction patterns")
    """
    Applies additional resampling to the diffraction patterns by ``'scale_factors'`` along ky and kx directions. 

    **Example:**
 
    .. code-block:: yaml
    
        'meas_resample': {'mode': 'on_the_fly', 'scale_factors': [2,2]}
    
    See :class:`MeasResample` for more details.
    
    .. note::
        Resampling the diffraction pattern does NOT change kMax (or dx), but it will modify the FOV of the probe array.
        Common usage of resampling is to upsample the diffraction pattern, so that the probe array can have more extended FOV to accomodate the probe.
        For very large probe size, or thick samples with large convergence angle, this is often needed to reduce edge artifacts of the probe.
    """
    
    meas_add_source_size: Optional[float] = Field(default=None, gt=0.0, description="Gaussian blur std for spatial partial coherence in Angstrom")
    """
    **[Ang]** Adds additional spatial partial coherence to diffraction patterns by applying Gaussian blur along scan directions.
    
    **Example:**

    .. code-block:: yaml
    
        'meas_add_source_size': 0.34
     
    The provided value is used as the std (sigma) of the Gaussian blurring kernel in real space (i.e., it's mixing nearby diffraction patterns).
    
    This is useful for making simulated datasets more realisitc in a flexible manner, without creating unnecessary copies. 
    
    .. note:: 
        Since FWHM ~ 2.355 std, so a std of 0.34 Ang is equivalent to a source size (FWHM) of 0.8 Ang
    """

    meas_add_detector_blur: Optional[float] = Field(default=None, gt=0.0, description="Gaussian blur std for detector in pixels")
    """
    **[k-space px]** Adds additional detector blur to diffraction patterns to emulate the PSF on detector.
    
    The provided value is used as the std (sigma) for the Gaussian blurring kernel in k-space. 
    
    **Example:**

    .. code-block:: yaml
    
        'meas_add_detector_blur': 1
        
    This is useful for making simulated datasets more realisitc in a flexible manner, without creating unnecessary copies. 
    
    .. note::
        This is applied to the "measured", or "reference" diffraction pattern, so is different from ``model_params['detector_blur_std']``
        that applies blurring to the forward simulated diffraction pattern.
    """
    
    meas_remove_neg_values: MeasRemoveNegValues = Field(default_factory=MeasRemoveNegValues, description="Preprocessing for negative values in measurements")
    """
    Removes negative values or in the measurements (i.e, diffraction patterns).
    
    The measurements array must contain only positive values, so this correction is enforced with a default value of ``clip_neg``.

    This can also be used to clip low intensity noises, or to offset the entire baseline by specified values.
    
    **Example:**

    .. code-block:: yaml
    
        'meas_remove_neg_values': {'mode': 'clip_neg'}

    See :class:`MeasRemoveNegValues` for more details.
        
    .. note::
        For low dose data set with many 0-electron events, it is recommended to use ``clip_value`` to clip the 0-electron peak so the background is truly 0.
        For example, if the 0-electron peak is around 20 counts, then use ``'meas_remove_neg_values': {'mode': 'clip_value', 'value': 20}`` to clip those values,
        otherwise the probe will absorb the background intensity, which may degrade or even fail the reconstruction.
    """
    
    meas_normalization: MeasNormalization = Field(default_factory=MeasNormalization, description="Normalization method for measurements")
    """
    Normalization methods of the measurements array.

    **Example:**
    
    .. code-block:: yaml
    
        'meas_normalization': {'mode': 'max_at_one'}

    See :class:`MeasNormalization` for more details.

    .. note::
        Many optimizers are scale dependent and generally works better for values around 1. 
        This normalization keeps the diffraction pattern in a comfortable range for the AD optimizers, 
        and ensures the edge intensities are low enough so that we don't suffer from significant wrap-around FFT artifacts.

    """
    
    meas_add_poisson_noise: Optional[MeasAddPoissonNoise] = Field(default=None, description="Poisson noise configuration")
    """
    Applies additional Poisson noise to diffraction patterns to emulate electron dose Poisson statistics.

    **Example:**

    .. code-block:: yaml

        'meas_add_poisson_noise': {'unit': 'e_per_Ang2', 'value': 10000}


    This is useful for making simulated datasets more realisitc in a flexible manner, without creating unnecessary copies. 
    
    See :class:`MeasAddPoissonNoise` for details.
    
    .. note::
        You can use this to simulate ptychographic reconstruction with different dose conditions using a single noise-free dataset.
    """
    
    meas_export: Optional[Union[bool, MeasExport]] = Field(default=None, description="Export configuration for measurements")
    """
    Exports the final initialized measurements array to disk for further processing, analysis, or visualization.

    **Example:**

    .. code-block:: yaml

        'meas_export': {'file_name': 'ptyrad_init_meas', 'file_format': 'hdf5', 'append_shape': True}


    .. code-block:: yaml
    
        'meas_export': true # or null to simply disable

    The exported data layout has the same Python convention with py4DGUI so there's no need to worry about orientation mismatch.

    By default the output layout is (N_scans, Ky, Kx), and dropping it to py4DGUI then reshape it into (Ny, Nx, Ky, Kx) keeps the correct orientation. 

    See :class:`MeasExport` for details.

    .. note::
        This can be used to interactively check whether the ``meas_flipT`` is correct.
    """

    probe_permute: Optional[List[int]] = Field(default=None, description="Permutation for probe")
    """
    Applies additional permutation (reorder axes) to the loaded probe array with a list of ints. 

    The syntax is the same as ``np.transpose()``.
    
    **Example:**

    .. code-block:: yaml
    
        'probe_permute': [1, 2, 0]

    .. note::
        This can be used to permute custom probe array. For common `probe_source`` like ``PtyShv``, 
        the permutation is done internally during initialization so this ``probe_permute`` is not needed.
    """

    probe_z_shift: Optional[float] = Field(default=None, description="Axially (z) shift the initialized probe")
    """
    **[Ang]** Axially shifts the loaded probe along the depth (z) dimension.

    **Example:**

    .. code-block:: yaml

        'probe_z_shift': -50.0

    The sign convention follows the propagation direction (PtyRAD progrates from top to bottom):
    
    - **Positive (+)**: Propagates the probe forward (further into the object).
    - **Negative (-)**: Rewinds the probe backward.

    .. note::
        This is typically used to conserve the relative geometry between the probe and object when the object is re-centered (e.g., via ``obj_z_pad``, ``obj_z_crop``). 
        
        For example, if you pad the object with a 50 Ang vacuum layer ON TOP via ``obj_z_pad``, 
        you should apply a ``-50`` shift to the reconstructed probe to maintain relative geometry of probe and object.
    """
    
    probe_normalization: ProbeNormalization = Field(default_factory=ProbeNormalization, description="Normalization method for probe intensity")
    """
    Normalization methods for probe intensity.

    **Example:**

    .. code-block:: yaml

        'probe_normalization': {'mode': 'mean_total_ints'}

    See :class:`ProbeNormalization` for more details.

    .. note::
        Ideally, the probe intensity should match the measurement intensity to ensure the reconstructed object amplitude stays close to 1.

    However, for thicker sample with shorter collection angles, 
    
    significant amount of electrons are scattered outside of the detector so probe_int > DP_int.
    
    For such cases, use

    .. code-block:: yaml

        'probe_normalization': {'mode': 'max_total_ints'}

    to normalize the probe intensity by the strongest diffraction pattern (ideally vacuum region) for more accurate reconstruction.

    """

    pos_scan_flipT: Optional[List[int]] = Field(default=None, description="Flip and transpose for scan patterns")
    """
    Applies additional flipping and transposing to the scan pattern with a list of 3 binary booleans (0 or 1)
    
    The operation syntax is [flipud, fliplr, transpose], which is the same as PtychoShleves / fold_slice.
     
    **Example:**

    .. code-block:: yaml
    
        'pos_scan_flipT': [1,0,0] 
    
    [1,0,0] means to flip the scan pattern vertically.
 
    .. note::
        Modifying ``pos_scan_flipT`` would change the image orientation, 
        so it's recommended to set this to null, and only use ``meas_flipT`` to get the orientation correct.
    """

    pos_scan_affine: Optional[List[float]] = Field(default=None, description="Affine transformation for scan patterns")
    """
    Applies an additional affine transformation to the initialized scan patterns to correct for sample drift or imperfect scan coils.

    **Example:**

    .. code-block:: yaml

        'pos_scan_affine': [1.0, 0.0, 2.0, 0.5]

    The list must contain 4 floats in the order: ``[scale, asymmetry, rotation, shear]``.

    - **scale**: Global scaling factor
    - **asymmetry**: Aspect ratio distortion
    - **rotation**: **[deg]** Rotation angle
    - **shear**: **[deg]** Shear angle

    See tutorial notebook: https://github.com/chiahao3/ptyrad/tree/main/tutorials/get_affine_from_image.ipynb
    
    .. note::
        Default is ``None`` (equivalent to ``[1, 0, 0, 0]``). This format matches the convention used in PtychoShelves.
    """
    
    pos_scan_rand_std: Optional[float] = Field(default=0.15, ge=0.0, description="Random displacement std for scan positions in pixels")
    """
    **[real space px]** Standard deviation of Gaussian distributed random displacements applied to the initial scan positions.

    **Example:**

    .. code-block:: yaml

        'pos_scan_rand_std': 0.15

    Randomizing the initial guess helps reduce raster grid pathology during reconstruction.
    """

    obj_z_crop: Optional[List[int]] = Field(default=None, description="Cropping object along depth dimension")
    """
    Applies additional cropping to the 4D object (omode, Nz, Ny, Nx) along the depth dimension to remove unselected layers (e.g., vacuum padding).

    **Example:**

    .. code-block:: yaml

        # Keep layers from index 5 to 25
        'obj_z_crop': [5, 25]

    Takes a list of 2 integers: ``[z_start, z_end]``.

    .. note::
        The syntax follows conventional numpy indexing, so the upper bound (``z_end``) is **not included**.
        This can be combined with ``obj_z_pad``, ``obj_z_resample`` for flexible multislice object initialization.
    """
    
    obj_z_pad: Optional[ObjZPad] = Field(default=None, description="Padding object along depth dimension")
    """
    Applies additional padding to the 4D object ``(omode, Nz, Ny, Nx)`` along the depth dimension (Nz).

    **Example:**

    .. code-block:: yaml

        'obj_z_pad': {'pad_layers': [2, 2], 'pad_types': ['vacuum', 'vacuum']}

    This will pad 2 vacuum layers on top, and another 2 vacuum layer on the bottom.
    
    The ``obj_Nlayer`` will be updated automatically.

    This is useful for expanding reconstructed object to test object positioning or total thickness. 

    See :class:`ObjZPad` for more details.

    .. note::
        Padding the top surface effectively changes the entrance plane.
        You must adjust the axial probe position to maintain the relative geometry:
        
        - **Simulated Probe:** Add a corresponding underfocus to ``probe_aberrations``.
        - **Loaded Probe:** Apply a negative z-shift to ``probe_z_shift`` (e.g., if padding 20 Ang (i.e., 2 nm) vacuum on top, use ``probe_z_shift: -20``).
    
    """

    obj_z_resample: Optional[ObjZResample] = Field(default=None, description="Resampling object along depth dimension")
    """
    Applies additional resampling to the 4D object along the depth dimension, 
    modifying slice thickness while preserving prod(amp), sum(phase), and total thickness.

    **Example:**

    .. code-block:: yaml

        'obj_z_resample': {'mode': 'scale_Nlayer', 'value': 2}

    This will double the number of ``obj_Nlayer`` while halving the ``obj_slice_thickness``.
    
    See :class:`ObjZResample` for details.

    This is useful for reslicing the reconstructed object into different slice thicknesses, including converting between single-slice and multislice objects.
    """
    
    # Input source and params
    meas_source: Literal['file', 'custom'] = Field(default="file", description="Data source for measurements")
    """
    Data source for the diffraction patterns.
    
    Available options: ``file``, ``custom``.
    """

    meas_params: Union[FilePathWithKey, np.ndarray] = Field(description="Parameters for measurement loading") # Required
    """
    Configuration for loading measurement data. 
    
    **Examples by Source Type:**

    - **HDF5 / MAT**:

      .. code-block:: yaml

          'meas_source': 'file'
          'meas_params': {'path': '/data/scan.h5', 'key': '/entry/data'}

    - **TIF**:

      .. code-block:: yaml

          'meas_source': 'file'
          'meas_params': {'path': '/data/scan.tif'}

    - **EMPAD / Raw** (With shape/offset/gap):

      .. code-block:: yaml

          'meas_source': 'file'
          'meas_params': {'path': '/data/scan.raw', 'offset': 0, 'gap': 1024}

          # The ``shape`` can be explictly passed in as a list of ints [N, height, width], 
          # or it will be automatically filled in from ``init_params``,
          # while ``'offset': 0``, and ``'gap': 1024`` are default values for EMPAD1 datasets. 

    - **Custom** (In-memory Numpy):

      .. code-block:: python
      
          # Python code block

          # Load the YAML params file as a dict
          params = load_params(params_path, validate=True)
          
          # Update corresponding fields
          params['init_params']['meas_source'] = 'custom'
          params['init_params']['meas_params] = custom_dataset # numpy.array

          # Pass params to PtyRADSolver
          ptycho_solver = PtyRADSolver(params, device=device, logger=logger)
          ptycho_solver.run()
    """
    
    probe_source: Literal['simu', 'PtyRAD', 'PtyShv', 'py4DSTEM','custom'] = Field(default="simu",description="Data source for probe")
    """
    Data source for the probe. 
    
    Available options: ``simu``, ``PtyRAD``, ``PtyShv``, ``py4DSTEM``, ``custom``.
    """
    
    probe_params: Optional[Union[pathlib.Path, np.ndarray]] = Field(default=None, description="Parameters for probe loading/initialization")
    """
    Configuration for loading or initializing the probe.

    **Examples by Source Type:**

    - **Simulation** (Default):

      .. code-block:: yaml

          'probe_source': 'simu'
          'probe_params': null  
          
          # Probe will be simulated based on values set in ``init_params`` 
          # (e.g., kV, conv_angle, aberrations).

    - **Load from Reconstruction** (PtyRAD / PtyShv / py4DSTEM):

      .. code-block:: yaml

          'probe_source': 'PtyRAD'
          'probe_params': '/path/to/previous_reconstruction.h5'

    - **Custom** (In-memory Numpy):

      .. code-block:: python
      
          # Python code block

          # Load the YAML params file as a dict
          params = load_params(params_path, validate=True)
          
          # Update corresponding fields
          params['init_params']['probe_source'] = 'custom'
          params['init_params']['probe_params'] = custom_probe_3d # numpy.array

          # Pass params to PtyRADSolver
          ptycho_solver = PtyRADSolver(params, device=device, logger=logger)
          ptycho_solver.run()
    """
    
    pos_source: Literal['simu', 'PtyRAD', 'PtyShv', 'py4DSTEM', 'foldslice_hdf5', 'custom'] = Field(default="simu", description="Data source for probe positions")
    """
    Data source for scan positions. 
    
    Available options: ``simu``, ``PtyRAD``, ``PtyShv``, ``py4DSTEM``, ``foldslice_hdf5``, ``custom``.
    """
    
    pos_params: Optional[Union[pathlib.Path, np.ndarray]] = Field(default=None, description="Parameters for probe positions loading/initialization")
    """
    Configuration for loading or initializing scan positions.

    **Examples by Source Type:**

    - **Simulation** (Default):

      .. code-block:: yaml

          'pos_source': 'simu'
          'pos_params': null

          # Positions will be simulated based on ``N_scan_slow``, 
          # ``N_scan_fast``, ``scan_step_size``, and ``scan_affine``.

    - **Load from Reconstruction** (PtyRAD / PtyShv / py4DSTE<):

      .. code-block:: yaml

          'pos_source': 'PtyRAD'
          'pos_params': '/path/to/previous_reconstruction.h5'

    - **APS Instrument Data** (fold_slice HDF5):

      .. code-block:: yaml

          'pos_source': 'foldslice_hdf5'
          'pos_params': '/path/to/positions.h5'
          
          # These HDF5 files are generated from APS instruments 
          # (previously handled in `fold_slice` via 'p.src_positions=hdf5_pos').

    - **Custom** (In-memory Numpy):

      .. code-block:: python
      
          # Python code block

          # Load the YAML params file as a dict
          params = load_params(params_path, validate=True)
          
          # Update corresponding fields
          params['init_params']['pos_source'] = 'custom'
          params['init_params']['pos_params'] = custom_positions_N_by_2 # numpy.array

          # Pass params to PtyRADSolver
          ptycho_solver = PtyRADSolver(params, device=device, logger=logger)
          ptycho_solver.run()
    """
    
    obj_source: Literal['simu', 'PtyRAD', 'PtyShv', 'py4DSTEM','custom'] = Field(default="simu", description="Data source for object")
    """
    Data source for the object. 
    
    Available options: ``simu``, ``PtyRAD``, ``PtyShv``, ``py4DSTEM``, ``custom``.
    """
    
    obj_params: Optional[Union[List[int], pathlib.Path, np.ndarray]] = Field(default=None, description="Parameters for object loading/initialization")
    """
    Configuration for loading or initializing the object.

    **Examples by Source Type:**

    - **Simulation** (Default):

      .. code-block:: yaml

          'obj_source': 'simu'
          'obj_params': [1, 100, 256, 256] 
          
          # Format is [omode, Nz, Ny, Nx].
          # Set to null to let PtyRAD automatically determine shape from scan/probe 
          # (consistent with PtychoShelves behavior).

    - **Load from Reconstruction** (PtyRAD / PtyShv / py4DSTEM):

      .. code-block:: yaml

          'obj_source': 'PtyRAD'
          'obj_params': '/path/to/previous_reconstruction.h5'

    - **Custom** (In-memory Numpy):

      .. code-block:: python
      
          # Python code block

          # Load the YAML params file as a dict
          params = load_params(params_path, validate=True)
          
          # Update corresponding fields
          params['init_params']['obj_source'] = 'custom'
          params['init_params']['obj_params'] = custom_object_4d # numpy.array

          # Pass params to PtyRADSolver
          ptycho_solver = PtyRADSolver(params, device=device, logger=logger)
          ptycho_solver.run()
    """

    tilt_source: Literal['simu', 'PtyRAD', 'file','custom'] = Field(default="simu", description="Data source for object tilts")
    """
    Data source for object tilts. 
    
    Available options: ``simu``, ``PtyRAD``, ``file``, ``custom``.
    """

    tilt_params: Union[TiltParams, FilePathWithKey, pathlib.Path, np.ndarray] = Field(default_factory=TiltParams, description="Parameters for object tilt loading/initialization")
    """
    Configuration for loading or initializing object tilts.

    **Examples by Source Type:**

    - **Simulation** (Default):

      .. code-block:: yaml

          'tilt_source': 'simu'
          'tilt_params': {'tilt_type': 'all', 'init_tilts': [[0.0, 0.0]]}
          
          # 'tilt_y' and 'tilt_x' are in mrad.
          # 'tilt_type': 'all' creates a global (1,2) tilt array.
          # 'tilt_type': 'each' creates a position-dependent (N_scans,2) array.

    - **Load from Reconstruction** (PtyRAD):

      .. code-block:: yaml

          'tilt_source': 'PtyRAD'
          'tilt_params': '/path/to/previous_reconstruction.h5'

    - **Load from File**:

      .. code-block:: yaml

          'tilt_source': 'file'
          'tilt_params': {'path': '/data/tilts.mat', 'key': 'tilt_angles'}
          
          # Supported formats: 'tif', 'mat', 'hdf5', 'npy'.
          # Must be 2D array with shape (1,2) or (N,2).

    - **Custom** (In-memory Numpy):

      .. code-block:: python
      
          # Python code block

          # Load the YAML params file as a dict
          params = load_params(params_path, validate=True)
          
          # Update corresponding fields
          params['init_params']['tilt_source'] = 'custom'
          params['init_params']['tilt_params'] = custom_tilts # numpy.array (N,2) or (1,2)

          # Pass params to PtyRADSolver
          ptycho_solver = PtyRADSolver(params, device=device, logger=logger)
          ptycho_solver.run()

    .. note::
        Object tilt is implemented via a tilted Fresnel propagator (accurate within ~ 1 deg, or 17 mrad).
        
        **Important:** Always provide an initial tilt guess (via hypertune or tutorial notebook: https://github.com/chiahao3/ptyrad/tree/main/tutorials/get_local_obj_tilts.ipynb). 
        Optimizing tilts from scratch with AD is slow and most likely arrive at barely corrected, slice-shifted object.
        
        The initialized object tilts can be either fixed, hypertuned (global only), or AD-optimized (either global or local) if the learning rate of ``obj_tilts`` != 0.
    """

################################################################################################################################
################################################################################################################################
################################################################################################################################

    @field_validator('probe_aberrations')
    @classmethod
    def validate_aberrations_logic(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """
        Delegates validation to the Aberrations class. 
        If the dict contains invalid keys or values, Aberrations will raise an error,
        which Pydantic will catch and display to the user.
        """
        try:
            # We don't keep the object, we just check if it CAN be created.
            Aberrations(v)
        except Exception as e:
            # Wrap the error so Pydantic knows why validation failed
            raise ValueError(f"Invalid aberration configuration: {str(e)}")
        return v


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
        if isinstance(v, np.ndarray):
            return v
        else:
            raise ValueError(f"{info.field_name} must be null, a valid file path string, or a NumPy array, got {type(v).__name__}")
        
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
            for field in ['probe_kv', 'probe_conv_angle', 'probe_aberrations']:
                setattr(self, field, None)

        return self
    
    @model_validator(mode="after")
    def validate_all_source_params(self):
        _validate_source_params_pair('meas', self.meas_source, self.meas_params)
        _validate_source_params_pair('obj', self.obj_source, self.obj_params)
        _validate_source_params_pair('probe', self.probe_source, self.probe_params)
        _validate_source_params_pair('pos', self.pos_source, self.pos_params)
        _validate_source_params_pair('tilt', self.tilt_source, self.tilt_params)
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

# Make explicit list so autodoc_pydantic can sort by this when go by `autodoc_pydantic_model_member_order = 'bysource'` in conf.py
__all__ = [
    "InitParams",
    "MeasCalibration",
    "ObjOmodeInitOccu",
    "MeasPad",
    "MeasResample",
    "MeasRemoveNegValues",
    "MeasNormalization",
    "MeasAddPoissonNoise",
    "MeasExport",
    "ProbeNormalization",
    "ObjZPad",
    "ObjZResample",
    "TiltParams",
    "FilePathWithKey"
]