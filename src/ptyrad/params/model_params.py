from __future__ import annotations

from typing import Any, Dict, Optional, Union

import torch.optim
from pydantic import BaseModel, Field, FilePath, field_validator, model_validator, model_serializer
import pathlib


class OptimizerParams(BaseModel):
    model_config = {"extra": "forbid"}
    
    name: str = Field(default="Adam", description="Optimizer name")
    configs: Dict[str, Any] = Field(default_factory=dict, description="Optimizer configurations")
    load_state: Optional[FilePath] = Field(
        default=None, description="Path str of a PtyRAD model file to load previous optimizer state"
    )

    @field_validator("name")
    @classmethod
    def validate_optimizer_name(cls, v: str) -> str:
        """Ensure optimizer name is a valid PyTorch optimizer."""
        if not hasattr(torch.optim, v) or not callable(getattr(torch.optim, v)):
            raise ValueError(f"Optimizer name '{v}' is not a valid PyTorch optimizer")
        return v
    
    @model_serializer
    def serialize_model(self):
        """Custom serializer to convert pathlib.Path back to str."""
        data = self.__dict__.copy()
        if data.get('load_state') is not None and isinstance(data['load_state'], pathlib.Path):
            data['load_state'] = str(data['load_state'])
        return data


class UpdateParams(BaseModel):
    model_config = {"extra": "forbid"}
    
    obja: Dict[str, Union[int, float, None]] = Field(
        default={"start_iter": 1, "lr": 5.0e-4}, description="Object amplitude update params"
    )
    objp: Dict[str, Union[int, float, None]] = Field(
        default={"start_iter": 1, "lr": 5.0e-4}, description="Object phase update params"
    )
    obj_tilts: Dict[str, Union[int, float, None]] = Field(
        default={"start_iter": None, "lr": 0.0}, description="Object tilts update params"
    )
    slice_thickness: Dict[str, Union[int, float, None]] = Field(
        default={"start_iter": None, "lr": 0.0}, description="Slice thickness update params"
    )
    probe: Dict[str, Union[int, float, None]] = Field(
        default={"start_iter": 1, "lr": 1.0e-4}, description="Probe update params"
    )
    probe_pos_shifts: Dict[str, Union[int, float, None]] = Field(
        default={"start_iter": 1, "lr": 5.0e-4},
        description="Sub-pixel probe position shifts update params",
    )

    @field_validator(
        "obja", "objp", "obj_tilts", "slice_thickness", "probe", "probe_pos_shifts", mode="after"
    )
    @classmethod
    def validate_update_params(cls, v: Dict[str, Any], field) -> Dict[str, Any]:
        """Validate start_iter and lr for update parameters."""
        start_iter = v.get("start_iter")
        lr = v.get("lr", 0.0)

        # start_iter must be None or >= 1
        if not (start_iter is None or (isinstance(start_iter, int) and start_iter >= 1)):
            raise ValueError(f"{field.field_name}.start_iter must be None or an integer >= 1")

        # If start_iter is not None, lr must be non-zero
        if start_iter is not None and lr == 0.0:
            raise ValueError(f"{field.field_name}.lr must be non-zero when start_iter is not None")

        # lr must be >= 0
        if not (isinstance(lr, (int, float)) and lr >= 0.0):
            raise ValueError(f"{field.field_name}.lr must be a non-negative number")

        return v

    @model_validator(mode="after")
    def validate_all_start_iter(self):
        """Ensure not all start_iter are None or all > 1."""
        fields = ["obja", "objp", "obj_tilts", "slice_thickness", "probe", "probe_pos_shifts"]
        start_iters = [self.__dict__[field].get("start_iter") for field in fields]

        # start_iter can not be all None or all > 1
        if all(si is None for si in start_iters):
            raise ValueError("start_iter values can not be all None")
        non_none_iters = [si for si in start_iters if si is not None]
        if non_none_iters and all(si > 1 for si in non_none_iters):
            raise ValueError(
                "Non-None start_iter values can not be all > 1"
            )  # Early iterations would have no gradients to work with

        return self


class ModelParams(BaseModel):
    """
    "model_params" determines the forward model behavior, the optimizer configuration, and the learning of the PyTorch model (PtychoAD)

    optimizer configurations are specified in 'optimizer_params', see https://pytorch.org/docs/stable/optim.html for detailed information of available optimizers and configs.
    update behaviors of optimizable variables (tensors) are specified in 'update_params'.
    'start_iter' specifies the iteration at which the variables (tensors) can start being updated by automatic differentiation (AD)
    'lr' specifies the learning rate for the variables (tensors)
    Usually slower learning rate leads to better convergence/results, but is also updating slower.
    The variable optimization has 2 steps, (1) calculate gradient and (2) apply update based on learning rate * gradient
    'start_iter: null' will disable grad calculation and would not update the variable regardless the learning rate through out the whole reconstruction
    'start_iter: N(int)' would only calculate the grad when iteration >= N, so no grad will be calculated when iteration < N
    Therefore, only the variable with non-zero learning rate would be optimized when iteration > start_iter.
    If you don't want/need to optimize certain parameters, set their start_iter to null AND learning rate to 0 for faster computation.
    Typical learning rate is 1e-3 to 1e-4.
    """

    model_config = {"extra": "forbid"}
    
    
    obj_preblur_std: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Gaussian blur std for object before forward pass. unit: px (real space)",
    )
    """
    This applies Gaussian blur to the object before simulating diffraction patterns. 
    Since the gradient would flow to the original "object" before blurring, it's essentially deconvolving the object with a Gaussian kernel of specified std. 
    This sort of deconvolution can generate sharp features, but the usage is not easily justifiable so treat it carefully as a visualization exploration
    """

    detector_blur_std: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Gaussian blur std for simulated diffraction patterns. unit: px (k-space)",
    )
    """
    This applies Gaussian blur to the forward model simulated diffraction patterns to emulate the PSF of high-energy electrons on detector for experimental data. 
    Typical value is 0-1 px (std) based on the acceleration voltage
    """

    preload_data: Optional[bool] = Field(
        default=True,
        description="Boolean flag for either to preload data into device memory or not",
    )
    """
    type: bool. 
    This flag determines how the measurements data is stored and transferred to device during reconstruciton. 
    If true, measurement data will be fully loaded into device memory during model initialization for best performance. 
    However, dataset larger than device memory (i.e., GPU VRAM) would throw Out-Of-Memory error. 
    If 'preload_data': false, measurement data is kept on host memory (CPU RAM) and only the mini-batch is transferred to device memory in a streaming way. 
    This would enable reconstruction of large dataset that doesn't fit into GPU VRAM with a little data transfer overhead. 
    The default is 'true' for performance although the difference is negligible on demo datasets.
    """

    optimizer_params: OptimizerParams = Field(
        default_factory=OptimizerParams, description="Optimizer configuration"
    )
    """
    Support all PyTorch optimizer. 
    The suggested optimizer is 'Adam' with default configs (null). 
    You can load the previous optimizer state by passing the path of `model.hdf5` to `load_state`, this way you can continue previous reconstruciton smoothly without abrupt gradients. 
    (Because lots of the optimizers are adaptive and have history-dependent learning rate manipulation, so loading the optimizer state is necessary if you want to continue the previous optimization trajectory). 
    However, the optimizer state must be coming from previous reconstructions with the same set of optimization variables with identical size of the dimensions otherwise it won't run.
    """

    update_params: UpdateParams = Field(
        default_factory=UpdateParams, description="Update parameters for optimizable tensors"
    )

__all__ = [
    name for name, obj in globals().items()
    if getattr(obj, "__module__", None) == __name__
    and hasattr(obj, "model_fields")  # pydantic
]