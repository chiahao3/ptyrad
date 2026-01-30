from __future__ import annotations

from pydantic import BaseModel, Field

from .constraint_params import ConstraintParams  # noqa: F401
from .hypertune_params import HypertuneParams  # noqa: F401
from .init_params import InitParams  # noqa: F401
from .loss_params import LossParams  # noqa: F401
from .model_params import ModelParams  # noqa: F401
from .recon_params import ReconParams  # noqa: F401


class PtyRADParams(BaseModel):
    """
    The params object for PtyRAD. This object is used to create params, fill defaults, or validate inputs.
    """
    
    model_config = {"extra": "forbid"}

    
    init_params: InitParams = Field(
        default_factory=InitParams, description="Initialization parameters"
    )
    hypertune_params: HypertuneParams = Field(
        default_factory=HypertuneParams, description="Hyperparameter tuning parameters"
    )
    model_params: ModelParams = Field(
        default_factory=ModelParams, description="Model parameters"
    )
    loss_params: LossParams = Field(
        default_factory=LossParams, description="Loss parameters"
    )
    constraint_params: ConstraintParams = Field(
        default_factory=ConstraintParams, description="Constraint parameters"
    )
    recon_params: ReconParams = Field(
        default_factory=ReconParams, description="Reconstruction parameters"
    )

__all__ = [
    name for name, obj in globals().items()
    if getattr(obj, "__module__", None) == __name__
    and hasattr(obj, "model_fields")  # pydantic
]