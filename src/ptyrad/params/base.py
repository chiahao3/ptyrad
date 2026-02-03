"""
Defines the integrated pydantic model ``PtyRADParams`` used for default filling and validation of PtyRAD params files
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .init_params import InitParams
from .hypertune_params import HypertuneParams
from .model_params import ModelParams
from .loss_params import LossParams
from .constraint_params import ConstraintParams
from .recon_params import ReconParams


class PtyRADParams(BaseModel):
    """
    The major params object for PtyRAD. 
    
    This object is primarily used to fill in defaults, or validate values from the input params file.
    
    However, this object can also be used as a skeleton to generate valid dict for ``PtyRADSolver``.
    
    Treat this ``PtyRADParams`` as the "schema" for the entire params file, or equivalently "the programmic definition of legal PtyRAD params".
    
    ``PtyRADParams`` contains 6 fields, which are:
    
    1. ``init_params``
    2. ``hypertune_params``
    3. ``model_params``
    4. ``loss_params``
    5. ``constraint_params``
    6. ``recon_params``
    
    These are the same with the nested dict defined in PtyRAD param files. 

    For detailed explanations of available options, check the API reference at:
    https://ptyrad.readthedocs.io/en/latest/_autosummary/ptyrad.params.html
    
    """
    
    model_config = {"extra": "forbid"}

    
    init_params: InitParams = Field(description="Initialization parameters")
    """
    Defines available options and validation rules for the ``init_params`` dictionary.
    
    Note that ``init_params`` contains **dataset-dependent required fields**.
    """
    
    hypertune_params: HypertuneParams = Field(default_factory=HypertuneParams, description="Hyperparameter tuning parameters")
    """
    Defines available options and validation rules for the ``hypertune_params`` dictionary.
    """
    
    model_params: ModelParams = Field(default_factory=ModelParams, description="Model parameters")
    """
    Defines available options and validation rules for the ``model_params`` dictionary.
    """

    loss_params: LossParams = Field(default_factory=LossParams, description="Loss parameters")
    """
    Defines available options and validation rules for the ``loss_params`` dictionary.
    """
    
    constraint_params: ConstraintParams = Field(default_factory=ConstraintParams, description="Constraint parameters")
    """
    Defines available options and validation rules for the ``constraint_params`` dictionary.
    """
    
    recon_params: ReconParams = Field(default_factory=ReconParams, description="Reconstruction parameters")
    """
    Defines available options and validation rules for the ``recon_params`` dictionary.
    """