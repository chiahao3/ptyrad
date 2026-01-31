"""
Defines available options and validation rules for the "loss_params" dictionary.
"""

from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field, model_validator


class LossSingle(BaseModel):
    model_config = {"extra": "forbid"}

    state: bool = Field(default=True, description="Enable/disable loss_single term")
    weight: float = Field(default=1.0, ge=0.0, description="Weight of loss_single term")
    dp_pow: float = Field(default=0.5, description="Power to raise diffraction pattern before NRMSE calculation")


class LossPoissn(BaseModel):
    model_config = {"extra": "forbid"}

    state: bool = Field(default=False, description="Enable/disable loss_poissn term")
    weight: float = Field(default=1.0, ge=0.0, description="Weight of loss_poissn term")
    dp_pow: float = Field(default=1.0, description="Power to raise diffraction pattern before Poisson likelihood")
    eps: float = Field(default=1.0e-6, gt=0.0, description="Small constant to avoid log(0) in Poisson")


class LossPacbed(BaseModel):
    model_config = {"extra": "forbid"}

    state: bool = Field(default=False, description="Enable/disable loss_pacbed term")
    weight: float = Field(default=0.5, ge=0.0, description="Weight of loss_pacbed term")
    dp_pow: float = Field(default=0.2, description="Power to raise PACBED before NRMSE calculation")


class LossSparse(BaseModel):
    model_config = {"extra": "forbid"}

    state: bool = Field(default=False, description="Enable/disable loss_sparse term")
    weight: float = Field(default=0.1, ge=0.0, description="Weight of loss_sparse term")
    ln_order: int = Field(default=1, ge=1, description="L_n norm order for object phase (n >= 1)")


class LossSimlar(BaseModel):
    model_config = {"extra": "forbid"}

    state: bool = Field(default=False, description="Enable/disable loss_simlar term")
    weight: float = Field(default=0.1, ge=0.0, description="Weight of loss_simlar term")
    obj_type: Literal["amplitude", "phase", "both"] = Field(default="both", description="Object type for similarity")
    scale_factor: List[float] = Field(default=[1.0, 1.0, 1.0], min_items=3, max_items=3, description="Scale factors for z, y, x")
    blur_std: float = Field(default=1.0, ge=0.0, description="Standard deviation for Gaussian blur")


class LossParams(BaseModel):
    """
    Generally, the reconstruction loss is the CombinedLoss = weight1 * loss1 + weight2 * loss2 + weight3 * loss3 ...
    Use 'state' to switch on/off each loss term, and use 'weight' to determine their relative importance. 
    Each loss term would generate their corresponding gradient to the variable, 
    and the final update is determined by the weighted sum of all gradients coming from all participating loss terms. 
    Data-error related loss terms ('loss_single', 'loss_poissn', and 'loss_pacbed') compare simulated DP with experimental DP, 
    and their 'dp_pow' would raise the diffraction pattern to a power before the calculation
    For ptychography purpose, you MUST have at least 1 out of the 3 data-error loss terms. 
    Although you can set all of them to true, typical dataset works fine with 'loss_single' alone
    Soft constraint (regularization)-like loss terms ('loss_sparse', and 'loss_simlar') are optional addition to the required data-error loss terms
    Common regularization terms for image reconstruction tasks are total variation (TV) for smoothness and L_n norm (L1 and L2) for sparsity (i.e., promote near-zero or zero in the reconstructed tensor)

    """
    
    model_config = {"extra": "forbid"}
    
    
    loss_single: LossSingle = Field(default_factory=LossSingle, description="NRMSE loss for single diffraction pattern")
    """
    NRMSE error between single simulated and experimental diffraction pattern. 
    'dp_pow' is commonly set at 0.5 so NRMSE(DP_sim^0.5 - DP_exp^0.5) is equivalent to the Gaussian noise model for typical dataset (dose-sufficient) under the maximum-likelihood formalism
    """
    
    loss_poissn: LossPoissn = Field(default_factory=LossPoissn, description="Poisson likelihood loss")
    """
    negative log likelihood between simulated and experimental diffraction pattern. 
    'dp_pow' is commonly set at 1 so - (DP_sim * (DP_exp) - DP_exp) is equivalent to the Poisson noise model for low dose dataset under maximum likelihood formalism. 
    See Odstrˇcil2018 https://doi.org/10.1364/OE.26.003108 for more details
    """
    loss_pacbed: LossPacbed = Field(default_factory=LossPacbed, description="NRMSE loss for PACBED")
    """
    NRMSE error between simulated and experimental position-averaved CBED (PACBED). 
    Similar to 'loss_single', except that it's comparing PACBED with PACBED and mostly focusing on the diffuse background when 'dp_pow' is set at 0.2
    """
    
    loss_sparse: LossSparse = Field(default_factory=LossSparse, description="L_n norm regularization for object phase")
    """
    L_n norm regularization calculated for object phase. 
    'ln_order' means the L_n norm :math:`(|a_i^n|^{(1/n)})` used to regularize object phase ('objp'). 
    Usually 'ln_order' is set at 1 for L1 norm :math:`(|a|)`, this promotes 0 in the objp and enhance the sparsity (i.e. discrete atoms). 
    'ln_order' = 2 would be equivalent to L2 norm that promotes near-0 values
    """
    
    loss_simlar: LossSimlar = Field(default_factory=LossSimlar, description="Similarity regularization across object modes")
    """
    std across omode dimension for obj. 
    This promotes similarity between object modes. 
    'obj_type' can be either 'amplitude', 'phase', or 'both'. 
    'scale_factor' as (zoom_z, zoom_y, zoom_x) is used to scale the object before calculating the std. 
    Setting 'scale_factor' to [1,0.5,0.5]  is equivalent to downsampling the obj 2x along y and x directions before calculating the std, which should encourage the obj modes to keep lateral atom shifts. 
    Similarly, 'blur_std' applies a 2D (lateral) Gaussian blur kernel with specified std to blur the obj before calculating std along omode dimension
    """
    
    
    @model_validator(mode="after")
    def check_at_least_one_data_error(self):
        """Ensure at least one data-error loss term is enabled."""
        fields = ["loss_single", "loss_poissn", "loss_pacbed"]
        data_error_states = [self.__dict__[field].state for field in fields]

        if not any(data_error_states):
            raise ValueError("At least one of loss_single, loss_poissn, or loss_pacbed must have state: true")

        return self

# Make explicit list so autodoc_pydantic can sort by this when go by `autodoc_pydantic_model_member_order = 'bysource'` in conf.py
__all__ = [
    "LossParams",
    "LossSingle",
    "LossPoissn",
    "LossPacbed",
    "LossSparse",
    "LossSimlar"
]