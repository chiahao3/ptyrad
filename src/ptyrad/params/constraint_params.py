from typing import List, Literal, Optional

from pydantic import BaseModel, Field

# 2025.08.10 CHL: 'freq' is deprecated since PtyRAD v0.1.0b11 and it might be removed at stable release of 0.1.0.

class OrthoPmode(BaseModel):
    model_config = {"extra": "forbid"}
    
    start_iter: Optional[int] = Field(default=1, ge=1, description="Start iteration of applying orthogonalization to mixed-state probe")
    step: Optional[int] = Field(default=1, ge=1, description="Interval of iterations of applying orthogonalization to mixed-state probe")
    end_iter: Optional[int] = Field(default=None, ge=1, description="End iteration of applying orthogonalization to mixed-state probe")

class ProbeMaskK(BaseModel):
    model_config = {"extra": "forbid"}
    
    start_iter: Optional[int] = Field(default=None, ge=1, description="Start iteration of applying k-space probe mask")
    step: Optional[int] = Field(default=1, ge=1, description="Interval of iterations of applying k-space probe mask")
    end_iter: Optional[int] = Field(default=None, ge=1, description="End iteration of applying k-space probe mask")
    radius: float = Field(default=0.22, ge=0.0, description="Radius of sigmoid mask in relative kMax units")
    width: float = Field(default=0.05, ge=0.0, description="Width of sigmoid mask transition")
    power_thresh: float = Field(default=0.95, ge=0.0, le=1.0, description="Power threshold for probe modes")


class FixProbeInt(BaseModel):
    model_config = {"extra": "forbid"}
    
    start_iter: Optional[int] = Field(default=1, ge=1, description="Start iteration of rescaling probe intensity")
    step: Optional[int] = Field(default=1, ge=1, description="Interval of iterations of rescaling probe intensity")
    end_iter: Optional[int] = Field(default=None, ge=1, description="End iteration of rescaling probe intensity")


class ObjRblur(BaseModel):
    model_config = {"extra": "forbid"}
    
    start_iter: Optional[int] = Field(default=1, ge=1, description="Start iteration of applying lateral Gaussian blur")
    step: Optional[int] = Field(default=1, ge=1, description="Interval of iterations of applying lateral Gaussian blur")
    end_iter: Optional[int] = Field(default=None, ge=1, description="End iteration of applying lateral Gaussian blur")
    obj_type: Literal["amplitude", "phase", "both"] = Field(default="both", description="Object type for blur")
    kernel_size: int = Field(default=5, ge=1, description="Kernel size for Gaussian blur (odd, >6*std+1)")
    std: float = Field(default=0.4, ge=0.0, description="Standard deviation for Gaussian blur")


class ObjZblur(BaseModel):
    model_config = {"extra": "forbid"}
    
    start_iter: Optional[int] = Field(default=1, ge=1, description="Start iteration of applying z-direction Gaussian blur")
    step: Optional[int] = Field(default=1, ge=1, description="Interval of iterations of applying z-direction Gaussian blur")
    end_iter: Optional[int] = Field(default=None, ge=1, description="End iteration of applying z-direction Gaussian blur")
    obj_type: Literal["amplitude", "phase", "both"] = Field(default="both", description="Object type for blur")
    kernel_size: int = Field(default=5, ge=1, description="Kernel size for Gaussian blur (odd, >6*std+1)")
    std: float = Field(default=1.0, ge=0.0, description="Standard deviation for Gaussian blur")


class KrFilter(BaseModel):
    model_config = {"extra": "forbid"}
    
    start_iter: Optional[int] = Field(default=None, ge=1, description="Start iteration of applying lateral Fourier filter")
    step: Optional[int] = Field(default=1, ge=1, description="Interval of iterations of applying lateral Fourier filter")
    end_iter: Optional[int] = Field(default=None, ge=1, description="End iteration of applying lateral Fourier filter")
    obj_type: Literal["amplitude", "phase", "both"] = Field(default="both", description="Object type for filter")
    radius: float = Field(default=0.15, ge=0.0, description="Radius of sigmoid filter in relative kMax units")
    width: float = Field(default=0.05, ge=0.0, description="Width of sigmoid filter transition")


class KzFilter(BaseModel):
    model_config = {"extra": "forbid"}
    
    start_iter: Optional[int] = Field(default=None, ge=1, description="Start iteration of applying arctan kz filter")
    step: Optional[int] = Field(default=1, ge=1, description="Interval of iterations of applying arctan kz filter")
    end_iter: Optional[int] = Field(default=None, ge=1, description="End iteration of applying arctan kz filter")
    obj_type: Literal["amplitude", "phase", "both"] = Field(default="both", description="Object type for filter")
    beta: float = Field(default=1.0, ge=0.0, description="Strength of arctan function")
    alpha: float = Field(default=1.0, ge=0.0, description="Lateral Fourier filtering constant")


class KrThresh(BaseModel):
    model_config = {"extra": "forbid"}
    
    start_iter: Optional[int] = Field(default=None, ge=1, description="Start iteration of applying freq threshold constraint")
    step: Optional[int] = Field(default=1, ge=1, description="Interval of iterations of applying freq threshold constraint")
    end_iter: Optional[int] = Field(default=None, ge=1, description="End iteration of applying freq threshold constraint")
    obj_type: Literal["amplitude", "phase", "both"] = Field(default="both", description="Object type for filter")
    thresh: float = Field(default=0.05, gt=0.0, le=1, description="Threshold ratio for spatial frequencies to keep")


class ComplexRatio(BaseModel):
    model_config = {"extra": "forbid"}
    
    start_iter: Optional[int] = Field(default=None, ge=1, description="Start iteration of applying complex constraint")
    step: Optional[int] = Field(default=1, ge=1, description="Interval of iterations of applying complex constraint")
    end_iter: Optional[int] = Field(default=None, ge=1, description="End iteration of applying complex constraint")
    obj_type: Literal["amplitude", "phase", "both"] = Field(default="both", description="Object type for constraint")
    alpha1: float = Field(default=1.0, description="Alpha1 parameter for complex constraint")
    alpha2: float = Field(default=0.0, description="Alpha2 parameter for complex constraint")


class MirroredAmp(BaseModel):
    model_config = {"extra": "forbid"}
    
    start_iter: Optional[int] = Field(default=1, ge=1, description="Start iteration of applying mirrored amplitude constraint")
    step: Optional[int] = Field(default=1, ge=1, description="Interval of iterations of applying mirrored amplitude constraint")
    end_iter: Optional[int] = Field(default=None, ge=1, description="End iteration of applying mirrored amplitude constraint")
    relax: float = Field(default=0.1, ge=0.0, le=1.0, description="Relaxation parameter for thresholding")
    scale: float = Field(default=0.03, ge=0.0, description="Scale parameter for amplitude constraint")
    power: float = Field(default=4.0, ge=0.0, description="Power parameter for amplitude constraint")

class ObjZRecenter(BaseModel):
    model_config = {"extra": "forbid"}
    
    start_iter: Optional[int] = Field(default=None, ge=1, description="Start iteration of applying obj Z recneter constraint")
    step: Optional[int] = Field(default=1, ge=1, description="Interval of iterations of applying obj Z recneter constraint")
    end_iter: Optional[int] = Field(default=None, ge=1, description="End iteration of applying obj Z recneter constraint")
    thresh: Optional[float] = Field(default=95, ge=0, le=100, description="Percentile for image thresholding")
    scale: Optional[float] = Field(default=1, ge=0.0, le=1.0, description="Scale parameter for measured shift")
    max_shift: Optional[float] = Field(default=10, ge=0.0, description="Maximum value for applied shift")

class ObjaThresh(BaseModel):
    model_config = {"extra": "forbid"}
    
    start_iter: Optional[int] = Field(default=1, ge=1, description="Start iteration of applying object amplitude thresholding")
    step: Optional[int] = Field(default=1, ge=1, description="Interval of iterations of applying object amplitude thresholding")
    end_iter: Optional[int] = Field(default=None, ge=1, description="End iteration of applying object amplitude thresholding")
    relax: float = Field(default=0.0, ge=0.0, le=1.0, description="Relaxation parameter for thresholding")
    thresh: List[float] = Field(
        default=[0.98, 1.02],
        min_items=2,
        max_items=2,
        description="Min and max thresholds for amplitude",)


class ObjpPostiv(BaseModel):
    model_config = {"extra": "forbid"}

    start_iter: Optional[int] = Field(default=1, ge=1, description="Start iteration of applying positivity constraint")
    step: Optional[int] = Field(default=1, ge=1, description="Interval of iterations of applying positivity constraint")
    end_iter: Optional[int] = Field(default=None, ge=1, description="End iteration of applying positivity constraint")
    relax: float = Field(default=0.0, ge=0.0, le=1.0, description="Relaxation parameter for positivity")

class PosRecenter(BaseModel):
    model_config = {"extra": "forbid"}

    start_iter: Optional[int] = Field(default=None, ge=1, description="Start iteration of applying position recentering constraint")
    step: Optional[int] = Field(default=1, ge=1, description="Interval of iterations of applying position recentering constraint")
    end_iter: Optional[int] = Field(default=None, ge=1, description="End iteration of applying position recentering constraint")
    relax: float = Field(default=0.0, ge=0.0, le=1.0, description="Relaxation parameter for position recentering")

class TiltSmooth(BaseModel):
    model_config = {"extra": "forbid"}

    start_iter: Optional[int] = Field(default=None, ge=1, description="Start iteration of applying tilt smoothing")
    step: Optional[int] = Field(default=1, ge=1, description="Interval of iterations of applying tilt smoothing")
    end_iter: Optional[int] = Field(default=None, ge=1, description="End iteration of applying tilt smoothing")
    std: float = Field(default=2.0, ge=0.0, description="Standard deviation for Gaussian blur of tilts")


class ConstraintParams(BaseModel):
    """
    "constraint_params" determines the individual iter-wise constraints for the CombinedConstraint used for PtyRAD reconstruction

    Generally, these constraint functions are applied after each (or a couple) iteration(s) to stabilize the optimization trajectories
    When applied, the target tensor is passed through the constraint function, and the tensor get directly modified by the constraint function
    Set 'start_iter' to a positive integer to specify when do we want to apply such constraint, and setting 'start_iter' to 'null' would disable that constraint.
    'step' can be set to specify the interval of applying those constraints when (niter - start_iter) % step == 0.
    'end_iter' can be used to specify when to stop applying such constraint. 
    Note that end_iter is exclusive, so if end_iter: 20, the constraint would not be applied at the end of Iter: 20.
    Most constraints are designed specifically for an optimizable tensor to make sure they're following some arbitrary preference
    In other words, we typically choose the constraints according to our optimizable tensors. 
    For example, we need 'ortho_pmode' if we're optimizing mixed-state probe. Similarly, we need either 'obj_zblur' or 'kz_filter' to stabilize the multislice ptychography reconstruciton along z-direction.
    A common combination of constrains for mixed-state probe multislice ptychography would be 'ortho_pmode', 'fix_probe_int', 'kz_filter', and 'objp_postiv'
    """
    model_config = {"extra": "forbid"}


    ortho_pmode: OrthoPmode = Field(
        default_factory=OrthoPmode, description="Orthogonalization of mixed-state probe"
    )
    """
    Apply a SVD decomposition and orthogonalization of the mixed-state probe similar to the PtychoShelves implementation (except it's vectorized in PtyRAD). 
    Turn this on when optimizing with mixed-state probe
    """
    
    probe_mask_k: ProbeMaskK = Field(default_factory=ProbeMaskK, description="K-space probe mask")
    """
    Apply a k-space sigmoid (similar to a top-hat) probe mask that cut off spatial frequencies beyond the probe-forming aperture. 
    This prevents the probe from absorbing the object structure in k-space. 
    It might cause high-frequency oscillations in the real-space object,
    if you have strong diffuse background in the diffraction pattern and did not provide mixed-object to properly reproduce it. 
    Recommend setting it to 'null' unless you're pursuing mixed-object with more physical probes. 
    k-radius should be larger than 2*rbf/Npix to avoid cutting out the BF disk. 
    'radius' and 'width' are used to define the sigmoid funciton in relative unit with kMax. 
    See 'utils/make_sigmoid_mask' for more details. 
    'power_thresh' is used to specify how far into the pmode should be masked. 
    If 'power_thresh': 0.95, the k-space mask would be applied from strongest probe modes to the one that adds uo to 95% total intensity. 
    This promotes a more physical mixed-probe while keeping a small fraction of probe modes to absorb unexpected errors.
    """
    
    fix_probe_int: FixProbeInt = Field(
        default_factory=FixProbeInt, description="Rescale probe intensity"
    )
    """
    Rescale the probe intensity to make it consistent with the total diffraction pattern intensity (so every probe electron hits on the detector). 
    This is needed to stabilize the object amplitude update because the probe update could potentially change the total intensity. 
    This removes the scaling constant ambiguity between probe and object, and should be applied if you're simultaneously optimizing the probe and object amplitude
    """
    
    obj_rblur: ObjRblur = Field(
        default_factory=ObjRblur, description="Lateral Gaussian blur for object"
    )
    """
    Apply a "lateral" 2D Gaussian blur to the object. 
    This removes some high frequency noise in the reconstructed object and make the apperance smoother. 
    'obj_type' can be either 'amplitude', 'phase', or 'both' with a specified 'std' and 'kernel_size' in unit of real-space px. 
    Ideally kernel size is odd (like 5) and larger than 6std+1 so it decays to 0. 
    This is usually not needed if your dataset contains sufficient dose and the kMax is not insanely high 
    (extremely high kMax would gives very fine dx which makes feature appear sharper and probably more seemingly noisy)
    """
    
    obj_zblur: ObjZblur = Field(
        default_factory=ObjZblur, description="Z-direction Gaussian blur for object"
    )
    """
    Apply a "z-direction" 1D Gaussian blur to the object. 
    This is a real-space alternative to the typical kz_filter 
    (or so called missing-wedge regularization that applies Fourier filtering to the object) designed for multislice ptychography. 
    Similar to 'obj_rblur', 'obj_type' can be either 'amplitude', 'phase', or 'both' with a specified 'std' and 'kernel_size' in unit of real-space px. 
    Note that the 'ptycho/engines/GPU_MS/private/regulation_multilayers.m' from PtychoShelves (fold_slice) 
    is a combination of 'obja_thresh', 'kr_filter', and 'kz_filter', 
    so you may want to activate all these constraints altogether in PtyRAD to get the most similar effect 
    """
    
    kr_filter: KrFilter = Field(
        default_factory=KrFilter, description="Lateral Fourier filter for object"
    )
    """
    Apply a "lateral" Fourier low-pass filtering to the object. 
    This is similar to the band-pass filter in Digital Micrograph that the k-space filter has a sigmoid-like profile, essentially a cut-off spatial frequency. 
    Typically we're reconstucting object all the way to kMax (Nyquist frequency),
    so there's not much room for us to filter out hence it's recommended to keep this off unless you want to exclude certain spatial frequencies. 
    'radius' and 'width' are used to define the sigmoid funciton in relative unit with kMax. 
    See 'utils/make_sigmoid_mask' for more details
    """
    
    kz_filter: KzFilter = Field(default_factory=KzFilter, description="Arctan kz filter for object")
    """
    Apply the arctan kz filter just like the 'regulation_multilayers.m' in PtychoShelves. 
    'beta' controls the strength of the arctan function and is the same as 'eng. regularize_layers' in PtychoShelves. 
    Typical value of 'beta' ranges from 0 to 1. 
    'alpha' is the implicit constant controls the lateral Fourier filtering that has similar effect as 'kr_filter', 
    usually set as 1 to be consistent with PtychoShelves. 
    Note that 'kz_filter' is designed to be a PtyRAD equivalence of the 'regulation_multilayers.m' so it also includes the arctan Fourier filer, 
    lateral Fourier filter, and the soft object amplitude thresholding if you set 'obj_type' to 'both'. 
    See 'optimization/kz_filter' for more details. 
    While this 'kz_filter' works very well for most multislice reconstructions of crystals, 
    you might prefer 'obj_zblur' if you have an object that has distinct top and bottom layer like twisted bilayer or tilted systems,
    because 'kz_filter' would introduce intermixing between the top and bottom layer due to the periodic boundary condition of Fourier transform. 
    Another solution to the intermixing is to pad vacuum layers to your object and remove them later, 
    although padding extensive vacuum layers tend to make object phase bleed into the vacuum layers and it's very hard to set the interface
    """
    
    kr_thresh: KrThresh = Field(default_factory=KrThresh, description="kr threshold for object")
    """
    Apply a thresholding to supress weak spatial frequency coefficients of the object along "lateral" (i.e., x, y) directions. 
    'thresh' is the ratio of spatial frequencies to keep and thresh is within (0,1]. 
    Typical 'thresh' value is 0.1, meaning keeping 10% of the strongest spatial frequencies and set the others to 0. 
    Smaller 'thresh' value will promote sparsity in frequency domain (implemented with DCT) and produce a smoother object with reduced noise. 
    Note that this constraint would also reduce / remove the disorder in periodic objects like strain and atomic defects as it only keep the strong spatial frequencies,
    so it's advised to only use it for purely amorphous specimen, low dose reconstructions, or as an initial stabilizing constraint. 
    The DCT is applied to the last 2 dimensions for object array shaped (...,H,W), hence the regularization effect is only within 2D and is sequentially repeated for each depth slice.
    See https://doi.org/10.1364/OE.551986 for example on low dose biological samples.    
    """
    
    complex_ratio: ComplexRatio = Field(
        default_factory=ComplexRatio, description="Complex constraint for object"
    )
    """
    Apply a complex constraint between object amplitude and phase based on https://doi.org/10.1016/j.ultramic.2024.114068 and https://doi.org/10.1364/OE.18.001981. 
    This will constrain the amplitude to as A' = exp(-C*phase), C is an estimated (adaptive) positive constant, 
    so the amplitude would look similar to phase and strongest phase shift would correspond to less than 1 amplitude. 
    Default value of phase object is alpha1 = 1, alpha2 = 0. 
    Note that the implementation considers the negative correlation between amplitude and phase for electctron ptychography, 
    hence it's not the exact same formula as in the papers.
    """
    
    mirrored_amp: MirroredAmp = Field(
        default_factory=MirroredAmp, description="Mirrored amplitude constraint"
    )
    """
    Apply a more flexible, ad hoc constraint for constraining amplitude using 1-scale*phase**power, 
    which provide more arbitrary parameters to tune the constrained amplitude based on the phase.
    """
    
    obj_z_recenter: ObjZRecenter = Field(
        default_factory=ObjZRecenter, description="Obj Z recenter constraint"
    )
    """
    Apply the z-recentering of object based on CoM to keep the object centered within the depth dimension. 
    The probe defocus is adjusted accordingly. 
    thresh is the threshold value in percentile to exclude weaker signals while calculating center-of-mass along depth. 
    The value of thresh should be around 90-95 to target values that are 2-3 sigma away from mean. 
    scale is a multplication scaling factor applied on the measured shift, keep it <= 1 to slowly approach the ideal center. 
    max_shift sets a hard limit on the applied shift, which should be less than half the number of z-slices. 
    The shift is in unit of slices, so shift=1 would re-center the object by 1 slice.
    """
    
    obja_thresh: ObjaThresh = Field(
        default_factory=ObjaThresh, description="Object amplitude thresholding"
    )
    """
    Apply a thresholding of object amplitude around 1. 
    The threshold is defined by 'thresh' and the value is determined by the min and max. 
    Note that wave amplitude is multiplicative when propagating through multislice object, 
    so even an objective amplitude of 0.95 can become 0.6 after 10 layers. 
    The thresholding can be relaxed by the `relax` param that is a weighted sum betwen the pre-threshold and post-threshold values.
    """
    
    objp_postiv: ObjpPostiv = Field(
        default_factory=ObjpPostiv, description="Positivity constraint for object phase"
    )
    """
    Apply a positivity constraint of the object phase, make it non-negative. 
    This clips the negative values (anything below 0) so the object is visually darker but with stronger constrast, 
    it's suggested to keep it on so that you can interpret, compare, and process your object phase with a simple baseline. 
    Besides, the positivity constraint makes it easier to compare with atomic potential ground truth after correct scaling for the scattering factor. 
    The clipping can be relaxed by the `relax` param that is a weighted sum betwen the pre-threshold and post-threshold values.
    """
    
    pos_recenter: PosRecenter = Field(
        default_factory=PosRecenter, description="Recentering the probe position shifts"
    )
    """
    Recenter the probe position shifts by making mean(probe_pos_shifts) = 0 so there's no global offset from the cropping position. 
    This would keep the probe, probe position, and object relatively fixed in place even with large position learning rates.
    """
    
    tilt_smooth: TiltSmooth = Field(
        default_factory=TiltSmooth, description="Smoothing of local object tilts"
    )
    """
    Apply a lateral Gaussian blur of the local object tilts in unit of "scan positions". 
    This smoothens the local tilts so that you don't have drastic changes of object tilts between scan positions.
    """
    
