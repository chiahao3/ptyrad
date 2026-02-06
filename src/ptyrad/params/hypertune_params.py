"""
Defines available options and validation rules for the ``hypertune_params`` dictionary.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_serializer, model_validator


class SamplerParams(BaseModel):
    model_config = {"extra": "forbid"}

    name: Literal["RandomSampler", "TPESampler", "GPSampler",
                  "CmaEsSampler", "NSGAIISampler", "NSGAIIISampler",
                  "GridSampler", "QMCSampler", "BruteForceSampler"] = Field(default="TPESampler", description="Sampler algorithm for hyperparameter tuning")
    configs: Dict[str, Any] = Field(default={}, description="Sampler configurations")

    @model_validator(mode='before')
    def set_default_sampler_config(cls, values: dict) -> dict:
        name = values.get('name', 'TPESampler')
        if name == 'TPESampler' and 'configs' not in values:
            values['configs'] = {
                'multivariate': True, 
                'group': True, 
                'constant_liar': True
            }
        return values


class PrunerParams(BaseModel):
    model_config = {"extra": "forbid"}

    name: Literal["MedianPruner", "PatientPruner", "PercentilePruner",
                  "SuccessiveHalvingPruner", "HyperbandPruner", "ThresholdPruner"] = Field(default="HyperbandPruner", description="Pruner algorithm for early termination")
    configs: Dict[str, Any] = Field(default={}, description="Pruner configurations")

    @model_validator(mode='before')
    def set_default_pruner_config(cls, values: dict) -> dict:
        name = values.get('name', 'HyperbandPruner')
        if name == 'HyperbandPruner' and 'configs' not in values:
            values['configs'] = {
                'min_resource': 5, 
                'reduction_factor': 2
            }
        return values


class TuneParam(BaseModel):
    model_config = {"extra": "forbid"}

    state: bool = Field(description="Enable/disable tuning of this parameter")
    suggest: Literal['int', 'float', 'cat'] = Field(description="Suggestion method (e.g., 'int', 'float', 'cat')")
    kwargs: Dict[str, Any] = Field(description="Parameters for suggestion method")

    @field_validator("kwargs")
    @classmethod
    def validate_kwargs(cls, v: Dict[str, Any], info) -> Dict[str, Any]:
        """Validate kwargs based on suggest type."""
        
        # Type checks for common keys
        if "low" in v and not isinstance(v["low"], (int, float)):
            raise ValueError(f"kwargs.low must be an int or float, got {type(v['low']).__name__}")
        if "high" in v and not isinstance(v["high"], (int, float)):
            raise ValueError(f"kwargs.high must be an int or float, got {type(v['high']).__name__}")
        if "log" in v and not isinstance(v["log"], bool):
            raise ValueError(f"kwargs.log must be a bool, got {type(v['log']).__name__}")
        
        suggest = info.data.get("suggest")
        if suggest == "int":
            if not all(k in v for k in ["low", "high"]):
                raise ValueError("For suggest='int', kwargs must include 'low', 'high', and either 'step' or 'log=True'")
            
            step = v.get("step")
            log = v.get("log", False)
            if (step is not None and step != 1 and log is True):
                raise ValueError("For suggest='int', kwargs either have (1) step=1, log=True or (2) step=<VALUE>, log=False")

        elif suggest == "float":
            if not all(k in v for k in ["low", "high"]):
                raise ValueError("For suggest='float', kwargs must include 'low', 'high'")
            
            step = v.get("step")
            log = v.get("log", False)
            if "step" in v and v["step"] is not None and not isinstance(v["step"], (int, float)):
                raise ValueError(f"kwargs.step must be an int or float or None, got {type(v['step']).__name__}")
            if (step is not None and log is True):
                raise ValueError("For suggest='float', kwargs either have (1) step=None, log=True or (2) step=<VALUE>, log=False")

        elif suggest == "cat":
            if "choices" not in v:
                raise ValueError("For suggest='cat', kwargs must include 'choices'")
        return v


class TuneParams(BaseModel):
    model_config = {"extra": "forbid"}
    
    # Optimizer and batch size
    optimizer:  TuneParam = Field(default_factory=lambda: TuneParam(state=False, suggest="cat",   kwargs={"choices": ["Adam", "AdamW", "RMSprop", "SGD"], "optim_configs": {}}), description="AD Optimizer")
    batch_size: TuneParam = Field(default_factory=lambda: TuneParam(state=False, suggest="int",   kwargs={"low": 16, "high": 512, "log": True}), description="Batch size")
    # Learning rates
    plr:        TuneParam = Field(default_factory=lambda: TuneParam(state=False, suggest="cat",   kwargs={"choices": [1.0e-2, 1.0e-4, 1.0e-4]}), description="Probe learning rate")
    oalr:       TuneParam = Field(default_factory=lambda: TuneParam(state=False, suggest="float", kwargs={"low": 1.0e-4, "high": 1.0e-2, "log": True}), description="Object amplitude learning rate")
    oplr:       TuneParam = Field(default_factory=lambda: TuneParam(state=False, suggest="float", kwargs={"low": 1.0e-4, "high": 1.0e-2, "log": True}), description="Object phase learning rate")
    slr:        TuneParam = Field(default_factory=lambda: TuneParam(state=False, suggest="float", kwargs={"low": 1.0e-4, "high": 1.0e-2, "log": True}), description="Slice thickness learning rate")
    tlr:        TuneParam = Field(default_factory=lambda: TuneParam(state=False, suggest="float", kwargs={"low": 1.0e-4, "high": 1.0e-2, "log": True}), description="Object Tilt learning rate")
    dzlr:       TuneParam = Field(default_factory=lambda: TuneParam(state=False, suggest="float", kwargs={"low": 1.0e-4, "high": 1.0e-2, "log": True}), description="Slice thickness learning rate")
    # Real space calibration
    dx:         TuneParam = Field(default_factory=lambda: TuneParam(state=False, suggest="float", kwargs={"low": 0.1400, "high": 0.1600, "step": 0.001}), description="Real space pixel size (Ang)")
    # Probe mode and aberration
    pmode_max:  TuneParam = Field(default_factory=lambda: TuneParam(state=False, suggest="int",   kwargs={"low": 1, "high": 8, "step": 1}), description="Probe modes")
    conv_angle: TuneParam = Field(default_factory=lambda: TuneParam(state=False, suggest="float", kwargs={"low": 24, "high": 26, "step": 1}), description="Convergence angle (mrad)")
    C10:        TuneParam = Field(default_factory=lambda: TuneParam(state=False, suggest="float", kwargs={"low": -100, "high": 100, "step": 1}), description="Defocus (C10 = -df)")
    C12:        TuneParam = Field(default_factory=lambda: TuneParam(state=False, suggest="float", kwargs={"low": -100, "high": 100, "step": 1}), description="2-fold astigmatism")
    C21:        TuneParam = Field(default_factory=lambda: TuneParam(state=False, suggest="float", kwargs={"low": -1000, "high": 1000, "step": 1}), description="Axial coma")
    C23:        TuneParam = Field(default_factory=lambda: TuneParam(state=False, suggest="float", kwargs={"low": -1000, "high": 1000, "step": 1}), description="3-fold astigmatism")
    C30:        TuneParam = Field(default_factory=lambda: TuneParam(state=False, suggest="float", kwargs={"low": -5000, "high": 5000, "step": 1}), description="Spherical aberration")
    C32:        TuneParam = Field(default_factory=lambda: TuneParam(state=False, suggest="float", kwargs={"low": -5000, "high": 5000, "step": 1}), description="Axial star aberration")
    C34:        TuneParam = Field(default_factory=lambda: TuneParam(state=False, suggest="float", kwargs={"low": -5000, "high": 5000, "step": 1}), description="4-fold astigmatism")
    C41:        TuneParam = Field(default_factory=lambda: TuneParam(state=False, suggest="float", kwargs={"low": -5.0e4, "high": 5.0e4, "step": 1}), description="Axial coma(4th)")
    C43:        TuneParam = Field(default_factory=lambda: TuneParam(state=False, suggest="float", kwargs={"low": -5.0e4, "high": 5.0e4, "step": 1}), description="3-lobe aberration")
    C45:        TuneParam = Field(default_factory=lambda: TuneParam(state=False, suggest="float", kwargs={"low": -5.0e4, "high": 5.0e4, "step": 1}), description="5-fold astigmatism")
    C50:        TuneParam = Field(default_factory=lambda: TuneParam(state=False, suggest="float", kwargs={"low": -5.0e5, "high": 5.0e5, "step": 1}), description="Spherical aberration (5th)")
    C52:        TuneParam = Field(default_factory=lambda: TuneParam(state=False, suggest="float", kwargs={"low": -5.0e5, "high": 5.0e5, "step": 1}), description="Axial star aberration(5th)")
    C54:        TuneParam = Field(default_factory=lambda: TuneParam(state=False, suggest="float", kwargs={"low": -5.0e5, "high": 5.0e5, "step": 1}), description="4-lobe aberration")
    C56:        TuneParam = Field(default_factory=lambda: TuneParam(state=False, suggest="float", kwargs={"low": -5.0e5, "high": 5.0e5, "step": 1}), description="6-fold astigmatism")
    z_shift:    TuneParam = Field(default_factory=lambda: TuneParam(state=False, suggest="float", kwargs={"low": -50, "high": 50, "step": 10}), description="Axially (z) shift the initialized probe (Ang)")
    # Multislice object
    Nlayer:     TuneParam = Field(default_factory=lambda: TuneParam(state=False, suggest="int",   kwargs={"low": 1, "high": 8, "step": 1}), description="Number of object layers")
    dz:         TuneParam = Field(default_factory=lambda: TuneParam(state=False, suggest="float", kwargs={"low": 4, "high": 8, "step": 0.5}), description="Slice thickness (Ang)")
    # Scan affine
    scale:      TuneParam = Field(default_factory=lambda: TuneParam(state=False, suggest="float", kwargs={"low": 0.8, "high": 1.2, "step": 0.02}), description="Scan affine step size scale")
    asymmetry:  TuneParam = Field(default_factory=lambda: TuneParam(state=False, suggest="float", kwargs={"low": -0.2, "high": 0.2, "step": 0.05}), description="Scan affine asymmetry")
    rotation:   TuneParam = Field(default_factory=lambda: TuneParam(state=False, suggest="float", kwargs={"low": -4, "high": 4, "step": 0.5}), description="Scan affine rotation (degree)")
    shear:      TuneParam = Field(default_factory=lambda: TuneParam(state=False, suggest="float", kwargs={"low": -4, "high": 4, "step": 0.5}), description="Scan affine shear (degree)")
    # Object tilts
    tilt_y:     TuneParam = Field(default_factory=lambda: TuneParam(state=False, suggest="float", kwargs={"low": -5, "high": 5, "step": 0.5}), description="Object tilt y (mrad)")
    tilt_x:     TuneParam = Field(default_factory=lambda: TuneParam(state=False, suggest="float", kwargs={"low": -5, "high": 5, "step": 0.5}), description="Object tilt x (mrad)")

    # --- DORMANT LEGACY FIELDS ---
    # Keep these so the old params files (prior v0.1.0b13) parses successfully. 
    # default=None allows newer clean files to omit them entirely.
    defocus: TuneParam | None = Field(default=None, description="DEPRECATED: Use C10")
    c3:      TuneParam | None = Field(default=None, description="DEPRECATED: Use C30")
    c5:      TuneParam | None = Field(default=None, description="DEPRECATED: Use C50")
    
    @model_validator(mode='after')
    def check_legacy_intent(self):
        legacy_map = {
            'defocus': 'C10', 
            'c3': 'C30', 
            'c5': 'C50'
        }
        
        for old_key, new_key in legacy_map.items():
            # Access the field (e.g., self.defocus)
            param = getattr(self, old_key)
            
            if param is not None:
                # CHECK 1: Is the user trying to USE it?
                if param.state is True:
                    raise ValueError(
                        f"Legacy tune_params '{old_key}' is set to active (state=True). "
                        f"This parameter has been migrated to the Krivanek notation. "
                        f"Please rename '{old_key}' to '{new_key}' in your params file."
                    )
                
                # CHECK 2: It exists but is inactive (state=False)
                # We allow it (so the file loads), but we warn the user to clean it up.
                warnings.warn(
                    f"hypertune_params.tune_params contains deprecated field '{old_key}'. "
                    f"It is currently inactive and ignored, but will be removed in future versions. "
                    f"Please update your params file to remove it.",
                    UserWarning
                )
        
        return self
    
    @model_serializer(mode='wrap')
    def scrub_legacy_keys(self, handler) -> Dict[str, Any]:
        dumped_data = handler(self)
        # Remove these keys if they are None (inactive)
        legacy_keys = ['defocus', 'c3', 'c5']
        for k in legacy_keys:
            if dumped_data.get(k) is None:
                dumped_data.pop(k, None)
        return dumped_data


class HypertuneParams(BaseModel):
    """
    Hypertune optimizable parameters are specified in 'tune_params', 
    these will override the corresponding values in 'exp_params' but follows the exact same definition and unit. 
    Set 'state' to true to enable hypertuning that parameter. 
    'min', 'max', and 'step' defines the search space and sampling rate. 
    For example, conv_angle with min:24, max:26, step:0.1 would have a search range between 24 and 26 with 0.1 minimal spacing
    It's a better practice to limit your hypertune optimization with no more than 4 parameters simultaneously,
    and some hierachical strategy (i.e., optimizing dz first, then scale with rotation, then all 4 scan_affine, lastly object tilts) could be much faster / stable
    For typical dataset with sufficient dose, both conv_angle and defocus can be automatically handled by probe optimization. 
    However for low dose dataset (like 1e2 e-/DP), some hypertune of probe parameters could be necessary
    """
    model_config = {"extra": "forbid"}


    if_hypertune: bool = Field(default=False, description="Enable/disable hypertune mode")
    """
    Default is false. 
    Set to true to run PtyRAD in hypertune (hyperparameter optimization) mode. 
    This is the main switch for hypertune mode so none of the settings in 'hypertune_params' would take effect if 'if_hypertune' is false.
    """
    
    collate_results: bool = Field(default=True, description="Collect results from hypertune trials")
    """
    Default is true. 
    Set to true to collect results/figs specified under 'recon_params' from each hypertune trial to 'output_dir/<HYPERTUNE_FOLDER>',
    and name them by error metrics followed by trial parameters. 
    This provides a convenient way to quickly examine the hypertune results inside the hypertune folder under the main output directory 'output_dir'. 
    This is an independent control than 'SAVE_ITERS' and will save by the end of the 'NITER' or when the trial is pruned.
    """
    
    append_params: bool = Field(default=True, description="Append hyperparameter names/values to output files")
    """
    Default is true. 
    Set to true to append the hyperparameter name/values to the file names of collated result. 
    If set to false, only the error, trial ID, and iter would be preserved. 
    Set this to false to reduce the length of the output file name. 
    The exact parameter values are stored in the .sqlite3 database file.
    """
    
    n_trials: int = Field(default=5, ge=1, description="Number of hypertune trials")
    """
    Number of hypertune trials. 
    Each trial is a separate PtyRAD reconstruction with a set of optimizable parameter values (i.e., a configuration). 
    Note that when the hypertune mode is run in parallel (i.e., multiple jobs on multiple GPU nodes accesing the same database/study), 
    each job will run for 'n_trials' times. 
    So submiting 5 jobs with 'n_trials': 200 will get you a total of 1000 trials in the database/study
    """
    
    timeout: Optional[float] = Field(default=None, ge=0.0, description="Timeout for hypertune study in seconds")
    """
    Stop "study" after the given number of second(s). 
    Null represents no limit in terms of elapsed time. 
    The study continues to create trials until the number of trials reaches n_trials, 
    timeout period elapses, stop() is called or, a termination signal such as SIGTERM or Ctrl+C is received.
    """
    
    sampler_params: SamplerParams = Field(default_factory=SamplerParams, description="Sampler configuration for hypertuning")
    """
    Sampler is the optimization algorithm used for hyperparameter tuning. 
    See https://optuna.readthedocs.io/en/stable/reference/samplers/index.html for more details.
    """
    
    pruner_params: Optional[PrunerParams] = Field(default_factory=PrunerParams, description="Pruner configuration for early termination")
    """
    Pruning is early termination of unpromising trials to save computation budget. 
    Set to `'pruner_params': null` to disable pruning (i.e., no early termination).
    The recommended prunner is HyperbandPruner, see Optuna document for more details
    """
    
    storage_path: str = Field(default="sqlite:///hypertune.sqlite3", description="Path to SQLite database for hypertune")
    """
    Path to the SQLite database file (i.e., sotrage) as 'sqlite:///<DATABASE_FILENAME>.sqlite3'. 
    This database file keeps the record of hypertune runs and will be automatically created with new hypertube run. 
    If you're doing distributed (e.g. multiple GPU nodes) hypertune by submitting this params file multiple times, 
    then all the GPU worker would be accessing this database to be aware of each other's progress
    """
    
    study_name: str = Field(default="study", description="Name of the hypertune study")
    """
    Name of the hypertune record (i.e., study). 
    Hypertune of different dataset or different search space (i.e., different optimizable parameters) are encouraged to use different study name or even separate database file
    """
    
    error_metric: Literal['loss', 'contrast'] = Field(default="loss", description="Optimization metric for hypertune")
    """
    Either use 'loss' or 'contrast'. 
    The current suggested approach is to use 'loss' for rough optimization, 
    while switch to 'contrast' with loaded reconstructed object/probe/pos to refine remaining hyperparameters. 
    'contrast' is simply calculated by std(img)/mean(std) to reflect reconstruction quality of the object because 'loss' doesn't correlate that well. 
    Note that `contrast` doesn't necessarily change monotonically with iterations, especially at early iterations so you may want to disable pruning and set NITER carefullly. 
    """
    
    tune_params: TuneParams = Field(default_factory=TuneParams, description="Hypertunable parameters")
    """
    See https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial 
    for the syntax to configure the search space ranges and types
    """
    
# Make explicit list so autodoc_pydantic can sort by this when go by `autodoc_pydantic_model_member_order = 'bysource'` in conf.py
__all__ = [
    "HypertuneParams",
    "SamplerParams",
    "PrunerParams",
    "TuneParams",
    "TuneParam"
]