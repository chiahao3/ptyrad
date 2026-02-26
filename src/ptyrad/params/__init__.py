"""
Pydantic schema models for filling defaults, and validating entries from PtyRAD params files

"""

# Import classes for users so they can do `from ptyrad.params import PtyRADParams, ConstraintParams...`
from .ptyrad_params import PtyRADParams # noqa: F401
from .constraint_params import ConstraintParams  # noqa: F401
from .hypertune_params import HypertuneParams  # noqa: F401
from .init_params import InitParams  # noqa: F401
from .loss_params import LossParams  # noqa: F401
from .model_params import ModelParams  # noqa: F401
from .recon_params import ReconParams  # noqa: F401
from .parser import load_params

# This list controls what shows up in the "Modules" table in API reference, but not the order. 
# Do NOT include the classes like PtyRADParams in the __all__ list.
# Those classes are correctly imported at runtime by the above imports.
# Including classes in the __all__ list will cram the autosummary table under ptyrad.params page.
__all__ = [
    "ptyrad_params",
    "constraint_params",
    "hypertune_params",
    "init_params",
    "loss_params",
    "model_params",
    "recon_params",
    "load_params"
]
