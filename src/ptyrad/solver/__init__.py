"""
Public-facing PtyRADSolver and workflow-related functions

"""

from .ptyrad_solver import PtyRADSolver

# This list controls what shows up in the "Modules" table in API reference, but not the order. 
# Do NOT include the classes like PtychoModel in the __all__ list.
# Those classes are correctly imported at runtime by the above imports.
# Including classes in the __all__ list will cram the autosummary table and make Sphinx panicking.

__all__ = [
    "ptyrad_solver",
]