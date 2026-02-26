"""
Pytorch models for ptychography geometry, object, probe, scan patterns, etc.

"""

from .ptycho import PtychoModel

# This list controls what shows up in the "Modules" table in API reference, but not the order. 
# Do NOT include the classes like PtychoModel in the __all__ list.
# Those classes are correctly imported at runtime by the above imports.
# Including classes in the __all__ list will cram the autosummary table and make Sphinx panicking.

__all__ = [
    "ptycho",
]