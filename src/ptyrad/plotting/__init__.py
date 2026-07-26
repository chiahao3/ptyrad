"""
Visualization functions for summary figures

"""

# Only promote .basic for easier plotting in CLI and notebooks without pulling torch
from .basic import (
    plot_affine_transformation,
    plot_convergence_dashboard,
    plot_loss_curves,
    plot_obj_tilts,
    plot_obj_tilts_avg,
    plot_pos_grouping,
    plot_probe_modes,
    plot_scan_positions,
    plot_sigmoid_mask,
    plot_slice_thickness,
    plot_learning_rates_schedule,
)

__all__ = [
    "plot_affine_transformation",
    "plot_convergence_dashboard",
    "plot_loss_curves",
    "plot_obj_tilts",
    "plot_obj_tilts_avg",
    "plot_pos_grouping",
    "plot_probe_modes",
    "plot_scan_positions",
    "plot_sigmoid_mask",
    "plot_slice_thickness",
    "plot_learning_rates_schedule",
]