"""A screened-scattering intensity bound outside the measured detector crop."""

import logging

import numpy as np
import torch
from scipy.optimize import least_squares


logger = logging.getLogger(__name__)


def screened_intensity(radius, background, amplitude, screening):
    """Radial intensity at q/q_edge, with the elastic term normalized at q_edge."""
    return background + amplitude * ((1 + screening**2) / (radius**2 + screening**2))**2


class OutsideScatteringLoss(torch.nn.Module):
    """Penalize excess shell-averaged intensity in the unmeasured simulation area."""

    def __init__(self, measurements, simulation_size, transition_px, margin, device):
        super().__init__()
        mean_dp = np.mean(measurements, axis=0, dtype=np.float64)
        size = mean_dp.shape[-1]
        scale = mean_dp.mean()
        if mean_dp.shape != (size, size) or size < 32 or simulation_size <= size:
            raise ValueError("loss_outside requires a square detector of at least 32 px and a larger simulation grid")
        if (simulation_size - size) % 2:
            raise ValueError("loss_outside requires a centered crop with matching detector and simulation grid parity")
        if not np.isfinite(mean_dp).all() or mean_dp.min() < 0 or scale <= 0:
            raise ValueError("loss_outside requires finite, nonnegative measurements with positive mean intensity")

        # Use the same diagonal sectors at every radius. Beyond the inscribed circle
        # the corners still supply real measurements and can check the extrapolation.
        edge = size / 2
        coords = np.arange(size) - size // 2
        y, x = np.meshgrid(coords, coords, indexing="ij")
        radius = np.hypot(y, x) / edge
        diagonal = np.minimum(abs(y), abs(x)) >= 0.65 * np.maximum(abs(y), abs(x))
        fit_band = diagonal & (radius >= 0.78) & (radius < 1.0)
        check_band = diagonal & (radius >= 1.0) & (radius < 1.40)

        def annular_means(mask):
            rings = np.floor(radius[mask] * 32).astype(int)
            count = np.bincount(rings)
            present = count > 0
            q = np.bincount(rings, weights=radius[mask])[present] / count[present]
            intensity = np.bincount(rings, weights=mean_dp[mask])[present] / count[present]
            return q, intensity

        q_fit, fit = annular_means(fit_band)
        q_check, check = annular_means(check_band)
        fit_scale = fit.mean() if fit.size else 0.0
        if fit.size < 4 or check.size < 3 or fit_scale <= 0 or check.max() <= 0:
            raise ValueError("loss_outside cannot calibrate a bound from the measured diffraction corners")
        result = least_squares(
            lambda p: screened_intensity(q_fit, *p) - fit / fit_scale,
            x0=(0.1, 0.8, 0.4), bounds=((0, 0, 0), (np.inf, np.inf, 2)),
            loss="soft_l1", f_scale=0.1,
        )
        if not result.success:
            raise ValueError(f"loss_outside could not fit the scattering envelope: {result.message}")
        checked = screened_intensity(q_check, *result.x) * fit_scale
        corner_ratio = np.quantile(check / np.maximum(checked, fit_scale * 1e-6), 0.90)
        allowance = margin * max(1.0, corner_ratio)
        logger.info(
            "loss_outside calibration: %d fit rings, %d check rings, corner MAE / mean DP %.4g, "
            "background / mean DP %.4g, elastic scale / mean DP %.4g, q_screen / q_edge %.4g, allowance %.3g",
            fit.size, check.size, np.abs(check - checked).mean() / scale,
            result.x[0] * fit_scale / scale, result.x[1] * fit_scale / scale,
            result.x[2], allowance,
        )

        # Match center_crop exactly. Distance is zero in the measured square.
        start = (simulation_size - size) // 2
        coordinate = np.arange(simulation_size)
        outside_axis = np.maximum(np.maximum(start - coordinate, coordinate - (start + size - 1)), 0)
        distance = np.hypot(outside_axis[:, None], outside_axis[None, :])
        outside = distance > 0
        centered = coordinate - simulation_size // 2
        q = np.hypot(centered[:, None], centered[None, :])[outside] / edge
        gate = -np.expm1(-(distance[outside] / transition_px)**2)
        # Pool in bands of two detector pixels. Pool the bound with the same weights.
        ring = np.floor(q * edge / 2).astype(np.int64)
        mass = np.bincount(ring, weights=gate)
        present = mass > 0
        ring_ids = np.cumsum(present) - 1
        bound = screened_intensity(q, *result.x) * (fit_scale * allowance / scale)
        bound = np.bincount(ring, weights=gate * bound)[present] / mass[present]
        bound = np.maximum(bound, np.finfo(np.float32).tiny)
        self.scale = float(scale)
        self.register_buffer("outside", torch.as_tensor(np.flatnonzero(outside), device=device), persistent=False)
        self.register_buffer("ring", torch.as_tensor(ring_ids[ring], device=device), persistent=False)
        self.register_buffer("gate", torch.as_tensor(gate, dtype=torch.float32, device=device), persistent=False)
        self.register_buffer("mass", torch.as_tensor(mass[present], dtype=torch.float32, device=device), persistent=False)
        self.register_buffer("bound", torch.as_tensor(bound, dtype=torch.float32, device=device), persistent=False)

    def forward(self, raw_dp):
        mean_dp = raw_dp.mean(dim=0, dtype=torch.float32).flatten()
        exterior = mean_dp.index_select(0, self.outside) * self.gate
        predicted = torch.zeros_like(self.mass).scatter_add(0, self.ring, exterior)
        predicted = predicted / (self.mass * self.scale)
        # Clamp at the bound before sqrt so a zero-intensity shell has finite gradients.
        excess = torch.maximum(predicted, self.bound).sqrt() - self.bound.sqrt()
        return (self.mass * excess.square()).sum() / self.mass.sum()
