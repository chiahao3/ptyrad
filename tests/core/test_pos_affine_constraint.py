"""Tests for the ``pos_affine`` iteration-wise constraint."""

from collections import defaultdict

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ptyrad.core.constraints import (  # noqa: E402
    CombinedConstraint,
    fit_affine_pos,
    get_pos_rms_extent,
    get_probe_pos,
    is_2d_pos,
)
from ptyrad.utils.affine import compose_affine_matrix  # noqa: E402


class DummyModel:
    """Minimal stand-in exposing only the attributes ``apply_pos_affine`` touches."""

    def __init__(self, pos_init, pos_current, active_indices=None):
        crop_pos = np.round(pos_init).astype("int32")
        self.crop_pos = torch.tensor(crop_pos, dtype=torch.int32)
        self.init_probe_pos_shifts = torch.tensor(pos_init - crop_pos, dtype=torch.float32)
        self.opt_probe_pos_shifts = torch.nn.Parameter(
            torch.tensor(pos_current - crop_pos, dtype=torch.float32)
        )
        self.active_indices = (
            None if active_indices is None else torch.as_tensor(active_indices, dtype=torch.long)
        )
        self.convergence_iters = defaultdict(list)

    @property
    def probe_pos(self):
        """Current probe positions as float64, matching what the constraint fits."""
        return get_probe_pos(self.crop_pos, self.opt_probe_pos_shifts.detach())


def make_scan(n_slow=12, n_fast=10, step=4.0, offset=64.0, seed=0):
    """A regular raster scan (N, 2) in (y, x) object pixel coordinates."""
    rng = np.random.default_rng(seed)
    pos = step * np.array([(y, x) for y in range(n_slow) for x in range(n_fast)], dtype=np.float64)
    pos = pos - pos.mean(0) + offset
    # A small sub-px jitter of the initial scan, which the constraint must preserve
    return pos + 0.05 * rng.standard_normal(pos.shape)


def apply_affine(pos, scale=1.0, asymmetry=0.0, rotation=0.0, shear=0.0):
    """Apply the ptyrad affine convention (row vector @ matrix) about the scan center."""
    center = pos.mean(0)
    return (pos - center) @ compose_affine_matrix(scale, asymmetry, rotation, shear) + center


def make_constraint(**overrides):
    params = {"pos_affine": {"start_iter": 1, "step": 1, "end_iter": None, "relax": 0.0}}
    params["pos_affine"].update(overrides)
    return CombinedConstraint(params, device="cpu")


def as_tensors(model):
    return get_probe_pos(model.crop_pos, model.init_probe_pos_shifts), model.probe_pos


# ---------------------------------------------------------------------------
# fit_affine_pos
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "affine_kwargs",
    [
        {"rotation": 1.5},
        {"scale": 1.02},
        {"asymmetry": 0.03},
        {"shear": 0.8},
        {"scale": 0.99, "asymmetry": -0.02, "rotation": -2.0, "shear": 0.5},
    ],
)
def test_fit_recovers_pure_affine_deformation(affine_kwargs):
    """A purely affine deformation of the initial positions must be reproduced exactly."""
    pos_init = make_scan()
    pos_current = apply_affine(pos_init, **affine_kwargs)

    model = DummyModel(pos_init, pos_current)
    pos_fit, affine_mat = fit_affine_pos(*as_tensors(model))

    expected = compose_affine_matrix(
        affine_kwargs.get("scale", 1.0),
        affine_kwargs.get("asymmetry", 0.0),
        affine_kwargs.get("rotation", 0.0),
        affine_kwargs.get("shear", 0.0),
    )
    assert np.allclose(affine_mat.numpy(), expected, atol=1e-8)
    assert np.allclose(pos_fit.numpy(), model.probe_pos.numpy(), atol=1e-4)


def test_fit_removes_local_noise_but_keeps_global_rotation():
    """Per-position noise is discarded while the global rotation survives the fit."""
    rng = np.random.default_rng(42)
    pos_init = make_scan()
    pos_clean = apply_affine(pos_init, rotation=1.0)
    pos_noisy = pos_clean + 0.3 * rng.standard_normal(pos_clean.shape)

    model = DummyModel(pos_init, pos_noisy)
    pos_fit, _ = fit_affine_pos(*as_tensors(model))

    err_before = np.sqrt(((pos_noisy - pos_clean) ** 2).sum(-1).mean())
    err_after = np.sqrt(((pos_fit.numpy() - pos_clean) ** 2).sum(-1).mean())
    assert err_after < 0.2 * err_before


def test_fit_preserves_the_mean_position():
    """The translation is absorbed by the centering, so the global offset is untouched."""
    pos_init = make_scan()
    pos_current = apply_affine(pos_init, rotation=0.7) + np.array([2.5, -1.25])

    model = DummyModel(pos_init, pos_current)
    pos_fit, _ = fit_affine_pos(*as_tensors(model))
    assert np.allclose(pos_fit.numpy().mean(0), pos_current.mean(0), atol=1e-4)


def test_fit_ignores_positions_excluded_by_indices_mode():
    """Positions that never received a gradient must not drag the fit towards identity.

    Mirrors ``INDICES_MODE: 'sub'`` with the default half-by-half subsampling, where only a
    quarter of the positions are optimized and the rest still sit at their initial values.
    """
    n_slow, n_fast = 12, 10
    pos_init = make_scan(n_slow=n_slow, n_fast=n_fast)
    active = (np.arange(n_slow * n_fast).reshape(n_slow, n_fast)[::2, ::2]).reshape(-1)

    # Only the active positions carry the global rotation, the rest are untouched
    pos_current = pos_init.copy()
    pos_current[active] = apply_affine(pos_init, rotation=1.0)[active]

    model = DummyModel(pos_init, pos_current, active_indices=active)
    _, affine_subset = fit_affine_pos(*as_tensors(model), fit_indices=model.active_indices)
    _, affine_all = fit_affine_pos(*as_tensors(model))

    expected = compose_affine_matrix(1.0, 0.0, 1.0, 0.0)
    assert np.allclose(affine_subset.numpy(), expected, atol=1e-3)
    # Fitting everything is dragged towards identity by the untouched positions
    assert not np.allclose(affine_all.numpy(), expected, atol=1e-3)
    identity = np.eye(2)
    assert np.abs(affine_all.numpy() - identity).max() < 0.5 * np.abs(expected - identity).max()


def test_fit_extrapolates_to_positions_excluded_by_indices_mode():
    """The fitted transformation is applied to all positions, not just the fitted ones."""
    n_slow, n_fast = 12, 10
    pos_init = make_scan(n_slow=n_slow, n_fast=n_fast)
    active = (np.arange(n_slow * n_fast).reshape(n_slow, n_fast)[::2, ::2]).reshape(-1)
    inactive = np.setdiff1d(np.arange(n_slow * n_fast), active)

    pos_rotated = apply_affine(pos_init, rotation=1.0)
    pos_current = pos_init.copy()
    pos_current[active] = pos_rotated[active]

    model = DummyModel(pos_init, pos_current, active_indices=active)
    pos_fit, _ = fit_affine_pos(*as_tensors(model), fit_indices=model.active_indices)

    # The excluded positions end up on the rotated scan pattern rather than staying at their initial value
    assert np.allclose(pos_fit.numpy()[inactive], pos_rotated[inactive], atol=1e-2)
    assert not np.allclose(pos_fit.numpy()[inactive], pos_init[inactive], atol=1e-2)


# ---------------------------------------------------------------------------
# 2D scan gating
# ---------------------------------------------------------------------------


def test_rms_extent_separates_line_scans_from_thin_2d_scans():
    """The RMS extent is what makes the line-scan gate safe for thin but legitimate scans."""
    line = torch.tensor(make_scan(n_slow=1, n_fast=16))  # only sub-px jitter across the scan
    assert get_pos_rms_extent(line)[1].item() < 1.0
    assert not is_2d_pos(line)

    for n_slow, n_fast in ((256, 8), (512, 4)):
        thin = torch.tensor(make_scan(n_slow=n_slow, n_fast=n_fast))
        assert get_pos_rms_extent(thin)[1].item() > 1.0
        assert is_2d_pos(thin)


def test_constraint_is_skipped_for_a_line_scan(caplog):
    """A line scan cannot determine a 2D affine model, so the positions are left untouched."""
    pos_init = make_scan(n_slow=1, n_fast=16)
    pos_current = pos_init + np.stack([np.zeros(16), 0.2 * np.arange(16)], axis=-1)

    model = DummyModel(pos_init, pos_current)
    before = model.opt_probe_pos_shifts.detach().clone()

    constraint = make_constraint()
    with caplog.at_level("WARNING", logger="ptyrad.core.constraints"), torch.no_grad():
        constraint.apply_pos_affine(model, niter=1)
        constraint.apply_pos_affine(model, niter=2)

    assert torch.equal(model.opt_probe_pos_shifts.detach(), before)
    assert model.convergence_iters == {}
    # Resolved once and cached, so the warning is not repeated every iteration
    assert caplog.text.count("cannot determine a 2D affine model") == 1


def test_constraint_is_skipped_for_a_single_row_indices_mode_subset():
    """``INDICES_MODE: 'center'`` with ``subscan_slow: 1`` fits a single row, which is a line scan."""
    n_slow, n_fast = 12, 10
    pos_init = make_scan(n_slow=n_slow, n_fast=n_fast)
    active = np.arange(5 * n_fast, 6 * n_fast)

    pos_current = pos_init.copy()
    pos_current[active, 1] *= 1.05

    model = DummyModel(pos_init, pos_current, active_indices=active)
    before = model.opt_probe_pos_shifts.detach().clone()

    with torch.no_grad():
        make_constraint().apply_pos_affine(model, niter=1)

    assert torch.equal(model.opt_probe_pos_shifts.detach(), before)


# ---------------------------------------------------------------------------
# CombinedConstraint.apply_pos_affine
# ---------------------------------------------------------------------------


def test_constraint_with_relax_zero_replaces_positions_with_the_fit():
    rng = np.random.default_rng(1)
    pos_init = make_scan()
    pos_current = apply_affine(pos_init, rotation=1.0) + 0.3 * rng.standard_normal(pos_init.shape)

    model = DummyModel(pos_init, pos_current)
    pos_fit, _ = fit_affine_pos(*as_tensors(model))
    expected = (pos_fit - model.crop_pos).to(torch.float32)

    with torch.no_grad():
        make_constraint(relax=0.0).apply_pos_affine(model, niter=1)

    assert torch.allclose(model.opt_probe_pos_shifts.detach(), expected, atol=1e-6)


def test_constraint_relax_linearly_mixes_current_and_fitted_positions():
    """The mixing happens on the positions, and the shifts follow by subtracting crop_pos."""
    rng = np.random.default_rng(2)
    pos_init = make_scan()
    pos_current = apply_affine(pos_init, rotation=1.0) + 0.3 * rng.standard_normal(pos_init.shape)

    model = DummyModel(pos_init, pos_current)
    pos_before = model.probe_pos
    pos_fit, _ = fit_affine_pos(*as_tensors(model))

    with torch.no_grad():
        make_constraint(relax=0.75).apply_pos_affine(model, niter=1)

    expected_pos = 0.75 * pos_before + 0.25 * pos_fit
    assert np.allclose(model.probe_pos.numpy(), expected_pos.numpy(), atol=1e-4)


def test_constraint_records_the_fitted_affine_in_convergence_iters():
    pos_init = make_scan()
    pos_current = apply_affine(pos_init, scale=1.02, asymmetry=0.03, rotation=1.0, shear=0.5)

    model = DummyModel(pos_init, pos_current)
    with torch.no_grad():
        make_constraint().apply_pos_affine(model, niter=7)

    recorded = {k: v for k, v in model.convergence_iters.items()}
    assert set(recorded) == {
        "pos_affine_scale",
        "pos_affine_asymmetry",
        "pos_affine_rotation",
        "pos_affine_shear",
    }
    for key, expected in (
        ("pos_affine_scale", 1.02),
        ("pos_affine_asymmetry", 0.03),
        ("pos_affine_rotation", 1.0),
        ("pos_affine_shear", 0.5),
    ):
        (niter, value), = recorded[key]
        assert niter == 7
        assert value == pytest.approx(expected, abs=1e-3)


def test_constraint_appends_one_entry_per_application():
    pos_init = make_scan()
    pos_current = apply_affine(pos_init, rotation=1.0)

    model = DummyModel(pos_init, pos_current)
    constraint = make_constraint(relax=0.5)
    with torch.no_grad():
        for niter in (1, 2, 3):
            constraint.apply_pos_affine(model, niter)

    assert [n for n, _ in model.convergence_iters["pos_affine_rotation"]] == [1, 2, 3]


@pytest.mark.parametrize(
    "overrides",
    [
        {"start_iter": None},  # disabled
        {"start_iter": 5},  # not started yet at niter=1
        {"start_iter": 1, "end_iter": 1},  # end_iter is exclusive
        {"start_iter": 1, "step": 3},  # niter=2 is not on the step grid
        {"relax": 1.0},  # fully relaxed is a no-op
    ],
)
def test_constraint_is_a_no_op_when_not_scheduled(overrides):
    rng = np.random.default_rng(3)
    pos_init = make_scan()
    pos_current = apply_affine(pos_init, rotation=1.0) + 0.3 * rng.standard_normal(pos_init.shape)

    model = DummyModel(pos_init, pos_current)
    before = model.opt_probe_pos_shifts.detach().clone()

    niter = 2 if overrides.get("step") == 3 else 1
    with torch.no_grad():
        make_constraint(**overrides).apply_pos_affine(model, niter=niter)

    assert torch.equal(model.opt_probe_pos_shifts.detach(), before)
    assert model.convergence_iters == {}


def test_missing_constraint_entry_is_treated_as_disabled():
    """Params files written before ``pos_affine`` existed must not raise a KeyError."""
    pos_init = make_scan()
    model = DummyModel(pos_init, pos_init)
    before = model.opt_probe_pos_shifts.detach().clone()

    with torch.no_grad():
        CombinedConstraint({}, device="cpu").apply_pos_affine(model, niter=1)

    assert torch.equal(model.opt_probe_pos_shifts.detach(), before)
