import numpy as np
import pytest
import torch

from ptyrad.core.scattering_prior import OutsideScatteringLoss, screened_intensity
from ptyrad.params.ptyrad_params import PtyRADParams


def test_exterior_loss_only_penalizes_excess_outside_crop():
    coords = (np.arange(64) - 32) / 32
    q = np.hypot(coords[:, None], coords[None, :])
    measurements = screened_intensity(q, 0.01, 0.2, 0.5)[None]
    prior = OutsideScatteringLoss(measurements, 128, transition_px=4, margin=1.25, device='cpu')

    prediction = torch.zeros(2, 128, 128)
    prediction[:, 32:96, 32:96] = 1000
    prediction.requires_grad_()
    assert prior(prediction).item() == 0
    prior(prediction).backward()
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad.count_nonzero() == 0

    prediction = torch.ones(2, 128, 128, requires_grad=True)
    loss = prior(prediction)
    assert loss.item() > 0
    loss.backward()
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad[:, 32:96, 32:96].count_nonzero() == 0
    assert prediction.grad[:, 0].min() > 0


def test_exterior_loss_requires_crop_with_real_detector_data(minimal_params_dict):
    params = minimal_params_dict
    assert not PtyRADParams.model_validate(params).loss_params.loss_outside.state
    params['loss_params'] = {'loss_outside': {'state': True}}
    with pytest.raises(ValueError, match='simu_match_mode'):
        PtyRADParams.model_validate(params)

    params['init_params'].update(simu_match_mode='crop', meas_pad={'mode': 'precompute'})
    with pytest.raises(ValueError, match='without meas_pad'):
        PtyRADParams.model_validate(params)
