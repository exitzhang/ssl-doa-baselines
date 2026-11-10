import numpy as np
import pytest
import torch

from ssl_doa_baselines.icocnn.accdoa import accdoa_pit_mse, decode_accdoa
from ssl_doa_baselines.icocnn.model import IcoCNNMultiSource

MIC = np.array([
    [0.04, 0.0, 1.2],
    [0.00, 0.0, 1.2],
    [-0.04, 0.0, 1.2],
], dtype=np.float64)


def tiny_model():
    return IcoCNNMultiSource(
        n_mics=3, sample_rate=48000, mic_positions=MIC, r=1,
        srp_win=1024, srp_hop=512, width=8, temporal_kernel=3,
        n_slots=10, smooth_vertices=False,
    )


def test_forward_shape_and_decode():
    model = tiny_model()
    model.eval()
    wav = torch.randn(2, 2048, 3)
    with torch.no_grad():
        out = model(wav)
    assert out["accdoa"].shape == (2, 3, 10, 3)
    dets = model.decode(wav, threshold_abs=0.0, max_sources=3)
    assert len(dets) == 2
    assert all(0 < len(d) <= 3 for d in dets)
    assert all(0.0 <= d["azimuth_deg"] <= 180.0 for row in dets for d in row)


@pytest.mark.parametrize("n_sources", [1, 2, 3])
def test_loss_finite_for_source_counts(n_sources):
    model = tiny_model()
    wav = torch.randn(2, 2048, 3)
    targets = [np.array([[20.0, 90.0], [90.0, 90.0], [150.0, 80.0]])[:n_sources]] * 2
    loss = model.training_loss(wav, targets)
    assert torch.isfinite(loss) and loss.item() >= 0.0


def test_pit_is_permutation_invariant():
    torch.manual_seed(0)
    pred = torch.randn(3, 4, 5, 3)
    a = np.array([[20.0, 90.0], [90.0, 90.0], [150.0, 80.0]])
    b = a[[2, 0, 1]]
    assert torch.allclose(accdoa_pit_mse(pred, [a, b, a]),
                          accdoa_pit_mse(pred, [b, a, b][:3]), atol=1e-6)


def test_empty_slot_is_zero_target():
    pred = torch.zeros(1, 2, 4, 3)
    loss = accdoa_pit_mse(pred, [np.zeros((0, 2), np.float32)])
    assert loss.item() == 0.0


def test_decode_respects_max_sources():
    acc = np.zeros((1, 1, 6, 3), dtype=np.float32)
    acc[0, 0, :, 0] = 1.0
    dets = decode_accdoa(acc, times=[[0.0]], threshold_abs=0.0, max_sources=3)
    assert len(dets[0]) == 3
