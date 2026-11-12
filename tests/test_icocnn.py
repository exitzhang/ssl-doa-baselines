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


def test_head_matches_reference_architecture():
    """arXiv:2210.14536: only the last temporal conv outputs M channels."""
    model = tiny_model()
    assert model.temp_cnn[-1].conv.out_channels == 10
    assert all(m.conv.out_channels == 8 for m in model.temp_cnn[:-1])
    assert not hasattr(model, "head")
    assert not hasattr(model, "head_norm")


def test_output_is_bounded_and_input_dependent():
    """Soft-ArgMax output norms are activities in [0, 1]; the trunk is not dead."""
    model = tiny_model()
    model.eval()
    torch.manual_seed(0)
    wav_a = torch.randn(1, 2048, 3)
    wav_b = torch.randn(1, 2048, 3)
    with torch.no_grad():
        out_a = model(wav_a)["accdoa"]
        out_b = model(wav_b)["accdoa"]
    norms = out_a.norm(dim=-1)
    assert float(norms.max()) <= 1.0 + 1e-4
    assert float((out_a - out_b).abs().max()) > 1e-6


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


def _dataset_source(pos, az_deg, el_deg, *, fs=48000, n=4096, seed=0):
    """Synthetic clip in the dataset label frame (see ``make_synthetic.py``)."""
    rng = np.random.default_rng(seed)
    src = rng.standard_normal(n)
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    spectrum = np.fft.rfft(src)
    az, el = np.deg2rad(az_deg), np.deg2rad(el_deg)
    u = np.array([np.sin(el) * np.cos(az), np.sin(el) * np.sin(az), np.cos(el)])
    out = []
    for p in pos:
        delay = float(np.dot(p, u)) / 343.0
        out.append(np.fft.irfft(spectrum * np.exp(2j * np.pi * freqs * delay), n=n))
    return torch.from_numpy(np.stack(out, axis=-1)).float()[None]


def test_srp_steering_uses_unit_grid_in_label_frame():
    """A horizontal source must light up the matching azimuth, not its mirror."""
    from ssl_doa_baselines.icocnn.srp import IcosahedralSRP
    from ssl_doa_baselines.icocnn.grid import icosahedral_grid_coordinates

    pos = np.array([[0.04, 0.0, 1.2], [0.0, 0.0, 1.2], [-0.04, 0.0, 1.2]])
    grid = icosahedral_grid_coordinates(3).reshape(-1, 3)
    grid = grid / np.linalg.norm(grid, axis=-1, keepdims=True)
    horizontal = np.abs(np.degrees(np.arcsin(np.clip(grid[:, 2], -1.0, 1.0)))) < 15.0

    wav = _dataset_source(pos, az_deg=60.0, el_deg=90.0)
    srp = IcosahedralSRP(3, 48000, pos, r=3, win=4096, hop=2048)
    with torch.no_grad():
        maps, _ = srp(wav)
    scores = maps[0, 0].numpy().reshape(-1)
    peak = grid[np.flatnonzero(horizontal)[np.argmax(scores[horizontal])]]
    assert peak[0] == pytest.approx(np.cos(np.deg2rad(60.0)), abs=0.15)


def test_srp_ignores_self_pair_constant():
    """PHAT self-pairs add a constant; with 3 mics they must not flatten the map."""
    from ssl_doa_baselines.icocnn.srp import SRPIcosahedralMap

    srp = SRPIcosahedralMap(3, 1024, 3, MIC, 48000, tau_max=8)
    gcc = torch.randn(1, 3, 3, 17)
    maps = srp(gcc)
    changed = gcc.clone()
    for i in range(3):
        changed[:, i, i, :] = 100.0
    assert torch.allclose(maps, srp(changed))
