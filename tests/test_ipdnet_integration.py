import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from ssl_doa_baselines.api import (build_model, load_model, predict,
                                   predict_native, save_checkpoint)
from ssl_doa_baselines.dpipd_tracks import TrackDPIPD

MIC = [
    [0.04, 0.0, 1.2],
    [0.00, 0.0, 1.2],
    [-0.04, 0.0, 1.2],
]


def _ipdnet():
    return build_model("ipdnet", {
        "mic_positions": MIC, "n_az": 37, "hidden_size": 32,
        "hop": 256, "forgetting_frames": 280,
    })


def test_ipdnet_forward_shapes_and_axis_permutation():
    model = _ipdnet()
    wav = torch.randn(2, 16000, 3)
    captured = {}
    handle = model.backbone.conv.register_forward_hook(
        lambda _m, _i, out: captured.__setitem__("conv", out))
    try:
        out = model(wav)
    finally:
        handle.remove()

    assert out["stft_frames"] == 20
    assert tuple(out["frame_ipd"].shape) == (2, 2, 512, 2, 3)
    assert tuple(out["ipd"].shape) == (2, 512, 2, 3)

    conv = captured["conv"]                                  # (B,12,F,T)
    batch, t_out, two_f, pairs, tracks = out["frame_ipd"].shape
    n_freq = two_f // 2
    reconstructed = out["frame_ipd"].reshape(
        batch, t_out, 2, n_freq, pairs, tracks
    ).permute(0, 2, 4, 5, 3, 1).reshape(batch, 12, n_freq, t_out)
    assert torch.allclose(reconstructed, conv)


def test_ipdnet_training_loss_and_predict():
    model = _ipdnet()
    wav = torch.randn(2, 16000, 3)
    targets = [
        np.array([[20.0, 90.0], [150.0, 80.0]]),
        np.zeros((0, 2)),
    ]
    loss = model.training_loss(wav, targets)
    assert torch.isfinite(loss) and float(loss) >= 0.0
    loss.backward()

    detections = predict(model, wav, decode={"threshold_abs": -2.0})
    assert len(detections) == 2
    assert all(len(row) <= 3 for row in detections)


def test_ipdnet_checkpoint_roundtrip(tmp_path):
    model = _ipdnet()
    wav = torch.randn(1, 16000, 3)
    before = predict_native(model, wav, device="cpu")

    path = tmp_path / "checkpoint.pth"
    save_checkpoint(path, model)
    reloaded = load_model(str(path), device="cpu")
    after = predict_native(reloaded, wav, device="cpu")
    assert torch.allclose(before["ipd"], after["ipd"], atol=1e-6)
    assert torch.allclose(before["frame_ipd"], after["frame_ipd"], atol=1e-6)


class _TinyDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, index):
        return torch.randn(16000, 3), torch.tensor(
            [[20.0, 90.0], [150.0, 90.0]])


def test_ipdnet_candidate_cache():
    from integrations.locata import adapter

    model = _ipdnet()
    loader = DataLoader(_TinyDataset(), batch_size=2,
                        collate_fn=adapter.angle_batch_collate)
    cache = adapter.cache_candidates(model, loader, device="cpu",
                                     max_sources=3)
    assert cache["kind"] == "candidates"
    assert cache["frames"] == 2
    assert len(cache["candidates"]) == 2
    for row in cache["candidates"]:
        assert len(row) == 3
        assert row == sorted(row, key=lambda item: -item[0])
    assert cache["template_ms_per_frame"] >= 0.0


def test_ipdnet2_small_builds_and_reloads_on_cpu():
    pytest.importorskip("mamba_ssm")
    model = build_model("ipdnet2_small", {
        "mic_positions": MIC, "n_az": 37,
    })
    assert model.model_name == "ipdnet2_small"
    assert model.config["score_kind"] == "neg_mse"
    assert model.config["pool_factor"] == 5
    assert model.config["hop"] == 320


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Mamba needs CUDA")
def test_ipdnet2_small_forward_shapes():
    pytest.importorskip("mamba_ssm")
    model = build_model("ipdnet2_small", {
        "mic_positions": MIC, "n_az": 37,
    }).cuda()
    wav = torch.randn(1, 16000, 3, device="cuda")
    out = model(wav)
    assert out["stft_frames"] == 17
    assert tuple(out["frame_ipd"].shape) == (1, 4, 512, 2, 3)
    assert tuple(out["ipd"].shape) == (1, 512, 2, 3)
    loss = model.training_loss(wav, [np.array([[20.0, 90.0]])])
    assert torch.isfinite(loss)

    scores, azimuth, elevation = model.tracks.decode_candidates(
        out["ipd"], score_kind="neg_mse")
    assert scores.shape == (1, 3)
    assert np.allclose(elevation, 90.0)


def test_shared_tracks_separate_from_models():
    tracks = TrackDPIPD(np.asarray(MIC), np.arange(1, 257) * 16000 / 512,
                        n_az=37, pool_factor=5)
    target = tracks.target_from_angles([[40.0, 90.0]])
    _, azimuth, _ = tracks.decode_candidates(target[None], "neg_mse")
    assert 35.0 <= float(azimuth[0, 0]) <= 45.0
