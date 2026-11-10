import numpy as np
import pytest
import torch

from ssl_doa_baselines.tf_mamba.decode import decode_spectrum

mamba_ssm = pytest.importorskip("mamba_ssm")


def _cuda_available():
    return torch.cuda.is_available()


@pytest.mark.skipif(not _cuda_available(), reason="mamba-ssm requires CUDA")
def test_forward_shape_and_backward():
    from ssl_doa_baselines.tf_mamba.model import TFMambaSpectrum

    model = TFMambaSpectrum(
        num_mics=3, sample_rate=48000, n_fft=256, hop_length=64,
        freq_min=100.0, freq_max=8000.0, d_model=8, num_blocks=2,
        expand_factors=[2, 2], d_state=8, d_conv=4, output_dim=181,
        activation="sigmoid",
    ).cuda()
    x = torch.randn(2, 4096, 3, device="cuda")
    y = model(x)
    assert y.shape == (2, 181)
    assert torch.all(y >= 0) and torch.all(y <= 1)
    loss = y.mean()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


@pytest.mark.skipif(not _cuda_available(), reason="mamba-ssm requires CUDA")
def test_checkpoint_roundtrip(tmp_path):
    from ssl_doa_baselines.api import load_model, save_checkpoint
    from ssl_doa_baselines.tf_mamba.model import TFMambaSpectrum

    model = TFMambaSpectrum(
        num_mics=3, sample_rate=48000, n_fft=256, hop_length=64,
        d_model=8, num_blocks=2, expand_factors=[2, 2], d_state=8,
        output_dim=181,
    ).cuda().eval()
    path = tmp_path / "ckpt.pth"
    save_checkpoint(path, model, config={"model": dict(model.config, name="tf_mamba")})
    loaded = load_model(str(path), device="cuda").eval()
    x = torch.randn(1, 4096, 3, device="cuda")
    with torch.no_grad():
        assert torch.allclose(model(x), loaded(x), atol=1e-5)


def test_decode_multiple_peaks():
    spec = np.zeros((1, 181), dtype=np.float32)
    spec[0, [20, 90, 150]] = [1.0, 0.8, 0.6]
    dets = decode_spectrum(spec, max_sources=3, threshold_ratio=0.1,
                           neighborhood_size=3, min_peak_distance=3)
    assert [d["bin"] for d in dets[0]] == [20, 90, 150]


def test_decode_silence():
    dets = decode_spectrum(np.zeros((1, 181), np.float32), silence_max_thresh=0.0)
    assert dets[0] == []
