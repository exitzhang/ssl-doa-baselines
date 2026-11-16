import numpy as np
import pytest
import torch

from ssl_doa_baselines.dpipd_frontend import DPIPDFrontend


def test_resample_length_and_stft_frames():
    wav = torch.randn(2, 16000, 3)
    frontend = DPIPDFrontend(48000, 16000, hop_length=256, n_mics=3)
    out = frontend(wav)
    assert frontend.n_freqs == 256
    assert out["features"].shape == (2, 6, 256, 20)
    assert out["stft_frames"] == 20

    frontend2 = DPIPDFrontend(48000, 16000, hop_length=320, n_mics=3)
    out2 = frontend2(wav)
    assert out2["features"].shape == (2, 6, 256, 17)
    assert out2["stft_frames"] == 17

    resampled = frontend.resampler(torch.randn(1, 3, 16000))
    assert resampled.shape[-1] == 5334


def test_frequency_axis():
    frontend = DPIPDFrontend(48000, 16000, hop_length=256, n_mics=3)
    freqs = frontend.freqs_hz
    assert freqs.shape == (256,)
    assert abs(freqs[0] - 31.25) < 1e-9
    assert abs(freqs[-1] - 8000.0) < 1e-9
    assert np.allclose(np.diff(freqs), 31.25)


def test_channel_consistency_after_resample():
    frontend = DPIPDFrontend(48000, 16000, hop_length=256, n_mics=3)
    signal = torch.randn(1, 16000, 1).repeat(1, 1, 3)
    features = frontend(signal)["features"]
    # Real block: channels 0..2; imaginary block: channels 3..5.
    for ch in (1, 2):
        assert torch.allclose(features[:, 0], features[:, ch], atol=1e-6)
        assert torch.allclose(features[:, 3], features[:, 3 + ch], atol=1e-6)


def test_phase_sign_matches_delayed_channel():
    """X_j = X_i * exp(-2j pi f tau) for j delayed by tau samples."""
    frontend = DPIPDFrontend(16000, 16000, hop_length=256, n_mics=2)
    torch.manual_seed(0)
    base = torch.randn(1, 8000)
    wav = torch.stack([base[0], torch.roll(base[0], 1)], dim=-1)[None]
    features = frontend(wav)["features"]

    x0 = torch.complex(features[0, 0], features[0, 2])
    x1 = torch.complex(features[0, 1], features[0, 3])
    cross_phase = torch.angle((x1 * x0.conj()).mean(dim=-1))
    freqs = torch.as_tensor(frontend.freqs_hz)
    expected = -2.0 * np.pi * freqs / 16000.0
    for k in (8, 16, 32, 64, 128):
        assert abs(float(cross_phase[k]) - float(expected[k])) < 0.01
