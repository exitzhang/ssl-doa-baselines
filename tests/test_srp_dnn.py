import numpy as np
import torch

from ssl_doa_baselines.srp_dnn.dpipd import DPIPD
from ssl_doa_baselines.srp_dnn.idl import SourceDetectLocalize
from ssl_doa_baselines.srp_dnn.model import SRPDNN

MIC = np.array([
    [0.04, 0.0, 1.2],
    [0.00, 0.0, 1.2],
    [-0.04, 0.0, 1.2],
], dtype=np.float64)


def tiny_model():
    return SRPDNN(
        n_mics=3, sample_rate=48000, mic_positions=MIC,
        win_len=1536, hop=768, n_fft=1536, freq_min=31.25, freq_max=8000.0,
        max_num_sources=3, az_range_deg=(0.0, 180.0), n_az=37,
        el_range_deg=(60.0, 120.0), n_el=3,
    )


def test_forward_shapes_and_loss():
    model = tiny_model()
    wav = torch.randn(2, 4096, 3)
    out = model(wav)
    assert out["ipd"].shape[0] == 2
    assert out["ipd"].shape[-1] == 3
    assert out["ipd"].shape[2] == 2 * model.n_freq
    targets = [np.array([[20.0, 90.0], [150.0, 80.0]])] * 2
    loss = model.training_loss(wav, targets)
    assert torch.isfinite(loss) and loss.item() >= 0.0


def test_idl_recovers_three_sources():
    bins = np.arange(1, 257) * 48000 / 1536
    dpipd = DPIPD(MIC, bins, n_az=181, n_el=7)
    doa = np.array([[20.0, 90.0], [90.0, 90.0], [150.0, 90.0]])
    target = dpipd.mixed_target(doa, np.array([0.5, 0.3, 0.2]))
    idl = SourceDetectLocalize(3, dpipd.n_freq, len(dpipd.pairs))
    out = idl(torch.tensor(target[None, None]), dpipd.template,
              dpipd.az_deg, dpipd.el_deg)
    pred = out["doa"][0, 0].numpy().T
    scores = out["vad"][0, 0].numpy()
    assert np.all(scores > 0)
    # Collinear 3-mic array: azimuth resolvable, elevation ambiguous.
    for truth in doa[:, 0]:
        assert np.min(np.abs(pred[:, 0] - truth)) < 25.0


def test_decode_runs():
    model = tiny_model()
    dets = model.decode(torch.randn(2, 4096, 3), threshold_abs=0.0, max_sources=3)
    assert len(dets) == 2
    assert all(len(row) <= 3 for row in dets)
