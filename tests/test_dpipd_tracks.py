import numpy as np
import torch
from scipy.special import j0

from ssl_doa_baselines.dpipd_tracks import TrackDPIPD

MIC = np.array([
    [0.04, 0.0, 1.2],
    [0.00, 0.0, 1.2],
    [-0.04, 0.0, 1.2],
], dtype=np.float64)
FREQS = np.arange(1, 257) * 16000.0 / 512.0
SPEED = 340.0


def tiny_tracks(n_az=181):
    return TrackDPIPD(MIC, FREQS, n_az=n_az, pool_factor=12)


def test_target_axis_order_and_bessel_empty():
    tracks = tiny_tracks()
    target = tracks.target_from_angles([[0.0, 90.0]])
    assert target.shape == (512, 2, 3)

    # Pair (ref=0, other=1): tau = u.(p1-p0)/c, phase = -2*pi*f*tau.
    tau = (MIC[1, 0] - MIC[0, 0]) / SPEED
    phase = -2.0 * np.pi * FREQS * tau
    assert np.allclose(target[:256, 0, 0].numpy(), np.cos(phase), atol=1e-6)
    assert np.allclose(target[256:, 0, 0].numpy(), np.sin(phase), atol=1e-6)
    # Real block is first, imaginary block second.
    assert not np.allclose(target[:256, 0, 0].numpy(), target[256:, 0, 0].numpy())

    distance = abs(MIC[1, 0] - MIC[0, 0])
    expected_empty = j0(2.0 * np.pi * FREQS * distance / SPEED)
    assert np.allclose(target[:256, 0, 1].numpy(), expected_empty, atol=1e-6)
    assert np.allclose(target[256:, 0, 1].numpy(), 0.0, atol=1e-6)


def test_template_matches_continuous_target_at_grid_points():
    tracks = tiny_tracks()
    for az in (0.0, 90.0, 180.0):
        target = tracks.target_from_angles([[az, 90.0]])
        index = int(np.argmin(np.abs(tracks.az_deg - az)))
        assert np.allclose(tracks.templates[index], target[:, :, 0],
                           atol=1e-5)


def test_pit_is_permutation_invariant_and_handles_zero_to_three_sources():
    tracks = tiny_tracks(n_az=37)
    cases = [
        np.zeros((0, 2)),
        np.array([[20.0, 90.0]]),
        np.array([[20.0, 90.0], [150.0, 80.0]]),
        np.array([[20.0, 90.0], [90.0, 90.0], [150.0, 80.0]]),
    ]
    for case in cases:
        target = tracks.targets_from_angles([case])
        loss, perm = tracks.pit_loss(target, target)
        assert float(loss) < 1e-10
        assert perm.shape == (1, 3)

    target = tracks.targets_from_angles([cases[-1]])
    permuted = target[..., [2, 0, 1]]
    loss, _ = tracks.pit_loss(target, permuted)
    assert float(loss) < 1e-10


def test_decode_recovers_grid_angles_and_near_sources():
    tracks = tiny_tracks()
    target = tracks.target_from_angles([[80.0, 90.0], [100.0, 90.0]])
    scores, az, el = tracks.decode_candidates(target[None], score_kind="ratio")
    assert scores.shape == (1, 3)
    assert set(np.round(az[0, :2]).tolist()) == {80.0, 100.0}
    assert np.allclose(el, 90.0)
    assert scores[0, 0] >= scores[0, 1] >= scores[0, 2]


def test_silence_scores_and_threshold_direction():
    tracks = tiny_tracks()
    silent = torch.zeros(1, 512, 2, 3)
    ratio, _, _ = tracks.decode_candidates(silent, score_kind="ratio")
    neg_mse, _, _ = tracks.decode_candidates(silent, score_kind="neg_mse")
    assert np.allclose(ratio, 0.0)
    # The mean uses 2F*P components; only the real half is nonzero.
    assert np.allclose(neg_mse, -0.5, atol=1e-5)

    ideal = tracks.target_from_angles([[30.0, 90.0]])[None]
    ratio_ideal, _, _ = tracks.decode_candidates(ideal, score_kind="ratio")
    assert ratio_ideal[0, 0] > 0.99
    assert float(ratio_ideal[0, 0]) >= 0.9


def test_pool_weights_and_clip_aggregation():
    assert np.allclose(
        TrackDPIPD.pool_weights(20, 12, 2).numpy(), [12 / 20, 8 / 20])
    assert np.allclose(
        TrackDPIPD.pool_weights(17, 5, 4).numpy(), [5 / 17] * 3 + [2 / 17])

    tracks = tiny_tracks()
    frames = torch.arange(2.0).reshape(1, 2, 1, 1, 1).repeat(1, 1, 512, 2, 3)
    clip = tracks.pool_clip(frames, 20)
    assert torch.allclose(clip, torch.full((1, 512, 2, 3), 8.0 / 20))
