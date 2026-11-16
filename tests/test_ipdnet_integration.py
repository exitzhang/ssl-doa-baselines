import argparse
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from ssl_doa_baselines import cli, working_point
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


# ---------------------------------------------------------------------------
# Working-point identity
# ---------------------------------------------------------------------------

def _save(path, model, data_cfg=None):
    save_checkpoint(path, model, config={
        "model": dict(model.config, name=model.model_name),
        "data": data_cfg or {"mic_indices": [5, 6, 8]},
    })
    return path


def _point(threshold=0.2, f1=0.5):
    return {"key": str(threshold), "threshold": threshold, "F1": f1,
            "P": 0.5, "R": 0.5, "FAR": 0.1, "MAE": 1.0}


def _lock(path, model, checkpoint, score_kind="ratio", threshold=0.2):
    record = working_point.make_locked_record(
        _point(threshold), model=model.model_name, checkpoint=str(checkpoint),
        score_kind=score_kind, acc_threshold=10)
    Path(path).write_text(json.dumps({"locked": record, "curve": {}}))
    return record


def test_working_point_rejects_missing_and_mismatched_locks(tmp_path):
    model = _ipdnet()
    ckpt = _save(tmp_path / "a.pth", model)
    args = dict(model="ipdnet", checkpoint=str(ckpt), score_kind="ratio",
                acc_threshold=10)

    with pytest.raises(working_point.WorkingPointError,
                       match="validation file not found"):
        working_point.require_locked(tmp_path / "missing.json", **args)

    path = tmp_path / "validation.json"
    record = _lock(path, model, ckpt)
    loaded = working_point.require_locked(path, **args)
    assert loaded["threshold"] == record["threshold"]

    # Another epoch/weights of the same architecture: different digest.
    other = _save(tmp_path / "b.pth", _ipdnet())
    with pytest.raises(working_point.WorkingPointError,
                       match="lock belongs to checkpoint"):
        working_point.require_locked(path, **{**args, "checkpoint": str(other)})
    # Another model key.
    with pytest.raises(working_point.WorkingPointError,
                       match="lock belongs to model"):
        working_point.require_locked(path, **{**args, "model": "ipdnet2_small"})
    # Another score kind / selection tolerance.
    with pytest.raises(working_point.WorkingPointError, match="score_kind"):
        working_point.require_locked(path, **{**args, "score_kind": "neg_mse"})
    with pytest.raises(working_point.WorkingPointError, match="acc_threshold"):
        working_point.require_locked(path, **{**args, "acc_threshold": 5})
    # Incomplete lock.
    broken = json.loads(path.read_text())
    broken["locked"].pop("model")
    broken_path = tmp_path / "broken.json"
    broken_path.write_text(json.dumps(broken))
    with pytest.raises(working_point.WorkingPointError, match="missing"):
        working_point.require_locked(broken_path, **args)


def test_dicit_eval_fails_before_loading_data_without_validation(
        tmp_path, monkeypatch):
    from integrations.locata import run as runner

    model = _ipdnet()
    ckpt = _save(tmp_path / "checkpoint.pth", model)
    cfg = tmp_path / "cfg.toml"
    cfg.write_text("[model]\nname = \"ipdnet\"\n[decode]\nsource_column = 10\n")

    def _forbidden(*_args, **_kwargs):
        raise AssertionError("DICIT eval touched the dataset before the lock")

    monkeypatch.setattr(runner.adapter, "ensure_locata", lambda root: tmp_path)
    monkeypatch.setattr(runner.adapter, "build_dicit_loader", _forbidden)
    monkeypatch.setattr(runner.adapter, "check_audio_contract", _forbidden)
    monkeypatch.setattr(runner, "_evaluate", _forbidden)

    args = argparse.Namespace(
        config=str(cfg), data_config=None, seed=None, epochs=None,
        max_train_batches=None, output=None, locata_root=str(tmp_path),
        device="cpu", device_index=0, checkpoint=str(ckpt), split="dicit",
        validation=None)
    with pytest.raises(SystemExit, match="requires --validation"):
        runner.cmd_eval(args)


# ---------------------------------------------------------------------------
# Fixed-array contract
# ---------------------------------------------------------------------------

def _meta(root, positions):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "dataset_meta.json").write_text(json.dumps({
        "mic_pos": positions, "fs": 48000, "mic_output_num_samples": 16000,
    }))
    return root


def _array_model():
    return build_model("ipdnet", {
        "mic_positions": MIC, "n_mics": 3, "n_az": 37, "hidden_size": 8,
        "hop": 256, "forgetting_frames": 280,
    })


def test_fixed_array_rejects_swapped_order_and_spacing(tmp_path):
    from integrations.locata import adapter

    train = _meta(tmp_path / "train", MIC)
    val = _meta(tmp_path / "val", MIC)
    model = _array_model()
    data_cfg = {"train_root": str(train), "val_root": str(val),
                "mic_indices": [0, 1, 2]}
    record = adapter.check_fixed_array(model, data_cfg)
    assert record["mic_indices"] == [0, 1, 2]

    # Swap the first and third channel: the pair vectors reverse.
    swapped = dict(data_cfg, mic_indices=[2, 1, 0])
    with pytest.raises(ValueError, match="does not match the model's fixed array"):
        adapter.check_fixed_array(model, swapped)

    # Translate the whole array: allowed.
    shifted = np.asarray(MIC) + np.array([3.0, -2.0, 0.5])
    _meta(tmp_path / "train", shifted.tolist())
    _meta(tmp_path / "val", shifted.tolist())
    adapter.check_fixed_array(model, data_cfg)

    # Change one microphone spacing: rejected.
    wider = np.asarray(MIC).copy()
    wider[2, 0] -= 0.02
    _meta(tmp_path / "train", wider.tolist())
    _meta(tmp_path / "val", wider.tolist())
    with pytest.raises(ValueError, match="does not match the model's fixed array"):
        adapter.check_fixed_array(model, data_cfg)


def test_eval_array_rejects_permuted_loader_and_geometry(tmp_path):
    from integrations.locata import adapter

    model = _array_model()
    checkpoint_config = {"data": {
        "mic_indices": [0, 1, 2], "mic_positions": MIC, "ref_mic": 0}}
    data_cfg = {"mic_positions": MIC, "mic_indices": [0, 1, 2]}
    record = adapter.check_eval_array(model, checkpoint_config, data_cfg,
                                      split="dicit")
    assert record["mic_indices"] == [0, 1, 2]
    assert record["geometry_source"] == "data.mic_positions"

    permuted = dict(data_cfg, mic_indices=[2, 1, 0])
    with pytest.raises(ValueError, match="channel order mismatch"):
        adapter.check_eval_array(model, checkpoint_config, permuted,
                                 split="dicit")

    wider = np.asarray(MIC).copy()
    wider[2, 0] -= 0.02
    shifted = dict(data_cfg, mic_positions=wider.tolist())
    with pytest.raises(ValueError, match="does not match the checkpoint geometry"):
        adapter.check_eval_array(model, checkpoint_config, shifted,
                                 split="dicit")

    translated = dict(data_cfg,
                      mic_positions=(np.asarray(MIC) + 100.0).tolist())
    adapter.check_eval_array(model, checkpoint_config, translated,
                             split="dicit")


# ---------------------------------------------------------------------------
# Explicit operating point for standalone prediction
# ---------------------------------------------------------------------------

def test_predict_requires_an_explicit_finite_threshold(monkeypatch):
    model = _ipdnet()
    wav = torch.randn(1, 16000, 3)

    def _forward_called(*_args, **_kwargs):
        raise AssertionError("predict forwarded before validating the threshold")

    monkeypatch.setattr(model, "forward", _forward_called)
    with pytest.raises(ValueError, match="requires an explicit decode"):
        predict(model, wav, decode={})
    with pytest.raises(ValueError, match="does not match the model"):
        predict(model, wav, decode={"threshold_abs": -0.4,
                                    "score_kind": "neg_mse"})
    with pytest.raises(ValueError, match="must be finite"):
        predict(model, wav, decode={"threshold_abs": float("nan")})


def test_threshold_equality_is_accepted_like_the_sweep():
    from integrations.locata import adapter
    from ssl_doa_baselines.api import _filter_track_candidates

    scores = np.array([[-0.4, -0.400001, -0.3]], dtype=np.float32)
    azimuth = np.array([[30.0, 60.0, 90.0]], dtype=np.float32)
    elevation = np.full_like(azimuth, 90.0)
    threshold = float(scores[0, 0])       # exactly the first score
    dets = _filter_track_candidates(scores, azimuth, elevation,
                                    threshold=threshold, max_sources=3)
    assert [d["azimuth_deg"] for d in dets[0]] == [30.0, 90.0]

    cache = {"gts": [], "candidates": [
        [(-0.4, 30.0, 90.0), (-0.400001, 60.0, 90.0), (-0.3, 90.0, 90.0)]]}
    pred = adapter.decoded_at_threshold(cache, -0.4)
    assert pred == [[(30.0, 90.0), (90.0, 90.0)]]


def test_cli_validation_and_explicit_threshold_agree(tmp_path):
    import soundfile as sf

    model = _ipdnet()
    ckpt = _save(tmp_path / "checkpoint.pth", model)
    audio = tmp_path / "clip.wav"
    sf.write(audio, np.random.randn(16000, 3).astype(np.float32), 48000)
    validation = tmp_path / "validation.json"
    _lock(validation, model, ckpt, threshold=-0.4)
    out_explicit = tmp_path / "explicit.json"
    out_locked = tmp_path / "locked.json"

    base = ["predict", "--checkpoint", str(ckpt), "--audio", str(audio),
            "--device", "cpu"]
    assert cli.main(base + ["--threshold-abs=-0.4",
                            "--output", str(out_explicit)]) == 0
    assert cli.main(base + ["--validation", str(validation),
                            "--output", str(out_locked)]) == 0
    explicit = json.loads(out_explicit.read_text())
    locked = json.loads(out_locked.read_text())
    assert explicit["detections"] == locked["detections"]
    assert locked["locked_threshold"] == -0.4

    with pytest.raises(SystemExit, match="only one of"):
        cli.main(base + ["--threshold-abs=-0.4",
                         "--validation", str(validation)])

    other = _save(tmp_path / "other.pth", _ipdnet())
    with pytest.raises(SystemExit, match="lock belongs to checkpoint"):
        cli.main(["predict", "--checkpoint", str(other), "--audio", str(audio),
                  "--device", "cpu", "--validation", str(validation)])


# ---------------------------------------------------------------------------
# Parameters and timing
# ---------------------------------------------------------------------------

def test_parameter_count_comes_from_the_rebuilt_model(tmp_path):
    from integrations.locata import adapter

    model = build_model("ipdnet", {"mic_positions": MIC})
    assert adapter.count_parameters(model) == 725248
    ckpt = _save(tmp_path / "checkpoint.pth", model)
    reloaded = load_model(str(ckpt), device="cpu")
    assert adapter.count_parameters(reloaded) == 725248


def test_benchmark_denominator_is_the_accumulated_clip_count():
    from integrations.locata import adapter

    model = _ipdnet()
    batches = [torch.randn(2, 16000, 3), torch.randn(2, 16000, 3),
               torch.randn(1, 16000, 3)]
    timing = adapter.benchmark_inference(
        model, batches, decode_cfg={"score_kind": "ratio", "max_sources": 3},
        split="val", threshold=-2.0, warmup_batches=1, repeats=1)
    assert timing["inference_clips"] == 5      # 2 + 2 + 1
    assert timing["inference_batches"] == 3
    assert timing["inference_warmup_batches"] == 1
    assert timing["inference_ms_per_clip"] >= 0.0


def test_eval_payload_uses_the_validation_point_on_the_test_curve(tmp_path):
    from integrations.locata import run as runner

    model = _ipdnet()

    def row(f1):
        return {acc: {"F1": f1, "P": f1, "R": f1, "FAR": 0.1, "MAE": 1.0}
                for acc in (5, 10, 15)}

    result = {
        "frames": 5,
        "source_count_dist": {"0": 5},
        "gts": [],
        "metrics_by_key": {"0.2": row(0.4), "0.5": row(0.9)},
    }
    locked = working_point.make_locked_record(
        _point(0.2, f1=0.4), model="ipdnet",
        checkpoint=str(_save(tmp_path / "checkpoint.pth", model)),
        score_kind="ratio", acc_threshold=10)
    assert 0.2 in runner._merge_thresholds([0.0, 0.5], [0.2])
    payload = runner._eval_payload(
        model=model, result=result, pred_all=[[]] * 5, locked=locked,
        cfg={"model": {"name": "ipdnet"}, "decode": {}},
        decode_cfg={"source_column": 10, "far_target": 0.03},
        split="dicit", acc=(5, 10, 15),
        checkpoint=str(tmp_path / "checkpoint.pth"), decode_s=0.1,
        locata_root=str(tmp_path),
        inference={"inference_ms_per_clip": 1.0}, array_contract=None)
    assert payload["locked_threshold"] == 0.2
    assert payload["metrics"][10]["F1"] == 0.4          # validation point
    assert set(payload["curve"]) == {"0.2", "0.5"}      # both points kept
    from integrations.locata import adapter
    assert payload["num_params"] == adapter.count_parameters(model)


def test_negative_mse_boundary_matches_the_absolute_rule():
    """A negative-MSE score equal to the threshold is accepted."""
    from ssl_doa_baselines.api import _filter_track_candidates

    model = _ipdnet()
    target = model.tracks.target_from_angles([[40.0, 90.0]])
    perturbed = target[None].clone()
    perturbed[0, :, :, 0] += 0.05             # min MSE moves above zero
    scores, azimuth, elevation = model.tracks.decode_candidates(
        perturbed, "neg_mse")
    score = float(scores[0, 0])
    assert -score > 0.0                        # an actual raw MSE
    dets = _filter_track_candidates(
        scores, azimuth, elevation, threshold=score, max_sources=3)
    assert any(d["score"] == score for d in dets[0])
