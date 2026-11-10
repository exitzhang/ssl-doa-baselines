#!/usr/bin/env python
"""External runner for the three baselines on LOCATA data.

Subcommands::

    smoke  --config CFG --locata-root PATH
    train  --config CFG --data-config PATH --output DIR
    eval   --config CFG --data-config PATH --checkpoint CKPT --split dicit
           [--validation validation.json]

All training and evaluation goes through ``adapter.py``; the shared scorer,
peak decoder, trainer and datasets come from the LOCATA checkout passed with
``--locata-root`` (or ``LOCATA_ROOT``).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from ssl_doa_baselines.api import build_model, load_model, predict_native  # noqa: E402
from ssl_doa_baselines.config import deep_merge, load_configs, model_section  # noqa: E402

from integrations.locata import adapter  # noqa: E402


def _load_cfg(args) -> Dict[str, Any]:
    cfg = load_configs([args.config, args.data_config])
    if getattr(args, "seed", None) is not None:
        cfg.setdefault("run", {})["seed"] = args.seed
        cfg.setdefault("train", {})["seed"] = args.seed
    if getattr(args, "epochs", None) is not None:
        cfg.setdefault("train", {})["epochs"] = args.epochs
    if getattr(args, "max_train_batches", None) is not None:
        cfg.setdefault("train", {})["max_train_batches"] = args.max_train_batches
    return cfg


def _resolve_mic_positions(cfg: Dict[str, Any]) -> Optional[List[List[float]]]:
    data = cfg.get("data", {})
    if "mic_positions" in data:
        return data["mic_positions"]
    source = data.get("mic_positions_source")
    if not source:
        return None
    meta_path = Path(source)
    if meta_path.is_dir():
        meta_path = meta_path / "dataset_meta.json"
    meta = json.loads(meta_path.read_text())
    indices = list(data.get("mic_indices", [5, 6, 8]))
    return np.asarray(meta["mic_pos"], dtype=np.float64)[indices].tolist()


def _model_from_cfg(cfg: Dict[str, Any], device: str = "cpu"):
    name, kwargs = None, {}
    for key, value in model_section(cfg).items():
        if key in ("name", "model_name"):
            name = value
        else:
            kwargs[key] = value
    if name is None:
        raise SystemExit("[run] config has no [model].name")
    if name in ("icocnn", "srp_dnn") and "mic_positions" not in kwargs:
        positions = _resolve_mic_positions(cfg)
        if positions is None:
            raise SystemExit(
                f"[run] {name} needs [data].mic_positions or "
                "[data].mic_positions_source"
            )
        kwargs["mic_positions"] = positions
    model = build_model(name, kwargs)
    return model.to(device)


def _data_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return dict(cfg.get("data", {}))


def _train_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return dict(cfg.get("train", {}))


def _run_dir(cfg: Dict[str, Any], args, kind: str) -> Path:
    if args.output:
        return Path(args.output)
    base = Path(cfg.get("run", {}).get("output_root", "runs"))
    name = model_section(cfg).get("name", "model")
    stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    return base / f"{name}_{kind}_{stamp}"


def _resolved_config(cfg: Dict[str, Any], *, checkpoint: Optional[str] = None) -> Dict[str, Any]:
    return {
        "model": dict(model_section(cfg)),
        "train": _train_cfg(cfg),
        "data": _data_cfg(cfg),
        "decode": dict(cfg.get("decode", {})),
        "run": dict(cfg.get("run", {})),
        "checkpoint": checkpoint,
    }


def _device(args) -> str:
    if args.device:
        return args.device
    if torch.cuda.is_available():
        return f"cuda:{args.device_index}"
    return "cpu"


# ---------------------------------------------------------------------------
# smoke
# ---------------------------------------------------------------------------

def cmd_smoke(args) -> int:
    cfg = _load_cfg(args)
    device = _device(args)
    model = _model_from_cfg(cfg, device)
    q = int(model_section(cfg).get("output_dim", model_section(cfg).get("n_slots", 181)))
    n_mics = int(model_section(cfg).get("num_mics", model_section(cfg).get("n_mics", 3)))
    waveform = torch.randn(2, 16000, n_mics, device=device)
    with torch.no_grad():
        out = predict_native(model, waveform, device=device)

    summary: Dict[str, Any] = {
        "model": model.model_name,
        "device": device,
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "outputs": {},
    }
    for key, value in out.items():
        if torch.is_tensor(value):
            summary["outputs"][key] = list(value.shape)
    print(json.dumps(summary, indent=2))

    # native decode must produce one list per clip
    dets = adapter_decode_smoke(model, waveform)
    print(json.dumps({"detections": dets}, indent=2))

    # checkpoint round-trip
    out_dir = _run_dir(cfg, args, "smoke")
    out_dir.mkdir(parents=True, exist_ok=True)
    from ssl_doa_baselines.api import save_checkpoint

    ckpt = out_dir / "checkpoint.pth"
    save_checkpoint(ckpt, model, config={"model": dict(model.config,
                                                       name=model.model_name)})
    reloaded = load_model(str(ckpt), device=device)
    with torch.no_grad():
        again = predict_native(reloaded, waveform, device=device)
    for key, value in out.items():
        if torch.is_tensor(value):
            assert torch.allclose(value, again[key], atol=1e-5), f"reload mismatch: {key}"
    print(f"[smoke] checkpoint round-trip OK -> {ckpt}")
    return 0


def adapter_decode_smoke(model, waveform):
    from ssl_doa_baselines.api import predict

    cfg = {"max_sources": 3}
    if model.model_name == "tf_mamba":
        cfg.update({"threshold_ratio": 0.1, "neighborhood_size": 3,
                    "min_peak_distance": 3, "threshold_abs": 0.0,
                    "silence_max_thresh": 0.0})
    else:
        cfg.update({"threshold_abs": 0.0})
    return predict(model, waveform, decode=cfg)


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

def cmd_train(args) -> int:
    cfg = _load_cfg(args)
    data_cfg = _data_cfg(cfg)
    train_cfg = _train_cfg(cfg)
    decode_cfg = dict(cfg.get("decode", {}))
    device = _device(args)
    if device.startswith("cuda"):
        torch.cuda.set_device(args.device_index)

    locata_root = adapter.ensure_locata(args.locata_root)
    out_dir = _run_dir(cfg, args, "train")
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = int(train_cfg.get("seed", cfg.get("run", {}).get("seed", 42)))
    adapter.seed_everything(seed)

    model = _model_from_cfg(cfg, device)
    print(f"[train] model={model.model_name} params="
          f"{sum(p.numel() for p in model.parameters()):,} device={device}")

    if model.model_name == "tf_mamba":
        ckpt = adapter.train_tf_mamba(
            model, data_cfg, train_cfg, device_index=args.device_index,
            out_dir=str(out_dir), seed=seed,
            log_dir=str(out_dir / "tensorboard"),
            log_path=str(out_dir / "train.log"),
        )
    else:
        ckpt = adapter.train_native(model, data_cfg, train_cfg,
                                    out_dir=str(out_dir), seed=seed, device=device)

    config = _resolved_config(cfg, checkpoint=ckpt)
    adapter.write_json(out_dir / "config.json", {
        "config": config,
        "locata": adapter.git_state(str(locata_root)),
        "baselines": adapter.git_state(str(REPO_ROOT)),
        "seed": seed,
    })
    print(f"[train] checkpoint -> {ckpt}")

    # Lock the operating point on the simulated validation set.
    val_result = _evaluate(model, cfg, split="val", device=device,
                           device_index=args.device_index,
                           locata_root=str(locata_root))
    wp = adapter.lock_working_point(val_result["metrics_by_key"],
                                    int(decode_cfg.get("source_column", 10)))
    adapter.write_json(out_dir / "validation.json", {
        "locked": wp,
        "curve": val_result["metrics_by_key"],
        "r_at_far_3": adapter.interpolate_at_far(
            val_result["metrics_by_key"],
            float(decode_cfg.get("far_target", 0.03)),
            int(decode_cfg.get("source_column", 10)),
        ),
        "frames": val_result["frames"],
        "source_count_dist": val_result["source_count_dist"],
    })
    print(f"[train] validation locked at {wp['key']} (F1@10={wp['F1']:.4f})")
    adapter.write_json(out_dir / "train_result.json", {
        "checkpoint": str(Path(ckpt).resolve()),
        "validation": str((out_dir / "validation.json").resolve()),
        "locked": wp,
    })
    return 0


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------

def _evaluate(model, cfg: Dict[str, Any], *, split: str, device: str,
              device_index: int, locata_root: str) -> Dict[str, Any]:
    data_cfg = _data_cfg(cfg)
    decode_cfg = dict(cfg.get("decode", {}))
    batch_size = int(data_cfg.get("eval_batch_size", 4))
    workers = int(data_cfg.get("eval_num_workers", 4))
    max_sources = int(decode_cfg.get("max_sources", 3))
    acc = tuple(int(v) for v in decode_cfg.get("acc_thresholds", [5, 10, 15]))
    silence = float(decode_cfg.get(
        "silence_max_thresh_val" if split == "val" else "silence_max_thresh_dicit",
        0.0 if split == "val" else 0.1))

    if split == "val":
        ds = adapter.build_angle_dataset(data_cfg["val_root"], data_cfg, use_val=True)
        loader = adapter.build_loader(ds, batch_size=batch_size, num_workers=workers,
                                      shuffle=False)
    else:
        loader = adapter.build_dicit_loader(data_cfg["dicit_root"], data_cfg,
                                            batch_size=batch_size, num_workers=workers)

    if model.model_name == "tf_mamba":
        spectrum = adapter.eval_spectrum(
            model, loader, device=device,
            ratios=decode_cfg.get("ratio_thresholds", [0.01, 0.02, 0.05, 0.075, 0.1,
                                                        0.15, 0.2, 0.25, 0.3, 0.4,
                                                        0.5, 0.65, 0.8]),
            acc_thresholds=acc, max_sources=max_sources,
            silence_max_thresh=silence,
        )
        # run_forward_cache reports ms per batch; normalise to per clip so the
        # number is comparable with the candidate-path models.
        spectrum["forward_ms_per_frame"] = (
            float(spectrum["forward_ms_per_batch"]) / float(batch_size))
        return spectrum

    cache = adapter.cache_candidates(model, loader, device=device,
                                     max_sources=max_sources,
                                     frame_index=int(decode_cfg.get("frame_index", -1)))
    metrics = adapter.sweep_candidates(
        cache, decode_cfg.get("score_thresholds",
                              [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3,
                               0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9]),
        max_sources=max_sources, acc_thresholds=acc,
    )
    cache["metrics_by_key"] = metrics
    cache["source_count_dist"] = {
        str(n): int(sum(1 for g in cache["gts"] if len(g) == n))
        for n in (0, 1, 2, 3)
    }
    cache["forward_ms_per_batch"] = cache.get("forward_ms_per_frame", 0.0) * batch_size
    return cache


def _decoded(model, cache: Dict[str, Any], key: str, decode_cfg, split: str,
             max_sources: int):
    if cache["kind"] == "spectrum":
        return adapter.decoded_spectrum_at_ratio(
            cache, float(key), max_sources=max_sources,
            silence_max_thresh=float(decode_cfg.get(
                "silence_max_thresh_val" if split == "val" else "silence_max_thresh_dicit",
                0.0 if split == "val" else 0.1)),
        )
    return adapter.decoded_at_threshold(cache, float(key), max_sources=max_sources)


def _plain_angles(rows) -> List[List[float]]:
    """Coerce decoded/label angle sets to plain floats so JSON has no numpy."""
    return [[float(a), float(e)] for a, e in rows]


def _eval_payload(*, model, result: Dict[str, Any], pred_all, locked: dict,
                  cfg: Dict[str, Any], decode_cfg: Dict[str, Any], split: str,
                  acc, checkpoint: str, decode_s: float,
                  locata_root: str) -> Dict[str, Any]:
    """Assemble ``eval_<split>.json``; kept separate so it can be tested."""
    far_target = float(decode_cfg.get("far_target", 0.03))
    frames = max(int(result["frames"]), 1)
    return {
        "model": model.model_name,
        "checkpoint": str(Path(checkpoint).resolve()),
        "split": split,
        "frames": result["frames"],
        "source_count_dist": result["source_count_dist"],
        "working_point": locked,
        "metrics": locked,
        "by_source_count": adapter.score_by_source_count(
            result["gts"], pred_all, acc_thresholds=acc),
        "curve": result["metrics_by_key"],
        "r_at_far_3": adapter.interpolate_at_far(
            result["metrics_by_key"], far_target,
            int(decode_cfg.get("source_column", 10)),
        ),
        # Per-frame output at the locked point, for reviewing individual frames.
        "predictions": {
            "gt": [_plain_angles(g) for g in result["gts"]],
            "pred": [_plain_angles(p) for p in pred_all],
        },
        "forward_ms_per_frame": (
            None if result.get("forward_ms_per_frame") is None
            else float(result["forward_ms_per_frame"])),
        "forward_ms_per_batch": result.get("forward_ms_per_batch"),
        "decode_ms_per_frame": 1000.0 * float(decode_s) / frames,
        "elapsed_s": result.get("elapsed_s"),
        "config": _resolved_config(cfg, checkpoint=str(Path(checkpoint).resolve())),
        "locata": adapter.git_state(locata_root),
        "baselines": adapter.git_state(str(REPO_ROOT)),
    }


def cmd_eval(args) -> int:
    cfg = _load_cfg(args)
    decode_cfg = dict(cfg.get("decode", {}))
    acc = tuple(int(v) for v in decode_cfg.get("acc_thresholds", [5, 10, 15]))
    max_sources = int(decode_cfg.get("max_sources", 3))
    device = _device(args)
    locata_root = adapter.ensure_locata(args.locata_root)

    model = load_model(args.checkpoint, config=cfg, device=device)
    split = args.split
    result = _evaluate(model, cfg, split=split, device=device,
                       device_index=args.device_index, locata_root=str(locata_root))

    if args.validation:
        val = json.loads(Path(args.validation).read_text())
        locked = val["locked"]
    else:
        locked = adapter.lock_working_point(result["metrics_by_key"],
                                            int(decode_cfg.get("source_column", 10)))

    key = str(locked["key"])
    t_decode = time.time()
    pred_all = _decoded(model, result, key, decode_cfg, split, max_sources)
    decode_s = time.time() - t_decode
    payload = _eval_payload(
        model=model, result=result, pred_all=pred_all, locked=locked, cfg=cfg,
        decode_cfg=decode_cfg, split=split, acc=acc,
        checkpoint=args.checkpoint, decode_s=decode_s,
        locata_root=str(locata_root),
    )
    out = Path(args.output) if args.output else Path(args.checkpoint).parent / f"eval_{split}.json"
    adapter.write_json(out, payload)
    m = locked
    print(f"[eval] {model.model_name} {split}: F1@10={m['F1']:.4f} "
          f"P={m['P']:.4f} R={m['R']:.4f} FAR={m['FAR']:.4f} MAE={m['MAE']:.2f} "
          f"-> {out}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="locata-runner")
    p.add_argument("--locata-root", default=os.environ.get("LOCATA_ROOT"))
    p.add_argument("--config", required=True)
    p.add_argument("--data-config", default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--device-index", type=int, default=0)
    p.add_argument("--seed", type=int, default=None)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("smoke")
    s.set_defaults(func=cmd_smoke)

    t = sub.add_parser("train")
    t.add_argument("--epochs", type=int, default=None)
    t.add_argument("--max-train-batches", type=int, default=None)
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("eval")
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--split", choices=["val", "dicit"], default="dicit")
    e.add_argument("--validation", default=None,
                   help="validation.json from train; locks the working point")
    e.set_defaults(func=cmd_eval)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not args.locata_root:
        raise SystemExit("--locata-root (or LOCATA_ROOT) is required")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
