"""Command line entry points: ``predict`` and ``fit``.

These operate on plain wav files / a JSONL manifest and do not import LOCATA.
The LOCATA integration lives in ``integrations/locata/run.py``.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from .api import build_model, load_model, predict
from .config import deep_merge, load_toml, model_section
from .training import fit


def _parse_channels(text: Optional[str]):
    if not text:
        return None
    return [int(v) for v in text.split(",") if v.strip() != ""]


_NEW_MODEL_NAMES = ("ipdnet", "ipdnet2_small")


def _expected_sample_rate(model_name: Optional[str]) -> Optional[int]:
    """Models that resample internally expect a 48 kHz waveform."""
    return 48000 if str(model_name).lower() in _NEW_MODEL_NAMES else None


def _read_audio(path: str, channels: Optional[List[int]], normalize: bool = True,
                expected_sample_rate: Optional[int] = None):
    import soundfile as sf

    audio, sample_rate = sf.read(path, always_2d=True)  # (T, C)
    if (expected_sample_rate is not None
            and int(sample_rate) != int(expected_sample_rate)):
        raise ValueError(
            f"{path}: sample rate {sample_rate} Hz, expected "
            f"{expected_sample_rate} Hz")
    if channels is not None:
        audio = audio[:, channels]
    if normalize and audio.size:
        peak = float(np.abs(audio).max())
        if peak > 0:
            audio = audio / peak
    return audio.astype(np.float32)


def _load_entry_config(paths: List[Optional[str]]) -> Dict:
    cfg: Dict = {}
    for p in paths:
        if p:
            cfg = deep_merge(cfg, load_toml(p))
    return cfg


def cmd_predict(args) -> int:
    cfg = _load_entry_config([args.config, args.data_config])
    model = load_model(args.checkpoint, config=cfg, device=args.device,
                       strict=not args.allow_legacy)
    audio = _read_audio(
        args.audio, _parse_channels(args.channels),
        expected_sample_rate=_expected_sample_rate(
            getattr(model, "model_name", None)))
    decode = dict(cfg.get("decode", {}))
    if args.max_sources is not None:
        decode["max_sources"] = args.max_sources
    if args.threshold_abs is not None:
        decode["threshold_abs"] = args.threshold_abs
    if args.threshold_ratio is not None:
        decode["threshold_ratio"] = args.threshold_ratio
    detections = predict(model, audio, decode=decode, device=args.device)[0]
    result = {
        "audio": str(Path(args.audio).resolve()),
        "model": getattr(model, "model_name", None),
        "detections": detections,
    }
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(text)
    else:
        print(text)
    return 0


class ManifestDataset(torch.utils.data.Dataset):
    """JSONL manifest with ``{"audio": path, "angles": [[az, el], ...]}``."""

    def __init__(self, manifest: str, channels: Optional[List[int]] = None,
                 normalize: bool = True,
                 expected_sample_rate: Optional[int] = None):
        self.root = Path(manifest).resolve().parent
        self.channels = channels
        self.normalize = normalize
        self.expected_sample_rate = expected_sample_rate
        self.entries = []
        with Path(manifest).open() as f:
            for line in f:
                line = line.strip()
                if line:
                    self.entries.append(json.loads(line))
        if not self.entries:
            raise ValueError(f"Empty manifest: {manifest}")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        entry = self.entries[idx]
        audio_path = Path(entry["audio"])
        if not audio_path.is_absolute():
            audio_path = self.root / audio_path
        audio = _read_audio(str(audio_path), self.channels, self.normalize,
                            expected_sample_rate=self.expected_sample_rate)
        angles = np.asarray(entry.get("angles", []), dtype=np.float32).reshape(-1, 2)
        return {"audio": torch.from_numpy(audio), "angles": angles}


def _collate(batch):
    return {
        "audio": torch.stack([item["audio"] for item in batch], dim=0),
        "angles": [item["angles"] for item in batch],
    }


def cmd_fit(args) -> int:
    cfg = _load_entry_config([args.config])
    mcfg = dict(model_section(cfg))
    name = args.model or mcfg.pop("name", None) or mcfg.pop("model_name", None)
    if name is None:
        raise SystemExit("fit requires --model or [model].name in --config")
    mcfg.pop("name", None)
    model = build_model(name, mcfg)

    channels = _parse_channels(args.channels)
    expected_rate = _expected_sample_rate(name)
    train_ds = ManifestDataset(args.manifest, channels,
                               expected_sample_rate=expected_rate)
    val_ds = (ManifestDataset(args.val_manifest, channels,
                              expected_sample_rate=expected_rate)
              if args.val_manifest else None)
    train_cfg = cfg.get("train", {})
    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=int(args.batch_size or train_cfg.get("batch_size", 8)),
        shuffle=True, num_workers=args.workers, collate_fn=_collate,
    )
    val_loader = None
    if val_ds is not None:
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=int(args.batch_size or train_cfg.get("batch_size", 8)),
            shuffle=False, num_workers=args.workers, collate_fn=_collate,
        )
    fit(
        model, loader, val_loader,
        epochs=int(args.epochs or train_cfg.get("epochs", 100)),
        device=args.device,
        lr=float(train_cfg.get("lr", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        optimizer=str(train_cfg.get("optimizer", "adamw")),
        scheduler=train_cfg.get("scheduler", "cosine"),
        scheduler_gamma=float(train_cfg.get("scheduler_gamma", 0.975)),
        eta_min=float(train_cfg.get("eta_min", 1e-6)),
        amp=bool(train_cfg.get("amp", args.device.startswith("cuda"))),
        grad_clip=float(train_cfg.get("grad_clip", 1.0)),
        save_dir=args.output,
        eval_every=int(train_cfg.get("eval_every", 1)),
        patience=train_cfg.get("patience"),
        config=cfg,
        max_train_batches=args.max_train_batches,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ssl-doa-baselines")
    parser.add_argument("--version", action="version", version="0.1.0")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("predict", help="decode one wav with a checkpoint")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--audio", required=True)
    p.add_argument("--channels", default=None, help="comma-separated 0-based channel ids")
    p.add_argument("--config", default=None, help="model/decode TOML")
    p.add_argument("--data-config", default=None, help="machine-specific TOML")
    p.add_argument("--output", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-sources", type=int, default=None)
    p.add_argument("--threshold-abs", type=float, default=None)
    p.add_argument("--threshold-ratio", type=float, default=None)
    p.add_argument("--allow-legacy", action="store_true",
                   help="accept checkpoints without an embedded model_config")
    p.set_defaults(func=cmd_predict)

    f = sub.add_parser("fit", help="train on a JSONL manifest")
    f.add_argument("--config", required=True)
    f.add_argument("--manifest", required=True)
    f.add_argument("--val-manifest", default=None)
    f.add_argument("--model", default=None,
                   choices=["tf_mamba", "icocnn", "srp_dnn", "ipdnet",
                            "ipdnet2_small"])
    f.add_argument("--channels", default=None)
    f.add_argument("--output", required=True)
    f.add_argument("--epochs", type=int, default=None)
    f.add_argument("--batch-size", type=int, default=None)
    f.add_argument("--workers", type=int, default=0)
    f.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    f.add_argument("--max-train-batches", type=int, default=None)
    f.set_defaults(func=cmd_fit)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
