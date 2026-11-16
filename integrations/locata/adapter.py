"""LOCATA adapter: data, labels, training and the shared scorer.

The adapter is the only place that imports LOCATA.  It deliberately reuses
LOCATA's own components instead of re-implementing them:

* ``data.audio_dataset.AudioDataset`` for simulated training / validation;
* ``data.factory.create_locata_dataset`` for DICIT frames;
* ``engine.trainer.Trainer`` for TF-Mamba (same AMP / schedule / early stop);
* ``engine.sweep_eval`` and ``engine.matcher`` for every reported metric.

Angle convention: LOCATA simulated metadata stores elevation from the
horizontal plane, DICIT metadata stores zenith elevation.  Targets handed to
the baselines are always zenith (see ``geometry.py``); the conversion happens
in ``_to_zenith`` and is recorded in the resolved config.
"""

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ssl_doa_baselines.api import build_model, save_checkpoint
from ssl_doa_baselines.geometry import vectors_to_angles
from ssl_doa_baselines.training import fit

AZ_RANGE = (0.0, 180.0)
ANGLE_KEYS = ("azimuth_deg", "elevation_deg")
DEFAULT_MIC_INDICES = (5, 6, 8)
ARRAY_ATOL = 1e-6


# ---------------------------------------------------------------------------
# Fixed-array data contract
# ---------------------------------------------------------------------------

def resolve_mic_indices(data_cfg: Dict[str, Any], *,
                        n_mics: Optional[int] = None) -> List[int]:
    """Validate the raw ``mic_indices`` that will be passed to the dataset.

    Checks the count (against ``n_mics`` when known), non-negativity and
    distinctness.  Bounds are split-specific and are checked where the
    geometry/metadata is available.  The order is preserved: it is the channel
    order of the fixed array, not a set.
    """
    raw = data_cfg.get("mic_indices", list(DEFAULT_MIC_INDICES))
    if isinstance(raw, (str, bytes)) or isinstance(raw, (int, np.integer)):
        raise TypeError(f"[data].mic_indices must be a list of integers, got {raw!r}")
    try:
        values = list(raw)
    except TypeError:
        raise TypeError(f"[data].mic_indices must be a list of integers, got {raw!r}")
    indices: List[int] = []
    for value in values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)):
            raise TypeError(
                f"[data].mic_indices must contain integers, got {value!r}")
        indices.append(int(value))
    if not indices:
        raise ValueError("[data].mic_indices is empty")
    if n_mics is not None and len(indices) != int(n_mics):
        raise ValueError(
            f"[data].mic_indices {indices} selects {len(indices)} channels, "
            f"the model expects {int(n_mics)}")
    if any(index < 0 for index in indices):
        raise ValueError(f"[data].mic_indices {indices} has a negative index")
    if len(set(indices)) != len(indices):
        raise ValueError(f"[data].mic_indices {indices} repeats a channel")
    return indices


def load_mic_positions(source: str) -> np.ndarray:
    """Read ``mic_pos`` from a ``dataset_meta.json`` file or its directory."""
    path = Path(source)
    if path.is_dir():
        path = path / "dataset_meta.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found (needed for array geometry)")
    meta = json.loads(path.read_text())
    positions = meta.get("mic_pos")
    if positions is None:
        raise ValueError(f"{path}: no 'mic_pos' entry")
    arr = np.asarray(positions, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{path}: mic_pos must be (C, 3), got {arr.shape}")
    return arr


def selected_positions(data_cfg: Dict[str, Any],
                       split: str) -> Tuple[str, np.ndarray, List[int]]:
    """Geometry source label, full positions and validated indices for a split.

    Simulated train/val use ``<root>/dataset_meta.json``.  DICIT carries no
    coordinates, so it falls back to the configured training geometry
    (``[data].mic_positions_source``, or an explicit ``[data].mic_positions``).
    """
    indices = resolve_mic_indices(data_cfg)
    if split in ("train", "val"):
        root = Path(data_cfg[f"{split}_root"])
        source = str(root / "dataset_meta.json")
        positions = load_mic_positions(source)
    else:
        if data_cfg.get("mic_positions") is not None:
            source = "data.mic_positions"
            positions = np.asarray(data_cfg["mic_positions"], dtype=np.float64)
        elif data_cfg.get("mic_positions_source"):
            source = str(data_cfg["mic_positions_source"])
            positions = load_mic_positions(source)
        else:
            raise ValueError(
                "DICIT evaluation needs [data].mic_positions or "
                "[data].mic_positions_source for the array geometry")
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"{source}: positions must be (C, 3), got {positions.shape}")
    if max(indices) >= len(positions):
        raise ValueError(
            f"{source}: mic_indices {indices} exceed {len(positions)} mics")
    return source, positions, indices


def array_pair_vectors(positions, ref_mic: int = 0) -> np.ndarray:
    """Reference-mic-relative pair vectors ``(P, 3)``, translation invariant.

    The direction matters: ``[5, 6, 8]`` and ``[8, 6, 5]`` produce opposite
    vectors, while shifting the whole array leaves these differences intact.
    """
    mic = np.asarray(positions, dtype=np.float64)
    if mic.ndim != 2 or mic.shape[1] != 3:
        raise ValueError(f"positions must be (C, 3), got {mic.shape}")
    if mic.shape[0] < 2:
        raise ValueError("a fixed array needs at least two microphones")
    ref = int(ref_mic)
    if not 0 <= ref < mic.shape[0]:
        raise ValueError(f"ref_mic={ref} out of range for {mic.shape[0]} mics")
    return np.stack([mic[j] - mic[ref] for j in range(mic.shape[0]) if j != ref])


def _compare_pair_vectors(expected: np.ndarray, got: np.ndarray) -> bool:
    return bool(np.allclose(expected, got, rtol=0.0, atol=ARRAY_ATOL))


def _model_array(model) -> Tuple[np.ndarray, int, int]:
    config = dict(getattr(model, "config", {}) or {})
    positions = config.get("mic_positions")
    if positions is None:
        raise ValueError(
            f"{getattr(model, 'model_name', model.__class__.__name__)} has no "
            "mic_positions in its model config; the array cannot be verified")
    arr = np.asarray(positions, dtype=np.float64)
    n_mics = int(config.get("n_mics", arr.shape[0] if arr.ndim == 2 else 0))
    ref_mic = int(config.get("ref_mic", 0))
    return arr, n_mics, ref_mic


def check_fixed_array(model, data_cfg: Dict[str, Any], *,
                      splits: Tuple[str, ...] = ("train", "val")) -> Dict[str, Any]:
    """Compare the model's reference-pair geometry with one or more splits.

    Called before optimization.  Translation of the whole array is allowed;
    a permuted channel order or a changed microphone spacing is rejected.
    """
    positions, n_mics, ref_mic = _model_array(model)
    indices = resolve_mic_indices(data_cfg, n_mics=n_mics)
    expected = array_pair_vectors(positions, ref_mic)
    record: Dict[str, Any] = {
        "n_mics": n_mics,
        "mic_indices": indices,
        "mic_positions": positions.tolist(),
        "ref_mic": ref_mic,
        "splits": {},
    }
    for split in splits:
        source, dataset_positions, _ = selected_positions(data_cfg, split)
        selected = dataset_positions[indices]
        got = array_pair_vectors(selected, ref_mic)
        if not _compare_pair_vectors(expected, got):
            raise ValueError(
                f"{split} array ({source}) does not match the model's fixed "
                f"array (ref_mic={ref_mic}, mic_indices={indices}): "
                f"reference pair vectors differ. Use the training channel "
                f"order and geometry; expected {expected.tolist()}, got "
                f"{got.tolist()}")
        record["splits"][split] = {
            "source": source, "mic_positions": selected.tolist()}
    return record


def check_eval_array(model, checkpoint_config: Dict[str, Any],
                     data_cfg: Dict[str, Any], *, split: str) -> Dict[str, Any]:
    """Verify the checkpoint's channel order and paired geometry for a split.

    The checkpoint records ``mic_indices`` / ``mic_positions`` / ``ref_mic`` in
    ``config['data']``.  A reversed or permuted loader order is rejected even
    when the geometry source is the same.
    """
    positions, n_mics, ref_mic = _model_array(model)
    loader_indices = resolve_mic_indices(data_cfg, n_mics=n_mics)
    recorded = dict(checkpoint_config or {}).get("data") or {}
    ckpt_indices = recorded.get("mic_indices")
    if ckpt_indices is None:
        raise ValueError(
            "checkpoint has no config['data']['mic_indices']; the channel "
            "order cannot be verified. Re-export the checkpoint from a run "
            "that recorded the fixed array, or retrain with the current "
            "train path.")
    ckpt_indices = [int(value) for value in ckpt_indices]
    if ckpt_indices != loader_indices:
        raise ValueError(
            f"channel order mismatch: checkpoint {ckpt_indices} vs loader "
            f"{loader_indices}. Evaluate with the training microphone order "
            "from the data config.")
    ckpt_positions = recorded.get("mic_positions")
    if ckpt_positions is not None:
        ckpt_positions = np.asarray(ckpt_positions, dtype=np.float64)
        if not _compare_pair_vectors(array_pair_vectors(positions, ref_mic),
                                     array_pair_vectors(ckpt_positions, ref_mic)):
            raise ValueError(
                "checkpoint records a different mic geometry than its own "
                "model_config; the checkpoint is inconsistent")
    source, dataset_positions, _ = selected_positions(data_cfg, split)
    selected = dataset_positions[loader_indices]
    expected = array_pair_vectors(positions, ref_mic)
    got = array_pair_vectors(selected, ref_mic)
    if not _compare_pair_vectors(expected, got):
        raise ValueError(
            f"{split} array ({source}) does not match the checkpoint geometry: "
            f"reference pair vectors differ (expected {expected.tolist()}, got "
            f"{got.tolist()}). Check mic_positions_source and mic_indices.")
    return {
        "n_mics": n_mics,
        "mic_indices": loader_indices,
        "mic_positions": positions.tolist(),
        "ref_mic": ref_mic,
        "geometry_source": source,
        "dataset_mic_positions": selected.tolist(),
        "geometry_matches": True,
    }


def model_n_mics(model) -> Optional[int]:
    """Configured microphone count, if the model exposes one."""
    config = dict(getattr(model, "config", {}) or {})
    for key in ("n_mics", "num_mics"):
        if key in config:
            return int(config[key])
    return None


def attach_array_record(data_cfg: Dict[str, Any],
                        record: Dict[str, Any]) -> Dict[str, Any]:
    """Copy a data config with the resolved array written into it."""
    data = dict(data_cfg)
    data["mic_indices"] = [int(v) for v in record["mic_indices"]]
    data["mic_positions"] = [[float(x) for x in row]
                             for row in record["mic_positions"]]
    data["ref_mic"] = int(record["ref_mic"])
    return data


# ---------------------------------------------------------------------------
# LOCATA import boundary
# ---------------------------------------------------------------------------

def ensure_locata(root: str) -> Path:
    root = Path(root).resolve()
    required = ["engine/sweep_eval.py", "engine/trainer.py", "data/audio_dataset.py",
                "data/factory.py", "losses/doa_losses.py"]
    missing = [p for p in required if not (root / p).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Not a LOCATA checkout: {root} (missing {', '.join(missing)})"
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def locata_symbols():
    from data import AudioDataset, create_locata_dataset
    from engine import Trainer, angle_collate, run_forward_cache, eval_at_ratio, score_decoded
    from engine.heatmap_creator import DOASpectrumHeatmapCreatorV1
    from losses.doa_losses import mse_kl_loss

    return {
        "AudioDataset": AudioDataset,
        "create_locata_dataset": create_locata_dataset,
        "Trainer": Trainer,
        "angle_collate": angle_collate,
        "run_forward_cache": run_forward_cache,
        "eval_at_ratio": eval_at_ratio,
        "score_decoded": score_decoded,
        "heatmap_creator": DOASpectrumHeatmapCreatorV1,
        "mse_kl_loss": mse_kl_loss,
    }


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

def _to_zenith(label: torch.Tensor, source: str = "sim") -> torch.Tensor:
    """Convert angle labels to zenith elevation, preserving shape ``(N, 2)``."""
    if source == "sim":
        out = label.clone()
        if out.numel():
            out[:, 1] = 90.0 - out[:, 1]
        return out
    return label


class _ZenithDataset(Dataset):
    """Wrap an angle-mode LOCATA dataset and convert simulation labels."""

    def __init__(self, base: Dataset, source: str = "sim"):
        self.base = base
        self.source = source

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        audio, label = self.base[idx]
        return audio, _to_zenith(label.float(), self.source)


def angle_batch_collate(batch):
    """Stack audio, keep variable-length labels as a list."""
    audios = torch.stack([item[0] for item in batch], dim=0)
    labels = [item[1] for item in batch]
    return audios, labels


# ---------------------------------------------------------------------------
# Datasets / loaders
# ---------------------------------------------------------------------------

def _noise_kwargs(data_cfg: Dict[str, Any], use_val: bool) -> Dict[str, Any]:
    return {
        "enable_noise": bool(data_cfg.get("enable_noise", True)),
        "noise_snr": tuple(data_cfg.get("val_noise_snr" if use_val else "noise_snr",
                                          (5.0, 30.0))),
        "noise_mode_weights": tuple(data_cfg.get("noise_mix_weights", (1.0, 0.0, 1.0))),
        "diffuse_noise_path": data_cfg.get("diffuse_noise_dir"),
        "real_world_noise_path": data_cfg.get("real_noise_dir"),
    }


def build_angle_dataset(root: str, data_cfg: Dict[str, Any], *,
                        use_val: bool = False,
                        n_mics: Optional[int] = None) -> Dataset:
    sym = locata_symbols()
    base = sym["AudioDataset"](
        root_dir=root,
        split="all",
        label_mode="angle",
        use_aug=False if use_val else bool(data_cfg.get("use_aug", True)),
        mic_indices=resolve_mic_indices(data_cfg, n_mics=n_mics),
        normalize=True,
        **_noise_kwargs(data_cfg, use_val),
    )
    return _ZenithDataset(base, source=data_cfg.get("label_source", "sim"))


def build_heatmap_dataset(root: str, data_cfg: Dict[str, Any], *,
                          use_val: bool = False,
                          n_mics: Optional[int] = None) -> Dataset:
    sym = locata_symbols()
    creator = sym["heatmap_creator"](
        az_range_deg=AZ_RANGE,
        az_bins_1d=int(data_cfg.get("out_bins", 181)),
        sigma_pixels=float(data_cfg.get("label_sigma", 1.5)),
    )
    return sym["AudioDataset"](
        root_dir=root,
        split="all",
        label_mode="heatmap_1d",
        heatmap_creator=creator,
        use_aug=False if use_val else bool(data_cfg.get("use_aug", True)),
        mic_indices=resolve_mic_indices(data_cfg, n_mics=n_mics),
        normalize=True,
        **_noise_kwargs(data_cfg, use_val),
    )


def build_loader(dataset: Dataset, *, batch_size: int, num_workers: int,
                 shuffle: bool, collate=None, drop_last: bool = False) -> DataLoader:
    if collate is None:
        collate = angle_batch_collate
    elif collate is False:
        collate = None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        collate_fn=collate,
    )


def check_audio_contract(data_cfg: Dict[str, Any], *, split: str) -> Dict[str, Any]:
    """Verify the 48 kHz / 16,000-sample / 3-mic clip contract.

    Simulated splits are checked through ``dataset_meta.json``; DICIT frames
    are checked from WAV headers (all clips must share the rate and length),
    and every metadata file must have a WAV.  Called before IPDnet training
    and DICIT evaluation; header reads are cheap (~1 s for 926 frames).
    """
    import soundfile as sf

    target = int(data_cfg.get("target_num_samples", 16000))
    indices = resolve_mic_indices(data_cfg)

    if split in ("train", "val"):
        root = Path(data_cfg[f"{split}_root"])
        meta_path = root / "dataset_meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"{meta_path} not found")
        meta = json.loads(meta_path.read_text())
        rate = int(meta.get("fs", 0))
        if rate != 48000:
            raise ValueError(f"{root}: fs={rate}, expected 48000")
        num = int(meta.get("mic_output_num_samples",
                            meta.get("source_num_samples", target)))
        if num != target:
            raise ValueError(
                f"{root}: mic_output_num_samples={num}, expected {target}")
        positions = meta.get("mic_pos")
        if positions is not None and max(indices) >= len(positions):
            raise ValueError(
                f"{root}: mic_indices {indices} exceed {len(positions)} mics")
        return {"root": str(root), "sample_rate": rate, "num_samples": num,
                "mic_indices": indices, "source_num": meta.get("source_num")}

    if split == "dicit":
        root = Path(data_cfg["dicit_root"])
        wavs = sorted((root / "wavs").glob("*.wav"))
        metas = sorted((root / "metadata").glob("*.json"))
        if not wavs or not metas:
            raise FileNotFoundError(f"{root}: missing wavs/ or metadata/")
        wav_names = {p.stem for p in wavs}
        meta_names = {p.stem for p in metas}
        missing = sorted(meta_names - wav_names)
        if missing:
            raise FileNotFoundError(
                f"{root}: {len(missing)} metadata files have no WAV, "
                f"first: {missing[0]}")
        rates, lengths, channels = set(), set(), set()
        for path in wavs:
            info = sf.info(str(path))
            rates.add(int(info.samplerate))
            lengths.add(int(info.frames))
            channels.add(int(info.channels))
        if rates != {48000}:
            raise ValueError(f"{root}: WAV sample rates {sorted(rates)}")
        if lengths != {target}:
            raise ValueError(
                f"{root}: WAV lengths {sorted(lengths)}, expected {target}")
        if min(channels) <= max(indices):
            raise ValueError(
                f"{root}: WAVs have {min(channels)} channels, need "
                f"index {max(indices)}")
        return {"root": str(root), "sample_rate": 48000,
                "num_samples": target, "mic_indices": indices,
                "frames": len(wavs)}

    raise ValueError(f"Unknown split {split!r}")


def build_dicit_loader(root: str, data_cfg: Dict[str, Any], *, batch_size: int,
                       num_workers: int,
                       n_mics: Optional[int] = None) -> DataLoader:
    sym = locata_symbols()
    ds = sym["create_locata_dataset"](
        root_dir=root,
        split="all",
        min_sources=0,
        max_sources=int(data_cfg.get("max_sources", 3)),
        normalize=True,
        target_num_samples=int(data_cfg.get("target_num_samples", 16000)),
        mic_indices=resolve_mic_indices(data_cfg, n_mics=n_mics),
    )
    return build_loader(ds, batch_size=batch_size, num_workers=num_workers,
                        shuffle=False, collate=sym["angle_collate"])


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _optimizer_and_scheduler(model, train_cfg: Dict[str, Any], epochs: int):
    opt_name = str(train_cfg.get("optimizer", "adamw")).lower()
    lr = float(train_cfg.get("lr", 1e-4))
    wd = float(train_cfg.get("weight_decay", 0.01))
    if opt_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"Unsupported optimizer {opt_name!r}")
    scheduler = None
    if str(train_cfg.get("scheduler", "cosine")).lower() == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(epochs, 1),
            eta_min=float(train_cfg.get("eta_min", 1e-6)),
        )
    return optimizer, scheduler


def train_tf_mamba(model, data_cfg, train_cfg, *, device_index: int,
                   out_dir: str, seed: int, log_dir: str, log_path: str) -> str:
    """Train with LOCATA's ``Trainer``; returns the exported checkpoint path."""
    sym = locata_symbols()
    epochs = int(train_cfg.get("epochs", 200))
    batch_size = int(train_cfg.get("batch_size", 16))
    workers = int(train_cfg.get("num_workers", 8))

    n_mics = model_n_mics(model)
    train_ds = build_heatmap_dataset(data_cfg["train_root"], data_cfg,
                                     use_val=False, n_mics=n_mics)
    val_ds = build_heatmap_dataset(data_cfg["val_root"], data_cfg,
                                   use_val=True, n_mics=n_mics)
    train_loader = build_loader(train_ds, batch_size=batch_size, num_workers=workers,
                                shuffle=True, drop_last=True, collate=False)
    val_loader = build_loader(val_ds, batch_size=batch_size, num_workers=workers,
                              shuffle=False, collate=False)

    optimizer, scheduler = _optimizer_and_scheduler(model, train_cfg, epochs)
    trainer = sym["Trainer"](
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        loss_fn=sym["mse_kl_loss"],
        save_dir=out_dir,
        device_index=device_index,
        epochs=epochs,
        log_dir=log_dir,
        log_path=log_path,
        last_name="last.pth",
        best_name="best.pth",
        eval_every=int(train_cfg.get("eval_every", 5)),
        scheduler=scheduler,
        patience=train_cfg.get("patience", 6),
        use_amp=bool(train_cfg.get("amp", True)),
        seed=seed,
    )
    trainer.train()
    trainer.close()

    best = Path(out_dir) / "best.pth"
    if not best.is_file():
        # Trainer only writes best.pth when a validation loss improves.  A run
        # whose validation stayed non-finite (fp16 overflow) must still export
        # something rather than dying on a missing file.
        last = Path(out_dir) / "last.pth"
        if not last.is_file():
            raise FileNotFoundError(f"neither best.pth nor last.pth in {out_dir}")
        print(f"[train_tf_mamba] no best.pth (validation never improved); "
              f"exporting {last} instead")
        best = last
    ckpt = torch.load(best, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    export = Path(out_dir) / "checkpoint.pth"
    save_checkpoint(export, model, epoch=ckpt.get("epoch"), config={
        "model": dict(model.config, name="tf_mamba"),
        "train": train_cfg,
        "data": data_cfg,
    })
    return str(export)


def train_native(model, data_cfg, train_cfg, *, out_dir: str, seed: int,
                 device: str) -> str:
    """Train IcoCNN / SRP-DNN with the package loop on LOCATA angle labels.

    Follows ``train_tf_mamba``'s logging contract: epoch lines are written to
    ``<out_dir>/train.log`` in addition to stdout (which gpu_queue keeps in
    ``Record/queue/state/jobs/<job>/command.log``).
    """
    n_mics = model_n_mics(model)
    train_ds = build_angle_dataset(data_cfg["train_root"], data_cfg,
                                   use_val=False, n_mics=n_mics)
    val_ds = build_angle_dataset(data_cfg["val_root"], data_cfg,
                                 use_val=True, n_mics=n_mics)
    batch_size = int(train_cfg.get("batch_size", 8))
    workers = int(train_cfg.get("num_workers", 8))
    train_loader = build_loader(train_ds, batch_size=batch_size, num_workers=workers,
                                shuffle=True, drop_last=True)
    val_loader = build_loader(val_ds, batch_size=batch_size, num_workers=workers,
                              shuffle=False)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / "train.log", "w", buffering=1) as log_file:
        def log_fn(line: str) -> None:
            log_file.write(line + "\n")
            print(line, flush=True)

        fit(
            model, train_loader, val_loader,
            epochs=int(train_cfg.get("epochs", 100)),
            device=device,
            lr=float(train_cfg.get("lr", 1e-3)),
            weight_decay=float(train_cfg.get("weight_decay", 0.01)),
            optimizer=str(train_cfg.get("optimizer", "adamw")),
            scheduler=train_cfg.get("scheduler", "cosine"),
            scheduler_gamma=float(train_cfg.get("scheduler_gamma", 0.975)),
            eta_min=float(train_cfg.get("eta_min", 1e-6)),
            amp=bool(train_cfg.get("amp", True)),
            grad_clip=float(train_cfg.get("grad_clip", 1.0)),
            save_dir=out_dir,
            eval_every=int(train_cfg.get("eval_every", 1)),
            patience=train_cfg.get("patience"),
            config={"model": dict(model.config, name=model.model_name),
                    "train": train_cfg, "data": data_cfg},
            log_fn=log_fn,
        )
    best = Path(out_dir) / "best.pth"
    ckpt = torch.load(best, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    export = Path(out_dir) / "checkpoint.pth"
    save_checkpoint(export, model, epoch=ckpt.get("epoch"), config={
        "model": dict(model.config, name=model.model_name),
        "train": train_cfg,
        "data": data_cfg,
    })
    return str(export)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _curve_points(metrics_by_key: Dict[str, dict], acc: int) -> List[dict]:
    points = []
    for key, metrics in metrics_by_key.items():
        m = metrics[acc] if acc in metrics else metrics
        points.append({
            "key": key,
            "threshold": float(key),
            "F1": float(m["F1"]),
            "P": float(m["P"]),
            "R": float(m["R"]),
            "FAR": float(m["FAR"]),
            "MAE": float(m["MAE"]),
        })
    return sorted(points, key=lambda p: p["threshold"])


def lock_working_point(metrics_by_key: Dict[str, dict], acc: int = 10) -> dict:
    """Validation-locked operating point: max F1@10, tie-break by lower FAR."""
    points = _curve_points(metrics_by_key, acc)
    if not points:
        raise ValueError("No operating points to select from")
    return max(points, key=lambda p: (p["F1"], -p["FAR"]))


def interpolate_at_far(metrics_by_key: Dict[str, dict], far_target: float = 0.03,
                       acc: int = 10) -> Optional[dict]:
    """Linear interpolation of R and MAE at a target FAR, if bracketed."""
    pts = sorted(_curve_points(metrics_by_key, acc), key=lambda p: p["FAR"])
    for lo, hi in zip(pts, pts[1:]):
        if lo["FAR"] <= far_target <= hi["FAR"] and hi["FAR"] > lo["FAR"]:
            w = (far_target - lo["FAR"]) / (hi["FAR"] - lo["FAR"])
            return {
                "far_target": far_target,
                "FAR": far_target,
                "R": lo["R"] + w * (hi["R"] - lo["R"]),
                "MAE": lo["MAE"] + w * (hi["MAE"] - lo["MAE"]),
                "covered": True,
                "bracket": [lo["threshold"], hi["threshold"]],
            }
    return {"far_target": far_target, "covered": False,
            "FAR_min": pts[0]["FAR"] if pts else None,
            "FAR_max": pts[-1]["FAR"] if pts else None}


def count_parameters(model) -> int:
    """Parameter count of the actually rebuilt model."""
    return int(sum(p.numel() for p in model.parameters()))


def _silence_max_thresh(decode_cfg: Dict[str, Any], split: str) -> float:
    key = "silence_max_thresh_val" if split == "val" else "silence_max_thresh_dicit"
    return float(decode_cfg.get(key, 0.0 if split == "val" else 0.1))


def collect_audio_batches(loader, device: str) -> List[torch.Tensor]:
    """Materialise the loader's waveforms on the device, outside the timer."""
    batches: List[torch.Tensor] = []
    for audios, _labels in loader:
        batches.append(audios.to(device))
    return batches


@torch.no_grad()
def inference_detections(model, audios: torch.Tensor, *,
                         decode_cfg: Dict[str, Any], split: str,
                         threshold: float):
    """One method's full inference + decode at ``threshold`` for a batch.

    This is the timed unit: front end, backbone, the method's own template
    matching / peak decoder and the final score filter.  Public scoring is
    outside; ``threshold`` is a ratio for TF-Mamba and an absolute score for
    the candidate-path models.
    """
    name = model.model_name
    max_sources = int(decode_cfg.get("max_sources", 3))
    if name == "tf_mamba":
        from engine import decode_at_ratio

        spectrum = model(audios).detach().cpu().numpy().astype(np.float32)
        return decode_at_ratio(
            spectrum, float(threshold), max_sources=max_sources,
            silence_max_thresh=_silence_max_thresh(decode_cfg, split))
    if name in ("icocnn", "srp_dnn"):
        return model.decode(
            audios, threshold_abs=float(threshold), max_sources=max_sources,
            frame_index=int(decode_cfg.get("frame_index", -1)))
    if name in ("ipdnet", "ipdnet2_small"):
        score_kind = str(decode_cfg.get(
            "score_kind", getattr(model, "score_kind", "ratio")))
        out = model(audios)
        scores, azimuth, elevation = model.tracks.decode_candidates(
            out["ipd"], score_kind=score_kind)
        detections = []
        for b in range(scores.shape[0]):
            row = []
            for k in range(scores.shape[1]):
                if float(scores[b, k]) < float(threshold):
                    continue
                row.append((float(azimuth[b, k]), float(elevation[b, k])))
                if len(row) >= max_sources:
                    break
            detections.append(row)
        return detections
    raise ValueError(f"Unsupported model for inference timing: {name}")


def _sync_device(model) -> None:
    if torch.cuda.is_available():
        device = next(model.parameters()).device
        if device.type == "cuda":
            torch.cuda.synchronize(device)


def benchmark_inference(model, batches: List[torch.Tensor], *,
                        decode_cfg: Dict[str, Any], split: str,
                        threshold: float, warmup_batches: int = 1,
                        repeats: int = 1) -> Dict[str, Any]:
    """End-to-end per-clip inference time, normalised by actual clip count.

    The waveform is already on the device (``collect_audio_batches``); timing
    runs warm-up batches untimed, then ``repeats`` passes over the same
    batches with a device sync around each batch.  The denominator is the
    accumulated clip count, so a short final batch is counted correctly.
    """
    if not batches:
        raise ValueError("benchmark_inference needs at least one batch")
    model.eval()
    warmup = max(0, min(int(warmup_batches), len(batches)))
    with torch.no_grad():
        for index in range(warmup):
            inference_detections(model, batches[index], decode_cfg=decode_cfg,
                                 split=split, threshold=threshold)
        _sync_device(model)
        clips = 0
        elapsed = 0.0
        for _ in range(max(1, int(repeats))):
            for audios in batches:
                clips += int(audios.shape[0])
                _sync_device(model)
                start = time.perf_counter()
                inference_detections(model, audios, decode_cfg=decode_cfg,
                                     split=split, threshold=threshold)
                _sync_device(model)
                elapsed += time.perf_counter() - start
    return {
        "inference_ms_per_clip": 1000.0 * elapsed / max(clips, 1),
        "inference_clips": clips,
        "inference_batches": len(batches),
        "inference_warmup_batches": warmup,
        "inference_repeats": max(1, int(repeats)),
        "inference_elapsed_s": elapsed,
    }


def score_by_source_count(gts, pred_all, *, acc_thresholds=(5, 10, 15)) -> Dict[str, dict]:
    out = {}
    labels = np.array([len(g) for g in gts])
    for n in (0, 1, 2, 3):
        idx = np.flatnonzero(labels == n)
        key = str(n)
        if idx.size == 0:
            out[key] = {"frames": 0}
            continue
        from engine import score_decoded

        sub_gts = [gts[i] for i in idx]
        sub_pred = [pred_all[i] for i in idx]
        metrics, _, peaks = score_decoded(sub_gts, sub_pred,
                                          acc_thresholds=acc_thresholds,
                                          want_perframe=False)
        out[key] = {"frames": int(idx.size), "metrics": metrics,
                    "peaks_per_frame": float(peaks)}
    return out


def eval_spectrum(model, loader, *, device: str, ratios: Sequence[float],
                  acc_thresholds=(5, 10, 15), max_sources: int = 3,
                  silence_max_thresh: float = 0.0) -> Dict[str, Any]:
    """TF-Mamba path: reuse LOCATA's forward cache and peak decoder."""
    from engine import run_forward_cache, eval_at_ratio

    t0 = time.time()
    hms, gts, time_ms = run_forward_cache(model, loader, device)
    total_s = time.time() - t0
    metrics_by_ratio = {}
    for ratio in ratios:
        metrics, _, peaks = eval_at_ratio(
            hms, gts, ratio, max_sources=max_sources,
            silence_max_thresh=silence_max_thresh,
            acc_thresholds=tuple(acc_thresholds), want_perframe=False,
        )
        metrics_by_ratio[str(ratio)] = metrics
    return {
        "kind": "spectrum",
        "frames": len(gts),
        "source_count_dist": {str(k): int(v) for k, v in
                              zip(*np.unique([len(g) for g in gts], return_counts=True))},
        "metrics_by_key": metrics_by_ratio,
        "forward_ms_per_batch": float(time_ms),
        "elapsed_s": float(total_s),
        "gts": gts,
        "hms": hms,
    }


def decoded_spectrum_at_ratio(cache: Dict[str, Any], ratio: float, *,
                              max_sources: int = 3,
                              silence_max_thresh: float = 0.0):
    from engine import decode_at_ratio

    return decode_at_ratio(cache["hms"], ratio, max_sources=max_sources,
                           silence_max_thresh=silence_max_thresh)


@torch.no_grad()
def cache_candidates(model, loader, *, device: str, max_sources: int = 3,
                     frame_index: int = -1) -> Dict[str, Any]:
    """IcoCNN / SRP-DNN path: cache native candidates ``(score, az, el)``."""
    model.eval()
    gts: List[List[Tuple[float, float]]] = []
    candidates: List[List[Tuple[float, float, float]]] = []
    elapsed = 0.0
    frontend_elapsed = 0.0
    template_elapsed = 0.0
    frames = 0

    def _sync(tensor):
        if tensor.is_cuda:
            torch.cuda.synchronize(tensor.device)

    native_tracks = model.model_name in ("ipdnet", "ipdnet2_small")
    for audios, labels in loader:
        audios = audios.to(device)
        frames += audios.shape[0]
        gts.extend([[tuple(float(x) for x in src) for src in sample] for sample in labels])

        if native_tracks:
            # Separate the resampler/STFT front end from the network forward.
            _sync(audios)
            t0 = time.time()
            features = model.frontend(audios)
            _sync(audios)
            frontend_elapsed += time.time() - t0

            _sync(audios)
            t0 = time.time()
            frame_ipd = model.backbone(features["features"])
            ipd = model.tracks.pool_clip(frame_ipd, features["stft_frames"])
            _sync(audios)
            elapsed += time.time() - t0
            out = {"ipd": ipd}
        else:
            _sync(audios)
            t0 = time.time()
            out = model(audios)
            _sync(audios)
            elapsed += time.time() - t0

        if model.model_name == "icocnn":
            acc = out["accdoa"].detach().cpu().numpy()          # (B,T,M,3)
            act = np.linalg.norm(acc, axis=-1)                   # (B,T,M)
            vec = acc[:, frame_index] if acc.shape[1] > 0 else acc[:, 0]
            score = act[:, frame_index] if act.shape[1] > 0 else act[:, 0]
            ang = vectors_to_angles(vec.reshape(-1, 3)).reshape(vec.shape[0], -1, 2)
            for b in range(ang.shape[0]):
                cands = []
                for m in range(ang.shape[1]):
                    az = float(ang[b, m, 0])
                    az = az if az <= 180.0 else 360.0 - az
                    cands.append((float(score[b, m]), az, float(ang[b, m, 1])))
                candidates.append(sorted(cands, key=lambda c: -c[0])[:max_sources])
        elif model.model_name == "srp_dnn":
            idl = model.idl(out["ipd"], model.dpipd.template,
                            model.dpipd.az_deg, model.dpipd.el_deg)
            doa = idl["doa"].detach().cpu().numpy()              # (B,T,2,S)
            vad = idl["vad"].detach().cpu().numpy()              # (B,T,S)
            fi = doa.shape[1] + frame_index if frame_index < 0 else frame_index
            fi = int(np.clip(fi, 0, doa.shape[1] - 1))
            for b in range(doa.shape[0]):
                order = np.argsort(-vad[b, fi])
                candidates.append([
                    (float(vad[b, fi, s]), float(doa[b, fi, 0, s]),
                     float(doa[b, fi, 1, s])) for s in order[:max_sources]
                ])
        elif model.model_name in ("ipdnet", "ipdnet2_small"):
            # The native clip-level DP-IPD is already decoded per track by the
            # model; time the template matching separately from the forward.
            score_kind = str(model.config.get("score_kind",
                                              getattr(model, "score_kind", "ratio")))
            _sync(audios)
            t0 = time.time()
            scores, az, el = model.tracks.decode_candidates(
                out["ipd"], score_kind=score_kind)
            _sync(audios)
            template_elapsed += time.time() - t0
            for b in range(scores.shape[0]):
                candidates.append([
                    (float(scores[b, k]), float(az[b, k]), float(el[b, k]))
                    for k in range(min(max_sources, scores.shape[1]))
                ])
        else:
            raise ValueError(f"Unsupported model for candidate caching: {model.model_name}")

    payload = {"kind": "candidates", "frames": len(gts), "gts": gts,
               "candidates": candidates, "elapsed_s": elapsed,
               "forward_ms_per_frame": 1000.0 * elapsed / max(frames, 1),
               "template_ms_per_frame": 1000.0 * template_elapsed / max(frames, 1)}
    if native_tracks:
        payload["frontend_ms_per_frame"] = (
            1000.0 * frontend_elapsed / max(frames, 1))
    return payload


def sweep_candidates(cache: Dict[str, Any], thresholds: Sequence[float], *,
                     max_sources: int = 3,
                     acc_thresholds=(5, 10, 15)) -> Dict[str, dict]:
    from engine import score_decoded

    gts, cands = cache["gts"], cache["candidates"]
    out = {}
    for thr in thresholds:
        pred_all = [
            [(az, el) for score, az, el in row if score >= thr][:max_sources]
            for row in cands
        ]
        metrics, _, _ = score_decoded(gts, pred_all,
                                      acc_thresholds=tuple(acc_thresholds),
                                      want_perframe=False)
        out[str(thr)] = metrics
    return out


def decoded_at_threshold(cache: Dict[str, Any], threshold: float,
                         max_sources: int = 3) -> List[List[Tuple[float, float]]]:
    return [
        [(az, el) for score, az, el in row if score >= threshold][:max_sources]
        for row in cache["candidates"]
    ]


# ---------------------------------------------------------------------------
# Provenance and config snapshot
# ---------------------------------------------------------------------------

def git_state(repo: str) -> Dict[str, Any]:
    def run(*args):
        try:
            return subprocess.run(["git", "-C", repo, *args], check=True,
                                  capture_output=True, text=True).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            return f"<error: {exc}>"

    return {
        "root": str(Path(repo).resolve()),
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }


def write_json(path: str, payload: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    tmp.replace(p)
