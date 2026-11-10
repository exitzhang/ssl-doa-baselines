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
                        use_val: bool = False) -> Dataset:
    sym = locata_symbols()
    base = sym["AudioDataset"](
        root_dir=root,
        split="all",
        label_mode="angle",
        use_aug=False if use_val else bool(data_cfg.get("use_aug", True)),
        mic_indices=list(data_cfg.get("mic_indices", [5, 6, 8])),
        normalize=True,
        **_noise_kwargs(data_cfg, use_val),
    )
    return _ZenithDataset(base, source=data_cfg.get("label_source", "sim"))


def build_heatmap_dataset(root: str, data_cfg: Dict[str, Any], *,
                          use_val: bool = False) -> Dataset:
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
        mic_indices=list(data_cfg.get("mic_indices", [5, 6, 8])),
        normalize=True,
        **_noise_kwargs(data_cfg, use_val),
    )


def build_loader(dataset: Dataset, *, batch_size: int, num_workers: int,
                 shuffle: bool, collate=None, drop_last: bool = False) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        collate_fn=collate or angle_batch_collate,
    )


def build_dicit_loader(root: str, data_cfg: Dict[str, Any], *, batch_size: int,
                       num_workers: int) -> DataLoader:
    sym = locata_symbols()
    ds = sym["create_locata_dataset"](
        root_dir=root,
        split="all",
        min_sources=0,
        max_sources=int(data_cfg.get("max_sources", 3)),
        normalize=True,
        target_num_samples=int(data_cfg.get("target_num_samples", 16000)),
        mic_indices=list(data_cfg.get("mic_indices", [5, 6, 8])),
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

    train_ds = build_heatmap_dataset(data_cfg["train_root"], data_cfg, use_val=False)
    val_ds = build_heatmap_dataset(data_cfg["val_root"], data_cfg, use_val=True)
    train_loader = build_loader(train_ds, batch_size=batch_size, num_workers=workers,
                                shuffle=True, drop_last=True)
    val_loader = build_loader(val_ds, batch_size=batch_size, num_workers=workers,
                              shuffle=False)

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
    """Train IcoCNN / SRP-DNN with the package loop on LOCATA angle labels."""
    train_ds = build_angle_dataset(data_cfg["train_root"], data_cfg, use_val=False)
    val_ds = build_angle_dataset(data_cfg["val_root"], data_cfg, use_val=True)
    batch_size = int(train_cfg.get("batch_size", 8))
    workers = int(train_cfg.get("num_workers", 8))
    train_loader = build_loader(train_ds, batch_size=batch_size, num_workers=workers,
                                shuffle=True, drop_last=True)
    val_loader = build_loader(val_ds, batch_size=batch_size, num_workers=workers,
                              shuffle=False)
    fit(
        model, train_loader, val_loader,
        epochs=int(train_cfg.get("epochs", 100)),
        device=device,
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        scheduler=train_cfg.get("scheduler", "cosine"),
        eta_min=float(train_cfg.get("eta_min", 1e-6)),
        amp=bool(train_cfg.get("amp", True)),
        grad_clip=float(train_cfg.get("grad_clip", 1.0)),
        save_dir=out_dir,
        eval_every=int(train_cfg.get("eval_every", 1)),
        patience=train_cfg.get("patience"),
        config={"model": dict(model.config, name=model.model_name),
                "train": train_cfg, "data": data_cfg},
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


def score_by_source_count(gts, pred_all, *, acc_thresholds=(5, 10, 15)) -> Dict[str, dict]:
    from engine import score_decoded

    out = {}
    labels = np.array([len(g) for g in gts])
    for n in (0, 1, 2, 3):
        idx = np.flatnonzero(labels == n)
        key = str(n)
        if idx.size == 0:
            out[key] = {"frames": 0}
            continue
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
    frames = 0
    for audios, labels in loader:
        audios = audios.to(device)
        t0 = time.time()
        out = model(audios)
        elapsed += time.time() - t0
        frames += audios.shape[0]
        gts.extend([[tuple(float(x) for x in src) for src in sample] for sample in labels])

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
        else:
            raise ValueError(f"Unsupported model for candidate caching: {model.model_name}")

    return {"kind": "candidates", "frames": len(gts), "gts": gts,
            "candidates": candidates, "elapsed_s": elapsed,
            "forward_ms_per_frame": 1000.0 * elapsed / max(frames, 1)}


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
