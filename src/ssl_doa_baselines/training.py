"""Lightweight supervised loop for standalone baseline training.

The LOCATA integration does not use this module: it reuses the existing
LOCATA ``Trainer`` so that preprocessing, AMP, logging and checkpointing stay
identical to the internal models.  This loop exists so the repository is
usable on any manifest of ``wav + angle labels`` without LOCATA.
"""

import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch

from .api import save_checkpoint


def _unpack_batch(batch):
    if isinstance(batch, dict):
        return batch["audio"], batch.get("angles", batch.get("labels"))
    return batch[0], batch[1]


def _run_epoch(model, loader, device, optimizer=None, amp=False, grad_clip=1.0,
               max_batches: Optional[int] = None) -> float:
    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    for step, batch in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        audio, targets = _unpack_batch(batch)
        audio = audio.to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            with torch.cuda.amp.autocast(enabled=amp):
                loss = model.training_loss(audio, targets)
        if training:
            optimizer.zero_grad(set_to_none=True)
            if amp:
                scaler.scale(loss).backward()
                if grad_clip:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
        else:
            with torch.no_grad():
                pass
        total += float(loss.detach()) * audio.shape[0]
        count += audio.shape[0]
    return total / max(count, 1)


def fit(model: torch.nn.Module, train_loader, val_loader=None, *,
        epochs: int = 100, device: str = "cuda", lr: float = 1e-4,
        weight_decay: float = 0.01, scheduler: Optional[str] = "cosine",
        eta_min: float = 1e-6, amp: bool = False, grad_clip: float = 1.0,
        save_dir: Optional[str] = None, eval_every: int = 1,
        patience: Optional[int] = None, config: Optional[Dict[str, Any]] = None,
        log_fn: Callable[[str], None] = print, max_train_batches: Optional[int] = None,
        max_val_batches: Optional[int] = None, history: Optional[List[dict]] = None,
        ) -> List[dict]:
    """Fit ``model`` whose ``training_loss(waveform, targets)`` returns a scalar.

    Saves ``last.pth`` and ``best.pth`` (lowest validation loss, or last epoch
    without validation) under ``save_dir``.  Returns the per-epoch history.
    """
    device = torch.device(device)
    amp = bool(amp) and device.type == "cuda"
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if scheduler == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(epochs, 1), eta_min=eta_min
        )
    elif scheduler in (None, "none"):
        sched = None
    else:
        raise ValueError(f"Unsupported scheduler {scheduler!r}")

    save_dir = Path(save_dir) if save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
    history = history if history is not None else []
    best = float("inf")
    stale = 0

    for epoch in range(epochs):
        t0 = time.time()
        train_loss = _run_epoch(model, train_loader, device, optimizer, amp,
                                grad_clip, max_train_batches)
        val_loss = None
        if val_loader is not None and (epoch % eval_every == 0 or epoch == epochs - 1):
            val_loss = _run_epoch(model, val_loader, device, None, False,
                                  grad_clip, max_val_batches)
        if sched is not None:
            sched.step()

        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
               "lr": optimizer.param_groups[0]["lr"], "seconds": time.time() - t0}
        history.append(row)
        log_fn(
            f"epoch {epoch + 1}/{epochs} train={train_loss:.6f} "
            f"val={'n/a' if val_loss is None else f'{val_loss:.6f}'} "
            f"lr={row['lr']:.2e} ({row['seconds']:.1f}s)"
        )

        monitor = val_loss if val_loss is not None else train_loss
        improved = monitor < best
        if improved:
            best = monitor
            stale = 0
        else:
            stale += 1
        if save_dir:
            save_checkpoint(save_dir / "last.pth", model, epoch=epoch,
                            optimizer=optimizer, config=config,
                            extra={"history": history})
            if improved:
                save_checkpoint(save_dir / "best.pth", model, epoch=epoch,
                                optimizer=optimizer, config=config,
                                extra={"history": history, "best_loss": best})
        if patience is not None and stale >= patience:
            log_fn(f"early stopping at epoch {epoch + 1} (best={best:.6f})")
            break
    return history


def infer_targets(labels) -> List[np.ndarray]:
    """Normalise a batch of label objects into ``(N, 2)`` float arrays."""
    out = []
    for label in labels:
        arr = np.asarray(label, dtype=np.float32)
        out.append(arr.reshape(-1, 2) if arr.size else np.zeros((0, 2), np.float32))
    return out
