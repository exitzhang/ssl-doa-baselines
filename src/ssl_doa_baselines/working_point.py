"""Checkpoint identity and validation-locked working points.

The LOCATA runner and the standalone CLI both consume the ``validation.json``
written by a simulated-validation run.  This module is the single place that
computes a checkpoint's content identity and verifies that a validation file
belongs to the checkpoint being evaluated.  Threshold sweeps and the F1-based
selection stay in the baseline adapters.

A checkpoint is identified by the SHA-256 of its file contents, so copying or
moving it does not change its identity.  A lock also records the model key, the
score kind and comparison direction, and the angle tolerance used to pick the
point; all of them must match at evaluation time.
"""

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional

import torch

SCORE_DIRECTION = "higher_is_better"
COMPARISON = ">="
REQUIRED_FIELDS = ("key", "threshold", "score_kind", "comparison",
                   "score_direction", "model", "checkpoint_sha256",
                   "acc_threshold")


class WorkingPointError(ValueError):
    """A validation lock is missing, incomplete or does not match."""


def checkpoint_sha256(path: str) -> str:
    """SHA-256 of the checkpoint file contents (copy/move invariant)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_meta(path: str) -> Dict[str, Any]:
    """Read a checkpoint once and return its identity plus embedded configs."""
    path = Path(path)
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    return {
        "checkpoint": str(path.resolve()),
        "sha256": checkpoint_sha256(path),
        "model_name": payload.get("model_name"),
        "model_config": dict(payload.get("model_config") or {}),
        "config": dict(payload.get("config") or {}),
        "epoch": payload.get("epoch"),
    }


def read_validation(path: str) -> Dict[str, Any]:
    """Read a validation file; a missing/unreadable file is actionable."""
    path = Path(path)
    if not path.is_file():
        raise WorkingPointError(
            f"validation file not found: {path}. DICIT scoring must reuse the "
            "threshold locked on the simulated validation set; run the "
            "validation for the evaluated checkpoint first:\n"
            "  locata-runner --config ... --data-config ... eval "
            f"--checkpoint <checkpoint> --split val --output {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkingPointError(f"{path}: cannot read validation file ({exc})")


def make_locked_record(point: Dict[str, Any], *, model: str,
                       checkpoint: str, score_kind: Optional[str],
                       acc_threshold: int,
                       checkpoint_sha: Optional[str] = None) -> Dict[str, Any]:
    """Attach identity and comparison metadata to a selected curve point.

    ``point`` is one row from ``adapter.lock_working_point``.  ``score_kind`` is
    ``None`` for models without a thresholded score (e.g. TF-Mamba's ratio
    curve is still a score, so it passes ``"ratio"`` for the spectrum path).
    """
    threshold = float(point["threshold"])
    if not math.isfinite(threshold):
        raise ValueError(f"locked threshold must be finite, got {threshold!r}")
    record = dict(point)
    record["key"] = str(point.get("key", threshold))
    record["threshold"] = threshold
    record["score_kind"] = None if score_kind is None else str(score_kind).lower()
    record["comparison"] = COMPARISON
    record["score_direction"] = SCORE_DIRECTION
    record["model"] = str(model)
    record["checkpoint_sha256"] = (
        checkpoint_sha if checkpoint_sha is not None
        else checkpoint_sha256(checkpoint))
    record["acc_threshold"] = int(acc_threshold)
    return record


def _regenerate_hint(checkpoint: str) -> str:
    return (
        "Regenerate the validation lock from this checkpoint:\n"
        "  locata-runner --config ... --data-config ... eval "
        f"--checkpoint {checkpoint} --split val --output <path>/validation.json")


def require_locked(validation: str, *, model: str, checkpoint: str,
                   score_kind: Optional[str] = None,
                   acc_threshold: Optional[int] = None,
                   checkpoint_sha: Optional[str] = None) -> Dict[str, Any]:
    """Return the lock iff it belongs to ``checkpoint`` and matches ``model``.

    Raises :class:`WorkingPointError` with a regeneration hint when the file is
    missing, the lock is incomplete, or any recorded identity differs.
    """
    data = read_validation(validation)
    locked = data.get("locked")
    if not isinstance(locked, dict):
        raise WorkingPointError(
            f"{validation}: no 'locked' block. {_regenerate_hint(checkpoint)}")
    missing = [field for field in REQUIRED_FIELDS if field not in locked]
    if missing:
        raise WorkingPointError(
            f"{validation}: locked working point is missing {missing}. "
            f"{_regenerate_hint(checkpoint)}")

    sha = checkpoint_sha if checkpoint_sha is not None \
        else checkpoint_sha256(checkpoint)
    if str(locked["checkpoint_sha256"]) != sha:
        raise WorkingPointError(
            f"{validation}: lock belongs to checkpoint "
            f"{locked['checkpoint_sha256']}, not {sha} ({checkpoint}). "
            f"{_regenerate_hint(checkpoint)}")
    if str(locked["model"]).lower() != str(model).lower():
        raise WorkingPointError(
            f"{validation}: lock belongs to model {locked['model']!r}, not "
            f"{model!r}. {_regenerate_hint(checkpoint)}")
    if score_kind is not None:
        recorded = locked["score_kind"]
        if recorded is None or str(recorded).lower() != str(score_kind).lower():
            raise WorkingPointError(
                f"{validation}: lock score_kind={recorded!r}, expected "
                f"{score_kind!r}. {_regenerate_hint(checkpoint)}")
    if str(locked["comparison"]) != COMPARISON:
        raise WorkingPointError(
            f"{validation}: locked comparison {locked['comparison']!r}, "
            f"expected {COMPARISON!r}. {_regenerate_hint(checkpoint)}")
    if str(locked["score_direction"]) != SCORE_DIRECTION:
        raise WorkingPointError(
            f"{validation}: locked score_direction "
            f"{locked['score_direction']!r}, expected {SCORE_DIRECTION!r}. "
            f"{_regenerate_hint(checkpoint)}")
    if acc_threshold is not None and int(locked["acc_threshold"]) != int(acc_threshold):
        raise WorkingPointError(
            f"{validation}: lock was selected at acc_threshold="
            f"{locked['acc_threshold']}, this evaluation uses "
            f"{int(acc_threshold)}. {_regenerate_hint(checkpoint)}")
    threshold = float(locked["threshold"])
    if not math.isfinite(threshold):
        raise WorkingPointError(
            f"{validation}: locked threshold is not finite ({threshold!r}). "
            f"{_regenerate_hint(checkpoint)}")
    locked = dict(locked)
    locked["threshold"] = threshold
    return locked