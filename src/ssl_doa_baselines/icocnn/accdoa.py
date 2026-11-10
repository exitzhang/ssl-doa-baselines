"""ACCDOA targets, permutation-invariant loss and decoding."""

from typing import Dict, List, Sequence

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from ..geometry import angles_to_vectors


def accdoa_pit_mse(
    pred: torch.Tensor,
    targets: Sequence[np.ndarray],
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Frame-level PIT MSE for ACCDOA slots.

    Args:
        pred: ``(B, T, M, 3)`` predicted slot vectors.
        targets: length-``B`` sequence, each ``(N, 2)`` (az, el) in degrees
            (``N=0`` for silence).
        reduction: ``"mean"`` or ``"none"`` (returns per-frame loss ``(B, T)``).

    Empty slots are supervised towards the zero vector.  Matching is computed
    from the 3-D vector distance matrix with Hungarian assignment, avoiding
    ``M!`` permutations.
    """
    if pred.dim() != 4 or pred.shape[-1] != 3:
        raise ValueError(f"Expected (B, T, M, 3) predictions, got {tuple(pred.shape)}")
    B, T, M, _ = pred.shape
    per_frame = pred.new_zeros((B, T))
    for b in range(B):
        vec = angles_to_vectors(targets[b])          # (N, 3)
        n = vec.shape[0]
        if n > M:
            raise ValueError(f"{n} sources exceed {M} ACCDOA slots")
        for t in range(T):
            p = pred[b, t]                            # (M, 3)
            if n == 0:
                per_frame[b, t] = (p ** 2).mean()
                continue
            cost = ((p[:, None, :] - torch.as_tensor(vec, device=pred.device)[None, :, :]) ** 2).sum(-1)
            rows, cols = linear_sum_assignment(cost.detach().cpu().numpy())
            target = p.new_zeros((M, 3))
            for r, c in zip(rows, cols):
                target[r] = torch.as_tensor(vec[c], device=pred.device)
            per_frame[b, t] = ((p - target) ** 2).mean()
    if reduction == "none":
        return per_frame
    return per_frame.mean()


def slots_to_angles(vec: np.ndarray) -> np.ndarray:
    """``(..., 3)`` slot vectors -> ``(..., 2)`` (az, el) degrees."""
    from ..geometry import vectors_to_angles

    return vectors_to_angles(vec)


def decode_accdoa(
    accdoa: np.ndarray,
    *,
    times: Sequence[Sequence[float]] = None,
    threshold_abs: float = 0.3,
    max_sources: int = 3,
    frame_index: int = -1,
    az_range: Sequence[float] = (0.0, 180.0),
) -> List[List[Dict[str, float]]]:
    """Decode ACCDOA slots into per-clip detections.

    Each slot's vector norm is its activity score.  For clip-level evaluation
    we select one output frame (default: last).  Azimuth is folded into the
    front hemisphere ``[0, 180]`` (mirror across the x-z plane); scores are
    unchanged, so a backward-pointing slot still competes as a detection.
    ``az_range`` clips the returned list for reporting.
    """
    accdoa = np.asarray(accdoa, dtype=np.float32)
    if accdoa.ndim != 4 or accdoa.shape[-1] != 3:
        raise ValueError(f"Expected (B, T, M, 3), got {accdoa.shape}")
    B, T, M, _ = accdoa.shape
    fi = frame_index if frame_index >= 0 else T + frame_index
    fi = int(np.clip(fi, 0, T - 1))

    out: List[List[Dict[str, float]]] = []
    for b in range(B):
        vec = accdoa[b, fi]
        norm = np.linalg.norm(vec, axis=-1)
        order = np.argsort(-norm)
        dets: List[Dict[str, float]] = []
        for k in order:
            score = float(norm[k])
            if score < threshold_abs:
                continue
            az, el = slots_to_angles(vec[k][None, :])[0]
            az = float(az) if az <= 180.0 else float(360.0 - az)
            det = {"azimuth_deg": az, "elevation_deg": float(el), "score": score}
            if times is not None:
                try:
                    det["time_s"] = float(times[b][fi])
                except Exception:
                    pass
            dets.append(det)
            if len(dets) >= int(max_sources):
                break
        out.append(dets)
    return out
