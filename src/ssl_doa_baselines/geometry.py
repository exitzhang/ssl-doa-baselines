"""Shared direction conventions.

The baselines use **zenith elevation**: elevation is measured from the +z axis,
so a horizontal source has ``el = 90``.  Azimuth is measured in the horizontal
plane from the +x axis.  The unit vector is therefore

    u(az, el) = [sin(el) cos(az), sin(el) sin(az), cos(el)]

DICIT metadata already uses this convention (``mycart2sph`` computes
``arccos(z/r)``).  LOCATA simulated metadata uses elevation from the horizontal
plane instead (``arcsin(z/r)``), so the LOCATA adapter converts simulated
labels with ``el_zenith = 90 - el_sim`` before building 3-D targets; see
``integrations/locata/adapter.py``.  Azimuth is unchanged.
"""

import numpy as np
import torch


def angles_to_vectors(angles_deg: np.ndarray) -> np.ndarray:
    """``(N, 2)`` (az, el) degrees -> ``(N, 3)`` unit vectors."""
    a = np.asarray(angles_deg, dtype=np.float64)
    if a.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    az = np.deg2rad(a[:, 0])
    el = np.deg2rad(a[:, 1])
    v = np.stack([np.sin(el) * np.cos(az), np.sin(el) * np.sin(az), np.cos(el)], axis=-1)
    return v.astype(np.float32)


def vectors_to_angles(vectors: np.ndarray) -> np.ndarray:
    """``(..., 3)`` vectors -> ``(..., 2)`` (az in [0, 360), el in [0, 180]) degrees."""
    v = np.asarray(vectors, dtype=np.float64)
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    az = np.rad2deg(np.arctan2(y, x)) % 360.0
    el = np.rad2deg(np.arccos(np.clip(z / (np.linalg.norm(v, axis=-1) + 1e-12), -1.0, 1.0)))
    return np.stack([az, el], axis=-1)


def angles_to_vectors_torch(angles_deg: torch.Tensor) -> torch.Tensor:
    az = torch.deg2rad(angles_deg[..., 0])
    el = torch.deg2rad(angles_deg[..., 1])
    return torch.stack(
        [torch.sin(el) * torch.cos(az), torch.sin(el) * torch.sin(az), torch.cos(el)], dim=-1
    )
