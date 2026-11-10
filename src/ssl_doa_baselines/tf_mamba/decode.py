"""Spectrum peak decoder shared by TF-Mamba and LOCATA-style spatial spectra."""

from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.ndimage import maximum_filter1d


def decode_spectrum(
    spectra: np.ndarray,
    *,
    az_range: Sequence[float] = (0.0, 180.0),
    max_sources: int = 3,
    threshold_ratio: float = 0.1,
    threshold_abs: float = 0.0,
    silence_max_thresh: float = 0.0,
    neighborhood_size: int = 3,
    min_peak_distance: int = 3,
) -> List[List[Dict[str, float]]]:
    """Decode ``(B, Q)`` spectra into per-clip azimuth detections.

    A peak must be a local maximum in ``neighborhood_size`` and above
    ``max(threshold_abs, threshold_ratio * frame_max)``.  Peaks are greedily
    accepted in descending score order with a minimum grid distance, then the
    top ``max_sources`` are kept.  Frames whose maximum is below
    ``silence_max_thresh`` yield an empty list.
    """
    spectra = np.asarray(spectra, dtype=np.float32)
    if spectra.ndim != 2:
        raise ValueError(f"Expected (B, Q) spectra, got {spectra.shape}")
    n_bins = spectra.shape[1]
    angles = np.linspace(float(az_range[0]), float(az_range[1]), n_bins, dtype=np.float64)

    out: List[List[Dict[str, float]]] = []
    for frame in spectra:
        peak_max = float(frame.max())
        if peak_max < float(silence_max_thresh):
            out.append([])
            continue
        thresh = max(float(threshold_abs), float(threshold_ratio) * peak_max, 1e-7)
        local_max = maximum_filter1d(frame, size=int(neighborhood_size), mode="nearest")
        idx = np.flatnonzero((frame >= local_max) & (frame > thresh))
        if idx.size == 0:
            out.append([])
            continue
        order = idx[np.argsort(-frame[idx])]
        selected: List[int] = []
        for i in order:
            if all(abs(int(i) - j) >= int(min_peak_distance) for j in selected):
                selected.append(int(i))
            if len(selected) >= int(max_sources):
                break
        dets = [
            {
                "azimuth_deg": float(angles[i]),
                "score": float(frame[i]),
                "bin": int(i),
            }
            for i in selected
        ]
        out.append(dets)
    return out


def spectrum_to_angle_sets(
    spectra: np.ndarray,
    *,
    az_range: Sequence[float] = (0.0, 180.0),
    max_sources: int = 3,
    threshold_ratio: float = 0.1,
    threshold_abs: float = 0.0,
    silence_max_thresh: float = 0.0,
    neighborhood_size: int = 3,
    min_peak_distance: int = 3,
) -> List[List[tuple]]:
    """Convenience wrapper returning LOCATA ``(N, 2)``-style ``(az, 0.0)`` lists."""
    decoded = decode_spectrum(
        spectra,
        az_range=az_range,
        max_sources=max_sources,
        threshold_ratio=threshold_ratio,
        threshold_abs=threshold_abs,
        silence_max_thresh=silence_max_thresh,
        neighborhood_size=neighborhood_size,
        min_peak_distance=min_peak_distance,
    )
    return [[(d["azimuth_deg"], 0.0) for d in frame] for frame in decoded]
