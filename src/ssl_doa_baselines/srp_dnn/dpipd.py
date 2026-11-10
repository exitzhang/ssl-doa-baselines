"""Direct-path IPD templates and mixed DP-IPD targets.

Ported from ``SRP-DNN/code/Module.py::DPIPD`` (MIT), adapted to:

* an explicit physical frequency list ``k * fs / n_fft`` (default 31.25--8000 Hz);
* azimuth in ``[0, 180]`` degrees and zenith elevation (DICIT convention,
  90 deg = horizontal);
* unordered microphone pairs (``ch_mode='MM'``).
"""

from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn


class DPIPD(nn.Module):
    def __init__(self, mic_positions: np.ndarray, freqs: Sequence[float],
                 az_range_deg=(0.0, 180.0), n_az: int = 181,
                 el_range_deg=(60.0, 120.0), n_el: int = 7,
                 speed: float = 343.0):
        super().__init__()
        self.mic_positions = np.asarray(mic_positions, dtype=np.float64)
        self.freqs = np.asarray(freqs, dtype=np.float64)
        self.n_freq = int(self.freqs.size)
        self.az_range_deg = tuple(az_range_deg)
        self.el_range_deg = tuple(el_range_deg)
        self.n_az = int(n_az)
        self.n_el = int(n_el)
        self.speed = float(speed)

        self.az_deg = np.linspace(az_range_deg[0], az_range_deg[1], n_az)
        self.el_deg = np.linspace(el_range_deg[0], el_range_deg[1], n_el)
        az = np.deg2rad(self.az_deg)
        el = np.deg2rad(self.el_deg)
        grid_el, grid_az = np.meshgrid(el, az, indexing="ij")
        r = np.stack(
            [np.sin(grid_el) * np.cos(grid_az),
             np.sin(grid_el) * np.sin(grid_az),
             np.cos(grid_el)],
            axis=-1,
        ).reshape(-1, 3)                      # (nele*nazi, 3)
        self.register_buffer("grid", torch.from_numpy(r).float(), persistent=False)

        nmic = self.mic_positions.shape[0]
        # (D, F, P) complex, upper triangular pairs
        self.pairs = [(i, j) for i in range(nmic) for j in range(i + 1, nmic)]
        ipd = np.empty((r.shape[0], len(self.freqs), len(self.pairs)), dtype=np.complex64)
        for p, (i, j) in enumerate(self.pairs):
            itd = np.dot(r, self.mic_positions[j] - self.mic_positions[i]) / self.speed
            ipd[:, :, p] = np.exp(-2j * np.pi * self.freqs[None, :] * itd[:, None])
        # (D, 2F, P) real/imag interleaved along frequency
        tpl = np.concatenate([ipd.real, ipd.imag], axis=1).astype(np.float32)
        self.register_buffer("template", torch.from_numpy(tpl), persistent=False)

    def mixed_target(self, doa_deg: np.ndarray, beta: Optional[np.ndarray] = None) -> np.ndarray:
        """Sum weighted per-source templates.

        Args:
            doa_deg: ``(N, 2)`` (az near [0,180], zenith elevation).
            beta: ``(N,)`` activity weights (default all ones).
        Returns:
            ``(2F, P)`` real/imag stacked target.
        """
        doa = np.asarray(doa_deg, dtype=np.float64)
        if doa.size == 0:
            return np.zeros(
                (2 * len(self.freqs), len(self.pairs)), dtype=np.float32
            )
        if doa.ndim != 2 or doa.shape[1] != 2:
            raise ValueError(f"doa_deg must be (N,2), got {doa.shape}")
        if beta is None:
            beta = np.ones(doa.shape[0], dtype=np.float64)
        beta = np.asarray(beta, dtype=np.float64).reshape(-1)
        az = np.deg2rad(doa[:, 0])
        el = np.deg2rad(doa[:, 1])
        r = np.stack([np.sin(el) * np.cos(az), np.sin(el) * np.sin(az), np.cos(el)], axis=-1)
        out = np.zeros((len(self.freqs), len(self.pairs)), dtype=np.complex64)
        for n in range(doa.shape[0]):
            for p, (i, j) in enumerate(self.pairs):
                itd = np.dot(r[n], self.mic_positions[j] - self.mic_positions[i]) / self.speed
                out[:, p] += beta[n] * np.exp(-2j * np.pi * self.freqs * itd)
        return np.concatenate([out.real, out.imag], axis=0).astype(np.float32)
