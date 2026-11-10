"""Spherical SRP-PHAT front-end on the icosahedral grid.

Ported from David Diaz-Guerra's ``icoDOA`` (``acousticTrackingModules.py``:
``GCC`` / ``SRP_grid_map`` / ``SRP_icosahedral_map``), AGPL-3.0.  The port is
restricted to the inference/training path needed here; see ``THIRD_PARTY.md``.
"""

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

from .grid import icosahedral_grid_coordinates


class GCC(nn.Module):
    """GCC-PHAT for ``N`` channels, returning lags ``[-tau_max, tau_max]``."""

    def __init__(self, N: int, K: int, tau_max: int = None, transform: str = "PHAT"):
        super().__init__()
        if transform not in (None, "PHAT"):
            raise ValueError("transform only supports None or 'PHAT'")
        self.N = int(N)
        self.K = int(K)
        self.tau_max = int(K // 2 if tau_max is None else tau_max)
        self.transform = transform

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., N, K)
        X = torch.fft.rfft(x, n=self.K, dim=-1)
        if self.transform == "PHAT":
            X = X / (X.abs() + 1e-12)
        cps = X.unsqueeze(-2) * X.conj().unsqueeze(-3)   # (..., N, N, F)
        gcc_full = torch.fft.irfft(cps, n=self.K, dim=-1)
        t = self.tau_max
        return torch.cat([gcc_full[..., -t:], gcc_full[..., : t + 1]], dim=-1)


class SRPIcosahedralMap(nn.Module):
    """Sum GCC-PHAT lags over all microphone pairs at every grid point."""

    def __init__(self, N: int, K: int, r: int, mic_positions: np.ndarray,
                 fs: int, c: float = 343.0, normalize: bool = True,
                 tau_max: int = None):
        super().__init__()
        self.N = int(N)
        self.K = int(K)
        self.fs = float(fs)
        self.normalize = bool(normalize)
        self.r = int(r)
        grid = icosahedral_grid_coordinates(r).astype(np.float64)   # (5,H,W,3)
        # Chart coordinates lie on icosahedron faces, not on the unit sphere;
        # steering delays need the corresponding unit direction.
        grid = grid / np.linalg.norm(grid, axis=-1, keepdims=True)
        self.register_buffer("grid", torch.from_numpy(grid).float(), persistent=False)

        rn = np.asarray(mic_positions, dtype=np.float64)[:N]
        rn = rn - rn.mean(axis=0, keepdims=True)
        # Valid discrete lags, ordered as the GCC output: [-t..-1, 0..t].
        tau_max = self.K // 2 if tau_max is None else int(tau_max)
        tau = np.arange(-tau_max, tau_max + 1) / float(fs)
        tau0 = np.empty(grid.shape[:-1] + (N, N), dtype=np.int64)
        for k in range(N):
            for l in range(N):
                # The simulated/DICIT label frame is mirrored relative to the
                # raw room coordinates: on real frames the GCC lag of
                # X_k conj(X_l) is -(p_k - p_l).u_label/c.  Steering by
                # (p_l - p_k).u_label therefore points the map at the label.
                itd = np.dot(grid, rn[l] - rn[k]) / c
                tau0[..., k, l] = np.argmin(np.abs(itd[..., None] - tau), axis=-1)
        # (N, N, 5, H, W) with tau index in [0, 2*tau_max]
        self.register_buffer("tau_idx",
                             torch.from_numpy(np.moveaxis(tau0, (-2, -1), (0, 1))).long(),
                             persistent=False)
        self.grid_shape = grid.shape[:-1]

    def forward(self, gcc: torch.Tensor) -> torch.Tensor:
        # gcc: (..., N, N, 2*tau_max+1)
        batch_shape = gcc.shape[:-3]
        n_grid = len(self.grid_shape)
        maps = torch.zeros(batch_shape + self.grid_shape, device=gcc.device, dtype=gcc.dtype)
        for n in range(self.N):
            for m in range(self.N):
                vals = gcc[..., n, m, :].reshape(
                    batch_shape + (1,) * n_grid + (gcc.shape[-1],)
                ).expand(batch_shape + self.grid_shape + (gcc.shape[-1],))
                idx = self.tau_idx[n, m].reshape(
                    (1,) * len(batch_shape) + self.grid_shape + (1,)
                ).expand(batch_shape + self.grid_shape + (1,))
                maps = maps + torch.gather(vals, dim=-1, index=idx).squeeze(-1)
        if self.normalize:
            flat = maps.reshape(maps.shape[:-len(self.grid_shape)] + (-1,))
            flat = flat / (flat.amax(-1, keepdim=True) + 1e-12)
            maps = flat.reshape(maps.shape)
        return maps


class IcosahedralSRP(nn.Module):
    """Waveform -> per-window spherical SRP-PHAT maps.

    Args:
        n_mics: number of input channels (3 for DICIT).
        sample_rate: waveform sample rate.
        r: icosahedral grid resolution.
        win: analysis window in samples (48 kHz: 12288 = 256 ms).
        hop: window hop in samples (1536 = 32 ms).
        c: speed of sound.
    """

    def __init__(self, n_mics: int, sample_rate: int, mic_positions: np.ndarray,
                 r: int = 3, win: int = 12288, hop: int = 1536, c: float = 343.0):
        super().__init__()
        self.n_mics = int(n_mics)
        self.sample_rate = int(sample_rate)
        self.win = int(win)
        self.hop = int(hop)
        tau_max = int(np.ceil(np.max(
            np.linalg.norm(
                np.asarray(mic_positions, dtype=np.float64)[:n_mics, None]
                - np.asarray(mic_positions, dtype=np.float64)[None, :n_mics], axis=-1)
        ) / c * sample_rate)) + 1
        tau_max = min(tau_max, self.win // 2)
        self.gcc = GCC(n_mics, self.win, tau_max=tau_max, transform="PHAT")
        self.srp = SRPIcosahedralMap(n_mics, self.win, r, mic_positions, sample_rate,
                                     c=c, tau_max=tau_max)
        self.tau_max = tau_max

    def window_starts(self, n_samples: int) -> List[int]:
        if n_samples < self.win:
            return [0]
        return list(range(0, n_samples - self.win + 1, self.hop))

    def forward(self, waveform: torch.Tensor) -> Tuple[torch.Tensor, List[float]]:
        # waveform: (B, T, C)
        if waveform.dim() != 3:
            raise ValueError(f"Expected (B, T, C), got {tuple(waveform.shape)}")
        B, T, C = waveform.shape
        if C != self.n_mics:
            raise ValueError(f"Expected {self.n_mics} channels, got {C}")
        starts = self.window_starts(T)
        if starts and starts[-1] + self.win <= T:
            frames = torch.stack([waveform[:, s:s + self.win, :] for s in starts], dim=1)
        else:
            pad = waveform[:, -1:, :].expand(B, self.win, C)
            frames = pad.unsqueeze(1)
            starts = [0]
        # (B, Tf, C, win) -> GCC (B, Tf, C, C, lags)
        gcc = self.gcc(frames.permute(0, 1, 3, 2).contiguous())
        maps = self.srp(gcc.reshape((-1,) + gcc.shape[2:]))
        maps = maps.reshape((B, len(starts)) + maps.shape[1:])
        times = [s / float(self.sample_rate) for s in starts]
        return maps, times
