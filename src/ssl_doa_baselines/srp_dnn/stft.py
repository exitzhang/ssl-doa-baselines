"""STFT / feature front-end for SRP-DNN.

Ported from ``SRP-DNN/code/Module.py`` (STFT, AddChToBatch,
RemoveChFromBatch) and ``SRP-DNN/code/utils.py`` (forgetting_norm), MIT.
"""

from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn


def forgetting_norm(input: torch.Tensor, num_frame_set: int = None) -> torch.Tensor:
    """Online per-frame magnitude normalisation, returns ``(B, 1, 1, T)``."""
    assert input.ndim == 4
    batch_size, num_channels, num_freqs, num_frames = input.size()
    input = input.reshape(batch_size, num_channels * num_freqs, num_frames)
    if num_frame_set is None:
        num_frame_set = deepcopy(num_frames)

    mu = 0
    mu_list = []
    for frame_idx in range(num_frames):
        if frame_idx <= num_frame_set:
            alpha = (frame_idx - 1) / (frame_idx + 1)
        else:
            alpha = (num_frame_set - 1) / (num_frame_set + 1)
        current = torch.mean(input[:, :, frame_idx], dim=1).reshape(batch_size, 1)
        mu = alpha * mu + (1 - alpha) * current
        mu_list.append(mu)
    mu = torch.stack(mu_list, dim=-1)
    return mu.reshape(batch_size, 1, 1, num_frames)


class STFT(nn.Module):
    """Upstream STFT: ``(B, T, C)`` -> complex ``(B, F, Tf, C)``, center=False."""

    def __init__(self, win_len: int = 1536, win_shift: int = 768, nfft: int = 1536,
                 win: str = "hann"):
        super().__init__()
        self.win_len = int(win_len)
        self.win_shift = int(win_shift)
        self.nfft = int(nfft)
        self.win = win

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        nb, nsample, nch = signal.shape
        nf = self.nfft // 2 + 1
        nt = int(np.floor((nsample - self.win_len) / self.win_shift + 1))
        if nt <= 0:
            raise ValueError(f"Signal too short ({nsample}) for win_len={self.win_len}")
        stft = torch.zeros((nb, nf, nt, nch), dtype=torch.complex64, device=signal.device)
        window = None
        if self.win == "hann":
            window = torch.hann_window(self.win_len, device=signal.device)
        for ch in range(nch):
            stft[:, :, :, ch] = torch.stft(
                signal[:, :, ch], n_fft=self.nfft, hop_length=self.win_shift,
                win_length=self.win_len, window=window, center=False,
                normalized=False, return_complex=True,
            )
        return stft


class AddChToBatch(nn.Module):
    """``(B, C, F, T)`` -> ``(B * n_pairs, 2, F, T)`` for unordered mic pairs."""

    def __init__(self, n_mics: int):
        super().__init__()
        self.n_mics = int(n_mics)
        self.pairs = [(i, j) for i in range(self.n_mics) for j in range(i + 1, self.n_mics)]
        self.n_pairs = len(self.pairs)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        nb, nch, nf, nt = data.shape
        out = torch.zeros((nb * self.n_pairs, 2, nf, nt), dtype=data.dtype, device=data.device)
        for p, (i, j) in enumerate(self.pairs):
            out[p::self.n_pairs, 0] = data[:, i]
            out[p::self.n_pairs, 1] = data[:, j]
        return out


class RemoveChFromBatch(nn.Module):
    """``(B * P, T, ...)`` -> ``(B, T, ..., P)``."""

    def __init__(self, n_mics: int):
        super().__init__()
        self.n_mics = int(n_mics)
        self.n_pairs = n_mics * (n_mics - 1) // 2

    def forward(self, data: torch.Tensor, nb: int) -> torch.Tensor:
        return data.reshape(nb, self.n_pairs, *data.shape[1:])

    def interleave_to_last(self, data: torch.Tensor, nb: int) -> torch.Tensor:
        """``(B*P, T, 2F)`` -> ``(B, T, 2F, P)``."""
        x = self.forward(data, nb)              # (B, P, T, 2F)
        return x.permute(0, 2, 3, 1)
