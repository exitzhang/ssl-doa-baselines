"""Iterative detection and localization (IDL).

Ported from ``SRP-DNN/code/Module.py::SourceDetectLocalize`` (MIT).  The
per-(batch,time) Python loop is replaced by vectorised gather/argmax; the
algorithm is unchanged: match the mixed DP-IPD against every candidate
template, take the best candidate, estimate its activity ratio, subtract the
scaled template, and repeat up to ``max_num_sources`` times.
"""

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn


class SourceDetectLocalize(nn.Module):
    def __init__(self, max_num_sources: int = 3, n_freq: int = 256, n_pairs: int = 3):
        super().__init__()
        self.max_num_sources = int(max_num_sources)
        self.n_freq = int(n_freq)
        self.n_pairs = int(n_pairs)
        self.denom = float(self.n_freq * self.n_pairs / 2.0)

    def forward(self, pred_ipd: torch.Tensor, template: torch.Tensor,
                az_deg: np.ndarray, el_deg: np.ndarray) -> Dict[str, torch.Tensor]:
        """Args:
            pred_ipd: ``(B, T, 2F, P)``.
            template: ``(D, 2F, P)``.
            az_deg / el_deg: candidate grids, ``D = len(el)*len(az)`` with
                elevation-major ordering.
        Returns:
            dict with ``doa`` ``(B,T,2,S)`` degrees, ``vad`` ``(B,T,S)``,
            ``spectrum`` ``(B,T,D)``.
        """
        device = pred_ipd.device
        B, T, K, P = pred_ipd.shape
        if K != 2 * self.n_freq or P != self.n_pairs:
            raise ValueError(
                f"pred_ipd shape {tuple(pred_ipd.shape)} incompatible with "
                f"n_freq={self.n_freq}, n_pairs={self.n_pairs}"
            )
        tpl = template.reshape(template.shape[0], -1)          # (D, K*P)
        x = pred_ipd.reshape(B, T, -1)                          # (B, T, K*P)

        doa = pred_ipd.new_zeros((B, T, 2, self.max_num_sources))
        vad = pred_ipd.new_zeros((B, T, self.max_num_sources))
        spectrum = None
        residual = x
        for s in range(self.max_num_sources):
            scores = torch.matmul(residual, tpl.t()) / self.denom  # (B,T,D)
            if s == 0:
                spectrum = scores
            best = scores.argmax(dim=-1)                            # (B,T)
            tpl_best = tpl[best.reshape(-1)].reshape(B, T, -1)      # (B,T,K*P)
            ratio = (tpl_best * residual).sum(-1) / (
                (tpl_best * tpl_best).sum(-1) + 1e-12
            )
            doa[:, :, :, s] = self._candidate_angles(best, az_deg, el_deg, device)
            vad[:, :, s] = ratio
            residual = residual - ratio.unsqueeze(-1) * tpl_best
        return {"doa": doa, "vad": vad, "spectrum": spectrum}

    @staticmethod
    def _candidate_angles(best: torch.Tensor, az_deg: np.ndarray,
                          el_deg: np.ndarray, device) -> torch.Tensor:
        n_az = len(az_deg)
        el_idx = (best // n_az).long()
        az_idx = (best % n_az).long()
        az = torch.as_tensor(az_deg, device=device, dtype=torch.float32)
        el = torch.as_tensor(el_deg, device=device, dtype=torch.float32)
        return torch.stack([az[az_idx], el[el_idx]], dim=-1)
