"""Multi-source IcoCNN: spherical SRP-PHAT -> icosahedral CNN -> ACCDOA slots.

The spherical convolution stack follows David Diaz-Guerra's ``icoCNN`` /
``icoDOA`` (AGPL-3.0).  The single-source Soft-ArgMax head is replaced by a
linear multi-slot ACCDOA head (``M`` 3-D vectors, empty slots zero), as used by
the multi-source IcoCNN paper (arXiv:2210.14536 / arXiv:2306.08510).

Deviation from the upstream tracking model: the final causal temporal
convolution keeps ``width`` channels instead of collapsing to one, because the
multi-slot head needs a wider per-frame feature.  No other layer changes.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, reduce

from .conv import CleanVertices, ConvIco, LNormIco, PoolIco, SmoothVertices
from .srp import IcosahedralSRP


class CausConv1d(nn.Module):
    """Causal 1-D convolution (left padding, right side trimmed)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=self.pad, dilation=dilation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)[:, :, : -self.pad] if self.pad else self.conv(x)


class IcoCNNMultiSource(nn.Module):
    """Waveform -> ACCDOA slots at every SRP window.

    Args:
        n_mics: input channel count (3 for DICIT).
        sample_rate: waveform sample rate.
        mic_positions: ``(C, 3)`` microphone coordinates in metres.
        r: icosahedral resolution (paper default 3).
        srp_win / srp_hop: SRP-PHAT window and hop in samples
            (48 kHz: 12288 / 1536 = 256 ms / 32 ms).
        width: convolutional width (paper default 128).
        temporal_kernel: causal temporal kernel (paper default 5).
        n_slots: number of ACCDOA output slots (paper default 10).
    """

    model_name = "icocnn"

    def __init__(
        self,
        n_mics: int = 3,
        sample_rate: int = 48000,
        mic_positions: Optional[torch.Tensor] = None,
        r: int = 3,
        srp_win: int = 12288,
        srp_hop: int = 1536,
        width: int = 128,
        temporal_kernel: int = 5,
        n_slots: int = 10,
        smooth_vertices: bool = True,
    ):
        super().__init__()
        if mic_positions is None:
            raise ValueError("IcoCNNMultiSource requires mic_positions (C, 3)")
        if isinstance(mic_positions, torch.Tensor):
            mic_np = mic_positions.detach().cpu().numpy()
        else:
            import numpy as np

            mic_np = np.asarray(mic_positions, dtype=np.float64)
        if mic_np.shape[0] != n_mics:
            raise ValueError(f"mic_positions has {mic_np.shape[0]} rows, expected {n_mics}")

        self.n_mics = int(n_mics)
        self.sample_rate = int(sample_rate)
        self.r = int(r)
        self.n_slots = int(n_slots)
        self.smooth_vertices = bool(smooth_vertices)
        self.srp = IcosahedralSRP(n_mics, sample_rate, mic_np, r=r, win=srp_win, hop=srp_hop)

        deep = 5 + (self.r - 1) * 2
        self.process_vertices = nn.ModuleList(
            [SmoothVertices(i + 1) if smooth_vertices else CleanVertices(i + 1)
             for i in range(self.r)]
        )
        self.ico_cnn = nn.ModuleList(
            [ConvIco(self.r, 1, width, 1, 6, smooth_vertices=smooth_vertices)]
            + [ConvIco(max(self.r - i // 2, 1), width, width, 6, 6,
                       smooth_vertices=smooth_vertices)
               for i in range(1, deep)]
        )
        self.temp_cnn = nn.ModuleList(
            [CausConv1d(width, width, temporal_kernel) for _ in range(deep)]
        )
        self.layer_norm = nn.ModuleList([LNormIco(width, 6) for _ in range(deep)])
        self.poolings = nn.ModuleList([PoolIco(self.r - i, 6) for i in range(self.r)])

        feat_dim = width * 5 * 2 * 4   # after r -> 1: 5 charts x 2 x 4
        self.head_norm = nn.LayerNorm(feat_dim)
        self.head = nn.Linear(feat_dim, self.n_slots * 3)

        self.config = {
            "n_mics": self.n_mics,
            "sample_rate": self.sample_rate,
            "mic_positions": mic_np.tolist(),
            "r": self.r,
            "srp_win": int(srp_win),
            "srp_hop": int(srp_hop),
            "width": int(width),
            "temporal_kernel": int(temporal_kernel),
            "n_slots": self.n_slots,
            "smooth_vertices": self.smooth_vertices,
        }

    def _apply_ico_cnn(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, charts, H, W) -> (B, T, C, 1, charts, H, W)
        B = x.shape[0]
        x = x.unsqueeze(2).unsqueeze(3)
        H, W = x.shape[-2], x.shape[-1]
        r = self.r
        for idx, (ico_conv, temp_conv, l_norm) in enumerate(
            zip(self.ico_cnn, self.temp_cnn, self.layer_norm)
        ):
            x = torch.relu(ico_conv(x))
            x = rearrange(x, "B T C R charts H W -> (B R charts H W) C T",
                          B=B, R=6, charts=5)
            x = temp_conv(x)
            x = rearrange(x, "(B R charts H W) C T -> B T C R charts H W",
                          R=6, charts=5, B=B, H=H, W=W)
            if idx < len(self.ico_cnn) - 1:
                x = l_norm(x)
            x = self.process_vertices[r - 1](x)
            if idx < len(self.ico_cnn) - 1:
                x = torch.relu(x)
            if idx % 2 == 1 and r > 1:
                x = self.poolings[idx // 2](x)
                r -= 1
                H //= 2
                W //= 2
        return reduce(x, "B T C R charts H W -> B T (C charts H W)", "max", R=6)

    def forward(self, waveform: torch.Tensor) -> Dict[str, object]:
        """Return ``{"accdoa": (B, T, M, 3), "times": List[List[float]]}``."""
        maps, times = self.srp(waveform)          # (B, T, 5, H, W)
        feats = self._apply_ico_cnn(maps)         # (B, T, D)
        out = self.head(self.head_norm(feats))
        accdoa = out.reshape(out.shape[0], out.shape[1], self.n_slots, 3)
        return {"accdoa": accdoa, "times": times}

    def predict_native(self, waveform: torch.Tensor) -> Dict[str, object]:
        return self.forward(waveform)

    def training_loss(self, waveform: torch.Tensor,
                      targets: Sequence[np.ndarray]) -> torch.Tensor:
        from .accdoa import accdoa_pit_mse

        out = self.forward(waveform)
        return accdoa_pit_mse(out["accdoa"], targets)

    @torch.no_grad()
    def decode(self, waveform: torch.Tensor, threshold_abs: float = 0.3,
               max_sources: int = 3, frame_index: int = -1) -> List[List[Dict[str, float]]]:
        from .accdoa import decode_accdoa

        out = self.forward(waveform)
        return decode_accdoa(
            out["accdoa"].detach().cpu().numpy(),
            times=out["times"],
            threshold_abs=threshold_abs,
            max_sources=max_sources,
            frame_index=frame_index,
        )
