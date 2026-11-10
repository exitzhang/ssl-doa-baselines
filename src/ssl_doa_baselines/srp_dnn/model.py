"""SRP-DNN: shared causal CRNN -> mixed DP-IPD -> IDL.

Ported from ``SRP-DNN/code/ModelSRPDNN.py`` (CRNN) and the preprocessing /
target construction in ``LearnerSRPDNN.py`` (MIT).  See ``THIRD_PARTY.md``.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from .dpipd import DPIPD
from .idl import SourceDetectLocalize
from .stft import STFT, AddChToBatch, RemoveChFromBatch, forgetting_norm


class CausCnnBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel=(3, 3), stride=(1, 1),
                 padding=(1, 2), use_res=True, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel, stride=stride,
                               padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel, stride=stride,
                               padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.pad = padding
        self.use_res = use_res

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if self.pad[1] != 0:
            out = out[:, :, :, : -self.pad[1]]
        out = self.conv2(out)
        out = self.bn2(out)
        if self.pad[1] != 0:
            out = out[:, :, :, : -self.pad[1]]
        if self.use_res:
            if self.downsample is not None:
                residual = self.downsample(x)
            out = out + residual
        return self.relu(out)


class CRNN(nn.Module):
    """Shared causal CRNN; input ``(B*P, 4, 256, T)``, output ``(B*P, T', 512)``."""

    def __init__(self, max_num_sources: int = 3):
        super().__init__()
        cnn_in_dim, cnn_dim = 4, 64
        self.blocks = nn.ModuleList([
            CausCnnBlock(cnn_in_dim, cnn_dim, padding=(1, 2), use_res=False),
            CausCnnBlock(cnn_dim, cnn_dim, padding=(1, 2), use_res=False),
            CausCnnBlock(cnn_dim, cnn_dim, padding=(1, 2), use_res=False),
            CausCnnBlock(cnn_dim, cnn_dim, padding=(1, 2), use_res=False),
            CausCnnBlock(cnn_dim, cnn_dim, padding=(1, 2), use_res=False),
        ])
        self.pools = nn.ModuleList([
            nn.MaxPool2d(kernel_size=(4, 1)),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.MaxPool2d(kernel_size=(2, 2)),
        ])
        self.rnn = nn.GRU(input_size=256, hidden_size=256, num_layers=1,
                          batch_first=True, bias=True, dropout=0.4, bidirectional=False)
        self.rnn_fc = nn.Sequential(nn.Linear(256, 512), nn.Tanh())
        self.max_num_sources = int(max_num_sources)

    def forward(self, fea: torch.Tensor) -> torch.Tensor:
        nb = fea.shape[0]
        x = fea
        # The upstream last pool has kernel (2, 3).  A 16000-sample clip yields
        # 19 STFT frames, which leaves only 2 frames before the last pool, so
        # the time kernel is clipped to the available length.  This changes no
        # parameters and preserves the 12x total time reduction for long clips.
        for block, pool in zip(self.blocks, self.pools):
            x = pool(block(x))
        kt = min(3, x.shape[-1])
        if x.shape[-1] >= 3:
            x = torch.nn.functional.max_pool2d(x, kernel_size=(2, 3))
        else:
            x = torch.nn.functional.max_pool2d(x, kernel_size=(2, kt))
        fea_rnn_in = x.reshape(nb, -1, x.shape[-1]).permute(0, 2, 1)
        fea_rnn, _ = self.rnn(fea_rnn_in)
        return self.rnn_fc(fea_rnn) * self.max_num_sources


class SRPDNN(nn.Module):
    """Waveform -> predicted mixed DP-IPD, with optional IDL decoding.

    Args:
        n_mics: channels (3 for DICIT).
        sample_rate: waveform rate.
        mic_positions: ``(C, 3)`` metres.
        win_len / hop / n_fft: STFT (48 kHz: 1536 / 768 / 1536).
        freq_min / freq_max: used STFT bins (31.25--8000 Hz -> k=1..256).
        max_num_sources: CRNN scale and IDL iteration limit.
        az_range_deg / el_range_deg / n_az / n_el: IDL candidate grid.
    """

    model_name = "srp_dnn"

    def __init__(
        self,
        n_mics: int = 3,
        sample_rate: int = 48000,
        mic_positions: Optional[np.ndarray] = None,
        win_len: int = 1536,
        hop: int = 768,
        n_fft: int = 1536,
        freq_min: float = 31.25,
        freq_max: float = 8000.0,
        max_num_sources: int = 3,
        az_range_deg: Sequence[float] = (0.0, 180.0),
        el_range_deg: Sequence[float] = (60.0, 120.0),
        n_az: int = 181,
        n_el: int = 7,
    ):
        super().__init__()
        if mic_positions is None:
            raise ValueError("SRPDNN requires mic_positions (C, 3)")
        mic = np.asarray(mic_positions, dtype=np.float64)
        if mic.shape[0] != n_mics:
            raise ValueError(f"mic_positions has {mic.shape[0]} rows, expected {n_mics}")
        self.n_mics = int(n_mics)
        self.sample_rate = int(sample_rate)
        self.win_len = int(win_len)
        self.hop = int(hop)
        self.n_fft = int(n_fft)
        self.max_num_sources = int(max_num_sources)
        self.n_az = int(n_az)
        self.n_el = int(n_el)

        n_bins = self.n_fft // 2 + 1
        k_min = max(1, int(np.ceil(freq_min * self.n_fft / self.sample_rate)))
        k_max = min(n_bins - 1, int(np.floor(freq_max * self.n_fft / self.sample_rate)))
        self.freq_slice = slice(k_min, k_max + 1)
        self.n_freq = k_max - k_min + 1
        self.freqs = np.arange(k_min, k_max + 1) * self.sample_rate / self.n_fft

        self.stft = STFT(self.win_len, self.hop, self.n_fft)
        self.pairs = AddChToBatch(self.n_mics)
        self.unpairs = RemoveChFromBatch(self.n_mics)
        self.crnn = CRNN(self.max_num_sources)
        self.dpipd = DPIPD(mic, self.freqs, az_range_deg=az_range_deg, n_az=n_az,
                           el_range_deg=el_range_deg, n_el=n_el)
        self.idl = SourceDetectLocalize(self.max_num_sources, self.n_freq, self.pairs.n_pairs)

        self.config = {
            "n_mics": self.n_mics,
            "sample_rate": self.sample_rate,
            "mic_positions": mic.tolist(),
            "win_len": self.win_len,
            "hop": self.hop,
            "n_fft": self.n_fft,
            "freq_min": float(freq_min),
            "freq_max": float(freq_max),
            "max_num_sources": self.max_num_sources,
            "az_range_deg": list(az_range_deg),
            "el_range_deg": list(el_range_deg),
            "n_az": self.n_az,
            "n_el": self.n_el,
        }

    def features(self, waveform: torch.Tensor):
        """Return model input, predicted time stamps and STFT length."""
        stft = self.stft(waveform)                          # (B,F,T,C)
        stft = stft.permute(0, 3, 1, 2)                      # (B,C,F,T)
        mag = torch.abs(stft)
        mean_value = forgetting_norm(mag, num_frame_set=256).expand(mag.shape)
        stft = stft / (mean_value + 1e-6)
        rebatched = self.pairs(stft)                         # (B*P,2,F,T)
        mag = torch.log10(torch.abs(rebatched) + 1e-6)
        phase = torch.angle(rebatched)
        feats = torch.cat((mag, phase), dim=1)[:, :, self.freq_slice, :]
        times = (torch.arange(stft.shape[-1], device=waveform.device)
                 * self.hop / float(self.sample_rate))
        return feats, times, stft.shape[-1]

    def forward(self, waveform: torch.Tensor) -> Dict[str, object]:
        feats, times, _ = self.features(waveform)
        pred = self.crnn(feats)                              # (B*P,T',2F)
        B = waveform.shape[0]
        T = pred.shape[1]
        ipd = pred.reshape(B, self.pairs.n_pairs, T, 2 * self.n_freq)
        ipd = ipd.permute(0, 2, 3, 1).contiguous()           # (B,T,2F,P)
        return {"ipd": ipd, "times": times}

    def predict_native(self, waveform: torch.Tensor) -> Dict[str, object]:
        return self.forward(waveform)

    def training_loss(self, waveform: torch.Tensor,
                      targets: Sequence[np.ndarray],
                      beta: Optional[Sequence[Optional[np.ndarray]]] = None,
                      ) -> torch.Tensor:
        out = self.forward(waveform)
        ipd = out["ipd"]
        B, T, F2, P = ipd.shape
        tgt = np.stack([
            self.dpipd.mixed_target(targets[b], None if beta is None else beta[b])
            for b in range(B)
        ])
        tgt = torch.as_tensor(tgt, device=ipd.device, dtype=ipd.dtype)
        tgt = tgt[:, None, :, :].expand(B, T, F2, P)
        return torch.nn.functional.mse_loss(ipd, tgt)

    @torch.no_grad()
    def decode(self, waveform: torch.Tensor, threshold_abs: float = 0.3,
               max_sources: int = 3, frame_index: int = -1) -> List[List[Dict[str, float]]]:
        out = self.forward(waveform)
        ipd = out["ipd"]
        idl = self.idl(ipd, self.dpipd.template, self.dpipd.az_deg, self.dpipd.el_deg)
        doa = idl["doa"].detach().cpu().numpy()              # (B,T,2,S)
        vad = idl["vad"].detach().cpu().numpy()              # (B,T,S)
        B, T = doa.shape[:2]
        fi = frame_index if frame_index >= 0 else T + frame_index
        fi = int(np.clip(fi, 0, T - 1))
        times = out["times"].detach().cpu().numpy() if torch.is_tensor(out["times"]) else np.asarray(out["times"])
        dets_all: List[List[Dict[str, float]]] = []
        for b in range(B):
            order = np.argsort(-vad[b, fi])
            dets = []
            for s in order:
                if float(vad[b, fi, s]) < float(threshold_abs):
                    continue
                dets.append({
                    "azimuth_deg": float(doa[b, fi, 0, s]),
                    "elevation_deg": float(doa[b, fi, 1, s]),
                    "score": float(vad[b, fi, s]),
                    "time_s": float(times[fi]) if len(times) > fi else 0.0,
                })
                if len(dets) >= int(max_sources):
                    break
            dets_all.append(dets)
        return dets_all
