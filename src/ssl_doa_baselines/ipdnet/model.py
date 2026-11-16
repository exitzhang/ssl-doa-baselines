"""IPDnet baseline wrapper: shared 48->16 kHz front end + fixed-array backbone.

Training and inference both use clip-level aggregation followed by
permutation-invariant DP-IPD MSE.  Decoding uses the per-track projection
ratio (``dot(template, track) / dot(template, template)``).
"""

from typing import Dict, Optional, Sequence

import numpy as np
import torch

from ..dpipd_frontend import DPIPDFrontend
from ..dpipd_tracks import DPIPDClipModel, TrackDPIPD
from .backbone import IPDnetBackbone


class IPDNetBaseline(DPIPDClipModel):
    """Waveform ``(B, T, 3)`` at 48 kHz -> per-track DP-IPD and detections."""

    model_name = "ipdnet"

    def __init__(
        self,
        mic_positions: Optional[np.ndarray] = None,
        n_mics: int = 3,
        input_sample_rate: int = 48000,
        stft_sample_rate: int = 16000,
        win_len: int = 512,
        n_fft: int = 512,
        hop: int = 256,
        forgetting_frames: int = 280,
        hidden_size: int = 128,
        max_tracks: int = 3,
        pool_factor: int = 12,
        ref_mic: int = 0,
        az_range_deg: Sequence[float] = (0.0, 180.0),
        n_az: int = 181,
        elevation_deg: float = 90.0,
        speed: float = 340.0,
        dropout: float = 0.2,
        score_kind: str = "ratio",
    ):
        if mic_positions is None:
            raise ValueError("IPDNetBaseline requires mic_positions (C, 3)")
        mic = np.asarray(mic_positions, dtype=np.float64)
        if mic.shape != (int(n_mics), 3):
            raise ValueError(
                f"mic_positions must be ({n_mics}, 3), got {mic.shape}")

        frontend = DPIPDFrontend(
            input_sample_rate=input_sample_rate,
            stft_sample_rate=stft_sample_rate,
            win_len=win_len, n_fft=n_fft, hop_length=hop,
            n_mics=n_mics, forgetting_frames=forgetting_frames)
        backbone = IPDnetBackbone(
            input_size=2 * int(n_mics), hidden_size=hidden_size,
            max_tracks=max_tracks, n_pairs=int(n_mics) - 1, dropout=dropout)
        tracks = TrackDPIPD(
            mic, frontend.freqs_hz, ref_mic=ref_mic,
            max_tracks=max_tracks, pool_factor=pool_factor,
            az_range_deg=az_range_deg, n_az=n_az,
            elevation_deg=elevation_deg, speed=speed)
        super().__init__(frontend, backbone, tracks)

        self.n_mics = int(n_mics)
        self.config = {
            "mic_positions": mic.tolist(),
            "n_mics": self.n_mics,
            "input_sample_rate": int(input_sample_rate),
            "stft_sample_rate": int(stft_sample_rate),
            "win_len": int(win_len),
            "n_fft": int(n_fft),
            "hop": int(hop),
            "forgetting_frames": int(forgetting_frames),
            "hidden_size": int(hidden_size),
            "max_tracks": int(max_tracks),
            "pool_factor": int(pool_factor),
            "ref_mic": int(ref_mic),
            "az_range_deg": list(az_range_deg),
            "n_az": int(n_az),
            "elevation_deg": float(elevation_deg),
            "speed": float(speed),
            "dropout": float(dropout),
            "score_kind": str(score_kind),
        }
        self.score_kind = str(score_kind)

    def predict_native(self, waveform: torch.Tensor) -> Dict[str, object]:
        return self.forward(waveform)
