"""IPDnet2-Small baseline wrapper.

Same front end, clip aggregation and clip-level PIT as IPDnet; the backbone is
the author's Mamba-based online spatial network and decoding uses the negative
template MSE.
"""

from typing import Dict, Optional, Sequence

import numpy as np
import torch

from ..dpipd_frontend import DPIPDFrontend
from ..dpipd_tracks import DPIPDClipModel, TrackDPIPD
from .backbone import OnlineSpatialNet

_DEFAULT_NORMS = ["LN", "LN", "GN", "LN", "LN", "LN"]


class IPDnet2SmallBaseline(DPIPDClipModel):
    """Waveform ``(B, T, 3)`` at 48 kHz -> per-track DP-IPD and detections."""

    model_name = "ipdnet2_small"

    def __init__(
        self,
        mic_positions: Optional[np.ndarray] = None,
        n_mics: int = 3,
        input_sample_rate: int = 48000,
        stft_sample_rate: int = 16000,
        win_len: int = 512,
        n_fft: int = 512,
        hop: int = 320,
        forgetting_frames: int = 249,
        resample_lowpass_width: int = 6,
        resample_rolloff: float = 0.99,
        dim_hidden: int = 96,
        num_layers: int = 8,
        dim_squeeze: int = 8,
        num_heads: int = 4,
        kernel_size: Sequence[int] = (5, 3),
        conv_groups: Sequence[int] = (8, 8),
        norms: Sequence[str] = _DEFAULT_NORMS,
        attention: str = "mamba(16,4)",
        rope: bool = False,
        fre_compression_ratio: int = 16,
        time_compression_ratio: int = 5,
        time_compression_layer: int = 0,
        encoder_kernel_size: int = 5,
        max_tracks: int = 3,
        pool_factor: int = 5,
        ref_mic: int = 0,
        az_range_deg: Sequence[float] = (0.0, 180.0),
        n_az: int = 181,
        elevation_deg: float = 90.0,
        speed: float = 340.0,
        score_kind: str = "neg_mse",
    ):
        if mic_positions is None:
            raise ValueError("IPDnet2SmallBaseline requires mic_positions (C, 3)")
        mic = np.asarray(mic_positions, dtype=np.float64)
        if mic.shape != (int(n_mics), 3):
            raise ValueError(
                f"mic_positions must be ({n_mics}, 3), got {mic.shape}")

        frontend = DPIPDFrontend(
            input_sample_rate=input_sample_rate,
            stft_sample_rate=stft_sample_rate,
            win_len=win_len, n_fft=n_fft, hop_length=hop,
            n_mics=n_mics, forgetting_frames=forgetting_frames,
            resample_lowpass_width=resample_lowpass_width,
            resample_rolloff=resample_rolloff)
        n_pairs = int(n_mics) - 1
        backbone = OnlineSpatialNet(
            dim_input=2 * int(n_mics),
            dim_output=2 * n_pairs * int(max_tracks),
            num_layers=num_layers,
            dim_squeeze=dim_squeeze,
            num_freqs=frontend.n_freqs,
            encoder_kernel_size=encoder_kernel_size,
            dim_hidden=dim_hidden,
            num_heads=num_heads,
            kernel_size=tuple(kernel_size),
            conv_groups=tuple(conv_groups),
            norms=list(norms),
            attention=attention,
            rope=rope,
            fre_compression_ratio=fre_compression_ratio,
            time_compression_ratio=time_compression_ratio,
            time_compression_layer=time_compression_layer,
            n_pairs=n_pairs,
            max_tracks=max_tracks,
        )
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
            "resample_lowpass_width": int(resample_lowpass_width),
            "resample_rolloff": float(resample_rolloff),
            "dim_hidden": int(dim_hidden),
            "num_layers": int(num_layers),
            "dim_squeeze": int(dim_squeeze),
            "num_heads": int(num_heads),
            "kernel_size": [int(v) for v in kernel_size],
            "conv_groups": [int(v) for v in conv_groups],
            "norms": list(norms),
            "attention": str(attention),
            "rope": bool(rope),
            "fre_compression_ratio": int(fre_compression_ratio),
            "time_compression_ratio": int(time_compression_ratio),
            "time_compression_layer": int(time_compression_layer),
            "encoder_kernel_size": int(encoder_kernel_size),
            "max_tracks": int(max_tracks),
            "pool_factor": int(pool_factor),
            "ref_mic": int(ref_mic),
            "az_range_deg": list(az_range_deg),
            "n_az": int(n_az),
            "elevation_deg": float(elevation_deg),
            "speed": float(speed),
            "score_kind": str(score_kind),
        }
        self.score_kind = str(score_kind)

    def predict_native(self, waveform: torch.Tensor) -> Dict[str, object]:
        return self.forward(waveform)
