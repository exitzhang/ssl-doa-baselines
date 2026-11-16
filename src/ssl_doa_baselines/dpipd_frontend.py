"""Shared 48 kHz -> 16 kHz DP-IPD front end for IPDnet and IPDnet2.

The recipe follows the authors' fixed-array entries
(``IPDnet/runIPDnetOn.py`` and ``IPDnet2/run_IPDnet2.py``, MIT): one
anti-aliasing resampler shared by all channels, a 512-point Hann window,
512-point FFT and online forgetting-magnitude normalisation.  Two details are
deliberate:

* ``center=False`` and an explicit right-hand zero pad complete the last
  window.  This keeps the authors' 32 ms window / 31.25 Hz spacing while
  making the number of STFT frames depend only on the clip length and hop.
* DC is included in the magnitude used for forgetting normalisation and then
  dropped, so the returned features cover ``k = 1 .. n_fft // 2`` (31.25--
  8000 Hz at 16 kHz and ``n_fft=512``).

The output is ``(B, 2 * n_mics, F, T)`` with every real part followed by every
imaginary part.  Downstream backbones reorder it to ``(B, T, 2F, P, K)``.
"""

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .srp_dnn.stft import forgetting_norm


class DPIPDFrontend(nn.Module):
    """Waveform ``(B, T, C)`` at ``input_sample_rate`` -> DP-IPD features.

    Args:
        input_sample_rate: waveform rate (48 kHz for LOCATA).
        stft_sample_rate: internal rate after the shared resampler (16 kHz).
        win_len / n_fft: analysis window and FFT size (both 512).
        hop_length: STFT hop; 256 for IPDnet, 320 for IPDnet2.
        n_mics: expected channel count.
        forgetting_frames: forgetting-norm sample length (280 / 249).
        eps: normalisation floor.
    """

    def __init__(self, input_sample_rate: int = 48000, stft_sample_rate: int = 16000,
                 win_len: int = 512, n_fft: int = 512, hop_length: int = 256,
                 n_mics: int = 3, forgetting_frames: int = 280, eps: float = 1e-6):
        super().__init__()
        import torchaudio

        if win_len > n_fft:
            raise ValueError(f"win_len ({win_len}) must not exceed n_fft ({n_fft})")
        self.input_sample_rate = int(input_sample_rate)
        self.stft_sample_rate = int(stft_sample_rate)
        self.win_len = int(win_len)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.n_mics = int(n_mics)
        self.forgetting_frames = int(forgetting_frames)
        self.eps = float(eps)

        self.resampler = torchaudio.transforms.Resample(
            self.input_sample_rate, self.stft_sample_rate)
        self.register_buffer("window", torch.hann_window(self.win_len),
                             persistent=False)

        self.n_bins = self.n_fft // 2 + 1
        self.bins = np.arange(1, self.n_bins, dtype=np.int64)
        self.freqs_hz = self.bins.astype(np.float64) * self.stft_sample_rate / self.n_fft

    @property
    def n_freqs(self) -> int:
        return int(self.bins.size)

    def forward(self, waveform: torch.Tensor) -> Dict[str, object]:
        if waveform.dim() != 3:
            raise ValueError(
                f"DPIPDFrontend expects (B, T, C), got {tuple(waveform.shape)}")
        if waveform.shape[-1] != self.n_mics:
            raise ValueError(
                f"DPIPDFrontend expects {self.n_mics} channels, got "
                f"{waveform.shape[-1]}")

        x = waveform.float().transpose(1, 2).contiguous()     # (B, C, T)
        # Resampling, FFT and forgetting norm stay in FP32 even under AMP.
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = self.resampler(x.float())                     # (B, C, T16)
            n_samples = x.shape[-1]
            if n_samples < self.win_len:
                x = F.pad(x, (0, self.win_len - n_samples))
                n_samples = self.win_len
            pad = (self.hop_length - (n_samples - self.win_len) % self.hop_length
                   ) % self.hop_length
            if pad:
                x = F.pad(x, (0, pad))
            n_samples = x.shape[-1]

            batch, channels = x.shape[0], x.shape[1]
            stft = torch.stft(
                x.reshape(batch * channels, n_samples),
                n_fft=self.n_fft, hop_length=self.hop_length,
                win_length=self.win_len, window=self.window, center=False,
                normalized=False, return_complex=True,
            )                                                 # (B*C, F, T)
            n_frames = stft.shape[-1]
            stft = stft.reshape(batch, channels, self.n_bins, n_frames)

            magnitude = torch.abs(stft)
            mean = forgetting_norm(magnitude, num_frame_set=self.forgetting_frames)
            normalized = stft / (mean + self.eps)

            real = normalized.real[:, :, self.bins, :]
            imag = normalized.imag[:, :, self.bins, :]
            features = torch.cat([real, imag], dim=1).contiguous()

        return {
            "features": features,                              # (B, 2C, F, T)
            "stft_frames": int(n_frames),
            "freqs_hz": self.freqs_hz.copy(),
            "input_sample_rate": self.input_sample_rate,
            "stft_sample_rate": self.stft_sample_rate,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
        }
