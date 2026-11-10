"""TF-Mamba multi-source azimuth spectrum model.

Adapted from the historical LOCATA implementation (commit ``b33a2ab``,
``models/TF_Mamba.py``) which in turn follows Xiao et al., Interspeech 2025.
The original model regresses a single-source azimuth spectrum from two
microphones; this adaptation keeps the DenseNet encoder, bidirectional
temporal/frequency Mamba blocks and same-axis skip connections, but:

* accepts an arbitrary channel count (3 mics for DICIT);
* selects a physical frequency band (default 100--8000 Hz);
* exposes the output activation as a configuration knob, because the LOCATA
  heatmap loss expects a probability-like ``[0, 1]`` spectrum (sigmoid), while
  the historical implementation used ``tanh``.
"""

from typing import List, Optional

import torch
import torch.nn as nn


class BiMamba(nn.Module):
    """Bidirectional Mamba over the sequence axis (forward + reversed, mean)."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        from mamba_ssm import Mamba

        self.forward_mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.backward_mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_fwd = self.forward_mamba(x)
        x_rev = torch.flip(x, dims=[1])
        out_rev = self.backward_mamba(x_rev)
        out_rev = torch.flip(out_rev, dims=[1])
        return (out_fwd + out_rev) / 2


class DilatedDenseLayer(nn.Module):
    def __init__(self, in_channels: int, growth_rate: int, dilation: int):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(
            in_channels, growth_rate, kernel_size=3, padding=dilation, dilation=dilation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(self.relu(self.bn1(x)))
        return torch.cat([x, out], dim=1)


class DilatedDenseNet(nn.Module):
    def __init__(self, in_channels: int, growth_rate: int = 16, num_layers: int = 4):
        super().__init__()
        layers = []
        curr = in_channels
        dilations = [1, 2, 4, 8]
        for i in range(num_layers):
            layers.append(DilatedDenseLayer(curr, growth_rate, dilations[i % len(dilations)]))
            curr += growth_rate
        self.densenet = nn.Sequential(*layers)
        self.out_channels = curr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.densenet(x)


class FeatureEncoder(nn.Module):
    """Two convolutional layers flanking a dilated DenseNet.

    Input  ``(B, 2C, T, F)`` (STFT real/imag), output ``(B, d_model, T, F)``.
    """

    def __init__(self, in_channels: int, d_model: int, growth_rate: int = 16,
                 densenet_layers: int = 4):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.densenet = DilatedDenseNet(32, growth_rate, densenet_layers)
        self.conv2 = nn.Conv2d(self.densenet.out_channels, d_model, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(d_model)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.densenet(out)
        out = self.relu2(self.bn2(self.conv2(out)))
        return out


class TemporalMambaBlock(nn.Module):
    """Mamba along the frequency axis for every time frame (named as in [1])."""

    def __init__(self, d_model: int, expand: int = 2, d_state: int = 16, d_conv: int = 4):
        super().__init__()
        self.bimamba = BiMamba(d_model, d_state, d_conv, expand)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D, T, F = x.shape
        x_perm = x.permute(0, 2, 3, 1).reshape(B * T, F, D)
        out = self.bimamba(x_perm)
        out = out.reshape(B, T, F, D).permute(0, 3, 1, 2)
        out = out + x
        return self.norm(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class FrequencyMambaBlock(nn.Module):
    """Mamba along the time axis for every frequency bin."""

    def __init__(self, d_model: int, expand: int = 2, d_state: int = 16, d_conv: int = 4):
        super().__init__()
        self.bimamba = BiMamba(d_model, d_state, d_conv, expand)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D, T, F = x.shape
        x_perm = x.permute(0, 3, 2, 1).reshape(B * F, T, D)
        out = self.bimamba(x_perm)
        out = out.reshape(B, F, T, D).permute(0, 3, 2, 1)
        out = out + x
        return self.norm(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class TFMambaBlock(nn.Module):
    """One block = temporal (frequency-axis) Mamba then frequency (time-axis) Mamba."""

    def __init__(self, d_model: int, expand: int = 2, d_state: int = 16, d_conv: int = 4):
        super().__init__()
        self.temporal_mamba = TemporalMambaBlock(d_model, expand, d_state, d_conv)
        self.frequency_mamba = FrequencyMambaBlock(d_model, expand, d_state, d_conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.frequency_mamba(self.temporal_mamba(x))


class OutputDecoder(nn.Module):
    """Clip-level readout: global time-frequency average -> linear -> activation."""

    def __init__(self, d_model: int, output_dim: int = 181, activation: str = "sigmoid"):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(d_model, output_dim)
        if activation not in {"sigmoid", "tanh", "none"}:
            raise ValueError(f"Unsupported activation: {activation}")
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.global_pool(x).squeeze(-1).squeeze(-1)
        x = self.fc(x)
        if self.activation == "sigmoid":
            return torch.sigmoid(x)
        if self.activation == "tanh":
            return torch.tanh(x)
        return x


class TFMambaSpectrum(nn.Module):
    """TF-Mamba returning a clip-level azimuth spectrum ``(B, out_bins)``.

    Args:
        num_mics: number of microphone channels in the input waveform.
        sample_rate: sampling rate of the input waveform.
        n_fft / hop_length: STFT parameters (48 kHz: 1536 / 480 = 32 / 10 ms).
        freq_min / freq_max: physical analysis band in Hz.
        d_model, num_blocks, expand_factors, d_state, d_conv: backbone.
        output_dim: number of azimuth spectrum points (Q).
        activation: output activation before the LOCATA heatmap loss.
    """

    model_name = "tf_mamba"

    def __init__(
        self,
        num_mics: int = 3,
        sample_rate: int = 48000,
        n_fft: int = 1536,
        hop_length: int = 480,
        freq_min: float = 100.0,
        freq_max: float = 8000.0,
        d_model: int = 64,
        num_blocks: int = 5,
        expand_factors: Optional[List[int]] = None,
        d_state: int = 16,
        d_conv: int = 4,
        output_dim: int = 181,
        activation: str = "sigmoid",
        window_fn=torch.hann_window,
    ):
        super().__init__()
        if expand_factors is None:
            expand_factors = [2, 2, 4, 4, 8]
        if len(expand_factors) != num_blocks:
            raise ValueError("expand_factors length must equal num_blocks")

        self.num_mics = int(num_mics)
        self.sample_rate = int(sample_rate)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.output_dim = int(output_dim)

        n_bins = self.n_fft // 2 + 1
        freqs = torch.arange(n_bins, dtype=torch.float64) * self.sample_rate / self.n_fft
        self.register_buffer("freqs", freqs.to(torch.float32), persistent=False)
        k_min = int(torch.ceil(torch.tensor(freq_min * self.n_fft / self.sample_rate)).item())
        k_max = int(torch.floor(torch.tensor(freq_max * self.n_fft / self.sample_rate)).item())
        k_min = max(1, k_min)
        k_max = min(n_bins - 1, k_max)
        if k_max <= k_min:
            raise ValueError("Empty analysis band; check freq_min/freq_max/n_fft/sample_rate")
        self.register_buffer("freq_slice", torch.arange(k_min, k_max + 1), persistent=False)
        self.freq_min = float(freq_min)
        self.freq_max = float(freq_max)

        self.register_buffer("window", window_fn(self.n_fft))
        self.encoder = FeatureEncoder(2 * self.num_mics, d_model)
        self.blocks = nn.ModuleList(
            [TFMambaBlock(d_model, expand=exp, d_state=d_state, d_conv=d_conv)
             for exp in expand_factors]
        )
        self.decoder = OutputDecoder(d_model, output_dim, activation=activation)
        self.config = {
            "num_mics": self.num_mics,
            "sample_rate": self.sample_rate,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "freq_min": self.freq_min,
            "freq_max": self.freq_max,
            "d_model": d_model,
            "num_blocks": num_blocks,
            "expand_factors": list(expand_factors),
            "d_state": d_state,
            "d_conv": d_conv,
            "output_dim": self.output_dim,
            "activation": activation,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4 and x.shape[1] == 1:
            x = x.squeeze(1)
        if x.dim() != 3:
            raise ValueError(f"Expected (B, T, C) input, got {tuple(x.shape)}")
        B, T, C = x.shape
        if C != self.num_mics:
            raise ValueError(f"Expected {self.num_mics} channels, got {C}")

        # (B, C, T) -> batched STFT over channels
        x = x.permute(0, 2, 1).reshape(B * C, T)
        spec = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=True,
            pad_mode="reflect",
            normalized=False,
            return_complex=True,
        )  # (B*C, F, Tf)
        spec = spec.reshape(B, C, spec.shape[-2], spec.shape[-1])
        real, imag = spec.real, spec.imag
        net_input = torch.cat([real, imag], dim=1)          # (B, 2C, F, Tf)
        net_input = net_input[:, :, self.freq_slice, :]      # physical band
        net_input = net_input.permute(0, 1, 3, 2)            # (B, 2C, Tf, F)

        feat = self.encoder(net_input)
        out = feat
        for block in self.blocks:
            out = block(out)
        return self.decoder(out)

    def predict_native(self, x: torch.Tensor) -> dict:
        return {"spectrum": self.forward(x)}

    def training_loss(self, x: torch.Tensor, targets, sigma_bins: float = 1.5) -> torch.Tensor:
        """Standalone-loss: MSE against a max-of-Gaussians azimuth heatmap."""
        pred = self.forward(x)
        target = self.gaussian_heatmap(
            targets, self.output_dim, sigma_bins, azimuth_range=(0.0, 180.0)
        ).to(device=pred.device, dtype=pred.dtype)
        return torch.nn.functional.mse_loss(pred, target)

    @staticmethod
    def gaussian_heatmap(targets, n_bins: int, sigma_bins: float = 1.5,
                         azimuth_range=(0.0, 180.0)) -> torch.Tensor:
        """``targets`` = sequence of ``(N, 2)`` (az, el) arrays -> ``(B, n_bins)``."""
        grid = torch.linspace(float(azimuth_range[0]), float(azimuth_range[1]), n_bins)
        rows = []
        for angles in targets:
            angles = torch.as_tensor(angles, dtype=torch.float32).reshape(-1, 2)
            heat = torch.zeros(n_bins)
            if angles.numel():
                bins = (angles[:, 0] - float(azimuth_range[0])) / (
                    float(azimuth_range[1]) - float(azimuth_range[0])
                ) * (n_bins - 1)
                dist = (grid[None, :] - bins[:, None]) / float(sigma_bins)
                heat = torch.exp(-0.5 * dist ** 2).max(dim=0).values
            rows.append(heat)
        return torch.stack(rows, dim=0)
