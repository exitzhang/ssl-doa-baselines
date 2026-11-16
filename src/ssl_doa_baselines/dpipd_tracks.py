"""Per-track direct-path IPD: targets, empty tracks, templates and decoding.

Shared by IPDnet and IPDnet2-Small.  The layout is ``(2F, P, K)``: the
frequency axis holds every real part first and every imaginary part second,
``P`` is the number of microphone pairs ``(reference, other)`` and ``K`` is the
track slot count.  All angles use the project convention (zenith elevation).

The physical relation follows ``geometry.py`` and SRP-DNN: for a source in
direction ``u`` and pair ``(i, j) = (reference, other)`` the delay relative to
the reference is ``tau = u . (p_j - p_i) / c`` and the direct-path IPD is
``exp(-2j pi f tau)``, i.e. ``(cos(2 pi f tau), -sin(2 pi f tau))``.

Empty (inactive) tracks use the diffuse-coherence value of the first-kind
Bessel function, ``J0(2 pi f d / c)`` real and zero imaginary, where ``d`` is
the spacing of that microphone pair.  This mirrors the authors' non-source
target and keeps every track in the same MSE.

Decoding returns one ``(score, azimuth, elevation)`` per track:

* ``score_kind="ratio"``   -- best direct-path projection ratio
  ``dot(template, track) / dot(template, template)`` (IPDnet);
* ``score_kind="neg_mse"`` -- negative mean squared error against every
  template component (IPDnet2), so ``score >= threshold`` implements
  ``MSE <= -threshold``.
"""

import itertools
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.special import j0


class TrackDPIPD:
    """Build per-track DP-IPD targets/templates and decode tracks."""

    def __init__(self, mic_positions: np.ndarray, freqs_hz: Sequence[float], *,
                 ref_mic: int = 0, max_tracks: int = 3, pool_factor: int = 12,
                 az_range_deg: Sequence[float] = (0.0, 180.0), n_az: int = 181,
                 elevation_deg: float = 90.0, speed: float = 340.0):
        mic = np.asarray(mic_positions, dtype=np.float64)
        if mic.ndim != 2 or mic.shape[1] != 3:
            raise ValueError(f"mic_positions must be (C, 3), got {mic.shape}")
        if mic.shape[0] < 2:
            raise ValueError("TrackDPIPD requires at least two microphones")
        self.mic_positions = mic
        self.ref_mic = int(ref_mic)
        self.max_tracks = int(max_tracks)
        if self.max_tracks < 1:
            raise ValueError("max_tracks must be positive")
        self.pool_factor = int(pool_factor)
        if self.pool_factor < 1:
            raise ValueError("pool_factor must be positive")
        self.speed = float(speed)
        self.az_range_deg = tuple(float(v) for v in az_range_deg)
        self.n_az = int(n_az)
        self.elevation_deg = float(elevation_deg)
        self.freqs_hz = np.asarray(freqs_hz, dtype=np.float64)
        self.n_freq = int(self.freqs_hz.size)
        self.n_pairs = mic.shape[0] - 1
        # Reference-relative pairs: (ref, other), matching the author's
        # ch_mode='M' arrangement.
        self.pair_indices = [(self.ref_mic, j) for j in range(mic.shape[0])
                             if j != self.ref_mic]
        self.pair_vectors = np.stack(
            [mic[j] - mic[self.ref_mic] for _, j in self.pair_indices]
        )                                                      # (P, 3)

        self.az_deg = np.linspace(self.az_range_deg[0], self.az_range_deg[1],
                                  self.n_az)
        self.el_deg = np.full(self.n_az, self.elevation_deg, dtype=np.float64)

        self.empty = self._empty_target()                      # (2F, P)
        self.templates = self._build_templates()               # (D, 2F, P)
        self._templates_flat = self.templates.reshape(self.n_az, -1).clone()
        self._az_tensor = torch.from_numpy(self.az_deg.astype(np.float32))
        self._el_tensor = torch.from_numpy(self.el_deg.astype(np.float32))
        if self.max_tracks <= 4:
            self.permutations = list(itertools.permutations(range(self.max_tracks)))
        else:
            raise ValueError("Sequential PIT is limited to max_tracks <= 4")

    # ------------------------------------------------------------------
    # Physics
    # ------------------------------------------------------------------

    def _unit_vectors(self, az_deg: np.ndarray, el_deg: np.ndarray) -> np.ndarray:
        az = np.deg2rad(np.asarray(az_deg, dtype=np.float64))
        el = np.deg2rad(np.asarray(el_deg, dtype=np.float64))
        return np.stack(
            [np.sin(el) * np.cos(az), np.sin(el) * np.sin(az), np.cos(el)],
            axis=-1)

    def _pair_phase(self, vectors: np.ndarray) -> np.ndarray:
        """``(..., 3)`` unit vectors -> ``(..., F, P)`` phase ``-2 pi f tau``."""
        tau = np.einsum("...d,pd->...p", vectors, self.pair_vectors) / self.speed
        return -2.0 * np.pi * tau[..., None, :] * self.freqs_hz[:, None]

    def _empty_target(self) -> torch.Tensor:
        out = np.zeros((2 * self.n_freq, self.n_pairs), dtype=np.float32)
        for p in range(self.n_pairs):
            distance = float(np.linalg.norm(self.pair_vectors[p]))
            out[:self.n_freq, p] = j0(
                2.0 * np.pi * self.freqs_hz * distance / self.speed)
        return torch.from_numpy(out)

    def _build_templates(self) -> torch.Tensor:
        vectors = self._unit_vectors(self.az_deg, self.el_deg)   # (D, 3)
        phase = self._pair_phase(vectors)                         # (D, F, P)
        templates = np.concatenate(
            [np.cos(phase), np.sin(phase)], axis=1).astype(np.float32)
        return torch.from_numpy(templates)                        # (D, 2F, P)

    # ------------------------------------------------------------------
    # Targets and PIT
    # ------------------------------------------------------------------

    def target_from_angles(self, angles) -> torch.Tensor:
        """One angle list ``(N, 2)`` -> ``(2F, P, K)``.

        Azimuth is used continuously (no grid quantisation); the target is a
        horizontal far-field one, so elevation is the configured 90 degrees.
        The first ``N`` slots are active and the remaining slots are empty.
        """
        arr = np.asarray(
            angles.detach().cpu() if torch.is_tensor(angles) else angles,
            dtype=np.float64)
        if arr.size == 0:
            azimuths = np.zeros((0,), dtype=np.float64)
        elif arr.ndim == 1:
            azimuths = arr.reshape(-1)
        else:
            azimuths = arr.reshape(-1, arr.shape[-1])[:, 0]
        if azimuths.size > self.max_tracks:
            raise ValueError(
                f"got {azimuths.size} sources for a {self.max_tracks}-track "
                "model; filter the clip list or raise max_tracks")

        out = self.empty[:, :, None].repeat(1, 1, self.max_tracks).clone()
        used = azimuths.size
        if used:
            vectors = self._unit_vectors(
                azimuths[:used], np.full(used, self.elevation_deg))
            phase = self._pair_phase(vectors)                    # (N, F, P)
            active = np.concatenate([np.cos(phase), np.sin(phase)], axis=1)
            out[:, :, :used] = torch.from_numpy(active.transpose(1, 2, 0))
        return out

    def targets_from_angles(self, angle_lists, *, device=None,
                            dtype=torch.float32) -> torch.Tensor:
        targets = [self.target_from_angles(a) for a in angle_lists]
        out = torch.stack(targets, dim=0) if targets else torch.zeros(
            0, 2 * self.n_freq, self.n_pairs, self.max_tracks)
        return out.to(device=device, dtype=dtype)

    def pit_loss(self, prediction: torch.Tensor,
                 target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Minimum-cost track permutation MSE, averaged over the batch.

        Args:
            prediction / target: ``(B, 2F, P, K)``.
        Returns:
            ``(loss, perm)`` where ``perm[b, k]`` is the target track assigned
            to predicted track ``k`` for the chosen permutation.
        """
        if prediction.shape != target.shape:
            raise ValueError(
                f"prediction {tuple(prediction.shape)} and target "
                f"{tuple(target.shape)} must match")
        best = None
        best_perm = None
        batch = prediction.shape[0]
        for perm in self.permutations:
            perm_index = torch.tensor(perm, device=target.device, dtype=torch.long)
            permuted = target[..., perm_index]
            mse = ((prediction - permuted) ** 2).mean(dim=(1, 2, 3))
            if best is None:
                best = mse
                best_perm = perm_index.expand(batch, -1).clone()
            else:
                better = mse < best
                best = torch.where(better, mse, best)
                best_perm = torch.where(
                    better[:, None], perm_index.expand_as(best_perm), best_perm)
        return best.mean(), best_perm

    def pit_track_errors(self, prediction: torch.Tensor, target: torch.Tensor,
                         perm: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """Active/empty track MSE after PIT alignment (diagnostics)."""
        _, chosen = self.pit_loss(prediction, target) if perm is None else (None, perm)
        aligned = torch.stack(
            [target[b, ..., list(chosen[b].tolist())] for b in range(target.shape[0])]
        )
        diff = ((prediction - aligned) ** 2).mean(dim=(1, 2))     # (B, K)
        empty = self.empty.to(target.device)[None, :, :, None]
        active_mask = (aligned - empty).abs().sum(dim=(1, 2)) > 1e-6  # (B, K)
        active = diff[active_mask]
        silent = diff[~active_mask]
        return {
            "active_mse": float(active.mean()) if active.numel() else None,
            "empty_mse": float(silent.mean()) if silent.numel() else None,
            "active_tracks": int(active_mask.sum()),
            "empty_tracks": int((~active_mask).sum()),
        }

    # ------------------------------------------------------------------
    # Clip aggregation and decoding
    # ------------------------------------------------------------------

    @staticmethod
    def pool_weights(num_stft_frames: int, pool_factor: int,
                     n_groups: int) -> torch.Tensor:
        """Weight per pooled time step, normalised by the actual frame count."""
        if num_stft_frames <= 0 or pool_factor <= 0:
            raise ValueError("num_stft_frames and pool_factor must be positive")
        weights = [max(0, min(pool_factor, num_stft_frames - i * pool_factor))
                   for i in range(n_groups)]
        total = float(sum(weights))
        if int(round(total)) != int(num_stft_frames):
            raise ValueError(
                f"pooled groups {n_groups} with factor {pool_factor} cover "
                f"{total} of {num_stft_frames} STFT frames")
        return torch.tensor([w / total for w in weights], dtype=torch.float32)

    def pool_clip(self, frame_ipd: torch.Tensor,
                  num_stft_frames: int) -> torch.Tensor:
        """``(B, T_out, 2F, P, K)`` -> clip ``(B, 2F, P, K)``."""
        weights = self.pool_weights(num_stft_frames, self.pool_factor,
                                    frame_ipd.shape[1]).to(
            frame_ipd.device, frame_ipd.dtype)
        return (frame_ipd * weights[None, :, None, None, None]).sum(dim=1)

    def decode_candidates(self, clip_ipd: torch.Tensor,
                          score_kind: str = "ratio"):
        """Per-track best candidate.

        Returns numpy ``(scores, az_deg, el_deg)`` of shape ``(B, K)``, sorted
        with the largest score first on each row.
        """
        if clip_ipd.dim() != 4:
            raise ValueError(
                f"clip_ipd must be (B, 2F, P, K), got {tuple(clip_ipd.shape)}")
        if clip_ipd.shape[1] != 2 * self.n_freq:
            raise ValueError(
                f"clip_ipd has {clip_ipd.shape[1]} components, expected "
                f"{2 * self.n_freq}")
        if clip_ipd.shape[2] != self.n_pairs:
            raise ValueError(
                f"clip_ipd has {clip_ipd.shape[2]} pairs, expected {self.n_pairs}")

        device = clip_ipd.device
        # Move K ahead of the component axes: a plain reshape would cut
        # contiguous chunks that still mix the tracks.
        x = clip_ipd.permute(0, 3, 1, 2).reshape(
            clip_ipd.shape[0], clip_ipd.shape[3], -1)
        templates = self._templates_flat.to(device=device, dtype=x.dtype)

        kind = str(score_kind).lower()
        if kind == "ratio":
            denom = (templates * templates).sum(dim=-1)
            scores = torch.einsum("bkm,dm->bkd", x, templates)
            scores = scores / denom[None, None, :]
        elif kind in ("neg_mse", "mse"):
            diff = x[:, :, None, :] - templates[None, None, :, :]
            mse = (diff ** 2).mean(dim=-1)
            scores = -mse if kind == "neg_mse" else mse
        else:
            raise ValueError(f"Unknown score_kind {score_kind!r}")

        best = scores.argmax(dim=-1)                              # (B, K)
        best_scores = scores.gather(-1, best[..., None]).squeeze(-1)
        order = torch.argsort(best_scores, dim=-1, descending=True)
        best = best.gather(-1, order)
        best_scores = best_scores.gather(-1, order)

        az = self._az_tensor.to(device)[best].cpu().numpy()
        el = self._el_tensor.to(device)[best].cpu().numpy()
        return best_scores.detach().cpu().numpy(), az, el


class DPIPDClipModel(nn.Module):
    """Shared wrapper: frontend -> backbone -> clip aggregation -> PIT loss."""

    def __init__(self, frontend, backbone, tracks: TrackDPIPD):
        super().__init__()
        self.frontend = frontend
        self.backbone = backbone
        self.tracks = tracks
        self.pool_factor = int(getattr(tracks, "pool_factor", 1))

    def forward(self, waveform: torch.Tensor) -> Dict[str, object]:
        features = self.frontend(waveform)
        frame_ipd = self.backbone(features["features"])
        ipd = self.tracks.pool_clip(frame_ipd, features["stft_frames"])
        return {
            "frame_ipd": frame_ipd,
            "ipd": ipd,
            "stft_frames": features["stft_frames"],
            "freqs_hz": features["freqs_hz"],
        }

    def training_loss(self, waveform: torch.Tensor, targets) -> torch.Tensor:
        out = self.forward(waveform)
        target = self.tracks.targets_from_angles(
            targets, device=out["ipd"].device, dtype=torch.float32)
        loss, _ = self.tracks.pit_loss(out["ipd"].float(), target)
        return loss
