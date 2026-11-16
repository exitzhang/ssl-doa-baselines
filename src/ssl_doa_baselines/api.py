"""Stable package API: build, load, predict and decode.

The models deliberately differ in their native output:

* ``tf_mamba``    -> clip-level azimuth spectrum ``(B, Q)``;
* ``icocnn``      -> ACCDOA slots ``(B, T, M, 3)``;
* ``srp_dnn``     -> mixed DP-IPD ``(B, T, 2F, P)`` decoded by IDL;
* ``ipdnet``      -> per-track DP-IPD ``(B, 2F, P, K)`` (projection ratio);
* ``ipdnet2_small`` -> per-track DP-IPD ``(B, 2F, P, K)`` (negative MSE).

``predict`` hides that difference and returns, for every input clip, a list of
``{"azimuth_deg", "elevation_deg"?, "score"}`` detections.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from .config import model_section
from .tf_mamba.decode import decode_spectrum

_MODEL_NAMES = ("tf_mamba", "icocnn", "srp_dnn", "ipdnet", "ipdnet2_small")


def _split_config(config: Optional[Dict[str, Any]], overrides: Dict[str, Any]):
    cfg = dict(model_section(config or {}))
    name = cfg.pop("name", None) or cfg.pop("model_name", None)
    cfg.pop("type", None)
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return name, cfg


def build_model(name: Optional[str] = None, config: Optional[Dict[str, Any]] = None,
                **overrides) -> torch.nn.Module:
    """Build a baseline from a name plus a ``[model]`` config mapping."""
    cfg_name, cfg = _split_config(config, overrides)
    name = name or cfg_name
    if name is None:
        raise ValueError("build_model requires a model name")
    name = str(name).lower()

    if name == "tf_mamba":
        from .tf_mamba.model import TFMambaSpectrum

        allowed = {"num_mics", "sample_rate", "n_fft", "hop_length", "freq_min",
                   "freq_max", "d_model", "num_blocks", "expand_factors", "d_state",
                   "d_conv", "output_dim", "activation"}
        return TFMambaSpectrum(**{k: v for k, v in cfg.items() if k in allowed})

    if name == "icocnn":
        from .icocnn.model import IcoCNNMultiSource

        allowed = {"n_mics", "sample_rate", "mic_positions", "r", "srp_win", "srp_hop",
                   "width", "temporal_kernel", "n_slots", "smooth_vertices"}
        kwargs = {k: v for k, v in cfg.items() if k in allowed}
        if "mic_positions" in kwargs:
            kwargs["mic_positions"] = torch.as_tensor(kwargs["mic_positions"],
                                                      dtype=torch.float32)
        return IcoCNNMultiSource(**kwargs)

    if name in {"srp_dnn", "srpdnn"}:
        from .srp_dnn.model import SRPDNN

        allowed = {"n_mics", "sample_rate", "mic_positions", "win_len", "hop", "n_fft",
                   "freq_min", "freq_max", "max_num_sources", "az_range_deg",
                   "el_range_deg", "n_az", "n_el"}
        return SRPDNN(**{k: v for k, v in cfg.items() if k in allowed})

    if name in {"ipdnet", "ipd_net"}:
        from .ipdnet.model import IPDNetBaseline

        allowed = {"mic_positions", "n_mics", "input_sample_rate",
                   "stft_sample_rate", "win_len", "n_fft", "hop",
                   "forgetting_frames", "hidden_size", "max_tracks",
                   "pool_factor", "ref_mic", "az_range_deg", "n_az",
                   "elevation_deg", "speed", "dropout", "score_kind"}
        return IPDNetBaseline(**{k: v for k, v in cfg.items() if k in allowed})

    if name in {"ipdnet2_small", "ipdnet2"}:
        from .ipdnet2.model import IPDnet2SmallBaseline

        allowed = {"mic_positions", "n_mics", "input_sample_rate",
                   "stft_sample_rate", "win_len", "n_fft", "hop",
                   "forgetting_frames", "dim_hidden", "num_layers",
                   "dim_squeeze", "num_heads", "kernel_size", "conv_groups",
                   "norms", "attention", "rope", "fre_compression_ratio",
                   "time_compression_ratio", "time_compression_layer",
                   "encoder_kernel_size", "max_tracks", "pool_factor",
                   "ref_mic", "az_range_deg", "n_az", "elevation_deg",
                   "speed", "score_kind"}
        return IPDnet2SmallBaseline(**{k: v for k, v in cfg.items() if k in allowed})

    raise ValueError(f"Unknown model {name!r}; expected one of {_MODEL_NAMES}")


def model_name_of(model: torch.nn.Module) -> str:
    name = getattr(model, "model_name", None)
    if name is None:
        name = model.__class__.__name__.lower()
    return str(name)


def save_checkpoint(path: str, model: torch.nn.Module, *, epoch: Optional[int] = None,
                    optimizer: Optional[torch.optim.Optimizer] = None,
                    config: Optional[Dict[str, Any]] = None,
                    extra: Optional[Dict[str, Any]] = None) -> None:
    payload = {
        "model_name": model_name_of(model),
        "model_config": dict(getattr(model, "config", {})),
        "state_dict": model.state_dict(),
    }
    if epoch is not None:
        payload["epoch"] = int(epoch)
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if config is not None:
        payload["config"] = config
    if extra:
        payload.update(extra)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_model(checkpoint: str, *, config: Optional[Dict[str, Any]] = None,
               device: str = "cpu", strict: bool = True) -> torch.nn.Module:
    """Load a baseline from a checkpoint carrying its own model config.

    A caller-supplied ``config`` only fills keys missing from the checkpoint;
    the embedded config wins so that state-dict shapes always match.  Pass
    ``strict=False`` to allow old checkpoints without an embedded config.
    """
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    name = ckpt.get("model_name")
    if name is None:
        raise ValueError(f"{checkpoint} does not contain 'model_name'")
    embedded = ckpt.get("model_config")
    if embedded is None:
        if strict:
            raise ValueError(
                f"{checkpoint} has no embedded model_config; pass strict=False and "
                "explicit config to load a legacy checkpoint"
            )
        embedded = model_section(config or {})
    else:
        merged = dict(model_section(config or {}))
        merged.update(embedded)
        embedded = merged

    model = build_model(name, {"name": name, **embedded})
    state = ckpt.get("state_dict", ckpt.get("model"))
    if state is None:
        raise ValueError(f"{checkpoint} does not contain a state_dict")
    model.load_state_dict(state, strict=strict)
    model.to(device)
    model.eval()
    return model


def _as_batch(waveform) -> torch.Tensor:
    if isinstance(waveform, np.ndarray):
        waveform = torch.from_numpy(waveform)
    waveform = waveform.float()
    if waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)
    if waveform.dim() != 3:
        raise ValueError(f"Expected (T, C) or (B, T, C), got {tuple(waveform.shape)}")
    return waveform


@torch.no_grad()
def predict_native(model: torch.nn.Module, waveform, device: Optional[str] = None,
                   **kwargs) -> Dict[str, Any]:
    """Run the model and return its native output dictionary."""
    device = device or next(model.parameters()).device
    model.eval()
    x = _as_batch(waveform).to(device)
    out = model(x)
    if isinstance(out, dict):
        return out
    return {"output": out}


@torch.no_grad()
def predict(model: torch.nn.Module, waveform, *, decode: Optional[Dict[str, Any]] = None,
            device: Optional[str] = None) -> List[List[Dict[str, float]]]:
    """Return decoded detections for each clip in ``waveform``."""
    decode = dict(decode or {})
    device = device or next(model.parameters()).device
    model.eval()
    x = _as_batch(waveform).to(device)
    name = model_name_of(model)

    if name == "tf_mamba":
        spectrum = model(x).detach().cpu().numpy()
        az_range = decode.pop("az_range", (0.0, 180.0))
        max_sources = int(decode.pop("max_sources", 3))
        return decode_spectrum(spectrum, az_range=az_range, max_sources=max_sources,
                               **decode)

    if name == "icocnn":
        return model.decode(
            x,
            threshold_abs=float(decode.get("threshold_abs", 0.3)),
            max_sources=int(decode.get("max_sources", 3)),
            frame_index=int(decode.get("frame_index", -1)),
        )

    if name == "srp_dnn":
        return model.decode(
            x,
            threshold_abs=float(decode.get("threshold_abs", 0.3)),
            max_sources=int(decode.get("max_sources", 3)),
            frame_index=int(decode.get("frame_index", -1)),
        )

    if name in {"ipdnet", "ipdnet2_small"}:
        out = model(x)
        ipd = out["ipd"]
        score_kind = str(decode.get("score_kind",
                                    getattr(model, "score_kind", "ratio")))
        threshold = float(decode.get("threshold_abs", 0.0))
        max_sources = int(decode.get("max_sources", ipd.shape[-1]))
        scores, azimuth, elevation = model.tracks.decode_candidates(
            ipd, score_kind=score_kind)
        detections: List[List[Dict[str, float]]] = []
        for b in range(scores.shape[0]):
            row: List[Dict[str, float]] = []
            for k in range(scores.shape[1]):
                if float(scores[b, k]) < threshold:
                    continue
                row.append({
                    "azimuth_deg": float(azimuth[b, k]),
                    "elevation_deg": float(elevation[b, k]),
                    "score": float(scores[b, k]),
                })
                if len(row) >= max_sources:
                    break
            detections.append(row)
        return detections

    raise ValueError(f"Unknown model {name!r}")


def fit(model: torch.nn.Module, train_loader, val_loader=None, **kwargs):
    """Thin wrapper around :func:`ssl_doa_baselines.training.fit`."""
    from .training import fit as _fit

    return _fit(model, train_loader, val_loader, **kwargs)
