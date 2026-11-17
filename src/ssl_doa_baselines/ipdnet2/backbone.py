"""IPDnet2-Small (online spatial network) backbone.

Ported from ``Audio-WestlakeU/FN-SSL@76fcb281`` ``IPDnet2/IPDnet2.py`` (MIT).
Local adaptations:

* imports are package-relative and the Mamba dependency is checked up front
  with a clear error;
* the per-layer shape ``print`` is removed;
* the 5x time pooling keeps its tail group (``ceil_mode``), so 17 STFT frames
  give 4 pooled steps with weights ``[5, 5, 5, 2]``;
* the final reshape uses explicit real/imaginary, mic-pair and track axes and
  returns ``(B, T_out, 2F, P, K)`` with every real part before every
  imaginary part.

The model is only constructed when ``ipdnet2_small`` is selected; importing
this module does not import Mamba unless ``mamba-ssm`` is installed.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import MultiheadAttention
from torch.nn.common_types import _size_1_t

from .arch.base.linear_group import LinearGroup
from .arch.base.non_linear import *
from .arch.base.norm import *
from .arch.base.retention import MultiScaleRetention, RetNetRelPos

try:
    from mamba_ssm import Mamba
    from mamba_ssm.utils.generation import InferenceParams
except Exception:  # mamba-ssm is an optional, CUDA-built dependency
    Mamba = None
    InferenceParams = None


class FreqInverse(nn.Module):
    """Compressed-frequency hidden states -> full ``nfreq`` output."""

    def __init__(self, nfreq=256, compression_ratio=16, hidden_dim=96,
                 out_dim=16, sample_rate=16000):
        super().__init__()
        self.nfreq = nfreq
        self.compression_ratio = compression_ratio
        self.nfilters = nfreq // compression_ratio
        self.sample_rate = sample_rate
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.bounds = [i * compression_ratio for i in range(self.nfilters)]
        self.bounds.append(nfreq)
        self.trans2 = nn.Conv1d(hidden_dim, compression_ratio * self.out_dim, 1)

    def forward(self, x):
        # Vectorised equivalent of the author's per-band loop: one 1x1 conv over
        # every compressed bin, then interleave each bin's ``compression_ratio``
        # rows back into the full frequency axis in the same ``bin * ratio +
        # sub`` order the loop writes.  The loop cost ~16 slices plus in-place
        # adds per call (~8 ms of a training step) on the host; this version is
        # bit-identical on the forward and input gradient.
        batch, hidden, steps, bins = x.shape
        if bins != self.nfilters:
            raise ValueError(
                f"expected {self.nfilters} compressed bins, got {bins}")
        ratio = self.compression_ratio
        y = self.trans2(x.permute(0, 3, 1, 2).reshape(batch * bins, hidden, steps))
        y = y.reshape(batch, bins, self.out_dim, ratio, steps)
        y = y.permute(0, 2, 1, 3, 4).reshape(batch, self.out_dim, self.nfreq, steps)
        return y.permute(0, 1, 3, 2).contiguous().tanh()


class CausalConv1d(nn.Conv1d):
    """Causal 1-D convolution (optionally with a small look-ahead)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: _size_1_t = 0,
        dilation: _size_1_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
        look_ahead: int = 0,
    ) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, padding,
                         dilation, groups, bias, padding_mode, device, dtype)
        self.look_ahead = look_ahead
        assert look_ahead <= self.kernel_size[0] - 1, (look_ahead, self.kernel_size)

    def forward(self, x: Tensor, state: Dict[int, Any] = None) -> Tensor:
        # x: [B, H, T]
        if state is None or id(self) not in state:
            x = F.pad(x, pad=(self.kernel_size[0] - 1 - self.look_ahead,
                              self.look_ahead))
        else:
            x = torch.concat([state[id(self)], x], dim=-1)
        if state is not None:
            state[id(self)] = x[..., -self.kernel_size + 1:]
        return super().forward(x)

    def extra_repr(self):
        if self.look_ahead == 0:
            return super().extra_repr()
        return super().extra_repr() + f", look ahead={self.look_ahead}"


class SpatialNetLayer(nn.Module):
    """Frequency conv + full-band linear + Mamba (time) + Mamba (FFN)."""

    def __init__(
        self,
        dim_hidden: int,
        dim_squeeze: int,
        num_freqs: int,
        dropout: Tuple[float, float, float] = (0, 0, 0),
        kernel_size: Tuple[int, int] = (5, 3),
        conv_groups: Tuple[int, int] = (8, 8),
        norms: List[str] = ["LN", "LN", "GN", "LN", "LN", "LN"],
        padding: str = "zeros",
        full: nn.Module = None,
        attention: str = "mhsa",
        is_first: bool = False,
    ) -> None:
        super().__init__()
        f_conv_groups = conv_groups[0]
        t_conv_groups = conv_groups[1]
        f_kernel_size = kernel_size[0]
        self.fconv1 = nn.ModuleList([
            new_norm(norms[3], dim_hidden, seq_last=True, group_size=None,
                     num_groups=f_conv_groups),
            nn.Conv1d(in_channels=dim_hidden, out_channels=dim_hidden,
                      kernel_size=f_kernel_size, groups=f_conv_groups,
                      padding="same", padding_mode=padding),
            nn.PReLU(dim_hidden),
        ])
        self.norm_full = new_norm(norms[5], dim_hidden, seq_last=False,
                                  group_size=None, num_groups=f_conv_groups)
        self.full_share = False if full is None else True
        self.dim_squeeze = dim_squeeze
        self.squeeze = nn.Sequential(
            nn.Conv1d(in_channels=dim_hidden, out_channels=dim_squeeze,
                      kernel_size=1), nn.SiLU())
        self.dropout_full = nn.Dropout2d(dropout[2]) if dropout[2] > 0 else None
        self.is_first = is_first
        self.full = nn.Linear(num_freqs, num_freqs) if full is None else full
        self.unsqueeze = nn.Sequential(
            nn.Conv1d(in_channels=dim_squeeze, out_channels=dim_hidden,
                      kernel_size=1), nn.SiLU())
        self.fconv2 = nn.ModuleList([
            new_norm(norms[4], dim_hidden, seq_last=True, group_size=None,
                     num_groups=f_conv_groups),
            nn.Conv1d(in_channels=dim_hidden, out_channels=dim_hidden,
                      kernel_size=f_kernel_size, groups=f_conv_groups,
                      padding="same", padding_mode=padding),
            nn.PReLU(dim_hidden),
        ])
        self.norm_mhsa = new_norm(norms[0], dim_hidden, seq_last=False,
                                  group_size=None, num_groups=t_conv_groups)
        attn_params = attention[6:-1].split(",")
        d_state, mamba_conv_kernel = int(attn_params[0]), int(attn_params[1])
        self.mhsa = Mamba(d_model=dim_hidden, d_state=d_state,
                          d_conv=mamba_conv_kernel, layer_idx=0)
        self.attention = attention
        self.dropout_mhsa = nn.Dropout(dropout[0])
        self.norm_tconvffn = new_norm(norms[1], dim_hidden, seq_last=False,
                                      group_size=None, num_groups=t_conv_groups)
        self.tconvffn = Mamba(d_model=dim_hidden, d_state=d_state,
                              d_conv=mamba_conv_kernel, layer_idx=0)
        self.dropout_tconvffn = nn.Dropout(dropout[1])
        self.fre_compress_second = nn.AvgPool2d(kernel_size=(1, 8))
        self.fre_compress_first = nn.AvgPool2d(kernel_size=(1, 2))

    def forward(self, x: Tensor, att_mask: Optional[Tensor] = None,
                chunkwise_recurrent: bool = True, rope: bool = True,
                state: Dict[int, Any] = None, inference: bool = False):
        """``x``: ``[B, F, T, H]`` -> ``([B, F, T, H], attn)``."""
        x = x + self._fconv(self.fconv1, x)
        if self.is_first:
            x = self.fre_compress_first(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = x + self._full(x)
        x = x + self._fconv(self.fconv2, x)
        if self.is_first:
            x = self.fre_compress_second(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        attn = None
        if isinstance(self.mhsa, Mamba):
            x = x + self._mamba(x, self.mhsa, self.norm_mhsa,
                                self.dropout_mhsa, inference)
        else:
            x_, attn = self._tsa(x, att_mask, chunkwise_recurrent, rope,
                                 state=state, inference=inference)
            x = x + x_
        if isinstance(self.tconvffn, Mamba):
            x = x + self._mamba(x, self.tconvffn, self.norm_tconvffn,
                                self.dropout_tconvffn, inference)
        else:
            x = x + self._tconvffn(x, state=state)
        return x, attn

    def _mamba(self, x: Tensor, mamba, norm: nn.Module, dropout: nn.Module,
               inference: bool = False):
        B, F, T, H = x.shape
        x = norm(x)
        x = x.reshape(B * F, T, H)
        if inference:
            inference_params = InferenceParams(T, B * F)
            xs = []
            for i in range(T):
                inference_params.seqlen_offset = i
                xs.append(mamba.forward(x[:, [i], :], inference_params))
            x = torch.concat(xs, dim=1)
        else:
            x = mamba.forward(x)
        x = x.reshape(B, F, T, H)
        return dropout(x)

    def _tsa(self, x: Tensor, attn_mask: Optional[Tensor],
             chunkwise_recurrent: bool, rope: bool = True,
             state: Dict[int, Any] = None, inference: bool = False):
        B, F, T, H = x.shape
        x = self.norm_mhsa(x)
        x = x.reshape(B * F, T, H)
        if isinstance(self.mhsa, MultiheadAttention):
            need_weights = False if hasattr(self, "need_weights") else self.need_weights
            x, attn = self.mhsa.forward(x, x, x, need_weights=need_weights,
                                        average_attn_weights=False,
                                        attn_mask=attn_mask, is_causal=True)
        else:
            if inference is False:
                x = self.mhsa.forward(x, rel_pos=attn_mask,
                                      incremental_state=state,
                                      chunkwise_recurrent=chunkwise_recurrent,
                                      rope=rope)
            else:
                xs, state = [], dict()
                for i in range(T):
                    xi = self.mhsa.forward(x[:, [i], :], rel_pos=attn_mask[i],
                                           incremental_state=state)
                    xs.append(xi)
                x = torch.concat(xs, dim=1)
            attn = None
        x = x.reshape(B, F, T, H)
        return self.dropout_mhsa(x), attn

    def _tconvffn(self, x: Tensor, state: Dict[int, Any] = None) -> Tensor:
        B, F, T, H0 = x.shape
        x = x.transpose(-1, -2)                                  # [B,F,H,T]
        x = x.reshape(B * F, H0, T)
        for m in self.tconvffn:
            if isinstance(m, CausalConv1d):
                x = m(x, state=state)
            elif isinstance(m, nn.GroupNorm) or "GroupNorm" in type(m).__name__:
                x = x.reshape(B, F, -1, T).transpose(1, -1).reshape(B * T, -1, F)
                x = m(x)
                x = x.reshape(B, T, -1, F).transpose(1, -1).reshape(B * F, -1, T)
            else:
                x = m(x)
        x = x.reshape(B, F, H0, T)
        x = x.transpose(-1, -2)                                  # [B,F,T,H]
        return self.dropout_tconvffn(x)

    def _fconv(self, ml: nn.ModuleList, x: Tensor) -> Tensor:
        B, F, T, H = x.shape
        x = x.permute(0, 2, 3, 1)                                # [B,T,H,F]
        x = x.reshape(B * T, H, F)
        for m in ml:
            if type(m) == GroupBatchNorm:
                x = m(x, group_size=T)
            else:
                x = m(x)
        x = x.reshape(B, T, H, F)
        return x.permute(0, 3, 1, 2)                             # [B,F,T,H]

    def _full(self, x: Tensor) -> Tensor:
        B, F, T, H = x.shape
        x = self.norm_full(x)
        x = x.permute(0, 2, 3, 1)                                # [B,T,H,F]
        x = x.reshape(B * T, H, F)
        x = self.squeeze(x)                                      # [B*T,H',F]
        if self.dropout_full:
            x = x.reshape(B, T, -1, F)
            x = x.transpose(1, 3)                                # [B,F,H',T]
            x = self.dropout_full(x)
            x = x.transpose(1, 3)                                # [B,T,H',F]
            x = x.reshape(B * T, -1, F)
        x = self.full(x)
        x = self.unsqueeze(x)
        x = x.reshape(B, T, H, F)
        return x.permute(0, 3, 1, 2)                             # [B,F,T,H]

    def extra_repr(self) -> str:
        return f"full_share={self.full_share}"


class OnlineSpatialNet(nn.Module):
    """``(B, C_in, F, T)`` features -> ``(B, T_out, 2F, P, K)`` DP-IPD."""

    def __init__(
        self,
        dim_input: int,
        dim_output: int,
        num_layers: int,
        dim_squeeze: int,
        num_freqs: int,
        encoder_kernel_size: int = 5,
        dim_hidden: int = 192,
        num_heads: int = 2,
        dropout: Tuple[float, float, float] = (0, 0, 0),
        kernel_size: Tuple[int, int] = (5, 3),
        conv_groups: Tuple[int, int] = (8, 8),
        norms: List[str] = ["LN", "LN", "GN", "LN", "LN", "LN"],
        padding: str = "zeros",
        attention: str = "mhsa(251)",
        chunkwise_recurrent: bool = True,
        rope: Union[bool, str] = False,
        fre_compression_ratio: int = 16,
        time_compression_ratio: int = 5,
        time_compression_layer: int = 0,
        n_pairs: int = 2,
        max_tracks: int = 3,
    ):
        super().__init__()
        if Mamba is None:
            raise ImportError(
                "ipdnet2_small requires mamba-ssm==2.1.0 with a CUDA build; "
                "install it with `pip install mamba-ssm==2.1.0 "
                "--no-build-isolation` or choose another model.")
        if num_freqs % fre_compression_ratio != 0:
            raise ValueError(
                f"num_freqs={num_freqs} must be divisible by "
                f"fre_compression_ratio={fre_compression_ratio}")
        if dim_output % (2 * n_pairs) != 0:
            raise ValueError(
                f"dim_output={dim_output} must equal 2 * n_pairs * tracks")

        self.num_heads = num_heads
        self.chunkwise_recurrent = chunkwise_recurrent
        self.pos = None
        self.attn_scope = 1
        self.rope = rope
        self.n_pairs = int(n_pairs)
        self.max_tracks = int(max_tracks)
        self.encoder = CausalConv1d(in_channels=dim_input,
                                    out_channels=dim_hidden,
                                    kernel_size=encoder_kernel_size,
                                    look_ahead=0)
        self.time_compression_layer = time_compression_layer
        layers = []
        for layer_idx in range(num_layers):
            layers.append(SpatialNetLayer(
                dim_hidden=dim_hidden,
                dim_squeeze=dim_squeeze,
                num_freqs=(num_freqs // 2 if layer_idx == 0
                           else num_freqs // fre_compression_ratio),
                dropout=dropout,
                kernel_size=kernel_size,
                conv_groups=conv_groups,
                norms=norms,
                padding=padding,
                full=None,
                attention=attention,
                is_first=(layer_idx == 0),
            ))
        self.layers = nn.ModuleList(layers)
        self.freq_inverse = FreqInverse(
            nfreq=num_freqs, compression_ratio=fre_compression_ratio,
            hidden_dim=dim_hidden, out_dim=dim_output)
        self.decoder = nn.Linear(in_features=dim_output, out_features=dim_output)
        # Keep the tail group: 17 frames -> 4 pooled steps.
        self.time_pooling = nn.AvgPool2d(
            kernel_size=(time_compression_ratio, 1), ceil_mode=True)

    def forward(self, x: Tensor, inference: bool = False,
                return_attn_score: bool = False):
        # x: (B, C_in, F, T)
        x = x.permute(0, 2, 3, 1)                                # [B,F,T,H0]
        B, F, T, H0 = x.shape
        x = self.encoder(x.reshape(B * F, T, H0).permute(0, 2, 1)).permute(0, 2, 1)
        H = x.shape[2]
        chunkwise_recurrent = True if inference is False else self.chunkwise_recurrent
        mask = self.get_causal_mask(
            slen=T, device=x.device, chunkwise_recurrent=chunkwise_recurrent,
            batch_size=B, inference=inference)
        attns = [] if return_attn_score else None
        x = x.reshape(B, F, T, H)
        for i, m in enumerate(self.layers):
            if i == self.time_compression_layer:
                setattr(m, "need_weights", return_attn_score)
                x, attn = m(x, mask, chunkwise_recurrent, self.rope, None, inference)
                B, F, T, H = x.shape
                x = x.reshape(B * F, T, H)
                x = self.time_pooling(x)
                BF, T_, H = x.shape
                x = x.reshape(B, F, T_, H)
                # The cursor/time mask must follow the pooled length.
                mask = self.get_causal_mask(
                    slen=T_, device=x.device,
                    chunkwise_recurrent=chunkwise_recurrent,
                    batch_size=B, inference=inference)
                if return_attn_score:
                    attns.append(attn)
            else:
                setattr(m, "need_weights", return_attn_score)
                x, attn = m(x, mask, chunkwise_recurrent, self.rope, None, inference)
                if return_attn_score:
                    attns.append(attn)
        B, F, T_, H = x.shape
        x = x.permute(0, 3, 2, 1)
        x = self.freq_inverse(x)
        x = x.permute(0, 3, 2, 1)                                # [B,F,T_,H]
        x = self.decoder(x)                                      # [B,F,T_,12]
        B, F, T_, H = x.shape

        # Explicit (B, T_out, 2F, P, K): channels are (real/imag, pair, track).
        x = x.permute(0, 2, 1, 3)                                # [B,T_,F,12]
        x = x.reshape(B, T_, F, 2, self.n_pairs, self.max_tracks)
        x = x.permute(0, 1, 3, 2, 4, 5)                          # [B,T_,2,F,P,K]
        x = x.reshape(B, T_, 2 * F, self.n_pairs, self.max_tracks)
        if return_attn_score:
            return x.contiguous(), attns
        return x.contiguous()

    def get_causal_mask(self, slen: int, device=None,
                        chunkwise_recurrent: bool = True,
                        batch_size: int = None, inference: bool = False):
        if isinstance(self.pos, RetNetRelPos):
            if inference is False:
                return self.pos.forward(slen=slen,
                                        chunkwise_recurrent=chunkwise_recurrent)
            return [self.pos.forward(slen=t, activate_recurrent=True)
                    for t in range(slen)]
        pos1 = torch.arange(start=0, end=slen, dtype=torch.long, device=device,
                            requires_grad=False).unsqueeze(1)
        pos2 = torch.arange(start=0, end=slen, dtype=torch.long, device=device,
                            requires_grad=False).unsqueeze(0)
        relative_pos = pos1 - pos2
        if self.rope == "ALiBi":
            assert batch_size is not None, batch_size
            m = (2.0 ** (-8 / torch.arange(1, self.num_heads + 1, 1,
                                           device=device))).reshape(
                self.num_heads, 1, 1)
            m = torch.concat([m] * batch_size, dim=0)
            relative_pos = torch.where(
                (relative_pos >= 0) * (relative_pos < self.attn_scope),
                relative_pos.abs() * -1, -torch.inf)
            return m * relative_pos
        return torch.where(
            (relative_pos >= 0) * (relative_pos < self.attn_scope), 0.0,
            -torch.inf)
