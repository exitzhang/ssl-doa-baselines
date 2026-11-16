"""IPDnet backbone: full-band / narrow-band fusion + causal CNN.

Ported from ``Audio-WestlakeU/FN-SSL@76fcb281`` ``IPDnet/FixedAarryIPDnet.py``
(MIT).  Local adaptations:

* ``input_size``/``max_track`` are the 3-microphone, 3-track configuration
  (6 input components from 2 mic pairs, 12 output components);
* the 3x4 time pooling keeps its tail group (``ceil_mode``), so a short clip
  does not require ``T`` to be a multiple of 12;
* the final reshape is written with explicit real/imaginary, mic-pair and
  track axes instead of the author's ``F*2`` layout.  The output is
  ``(B, T_out, 2F, P, K)`` with all real parts before all imaginary parts.
"""

import torch
import torch.nn as nn


class FNblock(nn.Module):
    """Full-band (per-frequency BLSTM) + narrow-band (per-time LSTM) block."""

    def __init__(self, input_size, hidden_size=128, dropout=0.2,
                 add_skip_dim=4, is_online=False, is_first=False):
        super().__init__()
        self.input_size = input_size
        self.full_hidden_size = hidden_size // 2
        self.is_first = is_first
        self.is_online = is_online
        self.narr_hidden_size = hidden_size if is_online else hidden_size // 2
        self.dropout = dropout
        self.dropout_full = nn.Dropout(p=self.dropout)
        self.dropout_narr = nn.Dropout(p=self.dropout)
        full_input = input_size if is_first else input_size + add_skip_dim
        self.fullLstm = nn.LSTM(input_size=full_input,
                                hidden_size=self.full_hidden_size,
                                batch_first=True, bidirectional=True)
        self.narrLstm = nn.LSTM(input_size=2 * self.full_hidden_size + add_skip_dim,
                                hidden_size=self.narr_hidden_size,
                                batch_first=True, bidirectional=not self.is_online)

    def forward(self, x, fb_skip, nb_skip):
        nb, nt, nf, nc = x.shape
        x = x.reshape(nb * nt, nf, -1)
        x, _ = self.fullLstm(x)
        x = self.dropout_full(x)
        x = torch.cat((x, fb_skip), dim=-1)
        x = x.view(nb, nt, nf, -1).permute(0, 2, 1, 3).reshape(nb * nf, nt, -1)
        x, _ = self.narrLstm(x)
        x = self.dropout_narr(x)
        x = torch.cat((x, nb_skip), dim=-1)
        x = x.view(nb, nf, nt, -1).permute(0, 2, 1, 3)
        return x


class CausalCnnBlock(nn.Module):
    """Causal convolution block with the author's 3x4 time pooling."""

    def __init__(self, inp_dim, out_dim, cnn_hidden_dim=128,
                 kernel=(3, 3), stride=(1, 1), padding=(1, 2)):
        super().__init__()
        self.conv1 = nn.Conv2d(inp_dim, cnn_hidden_dim, kernel_size=kernel,
                               stride=stride, padding=padding, bias=False)
        self.conv2 = nn.Conv2d(cnn_hidden_dim, cnn_hidden_dim, kernel_size=kernel,
                               stride=stride, padding=padding, bias=False)
        self.conv3 = nn.Conv2d(cnn_hidden_dim, out_dim, kernel_size=kernel,
                               stride=stride, padding=padding, bias=False)
        # Keep a short tail group instead of dropping it.
        self.pooling1 = nn.AvgPool2d(kernel_size=(1, 3), ceil_mode=True)
        self.pooling2 = nn.AvgPool2d(kernel_size=(1, 4), ceil_mode=True)
        self.pad = padding
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = out[:, :, :, :-self.pad[1]]
        out = self.pooling1(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = out[:, :, :, :-self.pad[1]]
        out = self.pooling2(out)
        out = self.conv3(out)
        out = out[:, :, :, :-self.pad[1]]
        out = self.tanh(out)
        return out


class IPDnetBackbone(nn.Module):
    """``(B, T, F, 2C)`` features -> ``(B, T_out, 2F, P, K)`` DP-IPD."""

    def __init__(self, input_size=6, hidden_size=128, max_tracks=3,
                 n_pairs=2, dropout=0.2):
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.max_tracks = int(max_tracks)
        self.n_pairs = int(n_pairs)
        self.block_1 = FNblock(input_size=self.input_size,
                               hidden_size=self.hidden_size,
                               dropout=dropout, add_skip_dim=self.input_size,
                               is_online=True, is_first=True)
        self.block_2 = FNblock(input_size=self.hidden_size,
                               hidden_size=self.hidden_size,
                               dropout=dropout, add_skip_dim=self.input_size,
                               is_online=True, is_first=False)
        self.cnn_out_dim = 2 * self.n_pairs * self.max_tracks
        self.cnn_inp_dim = self.hidden_size + self.input_size
        self.conv = CausalCnnBlock(inp_dim=self.cnn_inp_dim,
                                   out_dim=self.cnn_out_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: (B, T, F, C)
        x = features.permute(0, 3, 2, 1)                       # (B, C, F, T)
        nb, nt, nf, nc = x.shape

        fb_skip = x.reshape(nb * nt, nf, nc)
        nb_skip = x.permute(0, 2, 1, 3).reshape(nb * nf, nt, nc)

        x = self.block_1(x, fb_skip=fb_skip, nb_skip=nb_skip)
        x = self.block_2(x, fb_skip=fb_skip, nb_skip=nb_skip)
        x = x.permute(0, 3, 2, 1)                              # (B, hidden, F, T)
        x = self.conv(x)                                       # (B, C', F, T_out)
        x = x.permute(0, 3, 2, 1)                              # (B, T_out, F, C')

        batch, t_out, n_freq, channels = x.shape
        x = x.reshape(batch, t_out, n_freq, 2, self.n_pairs,
                      self.max_tracks)
        x = x.permute(0, 1, 3, 2, 4, 5)                        # (B,T,2,F,P,K)
        return x.reshape(batch, t_out, 2 * n_freq, self.n_pairs,
                         self.max_tracks).contiguous()
