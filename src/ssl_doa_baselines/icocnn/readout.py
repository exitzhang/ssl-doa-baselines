"""Icosahedral Soft-ArgMax readout.

Ported from David Diaz-Guerra's ``icoDOA`` (``acousticTrackingModules.py``:
``SoftArgMax``), AGPL-3.0, commit ``04d1a89594c78ae3cf42f07d94c3737bdc1f7c82``.
The multi-source IcoCNN paper (arXiv:2210.14536) adapts the single-source model
by increasing the last temporal convolution's output channels from 1 to ``M``;
the same Soft-ArgMax then maps each channel's spherical distribution to one 3-D
vector.  The vector direction is the estimated DOA and its norm is the activity
score, matching the paper's 0.5 detection threshold on the norm.
"""

import torch
import torch.nn as nn


class SoftArgMax(nn.Module):
    """Expected position of a spherical distribution under each channel.

    Args:
        input_shape: spatial shape of the map, e.g. ``(5, 2, 4)``.
        indexes: ``(n_dim_out,) + input_shape`` coordinates of every map cell.
        include_exp: apply ``exp`` before normalising, as in icoDOA.  Without
            it the input is treated as an already-normalised probability map.
    """

    def __init__(self, input_shape, indexes=None, include_exp=False):
        super().__init__()
        self.input_shape = tuple(int(x) for x in input_shape)
        self.include_exp = bool(include_exp)
        self.n_dim_in = len(self.input_shape)

        if indexes is not None:
            self.n_dim_out = int(indexes.shape[0])
            self.register_buffer("indexes", indexes.float(), persistent=False)
        else:
            self.n_dim_out = self.n_dim_in
            self.register_buffer("indexes", torch.empty((self.n_dim_out,) + self.input_shape),
                                 persistent=False)
            for i in range(self.n_dim_in):
                broadcaster_shape = [1] * self.n_dim_in
                broadcaster_shape[i] = self.input_shape[i]
                self.indexes[i, ...] = torch.arange(self.input_shape[i]).reshape(broadcaster_shape)

    def forward(self, x: torch.Tensor):
        assert x.shape[-self.n_dim_in:] == self.input_shape, (
            f"expected spatial shape {self.input_shape}, got {tuple(x.shape)}")
        spatial_dims = list(range(x.ndim - self.n_dim_in, x.ndim))
        batch_shape = x.shape[:-self.n_dim_in]

        if self.include_exp:
            x = torch.exp(x - x.max())
        x = x / (x.sum(spatial_dims, keepdim=True) + 1e-12)

        y = torch.empty(batch_shape + (self.n_dim_out,), device=x.device, dtype=x.dtype)
        for i in range(self.n_dim_out):
            y[..., i] = torch.sum(x * self.indexes[i], spatial_dims)
        return y
