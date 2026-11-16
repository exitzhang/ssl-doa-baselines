# Third-party code and papers

This repository re-implements / vendors components from the following
projects.  The repository as a whole is distributed under **AGPL-3.0** because
the icosahedral CNN code is AGPL-3.0.

## icoCNN — icosahedral spherical CNN

- Upstream: <https://github.com/DavidDiazGuerra/icoCNN>
- Commit used: `e9720a1f9b0af1cd056240dcea8dc7572397ece1` (2023-04-11)
- License: AGPL-3.0
- Vendored files:
  - `src/ssl_doa_baselines/icocnn/grid.py` ← `icoCNN/icoGrid.py`
  - `src/ssl_doa_baselines/icocnn/conv.py` ← `icoCNN/icoCNN.py`
- Changes: import paths only; attribution headers added.
- Citation: D. Diaz-Guerra, A. Miguel, J. R. Beltran, *"Direction of arrival
  estimation with two-dimensional convolutional neural networks"*, IEEE
  Signal Processing Letters, 2021.

## icoDOA — spherical SRP-PHAT front end

- Upstream: <https://github.com/DavidDiazGuerra/icoDOA>
- Commit used: `04d1a89594c78ae3cf42f07d94c3737bdc1f7c82` (2022-05-19)
- License: AGPL-3.0
- Ported into `src/ssl_doa_baselines/icocnn/srp.py` (`GCC`, spherical SRP map),
  `icocnn/readout.py` (`SoftArgMax`) and the causal temporal convolution in
  `icocnn/model.py`.
- Changes: 48 kHz / 12,288-sample window / 1,536-sample hop for 16,000-sample
  clips; vectorised SRP gather; SRP maps sum microphone *cross-pairs* only.
  The PHAT self-correlation is a constant at lag 0, i.e. a constant added to
  every grid point; with the reference 12-mic array it is a small part of the
  sum, but with the 3-mic DICIT array it is a third of the sum and flattens the
  map (mean 0.40 vs 0.08, std 0.16 vs 0.25).  It carries no directional
  information, so it is dropped.  The multi-source head follows
  arXiv:2210.14536 exactly: the last temporal convolution outputs `M` channels
  instead of one, and the author's Soft-ArgMax maps each channel to a 3-D
  ACCDOA vector (direction = DOA, norm = activity).  Training uses the author's
  learning rate (`lr=1e-4` in `1sourceTracking_icoCNN.py`); at 1e-3 the
  Soft-ArgMax's near-uniform initialisation collapses to the zero output.
- Related papers:
  - D. Diaz-Guerra, J. R. Beltran, *"Direction of arrival estimation in
    dynamic acoustic scenes with two-dimensional convolutional recurrent
    networks"*, arXiv:2210.14536.
  - Multi-source configuration, arXiv:2306.08510.

## SRP-DNN

- Upstream: <https://github.com/BingYang-20/SRP-DNN>
- Commit used: `e67c2a9d24d8707048f38e09146fb2d3cbc9be8d` (2024-09-28)
- License: MIT
- Ported files:
  - `src/ssl_doa_baselines/srp_dnn/model.py` ← `code/ModelSRPDNN.py` (CRNN)
  - `src/ssl_doa_baselines/srp_dnn/stft.py` ← `code/Module.py` / `code/utils.py`
    (STFT front end, `forgetting_norm`, channel-pair reshape)
  - `src/ssl_doa_baselines/srp_dnn/dpipd.py` ← `code/Module.py::DPIPD`
  - `src/ssl_doa_baselines/srp_dnn/idl.py` ← `code/Module.py::SourceDetectLocalize`
- Changes: explicit physical frequency list, 48 kHz / 1,536 FFT adaptation,
  unordered 3-mic pairs, vectorised IDL, azimuth in [0°, 180°] with zenith
  elevation.
- Citation: B. Yang et al., *"SRP-DNN: Learning direct-path phase difference
  for multiple sound source localization"*, ICASSP 2022, arXiv:2202.07859.

## TF-Mamba

- Paper: X. Xiao et al., *"TF-Mamba: ..."*, Interspeech 2025
  (<https://www.isca-archive.org/interspeech_2025/xiao25_interspeech.pdf>).
- Code provenance: adapted from the historical LOCATA-internal implementation
  at commit `b33a2ab` (`models/TF_Mamba.py`), which follows the paper's
  bidirectional temporal / frequency Mamba blocks, DenseNet encoder and
  same-axis skip connections.
- Changes: arbitrary microphone count (3 for DICIT), 100–8,000 Hz analysis
  band, configurable output activation (`sigmoid` to match the shared heatmap
  loss), clip-level spectrum readout.
- The model requires `mamba-ssm==2.1.0` and a CUDA build.

## IPDnet / IPDnet2-Small

- Upstream: <https://github.com/Audio-WestlakeU/FN-SSL>
- Commit used: `76fcb281be92caf068c712dfb015e354f437260f` (2026-11-16 fetch)
- License: the root `README.md` marks the project **MIT**.  That commit has no
  standalone root `LICENSE` file; the vendored `retention.py` retains its
  Microsoft / Westlake MIT headers.  The port keeps the applicable copyright
  and licence notices in the files below.
- Papers:
  - IPDnet: <https://arxiv.org/html/2405.07021>
  - IPDnet2: <https://arxiv.org/html/2509.21900v1>
- Ported files:
  - `src/ssl_doa_baselines/ipdnet/backbone.py` ←
    `IPDnet/FixedAarryIPDnet.py` (`FNblock`, `CausCnnBlock`, `IPDnet`);
  - `src/ssl_doa_baselines/ipdnet/model.py`, `dpipd_frontend.py`,
    `dpipd_tracks.py` ← the front end, target construction and PIT in
    `IPDnet/runIPDnetOn.py`;
  - `src/ssl_doa_baselines/ipdnet2/backbone.py` ← `IPDnet2/IPDnet2.py`
    (`FreqInverse`, `CausalConv1d`, `SpatialNetLayer`, `OnlineSpatialNet`);
  - `src/ssl_doa_baselines/ipdnet2/model.py` ← the Small configuration in
    `IPDnet2/run_IPDnet2.py`;
  - `src/ssl_doa_baselines/ipdnet2/arch/base/linear_group.py`,
    `non_linear.py`, `norm.py`, `retention.py` ← the files of the same names
    under `IPDnet2/arch/base/` (vendored verbatim);
  - `dpipd_tracks.py` reimplements the far-field `DPIPD` template and the
    per-track decode in `IPDnet2/Module.py`.
- Changes:
  - fixed 3-microphone, 3-track configuration (`input_size=6`,
    `dim_output=12`, `max_track=3`);
  - the authors' partially hard-coded `reshape`/`permute` is written with
    explicit real/imaginary, frequency, microphone-pair and track axes, giving
    `(B, T_out, 2F, P, K)` with every real part before every imaginary part;
  - both backbones preserve the last partially filled time-pooling group
    (`ceil_mode=True`), so a 20- (IPDnet) or 17-frame (IPDnet2) clip yields 2
    or 4 pooled steps;
  - the front end is shared: a 48 kHz -> 16 kHz anti-aliasing resampler,
    512-point Hann window / FFT, `center=False` with right-hand zero padding,
    and FP32 forgetting-magnitude normalisation;
  - activity decoding is per track: projection ratio for IPDnet, negative
    template MSE for IPDnet2 (so a shared `score >= threshold` sweep works);
  - relative imports and a clear Mamba dependency error; the per-layer shape
    `print` is removed.
- Unlike the paper's native DICIT recordings, this integration trains and
  evaluates on the repository's frozen 16,000-sample DICIT frame protocol.

## LOCATA / DICIT components used by the adapter (not vendored)

`integrations/locata/adapter.py` imports, at runtime and from an explicit
checkout, LOCATA's:

- `data/audio_dataset.py`, `data/factory.py` (datasets);
- `engine/trainer.py` (TF-Mamba training);
- `engine/sweep_eval.py`, `engine/matcher.py`, `engine/metrics.py`
  (shared scoring);
- `losses/doa_losses.py` (`mse_kl_loss`).

No LOCATA file is copied into this repository.
