# ssl-doa-baselines

Independent implementations of three external multi-source DOA baselines, with
a thin LOCATA adapter that reuses LOCATA's own datasets, trainer and scorer.

| Key | Model | Native output | Decoder |
|---|---|---|---|
| `tf_mamba` | TF-Mamba (bidirectional temporal / frequency Mamba) | azimuth spectrum `(B, 181)` | shared peak decoder |
| `icocnn` | multi-source IcoCNN (spherical SRP-PHAT + icosahedral CNN) | ACCDOA slots `(B, T, 10, 3)` | activity threshold, top-3 |
| `srp_dnn` | SRP-DNN (causal CRNN + mixed DP-IPD) | mixed DP-IPD `(B, T, 2F, P)` | iterative detection & localization |
| `ipdnet` | IPDnet (fixed 3-mic full/narrow-band LSTM + causal CNN) | per-track DP-IPD `(B, 2F, P, K)` | projection-ratio threshold |
| `ipdnet2_small` | IPDnet2-Small (fixed 3-mic online spatial network, Mamba) | per-track DP-IPD `(B, 2F, P, K)` | negative-MSE threshold |

The package itself never imports LOCATA.  `integrations/locata/` is the only
boundary and is loaded only by the external runner.

## Install

```bash
conda activate locata                     # Python 3.8, torch 2.2.1, CUDA
pip install -e ".[all]"                   # numpy/scipy/soundfile/tomli + einops
pip install mamba-ssm==2.1.0 --no-build-isolation   # if not already present
```

`mamba-ssm` is needed for `tf_mamba` and `ipdnet2_small`; it requires a CUDA
build.  `ipdnet` and `ipdnet2_small` additionally need `torchaudio`
(`pip install -e ".[ipdnet]"` / `".[ipdnet2]"`); the other models run on CPU
(slowly) or GPU.  The combination verified here is torch 2.2.1 /
torchaudio 2.2.1 / CUDA 11.8.

The two IPDnet ports share a 48 kHz -> 16 kHz resampler, 512-point Hann/FFT
front end (`center=False`, explicit right pad), per-track DP-IPD targets with
`J0` empty tracks and clip-level PIT.  IPDnet uses hop 256 (20 STFT frames,
2 pooled steps); IPDnet2-Small uses hop 320 (17 frames, 4 pooled steps).
Decoding returns one detection per active track: IPDnet scores by the
projection ratio, IPDnet2-Small by the negative template MSE (so the same
`score >= threshold` sweep applies; `-0.4` means `MSE <= 0.4`).

## Package API

```python
from ssl_doa_baselines.api import build_model, load_model, predict, predict_native

model = build_model("srp_dnn", {
    "n_mics": 3,
    "mic_positions": [[0.04, 0.0, 1.2], [0.0, 0.0, 1.2], [-0.04, 0.0, 1.2]],
})
model = load_model("runs/srp_dnn/checkpoint.pth", device="cuda")
detections = predict(model, waveform, decode={"threshold_abs": 0.3,
                                              "max_sources": 3})
# [[{"azimuth_deg": ..., "elevation_deg": ..., "score": ...}, ...], ...]
```

`waveform` is `(T, C)` or `(B, T, C)` at 48 kHz.  Checkpoints written by
`save_checkpoint` embed the model config and are reloadable without passing it.

`ipdnet` and `ipdnet2_small` have no safe default operating point: `predict`
requires an explicit finite `decode["threshold_abs"]` and raises before the
forward pass when it is missing.  The value is normally the threshold locked on
the simulated validation set (`validation.json["locked"]["threshold"]`); for
IPDnet2-Small `-0.4` means "keep tracks with `MSE <= 0.4`".  `predict_native`
still returns the raw tracks without thresholding.

```python
model = load_model("runs/ipdnet2_small_s42/checkpoint.pth", device="cuda")
detections = predict(model, waveform, decode={"threshold_abs": -0.4})
```

The standalone CLI accepts either the explicit value or the validation file
(the two are mutually exclusive):

```bash
ssl-doa-baselines predict \
  --checkpoint runs/ipdnet2_small_s42/checkpoint.pth \
  --audio clip.wav \
  --validation runs/ipdnet2_small_s42/validation.json

ssl-doa-baselines predict \
  --checkpoint runs/ipdnet2_small_s42/checkpoint.pth \
  --audio clip.wav --threshold-abs=-0.4
```

A `--validation` file is matched by checkpoint SHA-256 (copy/move invariant),
model key, score kind and comparison direction; a file from another model or
another epoch is rejected.

Angles use **zenith elevation**: `u = [sin(el)cos(az), sin(el)sin(az), cos(el)]`,
so a horizontal source has `el = 90`.  DICIT metadata already follows this.
LOCATA simulated metadata stores elevation from the horizontal plane; the
adapter converts it with `el_zenith = 90 - el_sim`.

Azimuth follows the project label frame.  On real DICIT frames the GCC lag of
`X_k conj(X_l)` is `-(p_k - p_l)·u_label/c`, i.e. the label frame is mirrored
relative to the raw room coordinates.  The SRP front end therefore steers with
`(p_l - p_k)·u`; with that convention its peaks track DICIT single-source
labels to ~6° MAE.  `examples/make_synthetic.py` generates clips in the same
convention.

## Smoke tests

CPU-only unit tests (TF-Mamba is skipped without CUDA):

```bash
python -m pytest tests -q
```

End-to-end GPU smoke for one model (forward, native decode, checkpoint reload):

```bash
python integrations/locata/run.py \
  --locata-root /path/to/LOCATA \
  --config integrations/locata/configs/icocnn_dicit.toml \
  --data-config integrations/locata/configs/data_local.toml \
  --output /tmp/icocnn_smoke --device cuda:0 smoke
```

## LOCATA training and evaluation

Copy the machine-specific data config and edit the paths:

```bash
cp integrations/locata/configs/data_local.example.toml \
   integrations/locata/configs/data_local.toml
```

Train, lock the operating point on the simulated validation set, then evaluate
on DICIT:

```bash
python integrations/locata/train_then_eval.py \
  --locata-root /path/to/LOCATA \
  --config integrations/locata/configs/tf_mamba_dicit.toml \
  --data-config integrations/locata/configs/data_local.toml \
  --output runs/tf_mamba_spectrum_s42 --seed 42 --device-index 0
```

`train` alone writes:

```
runs/<name>/
  checkpoint.pth      # embedded model config, best validation epoch
  validation.json     # full threshold curve + locked working point
  config.json         # resolved config, both git commits, dirty flags
  train.log           # TF-Mamba only (LOCATA Trainer log)
```

`eval --split dicit --validation .../validation.json` writes `eval_dicit.json`
with the locked point, per-source-count metrics, the full test curve and
`R@FAR 3%` interpolation (only when two measured points bracket 3% FAR).
It also stores `predictions.gt` / `predictions.pred` at the locked point (one
angle set per frame, plain floats) so individual frames can be re-checked, plus
`num_params` (counted from the rebuilt model), `forward_ms_per_frame`,
`decode_ms_per_frame` and `inference_ms_per_clip`.

`inference_ms_per_clip` is the unified end-to-end number: from a waveform
already on the device to the per-clip angle set at the locked threshold,
including the method's own front end, backbone, template matching and score
filter.  Warm-up, device synchronisation and the batch size are the same for
every method, and the denominator is the actual clip count (a short final
batch is not rounded up).  Public scoring, disk reads and the full threshold
sweep are excluded; the older `forward_ms_per_frame` and the IPDnet
`frontend` / `template` breakdowns remain for profiling.

For `ipdnet` / `ipdnet2_small`, DICIT evaluation **requires**
`--validation` and fails before touching the dataset otherwise: the test
threshold must be the one locked on the simulated validation set.  The locked
point is merged into the reported test curve numerically, so a changed scan
config or JSON key spelling cannot drop it.

``eval --split val --output PATH`` writes the same `validation.json` that
`train` produces (curve + identity-bearing lock), so a threshold can be
regenerated for an existing checkpoint without retraining.  Training records
`mic_indices`, `mic_positions` and `ref_mic` in the checkpoint config and run
record; evaluation rejects a swapped channel order or a changed pair spacing,
while allowing a translated copy of the same array.

### Metrics

All scoring calls LOCATA's `engine/sweep_eval.py` and
`engine/metrics.py` (`score_decoded`), with 10° one-to-one matching and
azimuth-only error.  TF-Mamba reuses the same spectrum cache + peak decoder as
`run_pr_sweep.py`; IcoCNN and SRP-DNN cache native `(score, az, el)`
candidates and sweep their own absolute threshold.  Operating points are locked
on the simulated validation set by maximum `F1@10°` (tie-break: lower FAR).

Grouping by true source count is computed with the same matcher.  FAR keeps
silence-frame false peaks, matching LOCATA's evaluator.

## Queue submission

Training must run through `gpu_queue` (foreground command, train + eval share
one slot).  `--locata-root` points at a dedicated worktree of the LOCATA repository
(`/home/t-zhangchaoqin/projects/ssl/wt/locata-dev`), **detached at a fixed
commit** (`d5e7c5d`, the `dev` tip on 2026-11-11 10:58).  It was created from
`dev` so the shared checkout used by the V5 ablations stays untouched, and it is
detached so that merge activity on `dev` cannot change the code a queued run
imports.  To move the baselines to a newer LOCATA revision, check out that
commit explicitly and resubmit.  Each run records the actual LOCATA
commit/branch/dirty flag under `config.json -> locata`.

Example job manifest:

```json
{
  "jobs": [
    {
      "id": "tf_mamba_spectrum_s42",
      "cwd": "/path/to/ssl-doa-baselines",
      "gpu": 0,
      "argv": [
        "/home/t-zhangchaoqin/.conda/envs/locata/bin/python",
        "integrations/locata/train_then_eval.py",
        "--locata-root", "/path/to/LOCATA",
        "--config", "integrations/locata/configs/tf_mamba_dicit.toml",
        "--data-config", "integrations/locata/configs/data_local.toml",
        "--output", "runs/tf_mamba_spectrum_s42",
        "--seed", "42", "--device-index", "0"
      ]
    }
  ]
}
```

```bash
PYTHONPATH=/path/to/Record/queue \
GPU_QUEUE_CONFIG=/path/to/Record/queue/gpu-queue.config.json \
python -m gpu_queue submit /path/to/Record/queue/state jobs.json
```

Submit `icocnn_multi_s42`, `srp_dnn_s42`, `ipdnet_s42` and
`ipdnet2_small_s42` the same way with their configs.  The two IPDnet manifests
are `queue/ipdnet_512_s42.json` and `queue/ipdnet2_small_512_s42.json`.

Both IPDnet jobs run from a dedicated detached worktree,
`/home/t-zhangchaoqin/projects/ssl/wt/ssl-doa-baselines-ipdnet`, pinned at the
integration merge commit, so code cannot change under a running job.  The
machine-local, git-ignored `data_local.toml` and the artifact root are passed
by absolute path into the main checkout; `runs/ipdnet_s42` and
`runs/ipdnet2_small_s42` therefore stay in the indexed checkout.

### IPDnet provenance

Both ports are fixed-array adaptations of
`Audio-WestlakeU/FN-SSL@76fcb281` (MIT; see `THIRD_PARTY.md`).  They train
from random initialisation on the shared LOCATA protocol; the authors'
paper-level results, native arrays and frame-level outputs are not reproduced
here.

### Shared training budget

All three methods run `--epochs 100` with `patience 6` eval rounds of early
stopping and seed 42, so they see the same sample budget.  A 15-epoch first
pass was dropped: the V5 mainline (B0 / Parallel) picks its best epoch around
30 and early-stops at 60, so 15 epochs sat at roughly a quarter of the
comparison budget.  The effective epoch count and budget are recorded in each
`config.json`.  The submitted manifests are
`queue/external_baselines_2026-11-11_v4.json` (TF-Mamba, IcoCNN) and
`queue/external_baselines_2026-11-11_srp100.json` (SRP-DNN).

## Layout

```
src/ssl_doa_baselines/
  api.py config.py geometry.py training.py cli.py
  dpipd_frontend.py dpipd_tracks.py     shared IPDnet front end / targets / decode
  tf_mamba/  model.py decode.py
  icocnn/    model.py srp.py grid.py conv.py accdoa.py
  srp_dnn/   model.py stft.py dpipd.py idl.py
  ipdnet/    model.py backbone.py
  ipdnet2/   model.py backbone.py arch/base/
integrations/locata/
  adapter.py            data / labels / training / scoring bridge
  run.py                smoke | train | eval
  train_then_eval.py    gpu_queue foreground wrapper
  configs/              one TOML recipe per model + data_local.example.toml
examples/make_synthetic.py
tests/
runs/                   active and completed run artifacts (git-ignored)
aborted/                artifacts of aborted attempts (git-ignored)
```

## Status

Smoke tests (GPU forward, native decode, checkpoint reload) pass for all five
models, including the IPDnet2-Small Mamba path.  The unit suite covers shapes,
PIT invariance, IDL on ideal DP-IPD, label conversion, SRP steering, the
five-block CRNN forward, the import boundary, the shared IPDnet front end
(resampling, frame counts, phase sign) and per-track targets/decoding.

On 2026-11-10 the three formal jobs (TF-Mamba `_dev`, IcoCNN `_dev`, SRP-DNN
`_dev`) were submitted to `gpu_queue` with the shared 15-epoch budget and
`--locata-root` pinned to the `dev` worktree.  That budget was abandoned: the
V5 mainline picks its best epoch around 30, so 15 epochs sat at a quarter of
the comparison budget.  TF-Mamba's `_dev` run also hit an fp16 overflow, and
IcoCNN's ran on a pathologically slow version of the vendored layers.

The current canonical runs are `tf_mamba_s42` (artifact
`runs/tf_mamba_spectrum_s42`, running on GPU0) and `icocnn_multi_s42` (paused
after epoch 1 — too slow to share a GPU; see `RESULTS.md`).  Both queued jobs
use 100 epochs / `patience 6` / seed 42 on `gpu_queue`:

- `srp_dnn_s42` (`queue/srp_dnn_2026-11-12.json`, GPU1): reruns the recipe with
  the fixed forward.  The previous run executed only four of the five CRNN
  blocks (`zip(self.blocks, self.pools)` stops after the four pools), so its
  artifacts are archived under `aborted/srp_dnn_block5_20261112/` and its
  metrics are not cited.
- `tf_mamba_512_s42` (`queue/tf_mamba_512_2026-11-12.json`, GPU1): the
  `n_fft=512` front-end variant, `tf_mamba_dicit_512.toml`, against the
  running 1536/480 run.

Both are pending until a GPU1 slot frees.  Every attempt's status is recorded
in the `RESULTS.md` run log; job ids in historical `queue/` files keep their
original `_vN` names.
