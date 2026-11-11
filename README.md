# ssl-doa-baselines

Independent implementations of three external multi-source DOA baselines, with
a thin LOCATA adapter that reuses LOCATA's own datasets, trainer and scorer.

| Key | Model | Native output | Decoder |
|---|---|---|---|
| `tf_mamba` | TF-Mamba (bidirectional temporal / frequency Mamba) | azimuth spectrum `(B, 181)` | shared peak decoder |
| `icocnn` | multi-source IcoCNN (spherical SRP-PHAT + icosahedral CNN) | ACCDOA slots `(B, T, 10, 3)` | activity threshold, top-3 |
| `srp_dnn` | SRP-DNN (causal CRNN + mixed DP-IPD) | mixed DP-IPD `(B, T, 2F, P)` | iterative detection & localization |

The package itself never imports LOCATA.  `integrations/locata/` is the only
boundary and is loaded only by the external runner.

## Install

```bash
conda activate locata                     # Python 3.8, torch 2.2.1, CUDA
pip install -e ".[all]"                   # numpy/scipy/soundfile/tomli + einops
pip install mamba-ssm==2.1.0 --no-build-isolation   # if not already present
```

`mamba-ssm` is only needed for `tf_mamba`; it requires a CUDA build.  The other
two models run on CPU (slowly) or GPU.

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
`forward_ms_per_frame` and `decode_ms_per_frame`.

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

Submit `icocnn_multi_s42` and `srp_dnn_idl_s42` the same way with their configs.

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
  tf_mamba/  model.py decode.py
  icocnn/    model.py srp.py grid.py conv.py accdoa.py
  srp_dnn/   model.py stft.py dpipd.py idl.py
integrations/locata/
  adapter.py            data / labels / training / scoring bridge
  run.py                smoke | train | eval
  train_then_eval.py    gpu_queue foreground wrapper
  configs/              one TOML recipe per model + data_local.example.toml
examples/make_synthetic.py
tests/
```

## Status

Smoke tests (GPU forward, native decode, checkpoint reload) pass for all three
models; the 20-case CPU unit suite covers shapes, PIT invariance, IDL on ideal
DP-IPD, label conversion, SRP steering and the import boundary.

On 2026-11-10 the three formal jobs (`tf_mamba_spectrum_s42_dev`,
`icocnn_multi_s42_dev`, `srp_dnn_idl_s42_dev`) were submitted to `gpu_queue`
with the shared 15-epoch budget and `--locata-root` pinned to the `dev`
worktree.  SRP-DNN finished; TF-Mamba hit an fp16 overflow and IcoCNN ran on a
pathologically slow version of the vendored layers, so on 2026-11-11 both were
resubmitted in fp32 as `queue/external_baselines_2026-11-11_v2.json`
(`tf_mamba_spectrum_s42_v2`, `icocnn_multi_s42_v2`).  Results are filled into
`RESULTS.md` as each job finishes.

On 2026-11-11 the budget was raised to 100 epochs / `patience 6` for all three
rows; SRP-DNN's 15-epoch run is archived and re-queued as
`srp_dnn_idl_s42_v4b` (`queue/external_baselines_2026-11-11_srp100.json`, GPU1).
