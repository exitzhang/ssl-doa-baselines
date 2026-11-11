# Main table (SRP-DNN complete; TF-Mamba and IcoCNN pending)

Protocol: 48 kHz, 16,000-sample clips, mics `5,6,8`, azimuth 0°–180°,
10° one-to-one matching, azimuth-only MAE.  Operating points are locked on the
simulated validation set; DICIT curves are reported separately.

**Label width.**  TF-Mamba's spectrum label uses `label_sigma = 3.0` bins
(1° per bin at Q=181), matching the mainline V5 baseline B0 and the Parallel
arms, which train with `--label_sigma 3`.  The σ=1.5 runs (A0/A1/A2) are the
coarse-to-fine controls, not the baseline.  IcoCNN (ACCDOA unit vectors) and
SRP-DNN (DP-IPD template regression + IDL) have no Gaussian label.

**Shared training budget.**  All three rows use `--epochs 100` with
`patience 6` eval rounds of early stopping (seed 42); the effective epoch count
and sample budget are recorded in each `config.json`.  The 15-epoch first pass
was abandoned: the V5 mainline (B0/Parallel) picks its best epoch around 30 and
early-stops at 60, so 15 epochs left the external baselines at roughly a
quarter of the comparison budget.  Jobs:
`queue/external_baselines_2026-11-11_v4.json` (TF-Mamba, IcoCNN) and
`queue/external_baselines_2026-11-11_srp100.json` (SRP-DNN).

**LOCATA pin.**  Every run imports LOCATA from the worktree
`/home/t-zhangchaoqin/projects/ssl/wt/locata-dev`, detached at `d5e7c5d`
(clean).  The shared checkout `/projects/ssl/LOCATA` stays on the V5 ablation
branch, which the running ablation jobs use for their own sweep step.

The pin was applied on 2026-11-11 after `dev` moved under the running jobs
(`73e95b3` at 10:48, `d5e7c5d` at 10:58): `tf_mamba_spectrum_s42_v2` imported
`73e95b3`, `icocnn_multi_s42_v2` imported `d5e7c5d`, and SRP-DNN ran on
`54ffaa6`.  The delta between them is additive coarse-to-fine support that is
inert when `coarse_supervision` / `coarse_heatmap_creator` are `None`, so the
three rows remain comparable; from now on every process (including the eval
step) imports the pinned commit.

## DICIT test — locked validation working point

| Method | Run | F1@10° | 1-src F1 | 2-src F1 | 3-src F1 | FAR | R@FAR 3% | MAE (°) | Params | Forward ms |
|---|---|---|---|---|---|---|---|---|---|---|
| TF-Mamba | `tf_mamba_spectrum_s42_v4` | — | — | — | — | — | — | — | 1.41 M | — |
| IcoCNN (multi-source) | `icocnn_multi_s42_v4` | — | — | — | — | — | — | — | 6.41 M | — |
| SRP-DNN | `srp_dnn_idl_s42_dev` | 0.7607 | 0.8727 | 0.7782 | 0.7251 | 0.1221 | not covered | 3.27 | 0.86 M | 1.66 |

SRP-DNN (threshold 0.45 locked on 9,462 simulated validation frames):
precision 0.849, recall 0.689, frames 926 (176 / 379 / 371).  `R@FAR 3%` is
not bracketed: the lowest FAR on the DICIT curve is 4.15%.  Forward time is
per 16,000-sample clip on one RTX 4090, including the STFT front end and IDL.

**SRP-DNN budget alignment.**  The numbers above are the 15-epoch `_dev` run.
`srp_dnn_idl_s42_v4b` (100 epochs, `patience 6`, same seed 42 and recipe) was
submitted to `gpu_queue` on 2026-11-11 and will replace this row when it
finishes.  The 15-epoch artifacts are archived under
`runs/_aborted_budget15_20261111/srp_dnn_dev`; `runs/srp_dnn_idl_s42` is reused
for the new run.

DICIT frame counts (for reference): 926 frames total, 176 / 379 / 371 for
1 / 2 / 3 sources; the actual counts are written in each `eval_dicit.json`.

## Validation working points

| Method | Locked threshold | Val F1@10° | Val FAR |
|---|---|---|---|
| TF-Mamba | — | — | — |
| IcoCNN | — | — | — |
| SRP-DNN | 0.45 | 0.6959 | 0.1531 |

## Run log

| Job | Outcome |
|---|---|
| `srp_dnn_idl_s42_dev` | done 2026-11-10 22:21, 15 epochs, Adam, lr 1e-3; results archived, superseded by `_v4b` |
| `srp_dnn_idl_s42_v4` | canceled 2026-11-11 16:40 — started on GPU0; moved to GPU1 after GPU0 was reduced to one slot |
| `srp_dnn_idl_s42_v4b` | submitted 2026-11-11 — 100 epochs / patience 6, seed 42, same Adam lr 1e-3 recipe, pinned to GPU1; output `runs/srp_dnn_idl_s42` |
| `tf_mamba_spectrum_s42_dev` | failed 2026-11-11 07:15 — fp16 autocast overflows (forward NaN from step ~132 on real batches); no checkpoint survived. fp32 at batch 8 is stable (400 steps, 12.8 GiB) |
| `icocnn_multi_s42_dev` | ran on the pre-fix code: epoch 1 took 39,032 s (the vendored icoCNN layers rebuilt their index tensors on the host every forward). Fixed in `4efc135` (25x faster step, bitwise-identical output); superseded by `_v2` |
| `tf_mamba_spectrum_s42_v2` | canceled 2026-11-11 11:18 after 20 min — trained with `label_sigma = 1.5`, the coarse-to-fine control width, not the mainline B0/Parallel width (3.0) |
| `icocnn_multi_s42_v2` | canceled 2026-11-11 11:47 after 2 epochs — 15-epoch budget replaced by the 100-epoch budget |
| `tf_mamba_spectrum_s42_v3` | canceled 2026-11-11 11:47 after 1 epoch — same budget change |
| `tf_mamba_spectrum_s42_v4` | submitted 2026-11-11 11:47 — fp32, batch 8, `label_sigma = 3.0`, 100 epochs / patience 6; ~44 min/epoch, ~0.22 s/step |
| `icocnn_multi_s42_v4` | submitted 2026-11-11 11:47 — fp32, batch 8, 100 epochs / patience 6; ~25.5 min/epoch (measured in `_v2`) |
| `srp_dnn_dicit_reeval_v2` | done 2026-11-11 11:02 — refreshed `runs/srp_dnn_idl_s42/eval_dicit.json` with the evaluated split's metrics (the first evaluation stored the validation working point in `metrics`) |
