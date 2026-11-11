# Main table (SRP-DNN complete; TF-Mamba and IcoCNN pending)

Protocol: 48 kHz, 16,000-sample clips, mics `5,6,8`, azimuth 0°–180°,
10° one-to-one matching, azimuth-only MAE.  Operating points are locked on the
simulated validation set; DICIT curves are reported separately.

**Shared training budget.**  All three rows are trained with 15 epochs
(1.41 M samples, `--epochs 15`, seed 42) so the comparison is budget-matched.
The upstream 200-epoch recipes remain in the configs; the actual values are
recorded in each `config.json`.  Jobs: `queue/external_baselines_2026-11-10.json`.

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
| TF-Mamba | `tf_mamba_spectrum_s42_v2` | — | — | — | — | — | — | — | 1.41 M | — |
| IcoCNN (multi-source) | `icocnn_multi_s42_v2` | — | — | — | — | — | — | — | 6.41 M | — |
| SRP-DNN | `srp_dnn_idl_s42_dev` | 0.7607 | 0.8727 | 0.7782 | 0.7251 | 0.1221 | not covered | 3.27 | 0.86 M | 1.66 |

SRP-DNN (threshold 0.45 locked on 9,462 simulated validation frames):
precision 0.849, recall 0.689, frames 926 (176 / 379 / 371).  `R@FAR 3%` is
not bracketed: the lowest FAR on the DICIT curve is 4.15%.  Forward time is
per 16,000-sample clip on one RTX 4090, including the STFT front end and IDL.

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
| `srp_dnn_idl_s42_dev` | done 2026-11-10 22:21, 15 epochs, Adam, lr 1e-3 |
| `tf_mamba_spectrum_s42_dev` | failed 2026-11-11 07:15 — fp16 autocast overflows (forward NaN from step ~132 on real batches); no checkpoint survived. fp32 at batch 8 is stable (400 steps, 12.8 GiB) |
| `icocnn_multi_s42_dev` | ran on the pre-fix code: epoch 1 took 39,032 s (the vendored icoCNN layers rebuilt their index tensors on the host every forward). Fixed in `4efc135` (25x faster step, bitwise-identical output); superseded by `_v2` |
| `tf_mamba_spectrum_s42_v2` | running since 2026-11-11 10:57 — fp32, batch 8, ~0.25 s/step (~50 min/epoch, ~12 h for 15 epochs) |
| `icocnn_multi_s42_v2` | running since 2026-11-11 11:00 — fp32, batch 8, ~0.18 s/step (~35 min/epoch) |
| `srp_dnn_dicit_reeval_v2` | done 2026-11-11 11:02 — refreshed `runs/srp_dnn_idl_s42/eval_dicit.json` with the evaluated split's metrics (the first evaluation stored the validation working point in `metrics`) |
