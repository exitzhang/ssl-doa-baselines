# Main table (pending formal runs)

Protocol: 48 kHz, 16,000-sample clips, mics `5,6,8`, azimuth 0°–180°,
10° one-to-one matching, azimuth-only MAE.  Operating points are locked on the
simulated validation set; DICIT curves are reported separately.

**Shared training budget.**  All three rows are trained with 15 epochs
(1.41 M samples, `--epochs 15`, seed 42) so the comparison is budget-matched.
The upstream 200-epoch recipes remain in the configs; the actual values are
recorded in each `config.json`.  Jobs: `queue/external_baselines_2026-11-10.json`.

## DICIT test — locked validation working point

| Method | Run | F1@10° | 1-src F1 | 2-src F1 | 3-src F1 | FAR | R@FAR 3% | MAE (°) | Params | Forward ms |
|---|---|---|---|---|---|---|---|---|---|---|
| TF-Mamba | `tf_mamba_spectrum_s42` | — | — | — | — | — | — | — | 1.41 M | — |
| IcoCNN (multi-source) | `icocnn_multi_s42` | — | — | — | — | — | — | — | 6.41 M | — |
| SRP-DNN | `srp_dnn_idl_s42` | — | — | — | — | — | — | — | 0.86 M | — |

DICIT frame counts (for reference): 926 frames total, 176 / 379 / 371 for
1 / 2 / 3 sources; the actual counts are written in each `eval_dicit.json`.

## Validation working points

| Method | Locked threshold | Val F1@10° | Val FAR |
|---|---|---|---|
| TF-Mamba | — | — | — |
| IcoCNN | — | — | — |
| SRP-DNN | — | — | — |
