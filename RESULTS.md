# Main table (SRP-DNN complete; TF-Mamba running; IcoCNN awaiting faithful rerun)

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
| IcoCNN (multi-source) | pending (`_v5c`, output `runs/icocnn_multi_s42_v5`) | — | — | — | — | — | — | — | 6.17 M | — |
| SRP-DNN | `srp_dnn_idl_s42` (v4c) | 0.7611 | 0.8842 | 0.7667 | 0.7330 | 0.1475 | not covered | 3.37 | 0.86 M | 1.66 |

SRP-DNN (threshold 0.40 locked on 9,462 simulated validation frames):
precision 0.827, recall 0.705, frames 926 (176 / 379 / 371).  `R@FAR 3%` is
not bracketed: the lowest FAR on the DICIT curve is 5.03% (at threshold 0.9,
where F1@10 = 0.7025).  Forward time is per 16,000-sample clip on one RTX 4090,
including the STFT front end and IDL.

**SRP-DNN budget alignment.**  The row is the 100-epoch `srp_dnn_idl_s42_v4c`
run (`patience 6`, seed 42).  Against the 15-epoch `_dev` run (F1@10 0.7607,
FAR 0.1221, MAE 3.27) the extra budget changed F1 by +0.0004 and did not
resolve `R@FAR 3%`; the 15-epoch artifacts are archived under
`runs/_aborted_budget15_20261111/srp_dnn_dev`.  Note: with the early-stopping
bug below, `_v4c` stopped at epoch 33 (3 validation rounds after the epoch-27
minimum) instead of the intended 6 rounds; epoch 27 is a true validation
minimum, so the metrics stand, but the run did not consume the full
12-round window.

**IcoCNN is being rerun.**  `icocnn_multi_s42_v4` is invalid: its head kept
`width` channels through the last temporal convolution and added a
`LayerNorm(5120) + Linear(30)` readout, which is not the reference model.  It
collapsed to a constant 3-beam output (926/926 DICIT frames byte-identical;
training loss flat at 0.0312 from epoch 2).  The model now follows
arXiv:2210.14536 (last temporal conv outputs `M`, then the author's
Soft-ArgMax).

Two further causes were found with fixed-batch screens on 2026-11-12:

1. **Learning rate.** The old config used `lr=1e-3`; the author's
   `1sourceTracking_icoCNN.py` uses `1e-4`. At 1e-3 the first AdamW step
   destroys the Soft-ArgMax's near-uniform initialisation (output ~1e-4),
   after which the 9 empty slots' zero targets pin the model at the trivial
   solution (`loss = 1/30` with one active source). Even an `M=1` model
   collapses at 1e-3, so this was not a multi-source effect.
2. **SRP self-pair pedestal.** The vendored SRP map summed all `N×N`
   microphone pairs; a PHAT self-pair is a constant delta, adding a constant
   to every grid cell. With 12 reference mics that is a small fraction, but
   with 3 mics it is a third of the sum (map mean 0.40, std 0.16 vs 0.08 /
   0.25 after removal). The front end now sums cross-pairs only.

With both fixes, an 80-step fixed-batch screen drops the loss from 0.0332 to
0.0023 and output norms grow to ~0.83; at 1e-4 with the pedestal still in,
training oscillates around the zero solution. The mixed 1/2/3-source
acceptance run passed (200 steps: loss 0.067 → 0.0069, best-match error
70° → 7.4°, distinct predictions per clip), and `icocnn_multi_s42_v5c` is
running on `gpu_queue`. The `_v4` artifacts are kept only as evidence of the
failure.

**Early stopping (all three rows).** `patience=6` counts validation rounds
(`eval_every` epochs apart), matching LOCATA's `Trainer`; the package loop had
been counting every epoch and using the training loss as the monitor on
non-eval epochs, so IcoCNN could stop after ~3 validation rounds. Fixed on
2026-11-12 (covered by `tests/test_training.py`); `_v5c` is the first run with
the corrected semantics.

DICIT frame counts (for reference): 926 frames total, 176 / 379 / 371 for
1 / 2 / 3 sources; the actual counts are written in each `eval_dicit.json`.

## Validation working points

| Method | Locked threshold | Val F1@10° | Val FAR |
|---|---|---|---|
| TF-Mamba | — | — | — |
| IcoCNN | — | — | — |
| SRP-DNN | 0.40 | 0.6904 | 0.1801 |

## Run log

| Job | Outcome |
|---|---|
| `srp_dnn_idl_s42_dev` | done 2026-11-10 22:21, 15 epochs, Adam, lr 1e-3; results archived, superseded by `_v4b` |
| `srp_dnn_idl_s42_v4` | canceled 2026-11-11 16:40 — started on GPU0; moved to GPU1 after GPU0 was reduced to one slot |
| `srp_dnn_idl_s42_v4b` | canceled 2026-11-11 17:20 — shared NFS dataset latency rose ~12x once three jobs loaded it concurrently (SRP epoch 4 > 21 min vs 3.7–5.8 min); taken off the GPU and requeued as `_v4c` |
| `srp_dnn_idl_s42_v4c` | done — 100 epochs / patience 6 / seed 42; DICIT F1@10 0.7611 (vs 0.7607 at 15 epochs), no meaningful gain |
| `tf_mamba_spectrum_s42_dev` | failed 2026-11-11 07:15 — fp16 autocast overflows (forward NaN from step ~132 on real batches); no checkpoint survived. fp32 at batch 8 is stable (400 steps, 12.8 GiB) |
| `icocnn_multi_s42_dev` | ran on the pre-fix code: epoch 1 took 39,032 s (the vendored icoCNN layers rebuilt their index tensors on the host every forward). Fixed in `4efc135` (25x faster step, bitwise-identical output); superseded by `_v2` |
| `tf_mamba_spectrum_s42_v2` | canceled 2026-11-11 11:18 after 20 min — trained with `label_sigma = 1.5`, the coarse-to-fine control width, not the mainline B0/Parallel width (3.0) |
| `icocnn_multi_s42_v2` | canceled 2026-11-11 11:47 after 2 epochs — 15-epoch budget replaced by the 100-epoch budget |
| `tf_mamba_spectrum_s42_v3` | canceled 2026-11-11 11:47 after 1 epoch — same budget change |
| `tf_mamba_spectrum_s42_v4` | submitted 2026-11-11 11:47 — fp32, batch 8, `label_sigma = 3.0`, 100 epochs / patience 6; ~44 min/epoch, ~0.22 s/step |
| `icocnn_multi_s42_v4` | **invalid** — non-reference head (`LayerNorm+Linear` over 128 channels instead of last conv → `M` + Soft-ArgMax); collapsed to a constant 3-beam output (identical DICIT predictions on all 926 frames, loss flat at 0.0312 from epoch 2). Superseded by the reference-head code and `_v5` |
| `icocnn_multi_s42_v5` | canceled 2026-11-12 15:49 after startup — early stopping counted epochs, not validation rounds, and monitored train loss on non-eval epochs; cancelled before the first epoch so the fix could take effect |
| `icocnn_multi_s42_v5b` | canceled 2026-11-12 16:01 after 9 min — the native loop logged epochs only to the queue `command.log`, so the run directory had no `train.log`; restarted as `_v5c` after adding the log tee |
| `icocnn_multi_s42_v5c` | submitted 2026-11-12 16:02 — reference head (last conv → `M` + Soft-ArgMax), `lr 1e-4`, cross-pair-only SRP maps, per-epoch `runs/icocnn_multi_s42_v5/train.log`; 100 epochs / patience 6 validation rounds / seed 42 on GPU1; fixed-batch acceptance passed (loss 0.067 → 0.0069, best-match error 70° → 7.4°) |
| `srp_dnn_dicit_reeval_v2` | done 2026-11-11 11:02 — refreshed `runs/srp_dnn_idl_s42/eval_dicit.json` with the evaluated split's metrics (the first evaluation stored the validation working point in `metrics`) |
