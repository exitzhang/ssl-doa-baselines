# Main table (SRP-DNN and TF-Mamba 512 queued; TF-Mamba 1536 running; IcoCNN paused — too slow to share a GPU)

Protocol: 48 kHz, 16,000-sample clips, mics `5,6,8`, azimuth 0°–180°,
10° one-to-one matching, azimuth-only MAE.  Operating points are locked on the
simulated validation set; DICIT curves are reported separately.

**Naming.**  Canonical runs use the model name plus the seed only
(`srp_dnn_s42`, `tf_mamba_s42`, `icocnn_multi_s42`); front-end or task variants
add a descriptive suffix (`tf_mamba_512_s42`).  `_vN` labels are retired.
Attempts aborted because of a confirmed bug or a wrong recipe are marked
`aborted — reason` in the run log and are not cited as results.  Their
artifacts live outside `runs/`, under `aborted/<reason>_<date>/`, so `runs/`
holds only active and completed runs.  Artifact directories created before
2026-11-12 keep their legacy names; new runs use the canonical name.

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
quarter of the comparison budget.  Historical queue files keep their original
names under `queue/`; new runs use the canonical names above.

**LOCATA pin.**  Every run imports LOCATA from the worktree
`/home/t-zhangchaoqin/projects/ssl/wt/locata-dev`, detached at `d5e7c5d`
(clean).  The shared checkout `/projects/ssl/LOCATA` stays on the V5 ablation
branch, which the running ablation jobs use for their own sweep step.

The pin was applied on 2026-11-11 after `dev` moved under the running jobs:
the then-running TF-Mamba and IcoCNN attempts imported `73e95b3` and
`d5e7c5d` respectively, and SRP-DNN ran on `54ffaa6`.  The delta between them
is additive coarse-to-fine support that is inert when `coarse_supervision` /
`coarse_heatmap_creator` are `None`, so the TF-Mamba and IcoCNN rows remain
comparable; the aborted SRP-DNN run was affected by neither commit, and its
rerun will use the pin.  From now on every process (including the eval step)
imports the pinned commit.

## DICIT test — locked validation working point

| Method | Run | F1@10° | 1-src F1 | 2-src F1 | 3-src F1 | FAR | R@FAR 3% | MAE (°) | Params | Forward ms |
|---|---|---|---|---|---|---|---|---|---|---|
| TF-Mamba (n_fft=1536) | `tf_mamba_s42` (running; artifact `runs/tf_mamba_spectrum_s42`) | — | — | — | — | — | — | — | 1.41 M | — |
| TF-Mamba (n_fft=512) | `tf_mamba_512_s42` (queued 2026-11-12, pending GPU1 slot) | — | — | — | — | — | — | — | 1.41 M | — |
| IcoCNN (multi-source) | `icocnn_multi_s42` (paused — rerun when a GPU is idle) | — | — | — | — | — | — | — | 6.17 M | — |
| SRP-DNN | `srp_dnn_s42` (queued 2026-11-12, pending GPU1 slot) | — | — | — | — | — | — | — | 0.86 M | — |

The previous SRP-DNN attempt is **aborted**: the forward applied only four
CRNN blocks because `zip(self.blocks, self.pools)` stops after the four pools,
leaving `blocks[4]` (73,984 parameters) unexecuted.  It scored F1@10 0.7611 /
FAR 0.1475 / MAE 3.37 at threshold 0.40, but those numbers are not cited as the
SRP-DNN row.  The artifacts are archived under
`aborted/srp_dnn_block5_20261112/`; `srp_dnn_s42` reruns the same recipe
with the fixed forward.

**IcoCNN is being rerun.**  The first attempt is **aborted**: its head kept
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
70° → 7.4°, distinct predictions per clip). The full run trains, but it needs
~37 min/epoch while sharing GPU1 (2/2 slots) against ~24 min solo, so it is
paused after epoch 1 and will rerun when a GPU is idle. Epoch 1 reached
train 0.0224 / val 0.0212, below the 1/30 zero solution — no collapse. The
aborted artifacts under `aborted/icocnn_head_20261112/` are kept only as
evidence of the failure; the paused attempt is archived under
`aborted/icocnn_slow_20261112/`.

**Early stopping (all three rows).** `patience=6` counts validation rounds
(`eval_every` epochs apart), matching LOCATA's `Trainer`; the package loop had
been counting every epoch and using the training loss as the monitor on
non-eval epochs, so IcoCNN could stop after ~3 validation rounds. Fixed on
2026-11-12 (covered by `tests/test_training.py`); the paused IcoCNN attempt is
the first run with the corrected semantics (it was stopped for speed before
early stopping could trigger).

DICIT frame counts (for reference): 926 frames total, 176 / 379 / 371 for
1 / 2 / 3 sources; the actual counts are written in each `eval_dicit.json`.

## Validation working points

| Method | Locked threshold | Val F1@10° | Val FAR |
|---|---|---|---|
| TF-Mamba | — | — | — |
| IcoCNN | — | — | — |
| SRP-DNN | queued | — | — |

## Queue status

| Plan | Status | Setting | Purpose |
|---|---|---|---|
| `srp_dnn_s42` | **queued 2026-11-12**, GPU1, waiting for a slot (`queue/srp_dnn_2026-11-12.json`) | `srp_dnn_dicit.toml` (win 1536 / hop 768 / n_fft 1536, 100–8000 Hz, 100 epochs / patience 6 / seed 42) | Replace the aborted 4-block run in the main table. `tests/test_srp_dnn.py` asserts all five CRNN blocks execute and that `blocks[4]` receives gradients. |
| `tf_mamba_512_s42` | **queued 2026-11-12**, GPU1, waiting for a slot (`queue/tf_mamba_512_2026-11-12.json`) | `tf_mamba_dicit_512.toml`: `n_fft=512`, `hop_length=480`, 100–8000 Hz (84 bins), 3 mics, Q=181, 100 epochs / patience 6 / seed 42 | Front-end ablation against the running 1536/480 run. The 512 front end is 10.7 ms / 93.75 Hz, not the paper's 32 ms / 31.25 Hz, so it is an adapted setting. Smoke: 1,408,725 params, output (2, 181), peak decode and checkpoint round-trip OK; random-input `mse_kl_loss` backward finite. |
| `icocnn_multi_s42` | backlog; **run when a GPU is idle** (too slow to share one) | reference head (last conv → `M` + Soft-ArgMax), `lr 1e-4`, cross-pair-only SRP maps, 100 epochs / patience 6 / seed 42 | Main-table IcoCNN row. ~37 min/epoch with GPU1 at 2/2 slots vs ~24 min solo; epoch 1 was healthy (train 0.0224 / val 0.0212). |

GPU0 runs `tf_mamba_s42` at 1/1 slots; GPU1 is at 2/2
(`azreg_feature_query_s42_v2`, `nfft1024_parallel_s42`), so both new jobs stay
pending until a GPU1 slot frees.

The aborted SRP-DNN attempt stays in the run log as evidence of the bug; do not
cite its metrics as the SRP-DNN row until `srp_dnn_s42` is available.

## Run log

The `Attempt` column keeps the job id that appears in `queue/` (legacy ids
included).  The `Artifact / note` column gives the canonical run name where one
exists, plus the output directory.  Only `running` and `done` attempts are
cited as results.

| Attempt | Status | Artifact / note |
|---|---|---|
| `srp_dnn_idl_s42_dev` | aborted — 15-epoch budget abandoned (quarter of the comparison budget) | `aborted/budget15_20261111/srp_dnn_dev`; done 2026-11-10 22:21, Adam, lr 1e-3, F1@10 0.7607 |
| `srp_dnn_idl_s42_v4` | aborted — scheduling, no result | canceled 2026-11-11 16:40; started on GPU0, moved to GPU1 after GPU0 was reduced to one slot |
| `srp_dnn_idl_s42_v4b` | aborted — NFS dataset contention, no result | canceled 2026-11-11 17:20; shared-dataset latency rose ~12x with three concurrent jobs (epoch 4 > 21 min vs 3.7–5.8 min) |
| `srp_dnn_idl_s42_v4c` | aborted — forward skipped `blocks[4]`; 73,984 dead parameters | `aborted/srp_dnn_block5_20261112/`; 100 epochs / patience 6 / seed 42, F1@10 0.7611, not cited |
| `srp_dnn_dicit_reeval_v2` | aborted — reevaluated the aborted 4-block checkpoint | done 2026-11-11 11:02; refreshed `eval_dicit.json` only |
| `tf_mamba_spectrum_s42_dev` | aborted — fp16 autocast overflow (forward NaN from step ~132) | no checkpoint survived (`aborted/tf_mamba_fp16_nan_20261111/`); fp32 at batch 8 is stable (400 steps, 12.8 GiB) |
| `icocnn_multi_s42_dev` | aborted — pre-fix forward rebuilt index tensors on the host (39,032 s per epoch) | `aborted/icocnn_slow_20261110/`; fixed in `4efc135` (25x faster step, bitwise-identical output); superseded |
| `tf_mamba_spectrum_s42_v2` | aborted — wrong label width (`label_sigma = 1.5`, the coarse-to-fine control, not mainline 3.0) | canceled 2026-11-11 11:18 after 20 min; `aborted/tf_mamba_sigma15_20261111/` |
| `icocnn_multi_s42_v2` | aborted — 15-epoch budget replaced by 100 epochs | canceled 2026-11-11 11:47 after 2 epochs |
| `tf_mamba_spectrum_s42_v3` | aborted — 15-epoch budget replaced by 100 epochs | canceled 2026-11-11 11:47 after 1 epoch |
| `tf_mamba_spectrum_s42_v4` | running — canonical `tf_mamba_s42` | `runs/tf_mamba_spectrum_s42`; submitted 2026-11-11 11:47, fp32, batch 8, `label_sigma = 3.0`, 100 epochs / patience 6, ~44 min/epoch |
| `icocnn_multi_s42_v4` | aborted — non-reference head (`LayerNorm+Linear` over 128 channels instead of last conv → `M` + Soft-ArgMax); collapsed to a constant 3-beam output | `aborted/icocnn_head_20261112/`; identical DICIT predictions on all 926 frames, loss flat at 0.0312 from epoch 2 |
| `icocnn_multi_s42_v5` | aborted — early stopping counted epochs, not validation rounds | canceled 2026-11-12 15:49 before the first epoch so the fix could take effect |
| `icocnn_multi_s42_v5b` | aborted — no `train.log` in the run directory | canceled 2026-11-12 16:01 after 9 min; restarted after adding the log tee |
| `icocnn_multi_s42_v5c` | paused — training too slow on a shared GPU; no result | `aborted/icocnn_slow_20261112/`; canceled 2026-11-12 17:04 after epoch 1 (train 0.0224 / val 0.0212, no collapse); ~37 min/epoch with GPU1 at 2/2 slots vs ~24 min solo; rerun when a GPU is idle |
| `srp_dnn_s42` | pending — submitted 2026-11-12, waiting for a GPU1 slot | output `runs/srp_dnn_s42`; fixed five-block forward |
| `tf_mamba_512_s42` | pending — submitted 2026-11-12, waiting for a GPU1 slot | output `runs/tf_mamba_512_s42`; `tf_mamba_dicit_512.toml` (`n_fft=512`) |
