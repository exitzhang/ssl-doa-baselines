# External baselines — results

Protocol: 48 kHz, 16,000-sample clips, mics `5,6,8`, azimuth 0°–180°,
10° one-to-one matching, azimuth-only MAE.  Operating points are locked on the
simulated validation set; DICIT curves are reported separately.

**Status 2026-11-16.** Three runs are complete and evaluated on DICIT:
`tf_mamba_s42` (n_fft=1536), `tf_mamba_512_s42`, `srp_dnn_s42`.  IcoCNN has
**no completed run** — every attempt so far is aborted or paused (see the run
log), so its main-table row stays empty.  Two new rows are queued as v3:
`ipdnet_s42` and `ipdnet2_small_s42` (fixed 3-microphone ports of
`FN-SSL@76fcb281`, clip-level DP-IPD PIT; see `THIRD_PARTY.md`).  Their cells
stay `pending` until the validation lock and DICIT evaluation finish.

**Review fix 2026-11-16.** The IPDnet rows now run under the corrected
contract: the DICIT point is the threshold locked on the simulated validation
set (checkpoint-identity checked), the fixed array / channel order is recorded
in the checkpoint and verified before evaluation, predictions require an
explicit threshold, `num_params` is counted from the rebuilt model
(725,248 / 1,246,764) and each eval writes a unified
`inference_ms_per_clip`.  Bounded train/val/DICIT smoke runs passed for both
models before resubmission, and the frozen train/val/DICIT manifests passed
the pre-run array check.

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

**Shared training budget.**  All rows use `--epochs 100` with `patience 6`
validation rounds of early stopping (seed 42); the effective epoch count and
sample budget are recorded in each `config.json`.  Completion:

| Run | Stopped at | Note |
|---|---|---|
| `tf_mamba_spectrum_s42` (1536) | epoch 100 | ran the full budget, best checkpoint at the last validation round |
| `tf_mamba_512_s42` | epoch 85 | early stopped, 6/6 patience exhausted |
| `srp_dnn_s42` | epoch 95 | early stopped, best validation loss 0.218812 |

The 15-epoch first pass was abandoned: the V5 mainline (B0/Parallel) picks its
best epoch around 30 and early-stops at 60, so 15 epochs left the external
baselines at roughly a quarter of the comparison budget.  Historical queue
files keep their original names under `queue/`; new runs use the canonical
names above.

**LOCATA pin.**  Every completed run imports LOCATA from the worktree
`/home/t-zhangchaoqin/projects/ssl/wt/locata-dev`, detached at `d5e7c5d`
(clean), recorded in each `eval_dicit.json` / `config.json` under `locata`.
The shared checkout `/projects/ssl/LOCATA` stays on the V5 ablation branch,
which the running ablation jobs use for their own sweep step.

The pin was applied on 2026-11-11 after `dev` moved under the running jobs:
the then-running TF-Mamba and IcoCNN attempts imported `73e95b3` and
`d5e7c5d` respectively, and SRP-DNN ran on `54ffaa6`.  The delta between them
is additive coarse-to-fine support that is inert when `coarse_supervision` /
`coarse_heatmap_creator` are `None`, so the TF-Mamba and IcoCNN rows remain
comparable; the aborted SRP-DNN run was affected by neither commit.  From now
on every process (including the eval step) imports the pinned commit.  The
completed `srp_dnn_s42` / `tf_mamba_512_s42` / `tf_mamba_spectrum_s42` runs all
record `d5e7c5d` with `dirty: false`.

## DICIT test — locked validation working point

All numbers are DICIT at the threshold locked on simulated validation
(`metrics` / `curve` in each `eval_dicit.json`), except `R@FAR 3%`, which is
interpolated on the DICIT P-R curve.  `FAR` is the pitch-accuracy-10 operating
FAR.  The `working_point` block in each JSON is the *validation* lock; it is
not a DICIT number.

| Method | Run | F1@10° | 1-src F1 | 2-src F1 | 3-src F1 | FAR | R@FAR 3% | MAE (°) | Params | Forward ms³ |
|---|---|---|---|---|---|---|---|---|---|---|
| TF-Mamba (n_fft=1536) | `tf_mamba_s42` → `runs/tf_mamba_spectrum_s42` | **0.9338** | 0.7848 | 0.9515 | 0.9520 | 0.0782 | —¹ | **1.47** | 1.41 M | 3.40 |
| TF-Mamba (n_fft=512) | `tf_mamba_512_s42` | 0.9294 | 0.8333 | 0.9522 | 0.9320 | **0.0630** | **0.7838** | 1.53 | 1.41 M | 3.41 |
| SRP-DNN | `srp_dnn_s42` | 0.7665 | **0.8579** | 0.7850 | 0.7354 | 0.1343 | —² | 3.26 | 0.86 M | **1.78** |
| IPDnet (fixed 3-mic, DP-IPD) | `ipdnet_s42` → `runs/ipdnet_s42` | pending | pending | pending | pending | pending | pending | pending | 725,248 | pending |
| IPDnet2-Small (fixed 3-mic, DP-IPD) | `ipdnet2_small_s42` → `runs/ipdnet2_small_s42` | pending | pending | pending | pending | pending | pending | pending | 1,246,764 | pending |
| IcoCNN (multi-source) | `icocnn_multi_s42` | — | — | — | — | — | — | — | 6.17 M | — |

¹ Lowest DICIT FAR on the TF-Mamba-1536 curve is 0.0308, just above the 3%
target, so `R@FAR 3%` is not covered. ² SRP-DNN's lowest DICIT FAR is 0.0484.
³ `Forward ms` is the pre-2026-11-16 measurement: method forward only,
batch-normalised, without that method's own decoding/filtering (IPDnet's
resampler/STFT front end and template matching were timed separately).  The
three completed rows keep their historical number; the main table moves to the
unified `inference_ms_per_clip` only once every row has been re-measured under
the same scope.  `num_params` is now counted from the rebuilt checkpoint in
each new `eval_*.json`; the IPDnet value 725,248 replaces the earlier
hand-written 0.35 M.

Thresholds locked on validation: TF-Mamba-1536 `0.3`, TF-Mamba-512 `0.3`,
SRP-DNN `0.45`, IPDnet `0.5`.  Accuracy-threshold breakdown at those locked
points:

| Method | F1@5° | F1@10° | F1@15° | MAE@10 (°) |
|---|---|---|---|---|
| TF-Mamba 1536 | 0.8845 | 0.9338 | 0.9396 | 1.47 |
| TF-Mamba 512 | 0.8733 | 0.9294 | 0.9348 | 1.53 |
| SRP-DNN | 0.5822 | 0.7665 | 0.8287 | 3.26 |
| IPDnet (3-mic) | 0.8806 | 0.9274 | 0.9337 | 1.49 |

**Reading.**  The two TF-Mamba front ends are effectively tied on DICIT
(ΔF1@10 = −0.004, ΔMAE = +0.06°), so the 1536/480 vs 512/480 STFT choice is not
what limits this baseline.  SRP-DNN trails by ~0.17 F1@10 and ~1.8° MAE; its
only win is the 1-source F1 (0.858 vs 0.785/0.833) and the ~1.8 ms forward,
since the other two spend 3.4 ms.  IPDnet (3-mic) lands within 0.002 F1@10 and
0.04° MAE of TF-Mamba-512 at 1.25 ms/clip, decoding three clip-level DP-IPD
tracks instead of a per-frame heat map.  All four produce varied outputs
(`unique_frames` 768 / 820 / 840 / 545 of 926), so none is in the collapsed
regime.

## Validation working points

| Method | Locked threshold | Val F1@10° | Val FAR |
|---|---|---|---|
| TF-Mamba (1536) | 0.30 | 0.9152 | 0.0357 |
| TF-Mamba (512) | 0.30 | 0.8874 | 0.0439 |
| SRP-DNN | 0.45 | 0.7006 | 0.1657 |
| IPDnet | 0.50 | 0.9012 | 0.0468 |
| IPDnet2-Small | pending | pending | pending |
| IcoCNN | — | — | — |

The validation lock is the only place the operating point is chosen; the DICIT
table above applies that threshold and re-scores on DICIT.  Do not copy the
`working_point` values from `eval_dicit.json` into the DICIT table — they are
the validation numbers by construction (`_eval_payload` in
`integrations/locata/run.py`).

## Queue status

| Plan | Status | Setting | Purpose |
|---|---|---|---|
| `srp_dnn_s42` | **done 2026-11-13 05:12** (`runs/srp_dnn_s42`) | `srp_dnn_dicit.toml` (win 1536 / hop 768 / n_fft 1536, 100–8000 Hz, 100 epochs / patience 6 / seed 42, five-block forward) | Main-table SRP-DNN row; replaced the aborted 4-block run. |
| `tf_mamba_512_s42` | **done 2026-11-15 00:40** (`runs/tf_mamba_512_s42`) | `tf_mamba_dicit_512.toml`: `n_fft=512`, `hop_length=480`, 100–8000 Hz (84 bins), 3 mics, Q=181, 100 epochs / patience 6 / seed 42 | Front-end ablation against the 1536/480 run. The 512 front end is 10.7 ms / 93.75 Hz, not the paper's 32 ms / 31.25 Hz, so it is an adapted setting. |
| `tf_mamba_s42` | **done 2026-11-14 15:45** (`runs/tf_mamba_spectrum_s42`) | `n_fft=1536`, `hop_length=480`, Q=181, 100 epochs / patience 6 / seed 42, fp32 batch 8, `label_sigma=3.0` | Main-table TF-Mamba row. |
| `ipdnet_s42` | **done 2026-11-17 10:59** (`runs/ipdnet_s42`) | `ipdnet_dicit.toml`: 48->16 kHz, 512 FFT / hop 256, 3 tracks, Adam lr 5e-4, exponential 0.975, 100 epochs / patience 6 / seed 42 | Main-table IPDnet row; clip-level DP-IPD PIT, projection-ratio decode. Ran all 100 epochs; validation locked 0.5, DICIT F1@10 0.9274. |
| `ipdnet2_small_s42` | **pending v4** — manifest kept at `queue/ipdnet2_small_512_s42_v4.json`, not submitted (paused 2026-11-17) | `ipdnet2_small_dicit.toml`: 48->16 kHz, 512 FFT / hop 320, 8 layers / hidden 96 / `mamba(16,4)`, AdamW, 100 epochs / patience 6 / seed 42 | Main-table IPDnet2-Small row; negative-MSE decode. Needs the `locata` env's `mamba-ssm==2.1.0`. v4 carries the vectorised `FreqInverse` (~1085 -> ~960 s/epoch measured). |
| `icocnn_multi_s42` | backlog; **run when a GPU is idle** (too slow to share one) | reference head (last conv → `M` + Soft-ArgMax), `lr 1e-4`, cross-pair-only SRP maps, 100 epochs / patience 6 / seed 42 | Main-table IcoCNN row. ~37 min/epoch with GPU1 at 2/2 slots vs ~24 min solo; the paused epoch-1 attempt was healthy (train 0.0224 / val 0.0212). |

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
| `tf_mamba_spectrum_s42_v4` | **done 2026-11-14 15:45 — canonical `tf_mamba_s42`** | `runs/tf_mamba_spectrum_s42`; submitted 2026-11-11 11:47, fp32, batch 8, `label_sigma = 3.0`, 100 epochs / patience 6, ran to epoch 100; DICIT F1@10 0.9338, MAE 1.47 |
| `icocnn_multi_s42_v4` | aborted — non-reference head (`LayerNorm+Linear` over 128 channels instead of last conv → `M` + Soft-ArgMax); collapsed to a constant 3-beam output | `aborted/icocnn_head_20261112/`; identical DICIT predictions on all 926 frames, loss flat at 0.0312 from epoch 2 |
| `icocnn_multi_s42_v5` | aborted — early stopping counted epochs, not validation rounds | canceled 2026-11-12 15:49 before the first epoch so the fix could take effect |
| `icocnn_multi_s42_v5b` | aborted — no `train.log` in the run directory | canceled 2026-11-12 16:01 after 9 min; restarted after adding the log tee |
| `icocnn_multi_s42_v5c` | paused — training too slow on a shared GPU; no result | `aborted/icocnn_slow_20261112/`; canceled 2026-11-12 17:04 after epoch 1 (train 0.0224 / val 0.0212, no collapse); ~37 min/epoch with GPU1 at 2/2 slots vs ~24 min solo; rerun when a GPU is idle |
| `srp_dnn_s42` | **done 2026-11-13 05:12** | `runs/srp_dnn_s42`; fixed five-block forward, early stopped epoch 95; DICIT F1@10 0.7665, MAE 3.26 |
| `tf_mamba_512_s42` | **done 2026-11-15 00:40** | `runs/tf_mamba_512_s42`; `tf_mamba_dicit_512.toml` (`n_fft=512`), early stopped epoch 85; DICIT F1@10 0.9294, MAE 1.53 |
| `ipdnet_s42` | canceled before start — worktree refreshed to the final commit | never ran; resubmitted as `ipdnet_s42_v2` |
| `ipdnet2_small_s42` | canceled before start — worktree refreshed to the final commit | never ran; resubmitted as `ipdnet2_small_s42_v2` |
| `ipdnet_s42_v2` | canceled 2026-11-16 before start — review fix (validation lock / array contract); resubmitted as v3 | never ran |
| `ipdnet2_small_s42_v2` | canceled 2026-11-16 before start — review fix (validation lock / array contract); resubmitted as v3 | never ran |
| `ipdnet_s42_v3` | **done 2026-11-17 10:59 — canonical `ipdnet_s42`** | `runs/ipdnet_s42`; all 100 epochs, validation locked 0.5 (F1@10 0.9012, FAR 0.0468); DICIT F1@10 0.9274 / MAE 1.49 / 1.25 ms per clip |
| `ipdnet2_small_s42_v3` | aborted — host-dispatch bound, ~1085 s/epoch; canceled at epoch 26 to ship the vectorised `FreqInverse` | `aborted/ipdnet2_slow_20261117/`; 2450 kernel launches per step vs IPDnet's 373 (GPU kernel time was lower, 16.5 vs 19.7 ms), so the step was CPU/launch bound, not GPU bound. Rejected speedups: `causal-conv1d`'s fused `mamba_inner` is slower at the post-pooling `T=4` (2.6 -> 4.2 ms per call, full step a wash), and `torch.compile(mode="reduce-overhead")` measured 70 vs 61 ms per step. No result cited |
| `ipdnet2_small_s42_v4` | paused (running-cancel) 2026-11-17 14:47 at epoch 12; marked pending, not resubmitted | `aborted/ipdnet2_paused_20261117/`; same recipe as v3 plus the vectorised `FreqInverse`; epoch times 1020 / 985 / 1014 / 1004 / 1024 / 1017 / 1054 / 1019 / 1046 / 852 / 797 / 770 s, so the fix is worth ~6% on a loaded box, not the 2x the launch profile suggested. Resubmit `queue/ipdnet2_small_512_s42_v4.json` when a slot is allocated |

**Early stopping (all rows).** `patience=6` counts validation rounds
(`eval_every` epochs apart), matching LOCATA's `Trainer`; the package loop had
been counting every epoch and using the training loss as the monitor on
non-eval epochs, so IcoCNN could stop after ~3 validation rounds. Fixed on
2026-11-12 (covered by `tests/test_training.py`); the paused IcoCNN attempt is
the first run with the corrected semantics.  The three completed runs use the
corrected semantics.

**IcoCNN invalidation (carried forward).**  The first attempt is **aborted**:
its head kept `width` channels through the last temporal convolution and added
a `LayerNorm(5120) + Linear(30)` readout, which is not the reference model.  It
collapsed to a constant 3-beam output (926/926 DICIT frames byte-identical;
training loss flat at 0.0312 from epoch 2).  The model now follows
arXiv:2210.14536 (last temporal conv outputs `M`, then the author's
Soft-ArgMax).  Two further causes were found with fixed-batch screens on
2026-11-12:

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
paused after epoch 1 and will rerun when a GPU is idle.

DICIT frame counts (for reference): 926 frames total, 176 / 379 / 371 for
1 / 2 / 3 sources; the actual counts are written in each `eval_dicit.json`.
