# 外部基线 — 结果

协议：48 kHz、16,000 采样片段、麦克风 `5,6,8`、方位 0°–180°、10° 一对一匹配、
仅方位 MAE。工作点在仿真验证集上锁定；DICIT 曲线单独报告。

**状态 2026-11-17。** 四个运行已完成并在 DICIT 上评估：`tf_mamba_s42`
（n_fft=1536）、`tf_mamba_512_s42`、`srp_dnn_s42` 和 `ipdnet_s42`。
`ipdnet2_small_s42`（`FN-SSL@76fcb281` 的固定三麦移植，片段级 DP-IPD PIT；
见 `THIRD_PARTY.md`）**没有结果**：v3 因每步 host-launch 受限（~1085 s/epoch）
在 epoch 26 取消，v4 带着向量化 `FreqInverse` 跑了 12 个 epoch 后按要求暂停；
`pending v4` 清单保留待提交。IcoCNN **尚无完成运行**——迄今所有尝试都中止或
暂停（见运行日志），因此主表对应行留空。

**Review fix 2026-11-16。** IPDnet 各行现在运行在修正后的契约下：DICIT 工作点
是仿真验证集上锁定的阈值（带 checkpoint 身份校验），固定阵列/通道顺序记录在
checkpoint 中并在评估前校验，预测必须显式给出阈值，`num_params` 由重建模型统计
（725,248 / 1,246,764），每次评估写出统一的 `inference_ms_per_clip`。两个模型
在重新提交前都通过了带边界的 train/val/DICIT 冒烟，冻结的 train/val/DICIT 清单
也通过了运行前阵列检查。

**命名。** 规范运行只用模型名加 seed（`srp_dnn_s42`、`tf_mamba_s42`、
`icocnn_multi_s42`）；前端或任务变体加描述性后缀（`tf_mamba_512_s42`）。`_vN`
标签已停用。因确认的 bug 或错误配方而中止的尝试在运行日志中标记为
`aborted — 原因`，不作为结果引用。它们的产物不放在 `runs/` 下，而在
`aborted/<原因>_<日期>/`，因此 `runs/` 只保留进行中和已完成的运行。2026-11-12
之前创建的产物目录沿用旧名；新运行使用规范名。

**标签宽度。** TF-Mamba 的频谱标签用 `label_sigma = 3.0` 个 bin（Q=181 时每 bin
1°），与主线 V5 基线 B0 及用 `--label_sigma 3` 训练的 Parallel 各臂一致。σ=1.5
的运行（A0/A1/A2）是 coarse-to-fine 对照，不是基线。IcoCNN（ACCDOA 单位向量）和
SRP-DNN（DP-IPD 模板回归 + IDL）没有高斯标签。

**共享训练预算。** 所有行都使用 `--epochs 100`、`patience 6` 轮验证早停
（seed 42）；实际 epoch 数和样本预算记录在每个 `config.json`。完成情况：

| 运行 | 停止于 | 说明 |
|---|---|---|
| `tf_mamba_spectrum_s42`（1536） | epoch 100 | 跑满预算，最佳 checkpoint 在最后一轮验证 |
| `tf_mamba_512_s42` | epoch 85 | 早停，6/6 patience 耗尽 |
| `srp_dnn_s42` | epoch 95 | 早停，最佳验证 loss 0.218812 |
| `ipdnet_s42` | epoch 100 | 跑满预算，最佳验证 loss 0.046565 |

15 epoch 的初版被弃用：V5 主线（B0/Parallel）在 30 epoch 左右取最佳、在 60
早停，15 epoch 只给外部基线留下约四分之一的对照预算。历史 queue 文件在
`queue/` 下保留原名；新运行用上述规范名。

**LOCATA 固定版本。** 每个完成的运行都从 worktree
`/home/t-zhangchaoqin/projects/ssl/wt/locata-dev` import LOCATA，detach 在
`d5e7c5d`（干净），记录在每个 `eval_dicit.json` / `config.json` 的 `locata`
下。共享 checkout `/projects/ssl/LOCATA` 保持在 V5 消融分支上，运行中的消融作业
用它做自己的 sweep 步骤。

该固定在 2026-11-11、`dev` 在运行作业下方移动之后施加：当时运行的 TF-Mamba 和
IcoCNN 尝试分别 import 了 `73e95b3` 和 `d5e7c5d`，SRP-DNN 跑在 `54ffaa6` 上。
它们之间的差异是增量式的 coarse-to-fine 支持，当 `coarse_supervision` /
`coarse_heatmap_creator` 为 `None` 时不起作用，因此 TF-Mamba 和 IcoCNN 行仍可
比；中止的 SRP-DNN 运行不受这两个提交影响。从现在起每个进程（包括评估步骤）都
import 固定的 commit。完成的 `srp_dnn_s42` / `tf_mamba_512_s42` /
`tf_mamba_spectrum_s42` / `ipdnet_s42` 运行都记录 `d5e7c5d` 且
`dirty: false`。

## DICIT 测试 — 锁定的验证工作点

所有数字都是 DICIT 在仿真验证集锁定阈值下的结果（各自 `eval_dicit.json` 的
`metrics` / `curve`），`R@FAR 3%` 例外，它在 DICIT 的 P-R 曲线上插值。`FAR` 是
pitch-accuracy-10 工作点的 FAR。每个 JSON 里的 `working_point` 块是**验证集**
锁定值，不是 DICIT 数字。

| 方法 | 运行 | F1@10° | 单源 F1 | 双源 F1 | 三源 F1 | FAR | R@FAR 3% | MAE (°) | 参数 | 前向 ms³ |
|---|---|---|---|---|---|---|---|---|---|---|
| TF-Mamba（n_fft=1536） | `tf_mamba_s42` → `runs/tf_mamba_spectrum_s42` | **0.9338** | 0.7848 | 0.9515 | 0.9520 | 0.0782 | —¹ | **1.47** | 1.41 M | 3.40 |
| TF-Mamba（n_fft=512） | `tf_mamba_512_s42` | 0.9294 | 0.8333 | 0.9522 | 0.9320 | **0.0630** | 0.7838 | 1.53 | 1.41 M | 3.41 |
| SRP-DNN | `srp_dnn_s42` | 0.7665 | **0.8579** | 0.7850 | 0.7354 | 0.1343 | —² | 3.26 | 0.86 M | **1.78** |
| IPDnet（固定三麦，DP-IPD） | `ipdnet_s42` → `runs/ipdnet_s42` | 0.9274 | 0.8093 | 0.9432 | 0.9398 | 0.0752 | **0.8123** | 1.49 | 725,248 | 0.89 |
| IPDnet2-Small（固定三麦，DP-IPD） | `ipdnet2_small_s42` → `runs/ipdnet2_small_s42` | pending | pending | pending | pending | pending | pending | pending | 1,246,764 | pending |
| IcoCNN（多源） | `icocnn_multi_s42` | — | — | — | — | — | — | — | 6.17 M | — |

¹ TF-Mamba-1536 曲线上 DICIT 的最低 FAR 是 0.0308，刚好高于 3% 目标，因此
`R@FAR 3%` 未被覆盖。² SRP-DNN 在 DICIT 上的最低 FAR 是 0.0484。
³ `前向 ms` 只计方法前向，按 batch 归一化，不含该方法自己的解码/过滤。2026-11-16
之前的三行保留历史数字；IPDnet 的 0.89 ms 是它的 `forward_ms_per_frame`
（重采样/STFT 前端和模板匹配单独计时），同一 checkpoint 的统一
`inference_ms_per_clip` 是 1.25 ms。只有在每一行都用同一口径重测之后，主表才会
切换到统一数字。`num_params` 现在由每个新 `eval_*.json` 中重建的 checkpoint
统计；IPDnet 的 725,248 取代了此前手写的 0.35 M。

阈值锁定于验证集：TF-Mamba-1536 `0.3`、TF-Mamba-512 `0.3`、SRP-DNN `0.45`、
IPDnet `0.5`。在这些锁定点的精度阈值分解：

| 方法 | F1@5° | F1@10° | F1@15° | MAE@10 (°) |
|---|---|---|---|---|
| TF-Mamba 1536 | 0.8845 | 0.9338 | 0.9396 | 1.47 |
| TF-Mamba 512 | 0.8733 | 0.9294 | 0.9348 | 1.53 |
| SRP-DNN | 0.5822 | 0.7665 | 0.8287 | 3.26 |
| IPDnet（三麦） | 0.8806 | 0.9274 | 0.9337 | 1.49 |

**解读。** 两个 TF-Mamba 前端在 DICIT 上基本打平（ΔF1@10 = −0.004，
ΔMAE = +0.06°），所以 1536/480 与 512/480 的 STFT 选择不是该基线的瓶颈。
SRP-DNN 落后约 0.17 F1@10 和约 1.8° MAE；它唯一的优势是单源 F1
（0.858 对 0.785/0.833）和约 1.8 ms 的前向（另外两个要 3.4 ms）。IPDnet（三麦）
与 TF-Mamba-512 的 F1@10 相差 0.002、MAE 相差 0.04°，R@FAR 3% 更高
（0.812 对 0.784），统一端到端口径下 1.25 ms/片段，且解码的是三条片段级 DP-IPD
轨而不是逐帧热图。四者输出都不塌缩（`unique_frames` 分别为 926 帧中的
768 / 820 / 840 / 545）。

## 验证工作点

| 方法 | 锁定阈值 | 验证 F1@10° | 验证 FAR |
|---|---|---|---|
| TF-Mamba（1536） | 0.30 | 0.9152 | 0.0357 |
| TF-Mamba（512） | 0.30 | 0.8874 | 0.0439 |
| SRP-DNN | 0.45 | 0.7006 | 0.1657 |
| IPDnet | 0.50 | 0.9012 | 0.0468 |
| IPDnet2-Small | pending | pending | pending |
| IcoCNN | — | — | — |

验证锁定是唯一选择工作点的地方；上面的 DICIT 表应用该阈值并在 DICIT 上重新评分。
不要把 `eval_dicit.json` 里的 `working_point` 值抄进 DICIT 表——按构造它们就是
验证集数字（`integrations/locata/run.py` 的 `_eval_payload`）。

## 队列状态

| 方案 | 状态 | 设置 | 用途 |
|---|---|---|---|
| `srp_dnn_s42` | **done 2026-11-13 05:12**（`runs/srp_dnn_s42`） | `srp_dnn_dicit.toml`（win 1536 / hop 768 / n_fft 1536，100–8000 Hz，100 epochs / patience 6 / seed 42，五块前向） | 主表 SRP-DNN 行；取代中止的四块运行。 |
| `tf_mamba_512_s42` | **done 2026-11-15 00:40**（`runs/tf_mamba_512_s42`） | `tf_mamba_dicit_512.toml`：`n_fft=512`、`hop_length=480`、100–8000 Hz（84 bin）、3 麦、Q=181、100 epochs / patience 6 / seed 42 | 对 1536/480 运行的前端消融。512 前端是 10.7 ms / 93.75 Hz，不是论文的 32 ms / 31.25 Hz，因此是改造后的设置。 |
| `tf_mamba_s42` | **done 2026-11-14 15:45**（`runs/tf_mamba_spectrum_s42`） | `n_fft=1536`、`hop_length=480`、Q=181、100 epochs / patience 6 / seed 42、fp32 batch 8、`label_sigma=3.0` | 主表 TF-Mamba 行。 |
| `ipdnet_s42` | **done 2026-11-17 10:59**（`runs/ipdnet_s42`） | `ipdnet_dicit.toml`：48->16 kHz、512 FFT / hop 256、3 轨、Adam lr 5e-4、指数衰减 0.975、100 epochs / patience 6 / seed 42 | 主表 IPDnet 行；片段级 DP-IPD PIT、投影比解码。跑满 100 epoch；验证锁定 0.5，DICIT F1@10 0.9274。 |
| `ipdnet2_small_s42` | **pending v4** — 清单保留在 `queue/ipdnet2_small_512_s42_v4.json`，未提交（2026-11-17 暂停） | `ipdnet2_small_dicit.toml`：48->16 kHz、512 FFT / hop 320、8 层 / hidden 96 / `mamba(16,4)`、AdamW、100 epochs / patience 6 / seed 42 | 主表 IPDnet2-Small 行；负 MSE 解码。需要 `locata` 环境的 `mamba-ssm==2.1.0`。v4 带向量化 `FreqInverse`（实测 ~1085 -> ~960 s/epoch）。实验 worktree 已于 2026-11-17 删除（其提交在 `feat/external-baselines` 上），因此提交前必须把清单的 `cwd` 指向新建的 worktree。 |
| `icocnn_multi_s42` | backlog；**等 GPU 空闲再跑**（太慢，不适合共享） | 参考头（last conv → `M` + Soft-ArgMax）、`lr 1e-4`、仅交叉麦克风对的 SRP 图、100 epochs / patience 6 / seed 42 | 主表 IcoCNN 行。GPU1 2/2 槽位时约 37 min/epoch，独占约 24 min/epoch；暂停的 epoch-1 尝试是健康的（train 0.0224 / val 0.0212）。 |

## 运行日志

`Attempt` 列保留出现在 `queue/` 中的作业 id（含历史 id）。`产物 / 备注` 列给出
规范运行名（若有）和输出目录。只有 `running` 和 `done` 的尝试会作为结果引用。

| 尝试 | 状态 | 产物 / 备注 |
|---|---|---|
| `srp_dnn_idl_s42_dev` | 中止 — 15 epoch 预算弃用（对照预算的四分之一） | `aborted/budget15_20261111/srp_dnn_dev`；done 2026-11-10 22:21，Adam、lr 1e-3、F1@10 0.7607 |
| `srp_dnn_idl_s42_v4` | 中止 — 排期问题，无结果 | 2026-11-11 16:40 取消；起初在 GPU0，GPU0 缩为一个槽位后移到 GPU1 |
| `srp_dnn_idl_s42_v4b` | 中止 — NFS 数据集争用，无结果 | 2026-11-11 17:20 取消；三个作业并发时共享数据集延迟上涨约 12 倍（epoch 4 超过 21 min，而正常为 3.7–5.8 min） |
| `srp_dnn_idl_s42_v4c` | 中止 — 前向跳过了 `blocks[4]`；73,984 个死参数 | `aborted/srp_dnn_block5_20261112/`；100 epochs / patience 6 / seed 42，F1@10 0.7611，不引用 |
| `srp_dnn_dicit_reeval_v2` | 中止 — 重新评估中止的四块 checkpoint | done 2026-11-11 11:02；只刷新了 `eval_dicit.json` |
| `tf_mamba_spectrum_s42_dev` | 中止 — fp16 autocast 溢出（约第 132 步起前向 NaN） | 没有 checkpoint 幸存（`aborted/tf_mamba_fp16_nan_20261111/`）；fp32 batch 8 稳定（400 步，12.8 GiB） |
| `icocnn_multi_s42_dev` | 中止 — 修复前的前向在 host 上重建索引张量（每 epoch 39,032 s） | `aborted/icocnn_slow_20261110/`；已在 `4efc135` 修复（单步快 25 倍，输出逐 bit 一致）；已被取代 |
| `tf_mamba_spectrum_s42_v2` | 中止 — 标签宽度错误（`label_sigma = 1.5`，coarse-to-fine 对照，不是主线 3.0） | 2026-11-11 11:18 取消，跑了 20 min；`aborted/tf_mamba_sigma15_20261111/` |
| `icocnn_multi_s42_v2` | 中止 — 15 epoch 预算被 100 epoch 取代 | 2026-11-11 11:47 取消，跑了 2 个 epoch |
| `tf_mamba_spectrum_s42_v3` | 中止 — 15 epoch 预算被 100 epoch 取代 | 2026-11-11 11:47 取消，跑了 1 个 epoch |
| `tf_mamba_spectrum_s42_v4` | **done 2026-11-14 15:45 — 规范运行 `tf_mamba_s42`** | `runs/tf_mamba_spectrum_s42`；2026-11-11 11:47 提交，fp32、batch 8、`label_sigma = 3.0`、100 epochs / patience 6，跑到 epoch 100；DICIT F1@10 0.9338、MAE 1.47 |
| `icocnn_multi_s42_v4` | 中止 — 非参考头（用 `LayerNorm+Linear` 处理 128 通道，而非 last conv → `M` + Soft-ArgMax）；塌缩为恒定的三波束输出 | `aborted/icocnn_head_20261112/`；926 帧 DICIT 预测完全相同，loss 从 epoch 2 起平在 0.0312 |
| `icocnn_multi_s42_v5` | 中止 — 早停按 epoch 计数而不是验证轮次 | 2026-11-12 15:49 在第一个 epoch 前取消，以便修复生效 |
| `icocnn_multi_s42_v5b` | 中止 — 运行目录里没有 `train.log` | 2026-11-12 16:01 取消，跑了 9 min；加上日志 tee 后重启 |
| `icocnn_multi_s42_v5c` | 暂停 — 共享 GPU 上训练太慢；无结果 | `aborted/icocnn_slow_20261112/`；2026-11-12 17:04 在 epoch 1 后取消（train 0.0224 / val 0.0212，未塌缩）；GPU1 2/2 槽位时约 37 min/epoch，独占约 24 min/epoch；等 GPU 空闲再跑 |
| `srp_dnn_s42` | **done 2026-11-13 05:12** | `runs/srp_dnn_s42`；修正后的五块前向，epoch 95 早停；DICIT F1@10 0.7665、MAE 3.26 |
| `tf_mamba_512_s42` | **done 2026-11-15 00:40** | `runs/tf_mamba_512_s42`；`tf_mamba_dicit_512.toml`（`n_fft=512`），epoch 85 早停；DICIT F1@10 0.9294、MAE 1.53 |
| `ipdnet_s42` | 启动前取消 — worktree 刷新到最终提交 | 从未运行；重新提交为 `ipdnet_s42_v2` |
| `ipdnet2_small_s42` | 启动前取消 — worktree 刷新到最终提交 | 从未运行；重新提交为 `ipdnet2_small_s42_v2` |
| `ipdnet_s42_v2` | 2026-11-16 启动前取消 — review fix（验证锁定 / 阵列契约）；重新提交为 v3 | 从未运行 |
| `ipdnet2_small_s42_v2` | 2026-11-16 启动前取消 — review fix（验证锁定 / 阵列契约）；重新提交为 v3 | 从未运行 |
| `ipdnet_s42_v3` | **done 2026-11-17 10:59 — 规范运行 `ipdnet_s42`** | `runs/ipdnet_s42`；跑满 100 epoch，验证锁定 0.5（F1@10 0.9012、FAR 0.0468）；DICIT F1@10 0.9274 / MAE 1.49 / 1.25 ms 每片段 |
| `ipdnet2_small_s42_v3` | 中止 — host-dispatch 受限，~1085 s/epoch；在 epoch 26 取消以交付向量化 `FreqInverse` | `aborted/ipdnet2_slow_20261117/`；每步 2450 次 kernel launch，而 IPDnet 为 373 次（GPU kernel 时间反而更低，16.5 vs 19.7 ms），说明瓶颈是 CPU/launch 而非 GPU。已否决的加速方案：`causal-conv1d` 的融合 `mamba_inner` 在池化后的 `T=4` 上更慢（每次调用 2.6 -> 4.2 ms，整步持平），`torch.compile(mode="reduce-overhead")` 实测每步 70 ms vs 61 ms。不引用结果 |
| `ipdnet2_small_s42_v4` | 2026-11-17 14:47 在 epoch 12 暂停（运行中取消）；标记为待跑，未重新提交 | `aborted/ipdnet2_paused_20261117/`；配方同 v3 加向量化 `FreqInverse`；epoch 时间为 1020 / 985 / 1014 / 1004 / 1024 / 1017 / 1054 / 1019 / 1046 / 852 / 797 / 770 s，说明该修复在负载机器上约值 6%，而非 launch 画像暗示的 2 倍。分配好槽位后重新提交 `queue/ipdnet2_small_512_s42_v4.json` |

**早停（所有行）。** `patience=6` 统计验证轮次（相隔 `eval_every` 个 epoch），与
LOCATA 的 `Trainer` 一致；包内的训练循环此前按每个 epoch 计数、并在非验证 epoch
用训练 loss 作为监控，导致 IcoCNN 可能在大约 3 轮验证后就停止。该问题在
2026-11-12 修复（由 `tests/test_training.py` 覆盖）；暂停的 IcoCNN 尝试是第一个
使用修正语义的运行。四个完成的运行都使用修正后的语义。

**IcoCNN 失效分析（沿用）。** 第一次尝试**中止**：它的头在最后一个时间卷积后仍
保留 `width` 通道，并加了 `LayerNorm(5120) + Linear(30)` 读出，不是参考模型。它
塌缩为恒定的三波束输出（926/926 帧 DICIT 预测逐字节相同；训练 loss 从 epoch 2
起平在 0.0312）。模型现在遵循 arXiv:2210.14536（最后一个时间卷积输出 `M`，再用
作者的 Soft-ArgMax）。2026-11-12 用固定 batch 的筛查又发现两个原因：

1. **学习率。** 旧配置用 `lr=1e-3`，而作者的 `1sourceTracking_icoCNN.py` 用
   `1e-4`。在 1e-3 下，第一个 AdamW 步就破坏了 Soft-ArgMax 的近均匀初始化
   （输出 ~1e-4），随后 9 个空槽的零目标把模型钉在平凡解上（单源时
   `loss = 1/30`）。即使 `M=1` 的模型在 1e-3 下也会塌缩，所以这不是多源效应。
2. **SRP 自对基座。** vendored 的 SRP 图对所有 `N×N` 麦克风对求和；PHAT 自对是
   一个常数 delta，会给每个网格单元加常数。参考 12 麦时占比很小，但 3 麦时占到
   总和的 1/3（去除前图均值 0.40、std 0.16，去除后 0.08 / 0.25）。前端现在只对
   交叉对求和。

两处修复后，固定 batch 的 80 步筛查把 loss 从 0.0332 降到 0.0023，输出范数增长
到 ~0.83；在 1e-4 但基座仍在时，训练在零解附近震荡。1/2/3 源混合的验收运行通过
（200 步：loss 0.067 → 0.0069，最佳匹配误差 70° → 7.4°，每片段预测各不相同）。
完整训练可以跑，但共享 GPU1（2/2 槽位）时需要约 37 min/epoch，独占约 24
min/epoch，因此在 epoch 1 后暂停，等 GPU 空闲再跑。

DICIT 帧数（供参考）：共 926 帧，1 / 2 / 3 源分别为 176 / 379 / 371；实际计数写
在每个 `eval_dicit.json` 里。