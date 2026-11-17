# ssl-doa-baselines

五个外部多源 DOA 基线的独立实现，附一层薄 LOCATA 适配层，复用 LOCATA 自带的
数据集、训练器和评分器。

| 键 | 模型 | 原生输出 | 解码方式 |
|---|---|---|---|
| `tf_mamba` | TF-Mamba（双向时间 / 频率 Mamba） | 方位谱 `(B, 181)` | 共享峰值解码器 |
| `icocnn` | 多源 IcoCNN（球面 SRP-PHAT + 二十面体 CNN） | ACCDOA 槽位 `(B, T, 10, 3)` | 活动度阈值 + top-3 |
| `srp_dnn` | SRP-DNN（因果 CRNN + 混合 DP-IPD） | 混合 DP-IPD `(B, T, 2F, P)` | 迭代检测与定位 |
| `ipdnet` | IPDnet（固定三麦，全带/窄带 LSTM + 因果 CNN） | 逐轨 DP-IPD `(B, 2F, P, K)` | 投影比阈值 |
| `ipdnet2_small` | IPDnet2-Small（固定三麦，在线空间网络，Mamba） | 逐轨 DP-IPD `(B, 2F, P, K)` | 负 MSE 阈值 |

包本身从不 import LOCATA。`integrations/locata/` 是唯一边界，只有外部 runner
会加载它。

## 安装

```bash
conda activate locata                     # Python 3.8 / torch 2.2.1 / CUDA
pip install -e ".[all]"                   # numpy/scipy/soundfile/tomli + einops
pip install mamba-ssm==2.1.0 --no-build-isolation   # 若尚未安装
```

`tf_mamba` 和 `ipdnet2_small` 需要 `mamba-ssm`，且必须是 CUDA 构建。`ipdnet` 和
`ipdnet2_small` 还需要 `torchaudio`（`pip install -e ".[ipdnet]"` /
`".[ipdnet2]"`）；其余模型可在 CPU（较慢）或 GPU 上运行。这里验证过的组合是
torch 2.2.1 / torchaudio 2.2.1 / CUDA 11.8。**不需要** `causal-conv1d`：它只会
启用 `mamba-ssm` 的融合 `mamba_inner` 路径，而该路径在本模型池化后的序列长度上
实测更慢（见 `RESULTS.md`）。

两个 IPDnet 移植共用 48 kHz -> 16 kHz 重采样器、512 点 Hann/FFT 前端
（`center=False`、显式右侧补零）、带 `J0` 空轨的逐轨 DP-IPD 目标和片段级 PIT。
IPDnet 用 hop 256（20 个 STFT 帧、2 个池化步），IPDnet2-Small 用 hop 320
（17 帧、4 个池化步）。解码为每个有效轨输出一个检测：IPDnet 按投影比打分，
IPDnet2-Small 按负模板 MSE 打分（因此同样适用 `score >= threshold` 的扫描；
`-0.4` 表示 `MSE <= 0.4`）。

## 包 API

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

`waveform` 是 48 kHz 的 `(T, C)` 或 `(B, T, C)`。`save_checkpoint` 写出的
checkpoint 内嵌模型配置，重载时无需再传配置。

`ipdnet` 和 `ipdnet2_small` 没有安全的工作点默认值：`predict` 要求显式且有限的
`decode["threshold_abs"]`，缺失时会在前向之前直接报错。该值通常是在仿真验证集上
锁定的阈值（`validation.json["locked"]["threshold"]`）；对 IPDnet2-Small，
`-0.4` 表示"保留 `MSE <= 0.4` 的轨"。`predict_native` 仍返回未做阈值过滤的
原始轨。

```python
model = load_model("runs/ipdnet2_small_s42/checkpoint.pth", device="cuda")
detections = predict(model, waveform, decode={"threshold_abs": -0.4})
```

独立 CLI 接受显式阈值或验证文件（二者互斥）：

```bash
ssl-doa-baselines predict \
  --checkpoint runs/ipdnet2_small_s42/checkpoint.pth \
  --audio clip.wav \
  --validation runs/ipdnet2_small_s42/validation.json

ssl-doa-baselines predict \
  --checkpoint runs/ipdnet2_small_s42/checkpoint.pth \
  --audio clip.wav --threshold-abs=-0.4
```

`--validation` 文件会按 checkpoint SHA-256（对拷贝/移动不敏感）、模型键、打分
类型和比较方向做匹配；来自其他模型或其他 epoch 的文件会被拒绝。

角度采用**天顶仰角**：`u = [sin(el)cos(az), sin(el)sin(az), cos(el)]`，因此
水平声源有 `el = 90`。DICIT 元数据本来就是该约定。LOCATA 仿真元数据存的是相对
水平面的仰角，适配层用 `el_zenith = 90 - el_sim` 转换。

方位遵循项目的标签坐标系。真实 DICIT 帧上，`X_k conj(X_l)` 的 GCC 时延为
`-(p_k - p_l)·u_label/c`，即标签坐标系相对房间原始坐标是镜像的。因此 SRP 前端
用 `(p_l - p_k)·u` 做导向；按此约定，它的峰与 DICIT 单源标签的平均误差约 6°。
`examples/make_synthetic.py` 也按同一约定生成片段。

## 冒烟测试

仅 CPU 的单元测试（无 CUDA 时 TF-Mamba 会被跳过）：

```bash
python -m pytest tests -q
```

单模型的端到端 GPU 冒烟（前向、原生解码、checkpoint 重载）：

```bash
python integrations/locata/run.py \
  --locata-root /path/to/LOCATA \
  --config integrations/locata/configs/icocnn_dicit.toml \
  --data-config integrations/locata/configs/data_local.toml \
  --output /tmp/icocnn_smoke --device cuda:0 smoke
```

## LOCATA 训练与评估

先复制机器相关的数据配置并改路径：

```bash
cp integrations/locata/configs/data_local.example.toml \
   integrations/locata/configs/data_local.toml
```

训练、在仿真验证集上锁定工作点、再在 DICIT 上评估：

```bash
python integrations/locata/train_then_eval.py \
  --locata-root /path/to/LOCATA \
  --config integrations/locata/configs/tf_mamba_dicit.toml \
  --data-config integrations/locata/configs/data_local.toml \
  --output runs/tf_mamba_spectrum_s42 --seed 42 --device-index 0
```

仅 `train` 会写出：

```
runs/<name>/
  checkpoint.pth      # 内嵌模型配置，最佳验证 epoch
  validation.json     # 完整阈值曲线 + 锁定的工作点
  config.json         # 解析后的配置、两个仓库的 git commit、dirty 标记
  train.log           # 仅 TF-Mamba（LOCATA Trainer 日志）
```

`eval --split dicit --validation .../validation.json` 会写出 `eval_dicit.json`，
包含锁定工作点、按源数分组的指标、完整测试曲线和 `R@FAR 3%` 插值（仅当两个实测
点能夹住 3% FAR 时）。它还会在锁定工作点存下 `predictions.gt` /
`predictions.pred`（每帧一组角度，纯 float），便于逐帧复查，以及 `num_params`
（由重建模型统计）、`forward_ms_per_frame`、`decode_ms_per_frame` 和
`inference_ms_per_clip`。

`inference_ms_per_clip` 是统一的端到端数字：从已在设备上的 waveform 到锁定阈值
下的逐片段角度集合，包含该方法自己的前端、骨干、模板匹配和分数过滤。预热、设备
同步和 batch 大小对所有方法一致，分母是实际片段数（末尾的短 batch 不会向上取
整）。公共评分、磁盘读取和完整阈值扫描不计入；旧的 `forward_ms_per_frame` 和
IPDnet 的 `frontend` / `template` 分解仍保留用于剖析。

对 `ipdnet` / `ipdnet2_small`，DICIT 评估**必须**提供 `--validation`，否则在
接触数据集之前就失败：测试阈值必须是在仿真验证集上锁定的那个。锁定点会被数值地
并入上报的测试曲线，因此即使扫描配置或 JSON 键的拼写变了也不会丢失。

`eval --split val --output PATH` 会写出与 `train` 相同的 `validation.json`
（曲线 + 带身份的锁定），因此无需重训即可为已有 checkpoint 重新生成阈值。训练会
把 `mic_indices`、`mic_positions` 和 `ref_mic` 记录进 checkpoint 配置和运行
记录；评估会拒绝通道顺序被换或麦克风对间距改变的模型，但允许同一阵列的平移副本。

### 指标

所有评分都调用 LOCATA 的 `engine/sweep_eval.py` 和 `engine/metrics.py`
（`score_decoded`），使用 10° 一对一匹配和纯方位误差。TF-Mamba 复用与
`run_pr_sweep.py` 相同的频谱缓存 + 峰值解码器；IcoCNN 和 SRP-DNN 缓存原生
`(score, az, el)` 候选并扫描各自的绝对阈值。工作点在仿真验证集上按最大
`F1@10°` 锁定（并列时取更低的 FAR）。

按真实源数分组用同一匹配器计算。FAR 保留静音帧的误峰，与 LOCATA 评估器一致。

## 队列提交

训练必须走 `gpu_queue`（前台命令，训练 + 评估共用一个槽位）。`--locata-root`
指向 LOCATA 仓库的专用 worktree
（`/home/t-zhangchaoqin/projects/ssl/wt/locata-dev`），**detach 在固定
commit**（`d5e7c5d`，2026-11-11 10:58 的 `dev` tip）。它从 `dev` 建出，以免 V5
消融使用的共享 checkout 被改动；同时又处于 detach 状态，`dev` 上的合并活动无法
改变排队作业 import 的代码。要把基线换到更新的 LOCATA 版本，需显式 checkout
那个 commit 再重新提交。每次运行都会把实际的 LOCATA commit/分支/dirty 标记记录在
`config.json -> locata`。

作业清单示例：

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

`icocnn_multi_s42`、`srp_dnn_s42`、`ipdnet_s42` 和 `ipdnet2_small_s42` 用各自
的配置按同样方式提交。IPDnet 的清单是 `queue/ipdnet_512_s42_v3.json`（已完成）
和 `queue/ipdnet2_small_512_s42_v4.json`（待跑，未提交）。

两个 IPDnet 作业都从专用 worktree
`/home/t-zhangchaoqin/projects/ssl/wt/ssl-doa-baselines-ipdnet`（分支
`fix/ipdnet-review-fix`）运行，因此运行中的作业不可能在不产生提交的情况下被改
代码。该 worktree 的 18 个提交已于 2026-11-17 合并进
`feat/external-baselines` 后删除；在提交待跑的 `ipdnet2_small_s42` v4 清单前需要
重建一个 worktree，因为清单的 `cwd` 仍指向旧路径。机器相关、被 git 忽略的
`data_local.toml` 和产物根目录通过绝对路径传入主 checkout；因此
`runs/ipdnet_s42` 和 `runs/ipdnet2_small_s42` 留在可索引的 checkout 里。

### IPDnet 出处

两个移植都是 `Audio-WestlakeU/FN-SSL@76fcb281`（MIT；见 `THIRD_PARTY.md`）的
固定阵列改造。它们在共享 LOCATA 协议上从随机初始化训练；不在此复现作者的论文级
结果、原生阵列与帧级输出。

### 共享训练预算

五行都用 `--epochs 100`、`patience 6` 轮验证早停和 seed 42，因此样本预算相同。
最初 15 epoch 的初版被弃用：V5 主线（B0 / Parallel）在 30 epoch 左右取最佳、在
60 早停，15 epoch 只有对照预算的四分之一。实际 epoch 数和预算记录在每个
`config.json` 中。已提交的清单是
`queue/external_baselines_2026-11-11_v4.json`（TF-Mamba、IcoCNN）和
`queue/external_baselines_2026-11-11_srp100.json`（SRP-DNN）。

## 目录结构

```
src/ssl_doa_baselines/
  api.py config.py geometry.py training.py cli.py
  dpipd_frontend.py dpipd_tracks.py     IPDnet 共享前端 / 目标 / 解码
  tf_mamba/  model.py decode.py
  icocnn/    model.py srp.py grid.py conv.py accdoa.py
  srp_dnn/   model.py stft.py dpipd.py idl.py
  ipdnet/    model.py backbone.py
  ipdnet2/   model.py backbone.py arch/base/
integrations/locata/
  adapter.py            数据 / 标签 / 训练 / 评分的桥接层
  run.py                smoke | train | eval
  train_then_eval.py    gpu_queue 前台包装
  configs/              每个模型一份 TOML 配方 + data_local.example.toml
examples/make_synthetic.py
tests/
runs/                   进行中和已完成的运行产物（git 忽略）
aborted/                中止/暂停尝试的产物（git 忽略）
```

## 当前状态

五个模型的冒烟测试（GPU 前向、原生解码、checkpoint 重载）全部通过，包括
IPDnet2-Small 的 Mamba 路径。单元测试覆盖形状、PIT 不变性、理想 DP-IPD 上的
IDL、标签转换、SRP 导向、五块 CRNN 前向、import 边界、共享 IPDnet 前端
（重采样、帧数、相位符号）以及逐轨目标/解码。

2026-11-10 曾把三个正式作业（TF-Mamba `_dev`、IcoCNN `_dev`、SRP-DNN `_dev`）
以共享的 15 epoch 预算提交到 `gpu_queue`，`--locata-root` 固定到 `dev`
worktree。该预算后被弃用：V5 主线在 30 epoch 左右取最佳，15 epoch 只有对照预算
的四分之一。TF-Mamba 的 `_dev` 还遇到了 fp16 溢出，IcoCNN 的 `_dev` 则跑在
vendored 层的病态慢版本上。

各规范运行及其 DICIT 结果（阈值锁定在仿真验证集；完整表格、按源数指标和运行日志
见 `RESULTS.md`）：

| 方法 | 运行 | DICIT F1@10° | MAE (°) | 状态 |
|---|---|---|---|---|
| TF-Mamba（n_fft=1536） | `runs/tf_mamba_spectrum_s42` | **0.9338** | **1.47** | done 2026-11-14，100 epoch |
| TF-Mamba（n_fft=512） | `runs/tf_mamba_512_s42` | 0.9294 | 1.53 | done 2026-11-15，epoch 85 早停 |
| SRP-DNN | `runs/srp_dnn_s42` | 0.7665 | 3.26 | done 2026-11-13，epoch 95 早停 |
| IPDnet（固定三麦） | `runs/ipdnet_s42` | 0.9274 | 1.49 | done 2026-11-17，100 epoch |
| IPDnet2-Small（固定三麦） | `queue/ipdnet2_small_512_s42_v4.json` | pending | pending | 已暂停；需从新 worktree 重新提交 v4 清单 |
| IcoCNN（多源） | backlog；最近一次尝试 `aborted/icocnn_slow_20261112` | — | — | 需要空闲 GPU（单跑约 24 min/epoch） |

每次尝试的状态都记录在 `RESULTS.md` 的运行日志里；历史 `queue/` 文件中的作业
id 保留原来的 `_vN` 名称。

IPDnet2-Small 训练到 ~960–1085 s/epoch，因为它的每一步是 host-launch 受限的：
每步约 2450 次 kernel launch，而 IPDnet 是 373 次，GPU kernel 时间反而更少
（16.5 vs 19.7 ms）。向量化后的 `FreqInverse`（`67cc6b2`）在这台负载较高的机器
上约省 6%；完整测量和已否决的加速方案（`causal-conv1d`、`torch.compile`）见
`RESULTS.md`。