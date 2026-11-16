# TF-Mamba、IcoCNN、SRP-DNN 外部复现与训练接入方案

三种方法统一维护在一个独立项目 `ssl-doa-baselines`。在训练机上把它作为 LOCATA checkout 的同级目录。基线模型、训练目标、解码、启动脚本、测试和依赖配置都归独立项目。LOCATA 提供数据、现有热图训练器和评测函数。首轮通过基线侧的适配器调用这些组件，`exitzhang/LOCATA` 无需增加模型注册、子模块或第三方源码。

本文约定目录名、包名与命令，实际实现时据此落地。实验范围和算法参数与[论文对比实验计划](V5DecoderOnly_baseline_experiment_plan_2026-11-10.md)配套。SRP-PHAT 已接入，沿用现有评测和结果，必要时按冻结协议重评。

## 1. 目录与依赖边界

```text
ssl/
├── LOCATA/                            # 现有项目 checkout
└── ssl-doa-baselines/                  # 独立 Git 仓库，建议先在训练机建立
    ├── pyproject.toml
    ├── README.md
    ├── THIRD_PARTY.md
    ├── configs/                       # 可分享的模型与训练配置
    ├── src/ssl_doa_baselines/
    │   ├── api.py                     # 建模、加载权重、原生预测与 DOA 解码
    │   ├── tf_mamba/                  # 模型、STFT、空间谱读出
    │   ├── icocnn/                    # 球面 SRP、二十面体 CNN、ACCDOA、PIT
    │   ├── srp_dnn/                   # 共享 CRNN、DP-IPD 目标和 IDL
    │   └── training.py                # 独立训练入口用的轻量训练循环
    ├── integrations/locata/
    │   ├── run.py                     # 组装训练、验证、DICIT 评测
    │   ├── adapter.py                 # LOCATA 数据、Trainer 和计分桥接
    │   └── configs/                   # DICIT 实验配方，数据路径由运行参数传入
    ├── examples/                      # 合成样例和独立使用示例
    └── tests/
```

| 层 | 职责 | 依赖 |
| --- | --- | --- |
| `ssl_doa_baselines` | 可从音频和麦坐标直接建模、加载权重、预测方向。支持原生训练目标 | PyTorch 和选定模型的依赖。`tf-mamba`、`icocnn`、`srp-dnn` 分开声明可选依赖，按所选模型延迟导入 |
| `integrations/locata` | 把 LOCATA 的数据、标签、训练与指标接到基线，写实验产物 | 基线包和通过 `--locata-root` 显式指定的 LOCATA checkout |
| LOCATA | 本文模型、原有 SRP-PHAT、共享数据与评分 | 现有环境。训练本文模型时仍使用现有入口 |

`--locata-root` 在外部 runner 进程内定位项目模块。启动时核对导入的 `data`、`engine` 确实来自指定 checkout，并记录 LOCATA 提交、基线提交和工作树状态。核心包独立运行时不加载这些模块。首轮使用兼容 LOCATA 的训练环境，记录 Python、PyTorch、CUDA、`mamba-ssm` 和 IcoCNN 后端版本。某个模型需要不同环境时为该模型单建环境，并先验证适配器可导入。若依赖确实不能共存，再用文件交换音频和标签清单与预测结果，评分在 LOCATA 环境执行。[Python 打包与可选依赖](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/)

## 2. 给其他使用者的接口

| 接口 | 约定 |
| --- | --- |
| `build_model(name, config)` | 可选 `tf_mamba`、`icocnn`、`srp_dnn`。明确传采样率和 `(C,3)` 麦坐标或模型所需的通道数。麦坐标用米，行顺序与音频通道一致。几何影响模板或球面图时，由配置构建并缓存 |
| `predict_native(audio)` | 输入浮点波形 `(B,T,C)`。TF-Mamba 返回 `(B,Q)` 空间谱。IcoCNN 返回原生 ACCDOA 与输出时间。SRP-DNN 返回混合 DP-IPD 与输出时间。训练保留这三种输出的原始含义 |
| `predict(audio, decode_config)` | 返回每段或每输出时刻的可变长检测列表。每条检测带 `azimuth_deg`、原生置信分数和时间，支持俯仰时另带 `elevation_deg`。报告最大输出源数、阈值类型和角度范围。三种模型的置信分数分别校准 |
| `load_model(checkpoint, device)` | 由权重配套配置重建，校验模型版本、输入通道和关键参数。对给定阵列训练的权重明确标注该阵列。切换麦坐标时，区分可重新生成的模板与需要重训的权重 |
| 独立训练 | 使用 `fit` 入口读取 WAV 和 JSONL 清单，清单至少含 `sample_id`、路径和角度标签。逐帧任务额外提供时戳及逐源活动信息，或明确选片段级监督。阵列、采样率和声源上限由配置声明。配一组合成音频生成示例，验证用户无需项目内数据即可跑通训练和推理 |

计划提供如下命令。`./ssl-doa-baselines` 代表独立仓库，安装参数按模型可选依赖分组。独立使用者可安装、跑合成示例并加载配套权重。

```bash
python -m pip install -e './ssl-doa-baselines[tf-mamba]'
ssl-doa-baselines predict --checkpoint /path/to/model.pt --audio /path/to/clip.wav
ssl-doa-baselines fit --config /path/to/model.toml --manifest /path/to/train.jsonl
```

复现版本固定后，可以发布独立 Git tag 或模型包。随版本提供支持的输入条件、配置、安装命令、合成示例和适用权重的获取方式。`THIRD_PARTY.md` 记录论文、上游提交、文件来源、改动和许可证。`icoDOA` 与 `icoCNN` 标注 AGPL-3.0，SRP-DNN 标注 MIT。公开分发前分别核对依赖、移植代码及权重的分发条件。TF-Mamba 可用 LOCATA 历史实现做内部参考，分享版先核对该实现的来源和许可。第三方源码和模型权重按对应条款单独处理。[IcoDOA](https://github.com/DavidDiazGuerra/icoDOA) · [icoCNN](https://github.com/DavidDiazGuerra/icoCNN) · [SRP-DNN](https://github.com/BingYang-20/SRP-DNN)

## 3. LOCATA 接入到哪一层

| 工作项 | 基线侧适配动作 |
| --- | --- |
| 训练数据 | 复用原实验的 **48 kHz、每段 16,000 点（约 333 ms）**、三麦顺序 `5,6,8` 和对应坐标。从原记录重放切片、混源、噪声和增益抽样，或使用缓存波形。固定种子用于复现抽样，逐样本是否相同用样本清单及前端前的波形校验值确认。若原抽样无法重建，冻结同一份新清单并复跑 V5 主模型。TF-Mamba 使用原有 `heatmap_1d` creator，另两条线使用 `angle` loader 及同一批标签的角度和活动信息 |
| TF-Mamba 训练 | 外部 runner 直接构造 `TFMambaSpectrum`、LOCATA DataLoader 和 V5 同款损失，再注入现有 `Trainer(model, train_dataloader, val_dataloader, optimizer, loss_fn, ...)`。直接复用其 AMP、梯度裁剪、验证选权重和断点恢复。结果由外部 runner 加载实例和 checkpoint |
| IcoCNN / SRP-DNN 训练 | 用外部 `training.py` 处理原生目标和时间对齐，复用 LOCATA 数据加载。循环只负责设备、优化、AMP、验证和 checkpoint，ACCDOA/PIT 与 DP-IPD/IDL 逻辑由对应模型模块提供。分别记录学习率、有效 batch、训练样本数和选 Best 的规则 |
| 仿真验证和 DICIT 计分 | 复用原实验的验证和 DICIT 片段清单，并从 WAV 头核对 DICIT 实际采样率，单独记录时长。TF-Mamba 复用 `run_forward_cache`、`decode_at_ratio` 和公共评分。另两条线用原生解码，再把预测投影到 LOCATA 期望的 `(N,2)` 角度格式并交给 `score_decoded`。校验预测和 GT 的样本 ID、行数与顺序一致，空检测保留空数组。阈值按方法各自校准，GT 源数只用于匹配及分组 |
| 评测产物 | 阈值扫描前缓存每个片段的谱、ACCDOA 或 DP-IPD，阈值循环只重做解码及计分。报告验证锁定点的测试指标、测试曲线 `R@FAR 3%` 和匹配 MAE、1/2/3 源分组、参数量及整条流水线耗时。结果 JSON 记录阈值类型、数据划分、两仓提交及模型配置，另附逐片段输出以便复核 |

当前 `run_train.py` 的 `--model` 选择来自 LOCATA 注册表，且该入口固定使用热图标签。首轮通过外部 runner 直接注入组件，既能沿用训练体系，也能让三种模型保留自己的训练目标。现有 `run_pr_sweep.py` 同样通过项目模型加载器读取 checkpoint，外部 runner 调用底层缓存与计分函数即可共享口径。[LOCATA 训练入口](https://github.com/exitzhang/LOCATA/blob/dev/run_train.py) · [可注入模型的 Trainer](https://github.com/exitzhang/LOCATA/blob/dev/engine/trainer.py#L36) · [共享评分函数](https://github.com/exitzhang/LOCATA/blob/dev/engine/sweep_eval.py#L94)

外部 runner 首轮只需要 `smoke`、`train`、`eval` 三个子命令。每次运行从独立基线仓库发起，输出目录指到实验记录区。仓库内的模型配置只保存可分享的参数。机器相关的 `--data-config` 在仓库外保存，提供 `train_root`、`val_root`、`dicit_root`、`mic_positions_source`、训练与测试划分清单，以及冻结配方要求的噪声目录。runner 将解析后的配置写入运行目录，记录 `mix`、`mic_indices`、`sample_rate`、片段长度、声源上限和标签配方。评测从同一个运行目录读取已锁定的配置与 checkpoint。

```bash
python integrations/locata/run.py smoke --locata-root ../LOCATA \
  --config integrations/locata/configs/tf_mamba_dicit.toml \
  --data-config /path/to/locata_data.toml
python integrations/locata/run.py train --locata-root ../LOCATA \
  --config integrations/locata/configs/tf_mamba_dicit.toml --seed 42 \
  --data-config /path/to/locata_data.toml \
  --output /path/to/Record/baselines/tf_mamba_s42
python integrations/locata/run.py eval --locata-root ../LOCATA \
  --run /path/to/Record/baselines/tf_mamba_s42 --split dicit-test
```

## 4. 三个方法的实现和验收清单

| 顺序 | 核心实现 | 任务适配 | 进入正式训练前的验收 |
| --- | --- | --- | --- |
| 1. TF-Mamba | 对照原文和历史实现。STFT 实虚部编码、DenseNet、五组时频 BiMamba、同轴跨块跳连。48 kHz 下使用 1,536 点窗和 FFT、480 点 hop，保留原文 32 ms 窗、10 ms hop 及分析频段 | 三麦输入转 6 通道时频特征。片段级读出 `Q=181`。接入 V5 主版本标签、损失和解码。记录激活与时间聚合的适配 | 检查轴序和跳连，输出 `(B,Q)`。小批次覆盖 1/2/3 源并能拟合多峰。用同一批冻结样本验证独立模型、外部 runner 与 LOCATA 原生解码路径的指标一致 |
| 2. IcoCNN | 从作者球面网格及卷积起步，复现多源纯卷积配置。基线包自行生成球面 SRP-PHAT，保留三维 ACCDOA 与空槽 | 48 kHz 下保留原 256 ms 窗（12,288 点），将 hop 设为 1,536 点（32 ms）以在约 333 ms 内得到三张完整球面图。从原文 `r=3` 和 `M=10` 起步。片段级标签在对应输出时刻做帧级 PIT | 核对球面采样方向、帧数、因果填充及标签时刻。用 0/1/2/3 源校验空槽、换序不变性和未知源数的三源输出。若三源适配未通过，结果放共同的 1–2 源子表 |
| 3. SRP-DNN | 按作者实现复现共享因果 CRNN、混合 DP-IPD 和 IDL。保留每轮模板扣除，设网络幅度上限和解码迭代上限为 3 | 直接使用同一 48 kHz、16,000 点片段。用 `1536/768` STFT 和 1,536 点 FFT，取 `k=1…256`（31.25–8000 Hz），保留原 CRNN 维度。STFT 特征、DP-IPD 监督与 IDL 模板使用同一频点，共有 19 个完整 STFT 帧。开训前核对 12 倍时间池化后的输出与参考时刻。优先由逐源逐帧活动算 `β`，仅有片段 DOA 时使用并注明片段活动近似 | 理想 DP-IPD 直送 IDL 检查已知方向、麦对顺序、相近双源、分离三源和静音停止。验证输入频点、目标、模板和标签时刻一致，再验收训练前后向、断点恢复和阈值扫描 |

三条线共享 **约 333 ms 的训练片段**和标签参考时刻。模型内部的重采样、帧移和填充只改变特征计算方式。填充使用片段内信息或零值，预测时刻按原始时间轴对齐。先跑形状检查、小样本过拟合和一次完整评测。IcoCNN 和 SRP-DNN 的原论文使用更长时序，论文复现说明记录本任务的短片段适配、活动标注可用性及有效输出时刻。[TF-Mamba 原文](https://www.isca-archive.org/interspeech_2025/xiao25_interspeech.pdf) · [多源 IcoCNN](https://arxiv.org/html/2210.14536) · [SRP-DNN](https://arxiv.org/pdf/2202.07859)

## 5. 实施顺序和完成标准

| 阶段 | 工作 | 完成标准 |
| --- | --- | --- |
| A. 骨架 | 建立独立仓库，固定两仓基线提交，落地包的最小 API、LOCATA 适配器和合成样例 | 在没有 LOCATA 的干净环境里，能安装基线包并运行一个合成输入的构建或推理测试。在训练机能通过 `--locata-root` 读到正确的音频、标签和三麦坐标 |
| B. 打通训练 | 优先完成 TF-Mamba 独立推理、`Trainer` 注入和 DICIT 评分。随后接入 IcoCNN 和 SRP-DNN 的原生任务 | 每种模型能独立构造及重载权重，基线侧 runner 能跑完一次训练、验证和评测。TF-Mamba 沿用现有 Trainer 的 epoch 级续训，核对模型、优化器、调度器和 AMP 状态。外部训练循环另记录其支持的随机状态与恢复粒度 |
| C. 正式对比 | 分别启动三条 `seed=42` 训练线，使用同一冻结划分与样本预算，锁定验证集选出的工作点后评 DICIT | 产物含每方法配置、训练记录、checkpoint、预测、指标和耗时。现有 SRP-PHAT 与两条空间谱基线按相同协议计分。按主表结论需要补配对种子 |
| D. 可复用交付 | 从独立仓库打版本，写最短安装和使用步骤，整理来源与许可记录 | 其他使用者用新的环境和合成样例，能只安装选定的模型并从 WAV、麦坐标与权重得到 DOA。提供可分享的模型配置、权重说明与复现记录 |
