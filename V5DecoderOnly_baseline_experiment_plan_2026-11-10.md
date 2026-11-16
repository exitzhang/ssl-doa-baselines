# V5DecoderOnly 论文对比实验计划

首轮复现并重训 **TF-Mamba、多源 IcoCNN 和 SRP-DNN**。复用已接入的 **SRP-PHAT**，并按同一协议评估现有的 **IPDnet 和 IPDnet2 单帧空间谱适配版**。完整主表覆盖 **1、2、3 源**。三条线的代码维护在独立基线项目，由基线侧适配器接入 LOCATA 数据、训练组件和评分。第 4–8 节说明各算法的复现参数，独立项目结构与接入步骤见[外部基线复现与训练接入方案](SSL_external_baselines_integration_plan_2026-11-10.md)。

`IPDnet_SpatialSpectrum` 是 **IPDnet 的单帧空间谱适配版**。`OfflineSpatialNet` 是 **IPDnet2 的离线单帧空间谱适配版**。这里的单帧指每个输入音频片段输出一个方位空间谱，谱中的多个峰可对应多个声源。[IPDnet 适配代码](../models/IPDnet_SpatialSpectrum.py#L62) · [IPDnet2 适配代码](../models/IPDnet2.py#L489)

## 1. 实验计划表

| 顺序 | 论文表中名称 | 当前状态 | 需要完成的工作 | 结果放置 |
| --- | --- | --- | --- | --- |
| P0 | V5DecoderOnly（本文方法） | 有历史结果 | 在验证集锁定论文主版本和 checkpoint。用下文统一协议重评。模型配置记录对齐方向数、输出谱点数与标签宽度 | 完整主表和 1–2 源子表 |
| P0 | IPDnet（单帧空间谱） | `IPDnet_SpatialSpectrum` 已训练 | 核对历史训练数据、麦克风顺序和 checkpoint 选取规则。兼容时复用 checkpoint 并重评，否则按统一划分补训 | 完整主表 |
| P0 | IPDnet2（离线单帧空间谱） | `OfflineSpatialNet` 已训练 | 同上。保留其非因果空间谱头，记录推理窗口和运行成本 | 完整主表 |
| P1 | SRP-PHAT | 已接入，无需训练 | 复用现有实现和结果。核对录音划分、三麦顺序、阈值及指标，口径不一致时只补评测 | 完整主表。衡量相对传统几何空间谱的收益 |
| **P2** | **TF-Mamba（多源单帧空间谱）** | 待复现、适配和重训 | 保留原文的时间 BiMamba、频率 BiMamba 和跳连。适配三麦输入及片段级读出，预测一个可含多个峰的方位空间谱。采用与本文方法一致的标签构造、训练损失和峰值解码规则 | 完整主表。比较同一空间谱任务下的时频建模架构 |
| **P3** | **IcoCNN（多源适配）** | 待复现、适配和重训 | 用相同 DICIT 三麦生成球面 SRP-PHAT 输入，保留二十面体卷积和多源排列不变训练。按相同参考时刻提取方位角，在验证集确定活动判决规则。验证模型能在未知真实源数时输出 0–3 个方向 | 支持三源时进完整主表。若复现仅支持至多两源，放入共同的 1–2 源子表 |
| **P4** | **SRP-DNN** | 待适配和重训 | 用相同训练数据和三麦几何，保留混合直达路径相位差估计与迭代定位。验证迭代最多能检出三个方向，由验证集确定停止阈值 | 完整主表。比较学习式多声源方法 |
| P5（按需） | IPDnet2-Small，DP-IPD（K=2） | 若需比较 DP-IPD 流程再补 | 用官方实现接入相同三麦，保留逐轨 DP-IPD 目标、模板匹配和声源活动判决。在 1–2 源数据上训练和评估，同步重评其他方法在完全相同片段上的结果 | 1–2 源子表。比较完整的 DP-IPD 定位流程 |
| P6（按需） | IPDnet2-Small，DP-IPD（K=3 适配） | P5 跑通后按需补 | 扩展输出轨数和对应训练目标，核对排列不变匹配、空轨迹和三源解码，用支持三源的统一训练数据重训 | 完整主表。用于与 DP-IPD 方法直接比较三源表现 |
| 按论文主张补充 | AGG-RL | 有官方模型代码和权重 | 如果主张涉及未见阵列或可变方向网格，先对齐其候选网格、三麦输入和评测时间，再用双方共同支持的声源数和数据划分进行比较 | 几何或网格泛化子表。单一 DICIT 子阵列的结果用于比较固定阵列表现 |

现有 checkpoint 可复用时，首轮计划新增 **TF-Mamba、多源 IcoCNN 和 SRP-DNN 三条训练线**。P5、P6 用于单独考察保留原论文 DP-IPD 输出的 IPDnet2，表中标注 `K` 和阵列适配情况。[现有 SRP-PHAT 实现](https://github.com/exitzhang/LOCATA/blob/dev/traditional_methods/srp_phat.py) · [SRP-DNN 论文](https://arxiv.org/abs/2202.07859) · [IPDnet2 官方实现](https://github.com/Audio-WestlakeU/FN-SSL/tree/main/IPDnet2) · [AGG-RL 官方实现](https://github.com/BaekMS/Audio-Geometry-Grid_Representation-Learning)

原版 TF-Mamba 用于双麦单源定位，直接回归方位空间谱。IPDnet2 论文沿用了它的 Mamba 骨干，改为逐声源轨迹估计 DP-IPD。本计划沿用同样的骨干适配思路，将输出统一为与本文方法相同的**单帧多峰空间谱**。正文报告改动过的输入通道、时间聚合和多峰监督，注明结果来自本任务重训。优先复现 TF-Mamba 论文主对比所用的双向配置，记录参数量和推理耗时。[TF-Mamba 原论文 §3–5](https://www.isca-archive.org/interspeech_2025/xiao25_interspeech.pdf) · [IPDnet2 多源适配说明 §3.2](https://arxiv.org/html/2509.21900v1#S3.SS2)

IPDnet2 在方法节把多源对照称为 IcoCNN，在结果表中写作 IcoDOA。本计划统一写作 **IcoCNN（多源适配）**。这一臂比较先构造 SRP-PHAT 球面空间谱再做神经建模的路线，推理耗时计入 SRP-PHAT 特征计算。[IPDnet2 §3.2 与表 1](https://arxiv.org/html/2509.21900v1#S3)

历史 checkpoint 复用候选为 `IPDnet_SpatialSpectrum` 的 `Best_2026-11-06_18-19.pth` 和 `OfflineSpatialNet` 的 `Best_2026-11-06_21-29.pth`。以远端训练记录核对实际参数与数据划分。[历史作业状态](../queue/v5_schedule_2026-11-06/state.json)

## 2. 统一评测协议

| 项目 | 首轮约定 |
| --- | --- |
| 输入与输出 | 共享训练输入为 **48 kHz、每段 16,000 点（约 333 ms）**。复用三麦顺序 `5,6,8`（项目索引）和对应麦坐标。DICIT 评测复用原实验的相同片段和参考标签，单独核对评测 WAV 采样率，`target_num_samples=16000` 本身只限定样本点数。输出范围为 0°–180°，空间谱用 `Q=181`。各基线保留原生前端与解码，统一最后的角度集合接口 |
| 训练与划分 | 固定共享训练数据、数据增强范围和验证划分。核对单帧空间谱模型的标签构造、损失、训练预算和 checkpoint 选择规则，TF-Mamba 与本文方法采用同一空间谱监督。SRP-DNN 和逐轨 DP-IPD 的 IPDnet2 保留各自的监督目标。IPDnet2 `K=2` 从共享训练池取可监督的至多两源样本，记录筛选规则及样本数。在 DICIT 上校准时按原始录音划分验证和测试，避免同一录音的相邻或重叠片段跨集合。若用仿真验证集校准并直接测试实录，明确写作仿真到实录评测 |
| 时间对齐 | 对每个输入片段沿用现有评估器的参考标签时刻或聚合规则。对逐帧输出的多源 IcoCNN、SRP-DNN 和 DP-IPD 版 IPDnet2，预先指定对应输出时刻，记录实际输入上下文和算法延迟 |
| 源数与匹配 | 推理输入包含音频和各方法所需的麦克风几何。真实源数用于测试分组及匹配。使用相同的 10° 容差和一对一匹配规则，冻结 FAR 分母与 MAE 的匹配条件。完整集分别汇总总体、1 源、2 源、3 源。1–2 源子表用相同录音划分下的共同片段重新聚合 |
| 固定工作点 | 按方法在验证集选择阈值和解码参数，锁定后报告测试 `F1@10°`、召回、实际 FAR、匹配 MAE。若验证集以 FAR 3% 为目标，报告测试实际达到的 FAR |
| 曲线比较 | 另报告测试 PR 曲线上的 `R@FAR 3%` 及同点匹配 MAE。只有相邻实测点覆盖 3% 时才插值。补扫后仍达不到时标记 `未覆盖`。测试扫阈值的峰值 F1 作为探索记录，与验证集锁定工作点分开 |
| 成本与复验 | 记录参数量、相同硬件和输入长度下的整条推理耗时，包括特征提取和解码。首轮可用单种子检查流程。对用于结论的关键神经网络比较，建议最终用三个种子报告均值和标准差 |

现有评测使用 `--split all`。历史日志的 926 个片段中，1、2、3 源分别有 176、379、371 个。它们提供当前实验规模的参考，最终按锁定后的划分重报样本数。若论文另有静音误报结论，单独加入 0 源子集并报告静音片段误报率。[当前评测配置](../queue/v5_schedule_2026-11-06/scheduler.py#L323) · [声源数分布](../queue/v5_schedule_2026-11-06/sweeps/A0_base_s15_181.log#L26) · [固定 FAR 评测约定](V5DecoderOnly_parallel_branch_ablation_plan_2026-11-10.md#4-配方与评估)

## 3. 主表填报模板

F1 四列采用验证集锁定的工作点。固定 FAR 两列来自测试曲线，单独列出。1–2 源子表沿用同一口径，并单独标明训练的最大源数和评估片段数。多源 IcoCNN 若仅支持至多两源，其结果整行移入 1–2 源子表，在相同片段上重算各方法指标。

| 方法 | F1@10° 总体 | 1 源 F1 | 2 源 F1 | 3 源 F1 | 实际 FAR | R@FAR 3% | 同点 MAE（°） | 参数量 | 推理耗时 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SRP-PHAT | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 0 | 待填 |
| SRP-DNN | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| IPDnet（单帧空间谱） | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| IPDnet2（离线单帧空间谱） | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| TF-Mamba（多源单帧空间谱） | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| IcoCNN（多源适配，三源验证后纳入） | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| IPDnet2-Small，DP-IPD（K=3 适配，按需） | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| V5DecoderOnly（本文方法） | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |

## 4. 开工前冻结数据和评测接口

| 工作项 | 具体操作与验收 |
| --- | --- |
| 冻结论文配方 | 先锁定本文模型的代码提交、数据划分、训练步数、音频增强、声源数比例和验证选点规则。现有记录有不同标签宽度的配方，TF-Mamba 取论文主版本的实际 `effective_config`，包括 `loss_fn`、`loss_kwargs`、`label_sigma` 和标签中心模式。首轮建议 `seed=42`，关键对照最终补 `43/44` |
| 冻结数据契约 | 三条线复用 `AudioDataset` 的切分、混音、通道选择、噪声增强。前端之前共享 `(B,16000,C)`、48 kHz 的波形和同一标签参考时刻。要与原实验逐样本一致，优先重放原始切片与混源、噪声、增益抽样清单，或重用相同的缓存波形。若历史抽样未保存且无法重建，冻结新清单并用它复跑本文主模型后再做逐样本对照。TF-Mamba 用现有 `heatmap_1d`。IcoCNN 和 SRP-DNN 从 `label_mode="angle"` 取每段 `(N,2)` 角度集合。逐帧逐源 VAD 需查实际数据来源 |
| 冻结时间和坐标 | 用已知方向的合成信号核对三麦坐标、通道顺序、角度正负、0°/180° 端点和原点。DICIT 实录标签沿用现有 loader 的坐标修正；各方法只把自己的预测转换到同一评测坐标。对有时间轴的模型，按 loader 的标签时刻或标签覆盖区间对齐输出，记录 STFT、窗口、卷积池化和时间索引。若训练只有片段级 DOA，在与标签对齐的位置监督，静态且持续有效的片段才可广播到多个输出帧 |
| 共用计分 | 外部 runner 直接调用 `engine/sweep_eval.py`。TF-Mamba 复用 `run_pr_sweep.py` 所使用的空间谱缓存和峰值解码函数。IcoCNN、SRP-DNN 各缓存原生预测，按各自阈值输出每片段的方位角集合，再调用 `score_decoded`，沿用项目 `(N,2)` 角度约定。三者对同一批样本按 10° 容差匹配。阈值在验证集锁定，测试曲线单独计算固定 FAR 指标 |

[训练入口](https://github.com/exitzhang/LOCATA/blob/dev/run_train.py) · [数据入口](https://github.com/exitzhang/LOCATA/blob/dev/data/audio_dataset.py) · [DICIT 评测入口与坐标说明](https://github.com/exitzhang/LOCATA/blob/dev/run_pr_sweep.py) · [共享计分](https://github.com/exitzhang/LOCATA/blob/dev/engine/sweep_eval.py)

重采样只在模型前端进行。保留每段约 333 ms 的原始音频范围、通道同步与标签时刻。若某个前端使用 16 kHz，按 `1/3` 重采样后约有 **5,333 点**，确切长度由固定的取整规则决定。内部填充用于满足卷积或窗口对齐，时间戳与有效区间仍对应原片段。评测配置分别记录 WAV 采样率、输入片段点数、模型内部采样率和前端时间参数。

## 5. TF-Mamba（多源单帧空间谱）

1. **从项目历史恢复骨干。** 已核对 `b33a2ab` 有 `models/TF_Mamba.py`，后续删除提交 `233b908` 的补丁也保留了最后一版代码。以历史实现的 DenseNet 编码器、双向 Mamba 和两个轴的 reshape 为起点。删除前代码默认两个块、`expand=[2,2]`。按原论文主实验恢复五块、`expand=[2,2,4,4,8]` 和同轴跨块跳连。首轮保留 48 kHz 输入，`win=1536`、`hop=480`、`n_fft=1536`，对应原文的 32 ms 窗、10 ms hop 和 31.25 Hz 频点间距，按物理频率筛选 100–8000 Hz。历史实现的隐藏宽度 64、`d_state=16`、`d_conv=4` 作为复现起始值，记录实际参数量，未由论文确定的细节注明采用历史实现。[历史源码](https://github.com/exitzhang/LOCATA/blob/b33a2abc90740caa3f189da893f230623a07d728/models/TF_Mamba.py) · [删除前版本补丁](https://github.com/exitzhang/LOCATA/commit/233b9083b3f4fd0d9e18fb4b081235affe7fe33b) · [TF-Mamba §3–5](https://www.isca-archive.org/interspeech_2025/xiao25_interspeech.pdf)
2. **接入同任务输出。** STFT 实虚部合计 `2C=6` 个输入通道。沿频率的 Mamba 序列形状为 `(B*T,F,H)`，沿时间为 `(B*F,T,H)`。保留双向处理和同轴跨块跳连。采用片段级平均聚合与线性 `Q` 点读出，激活与当前损失的接口一致。外部包直接实例化 `TFMambaSpectrum`，适配器把项目 `heatmap_1d` loader、损失和该模型交给 `Trainer`，共享峰值解码器。实验配置取 `Q=181` 和冻结的 V5 配方。
3. **验收。** 单测两个轴的 reshape 和跳连，三麦前向必须返回 `(B,Q)`。用固定 1/2/3 源小样本集做过拟合并检查解码能检出多个峰。跑一次 AMP 前后向和 checkpoint 严格重载。通过后用外部 runner 启动 `tf_mamba_spectrum_s42`，调用共享函数做仿真验证和 DICIT PR 评测。

## 6. 多源 IcoCNN（球面 SRP-PHAT → 多轨 ACCDOA）

1. **以多源论文的纯卷积配置为复现目标。** 使用作者 `icoDOA` 的球面网格、二十面体卷积及 Soft-ArgMax，扩展为 `M` 个三维 ACCDOA 输出，向量给出方向，范数表示活动。首选 `r=3`、卷积宽度 `128`、时间卷积核 `5`、输出槽数 `M=10`，评测最多保留三个高分候选。原文 16 kHz 下的 4096 点窗为 256 ms，换到 48 kHz 为 12,288 点。原 200 ms 步进在 16,000 点片段内只产生一张完整球面图。短片段适配首选保留 256 ms 窗、将 hop 设为 1,536 点（32 ms），可得到三张完整图。核对因果填充、输出参考时刻和短片段时间覆盖，报告该步进适配。[IcoDOA 作者代码](https://github.com/DavidDiazGuerra/icoDOA) · [多源卷积版与 sPIT](https://arxiv.org/html/2210.14536) · [多源 IcoCNN 配置与输出槽数](https://arxiv.org/html/2306.08510#S3.SS1)
2. **复用 GCC，保留球面拓扑。** 用同一 DICIT 三麦和麦坐标计算 GCC-PHAT。在作者二十面体顶点上计算每对麦克风的理论时延，按其 chart 布局生成球面输入。项目 `SRPMap` 目前接受规则 `theta × phi` 网格。可复用 `GCC`，球面投影沿用作者路径或复用相同的延时取样逻辑。先在单一方向用相同麦对和延时约定核对球面图峰值。[项目 GCC 和规则网格实现](https://github.com/exitzhang/LOCATA/blob/dev/traditional_methods/srp_phat.py) · [二十面体卷积实现](https://github.com/DavidDiazGuerra/icoCNN)
3. **训练 ACCDOA。** 用现有 `(azimuth,elevation)` 角度转单位向量，空槽为零。用三维向量误差矩阵做最小代价匹配，再对已配对向量算 MSE，避免枚举 `M!`。当前任务按片段定位，使用对应时刻的 **帧级 PIT**。将这个选择写进方法说明。若后续具备连续轨迹及时间标注，再单列 sPIT 跟踪复现。若训练 elevation 经过占位或角度约定不明确，先检查真实训练数据；若使用水平面假设，训练和测试遵循同一方位定义并标注该适配。
4. **解码及验收。** 从 ACCDOA 向量取范数作为活动分数，按验证阈值过滤，再转为项目方位角并应用相同的输出上限。先用 0/1/2/3 源样本验证活动槽、空槽、换序不变的损失和三源输出，再用仿真验证集的三源子集确认没有退化为固定两源。开训 `icocnn_multi_s42`。能稳定输出三源才进入完整主表，否则在同一 1–2 源子集重评各方法。推理计时包含球面 SRP-PHAT。

## 7. SRP-DNN（共享因果 CRNN → 混合 DP-IPD → IDL）

1. **移植作者实现的关键路径。** 复用 `ModelSRPDNN.py::CRNN`、`LearnerSRPDNN.py` 的幅度和相位前端、`Module.py` 的 DP-IPD 模板及 `SourceDetectLocalize`。固定三麦的三个无序麦对并统一对内通道次序。首轮直接读取共享的 48 kHz、16,000 点波形。用 `win=1536`、`hop=768`、`n_fft=1536`、`center=False`，保留作者的 32 ms 窗、16 ms hop 和 31.25 Hz 频点间隔。归一化使用 0–8 kHz 对应频带，随后显式取 `k=1…256`（31.25–8000 Hz），维持原 CRNN 每麦对 256 个输入频点和 512 维输出。约 333 ms 有 19 个完整 STFT 帧。开训前用固定样本核对 12 倍时域降采样后实际输出个数及覆盖区间。若输出未对应原标签时刻，先处理片段内对齐，再考虑调整时间池化并记录该适配。`max_num_sources=3` 同时设置 CRNN 输出幅度上限和 IDL 迭代上限。DP-IPD 监督和 IDL 模板都使用同一组 `k×48000/1536` 物理频率及三麦坐标。方位搜索在 0°–180°，评测只比较方位。若另做 48 kHz 全频带版本，需把 GRU 输入与回归头改为匹配 768 个频点，并单独标注。[作者 CRNN](https://github.com/BingYang-20/SRP-DNN/blob/main/code/ModelSRPDNN.py) · [前端与监督](https://github.com/BingYang-20/SRP-DNN/blob/main/code/LearnerSRPDNN.py) · [模板与 IDL](https://github.com/BingYang-20/SRP-DNN/blob/main/code/Module.py)
2. **按麦对构造回归目标。** 每个输出时刻，把该麦对上每个源的 DP-IPD 模板实虚部按活动率 `β` 加权求和，张量整理为 `(B,T_out,2F,P)`，`P=3`，以 MSE 训练。`β` 优先从同一混音中的逐源活动掩码按输出感受区间聚合。若训练数据只有片段级 DOA，在片段级目标上把标注有效源的 `β` 置 1，静音目标为零，并在结果标为片段活动近似。训练与推理取同一参考时刻的预测，验证时间池化后选中的输出覆盖标签对应区间。[SRP-DNN 论文式 8–12](https://arxiv.org/pdf/2202.07859#page=3)
3. **保留 IDL。** 对预测混合 DP-IPD 做模板匹配，取峰并减去该候选方向的加权模板，重复最多三次。首轮沿用作者 `IDL + unkNum` 的候选方向及活动分数，在验证集校准绝对分数阈值。作者代码可一次生成三轮候选，再按活动分数过滤；如果改为论文 Algorithm 1 的阈下立即停止，记录该选择并按顺序截断。将预测 DP-IPD 或 IDL 候选和分数缓存后，用公共计分器扫阈值。[作者 IDL 实现](https://github.com/BingYang-20/SRP-DNN/blob/main/code/Module.py) · [论文 Algorithm 1](https://arxiv.org/pdf/2202.07859#page=3)
4. **验收。** 先将已知方向合成的理想 DP-IPD 直接送给 IDL，检查单源峰值、相近双源、分离的三源、静音停止、麦对交换时模板符号一致。理想输入验收后再测 CRNN 小批量过拟合和 checkpoint 重载。开训 `srp_dnn_idl_s42`，用相同片段输出未知源数的角度集合和 PR 曲线。

## 8. 开训顺序与交付物

首跑的 TF-Mamba 沿用本项目 V5DecoderOnly 论文主版本的优化器和学习率配方，以便控制训练差异。IcoCNN 从 `AdamW`、学习率 `1e-3` 和全局梯度范数裁剪 `1.0` 起跑，其中后两项为本次建议初值。SRP-DNN 从作者的 `Adam` 和 `1e-3` 起跑。三条线使用同一训练样本预算。正式运行前若需调整学习率，在同一仿真验证集上用等额短跑预算选取并锁定。训练选择使用各方法原生监督下的验证结果，活动阈值另按第 2 节校准。[IcoCNN 训练设置](https://arxiv.org/html/2210.14536) · [SRP-DNN 作者训练选项](https://github.com/BingYang-20/SRP-DNN/blob/main/code/OptSRPDNN.py)

| 阶段 | 作业 | 完成条件 |
| --- | --- | --- |
| 0. 数据和接口 | 冻结数据清单与参考坐标，核对已有 SRP-PHAT 结果 | 三条线读到同一批音频、标签和几何。已有 SRP-PHAT 能按同一口径计分。检查目标数据的逐源活动标注，确定 SRP-DNN 的 `β` 路径 |
| 1. 最小验证 | 恢复 TF-Mamba，验证 IcoCNN 球面输入和 ACCDOA，验证 SRP-DNN 理想模板与 IDL | 通过各自的 0/1/2/3 源检查、前后向、小集过拟合及 checkpoint 往返。确认训练显存和每步耗时再设置 batch size，若调小 microbatch 则记录累积后的有效 batch |
| 2. 三条正式首跑 | `tf_mamba_spectrum_s42` → `icocnn_multi_s42` → `srp_dnn_idl_s42` | 统一由外部 runner 启动。TF-Mamba 直接调用项目 `Trainer`。IcoCNN、SRP-DNN 用各自的原生任务适配训练，三条线复用项目数据和评分组件。固定训练样本预算，记录模型实际优化器和学习率。先在仿真验证集选 checkpoint 和解码工作点 |
| 3. 统一汇总 | 锁定工作点后评 DICIT，汇总总体及按源数指标 | 每条线保存代码提交、有效配置、checkpoint、阈值、每片段输出、`pr_sweep.json` 或同口径 JSON、参数量与端到端耗时。按第 3 节填表。对支撑主要结论的模型补配对种子 |

模型、任务适配、启动配置和实验产物在独立基线项目内维护。基线侧适配器从显式指定的 LOCATA checkout 读取训练与评测组件。项目内保留实验计划及最终结果引用。
