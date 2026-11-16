# IPDnet 与 IPDnet2-Small 官方实现接入方案

在 `ssl-doa-baselines` 的主表和复现列表新增两行，从作者代码移植固定阵列模型，在 LOCATA 的冻结数据上从随机初始化重训。两行都回归**逐源 DP-IPD**，每个片段输出一组方位，覆盖 1、2、3 源。表名用 **IPDnet（固定三麦，DP-IPD）** 和 **IPDnet2-Small（固定三麦，DP-IPD）**。旧 Box 实验用于核对移植中出现过的坐标、频率和空轨问题。论文原始数据、阵列及逐帧结果可另作实验，本方案的验收目标是同一 LOCATA 协议下的基线。 [IPDnet 论文](https://arxiv.org/html/2405.07021) · [IPDnet2 论文](https://arxiv.org/html/2509.21900v1) · [作者实现](https://github.com/Audio-WestlakeU/FN-SSL/tree/76fcb281be92caf068c712dfb015e354f437260f)

## 1. 两仓接入点

| 仓库 | 核对到的现状 | 本方案的落点 |
| --- | --- | --- |
| [`ssl-doa-baselines`](https://github.com/exitzhang/ssl-doa-baselines/tree/d5ddc6ebdde344e5535bd864c10767ac314b24e7) | `feat/external-baselines` 为 `d5ddc6e`。已有 TF-Mamba、IcoCNN、SRP-DNN 的 API、LOCATA runner、配置和结果列表 | 模型、标签、解码、可选依赖、复现配置、队列任务和结果行都放此仓库 |
| [`LOCATA` 实验固定提交](https://github.com/exitzhang/LOCATA/tree/d5e7c5d8ebea3d1356eb937e2f56ebeff3251bde) | 已完成的外部基线使用此提交，包含适配器要求的 `engine/sweep_eval.py` 和原有包级导出 | 沿用已有 detached worktree，通过 `--locata-root` 注入数据与计分 |
| [`LOCATA/dev`](https://github.com/exitzhang/LOCATA/tree/78764a88137ccc31f1168309b9e82f97e53a04d5) | 当前查到 `78764a8`，评测已迁入 `engine/evaluation/`，现有基线适配器的路径检查和导入会失败 | 升级时在基线侧迁移适配器，并用冻结的 GT/预测对拍旧分数、静音和超容差样例 |

具体复用 `adapter.build_angle_dataset()`、`build_dicit_loader()`、`train_native()` 和公共一对一匹配。两种模型都实现现有的 `model.config`、`model.model_name`、`forward(audio)` 和 `training_loss(audio, targets)` 契约。作者的 Lightning 入口供核对训练逻辑，LOCATA 作业由现有 runner 组装。SRP-DNN 已有的远场相位公式和模板构造可对照复用，SRP-DNN 的混合目标和 IDL 属于该方法自身。 [现有适配器](https://github.com/exitzhang/ssl-doa-baselines/blob/d5ddc6ebdde344e5535bd864c10767ac314b24e7/integrations/locata/adapter.py) · [原生训练循环](https://github.com/exitzhang/ssl-doa-baselines/blob/d5ddc6ebdde344e5535bd864c10767ac314b24e7/src/ssl_doa_baselines/training.py) · [既有结果与 LOCATA 固定提交](https://github.com/exitzhang/ssl-doa-baselines/blob/d5ddc6ebdde344e5535bd864c10767ac314b24e7/RESULTS.md)

## 2. 移植作者实现

以 `Audio-WestlakeU/FN-SSL@76fcb281be92caf068c712dfb015e354f437260f` 为来源。使用包内相对导入，把两套代码中泛用的 `Module`、`utils_` 和 `arch` 名称限制在各自包内。移植文件、原提交、改动和许可记录进 `THIRD_PARTY.md`。作者根 README 标注 MIT，该提交的根目录未查到独立 `LICENSE` 文件。入库时随源码核对并保留适用的版权和许可声明。 [上游 README](https://github.com/Audio-WestlakeU/FN-SSL/blob/76fcb281be92caf068c712dfb015e354f437260f/README.md) · [基线仓库来源记录](https://github.com/exitzhang/ssl-doa-baselines/blob/d5ddc6ebdde344e5535bd864c10767ac314b24e7/THIRD_PARTY.md)

| 项目 | IPDnet | IPDnet2-Small |
| --- | --- | --- |
| 作者入口 | `IPDnet/FixedAarryIPDnet.py::IPDnet`，参考 `runIPDnetOn.py` 的前端、PIT 和标签 | `IPDnet2/IPDnet2.py::OnlineSpatialNet` 及依赖的 `arch/base` 模块，参考 `run_IPDnet2.py` |
| 主干 | 两个 FN block，沿频率 BLSTM、沿时间单向 LSTM，隐藏维以公开在线配置的 128 起步。保留因果 CNN 和内部 3×4 时间池化 | 8 层、隐藏维 96、`mamba(16,4)`，首层内 16 倍频率压缩，首层后 5 倍时间池化，保留频率恢复模块 |
| 三麦三轨 | `input_size=6, max_track=3, is_online=True`，卷积输出 12 个分量 | `dim_input=6, dim_output=12`，沿用 Small 的时频配置 |
| 必须改的输出整理 | 公开代码的部分 `reshape` 按二分量和默认实验布局写定。显式用 `F`、麦对数 `P` 和轨数 `K` 重排 | 同样把末端写死的 `F*2` 整理改成明确的频率、实虚部、麦对和轨轴 |
| 活动判决 | 逐轨选投影最大的方位，分数为与最佳模板的投影比值 | 逐轨选模板 MSE 最小的方位，公共候选缓存使用 `score = -min_mse`，便于复用“分数大于等于阈值”的逻辑 |

两种主干的帧输出统一为 `(B,T_out,2F,P,K)`，其中 `P=2`、`K=3`，频率轴先排全部实部，再排全部虚部。12 个输出分量按 `(实虚部, 麦对, 轨)` 定义。用可识别的索引值逐项验证 `reshape` 和 `permute`，仅检查形状容易漏掉轴序错误。IPDnet2 可复用项目已验证的 Mamba 环境，安装时只为 `ipdnet2` 增加可选依赖，IPDnet 不加载 Mamba。 [IPDnet 固定阵列模型](https://github.com/Audio-WestlakeU/FN-SSL/blob/76fcb281be92caf068c712dfb015e354f437260f/IPDnet/FixedAarryIPDnet.py#L76) · [IPDnet2 骨干](https://github.com/Audio-WestlakeU/FN-SSL/blob/76fcb281be92caf068c712dfb015e354f437260f/IPDnet2/IPDnet2.py#L259) · [IPDnet2-Small 配置](https://github.com/Audio-WestlakeU/FN-SSL/blob/76fcb281be92caf068c712dfb015e354f437260f/IPDnet2/run_IPDnet2.py#L103)

## 3. LOCATA 任务适配

### 前端和片段聚合

- 共用 **48 kHz、16,000 点**原始波形和 `[5,6,8]` 通道顺序。两套模型内部先对三通道使用同一个带抗混叠的重采样器降到 **16 kHz**，再做 **512 点 Hann 窗、512 点 FFT**、`center=False`。这是为了保留作者的 32 ms 窗和 31.25 Hz 间隔。现有 TF-Mamba-512 的 `48 kHz / hop 480` 是其独立配方。两种 IPDnet 的数据划分和音频片段仍可直接与该实验对齐。 [TF-Mamba-512 配方](https://github.com/exitzhang/ssl-doa-baselines/blob/d5ddc6ebdde344e5535bd864c10767ac314b24e7/integrations/locata/configs/tf_mamba_dicit_512.toml) · [Torchaudio 有限带宽重采样](https://docs.pytorch.org/audio/stable/generated/torchaudio.transforms.Resample.html)
- IPDnet 使用 hop 256（16 ms），IPDnet2 使用 hop 320（20 ms）。先用 257 个单边频点的幅度估计 forgetting norm，分别用作者的 280、249 帧参数，再取 `k=1…256` 的实虚部。STFT 特征、DP-IPD 标签和解码模板共用 `f_k=k×16000/512`，范围为 31.25–8,000 Hz。DICIT 开跑前检查原始 WAV 的采样率、长度和片段清单。
- 现有 `angle` 接口给每段一组角度。检查冻结数据标签承诺的是片段方位集合还是明确的参考时刻。已查的 DICIT 导出按源在片段内的 VAD 覆盖率筛选声源，并复用传入的声源角度，所核对的代码没有逐输出时刻重新取角。因此主方案在两个官方主干外使用**固定片段聚合**，再用片段级 PIT。训练和推理共用同一个聚合函数。
- 16,000 个输入点重采样后为 5,334 点。STFT 右端补零完成最后一窗，IPDnet 补 42 点，IPDnet2 补 298 点。两套内部池化均保留末端不足整组的输出。预期 IPDnet 得到 **20 个 STFT 帧、2 个池化输出**，IPDnet2 得到 **17 帧、4 个池化输出**。按池化时间步代表的 STFT 帧数加权，权重分别为 `[12,8]/20` 和 `[5,5,5,2]/17`。开训前验证实际帧数和尾部信号覆盖。池化和填充是短片段适配，保留作者主干和池化倍率。
- PIT 在聚合后执行，保证同一片段的轨道分工由同一个目标约束。若以后提供可靠的逐源逐帧 DOA/VAD，再开单独的逐帧复现实验，按输出时间重采样标签并逐帧 PIT。该版本评估时按时刻评分，跨帧合并前先处理换轨。 [IPDnet 前端及逐帧 PIT](https://github.com/Audio-WestlakeU/FN-SSL/blob/76fcb281be92caf068c712dfb015e354f437260f/IPDnet/runIPDnetOn.py#L188) · [DICIT 片段导出](https://github.com/exitzhang/LOCATA/blob/d5e7c5d8ebea3d1356eb937e2f56ebeff3251bde/locata_dicit_dataset/audio_frame_saver.py#L138)

### 有效轨、空轨和模板

1. **固定参考麦。** 参考所选第一个通道，麦对按 `(5,6),(5,8)` 排列。读元数据的米制坐标，保留麦对方向。仿真水平仰角转天顶角最多一次，先复用适配器的角度约定。主实验的方位模板取天顶角 90°，搜索 0°–180°，解码直接返回同一个 181 点角度数组。DICIT 方位修正由现有 loader 负责，模型接入层保留它的输出。
2. **有源轨。** 按连续标注方位用同一套远场几何生成 `cos(2πfτ), sin(2πfτ)`，有源标签避免先量化到搜索网格。IPDnet2 原训练入口的 `DPIPD2` 需要距离，作者推理模板则使用远场。现有角度接口没有提供距离，首轮两种模型都用水平远场目标，表注注明这个适配。如果后续数据能提供可靠位置和距离，再单独比较有源近场标签。用已知方向的有符号通道延迟验证相位符号，另外检查标签与模板的一致性。
3. **空轨与损失。** 将每段的 `N` 个有效源放入 `K` 个槽位，余下用 `J₀(2πfd/c)+0j`，`d` 为参考麦对的间距，频率与有源轨一致。声速 `c` 取作者入口的 340 m/s。将所有轨的实虚部、频率和麦对纳入 MSE，用最小代价排列后求平均。全静音时三轨均为空轨。指定三轨配置时遇到四源数据应当由冻结清单排除或显式报错。DP-IPD 误差诊断先做 PIT 对齐，再分别报告有效轨误差和空轨误差。
4. **逐轨解码。** 保留每轨最佳候选及活动分数，缓存后再做阈值扫描。第一版关闭跨轨角度抑制，保留对近角多源的检测能力。真值源数只进入 PIT 目标、评测匹配和分组。IPDnet2 阈值网格在缓存层用负 MSE，产物同时记原始 MSE 阈值和分数方向。例如 `-0.4` 对应 `MSE ≤ 0.4`，正式工作点由验证集选出。 [IPDnet 空轨和活动标签](https://github.com/Audio-WestlakeU/FN-SSL/blob/76fcb281be92caf068c712dfb015e354f437260f/IPDnet/runIPDnetOn.py#L208) · [IPDnet2 近场目标入口](https://github.com/Audio-WestlakeU/FN-SSL/blob/76fcb281be92caf068c712dfb015e354f437260f/IPDnet2/run_IPDnet2.py#L264) · [IPDnet2 模板解码](https://github.com/Audio-WestlakeU/FN-SSL/blob/76fcb281be92caf068c712dfb015e354f437260f/IPDnet2/Module.py#L595)

## 4. 逐文件修改清单

以下路径均相对于 `ssl-doa-baselines`。两套模型共享前端、物理模板及 PIT。LOCATA 侧直接使用已有固定 worktree 的数据和计分接口。

### 4.1 新增模型和共用模块

| 文件 | 具体修改 |
| --- | --- |
| `src/ssl_doa_baselines/dpipd_frontend.py` | 新增 `DPIPDFrontend`。接收 `(B,T,C)` 的 48 kHz 波形，转置成 `(B,C,T)`，用 `torchaudio.transforms.Resample(48000,16000)` 对整个张量重采样并缓存滤波器。将重采样、FFT 和 forgetting norm 置于 FP32。STFT 前补齐最后一窗，用 `torch.stft(..., n_fft=512, win_length=512, center=False)` 一次处理全部通道。用包含 DC 的幅度归一化，再去掉 DC，并按 `[real_mics, imag_mics]` 拼成 `(B,6,256,T_stft)`。将 FFT 正频点的 `bins` 与 `freqs_hz` 暴露给模板生成器。 |
| `src/ssl_doa_baselines/dpipd_tracks.py` | 新增 `TrackDPIPD`。接收三麦坐标、参考麦、`freqs_hz` 和轨数。复用现有 `geometry.angles_to_vectors` 的角度约定，构造有源标签、`J₀` 空轨及 `(181,512,2)` 候选模板。模型、目标和模板以 `(2F,P,K)` 为共同约定，先实部后虚部。对 `K≤3` 预先列出槽位排列，取每片段的最小 MSE，使用同一排列对齐诊断指标。提供 `pool_clip(frame_ipd, num_stft_frames, pool_factor)` 和 `decode_candidates(clip_ipd, score_kind)`。解码返回未阈值化的 `(score,az_deg,el_deg)`，每轨一项，按分数排序。 |
| `src/ssl_doa_baselines/ipdnet/__init__.py` | 建立子包，导出本方法包装类 `IPDnetBaseline`。 |
| `src/ssl_doa_baselines/ipdnet/backbone.py` | 从 `FixedAarryIPDnet.py` 移入 FN block、因果 CNN 和固定阵列在线主干。把 `input_size` 和 `max_track` 设为 6 和 3。3×4 池化启用末组保留，移除 `nt//12` 对输出长度的假设。读取卷积实际输出的 `T_out`，按 `(B,T_out,F,2,P,K)` 排列后合并实虚部与频率轴。保留频率 BLSTM、时间单向 LSTM、原卷积和跳连。 |
| `src/ssl_doa_baselines/ipdnet/model.py` | 构造共用前端（hop 256、forgetting norm 280）、IPDnet 主干、逐轨目标及解码。设置 `model_name="ipdnet"`。`forward()` 返回 `{"frame_ipd": (B,T_out,512,2,3), "ipd": (B,512,2,3)}`。`training_loss(audio, targets)` 由角度清单生成目标，对聚合后的 `ipd` 计算 PIT-MSE。解码调用共用模块的投影比值分支。 |
| `src/ssl_doa_baselines/ipdnet2/__init__.py` | 建立子包，导出 `IPDnet2SmallBaseline`。只在选用该模型时加载其包。 |
| `src/ssl_doa_baselines/ipdnet2/backbone.py` | 移入作者 `OnlineSpatialNet`、`SpatialNetLayer`、因果卷积和频率恢复。采用 Small 配置及 6 入、12 出。把 `arch.base.*` 改为包内相对导入。去掉逐次 shape 打印，Mamba 导入失败时给出依赖错误。5 倍时间池化保留末组，池化后根据实际长度更新时序掩码。恢复 256 个频点后显式整理到 `(B,T_out,512,2,3)`。 |
| `src/ssl_doa_baselines/ipdnet2/arch/__init__.py` | 建立内部子包。 |
| `src/ssl_doa_baselines/ipdnet2/arch/base/__init__.py` | 建立基础组件子包。 |
| `src/ssl_doa_baselines/ipdnet2/arch/base/linear_group.py` | 从官方同名文件移入分组线性实现，维持作者计算。 |
| `src/ssl_doa_baselines/ipdnet2/arch/base/non_linear.py` | 移入作者激活工厂及主干用到的激活。 |
| `src/ssl_doa_baselines/ipdnet2/arch/base/norm.py` | 移入作者归一化工厂及层实现。 |
| `src/ssl_doa_baselines/ipdnet2/arch/base/retention.py` | 保留官方主干模块导入的 Retention 类型和实现，与其他方法隔离。 |
| `src/ssl_doa_baselines/ipdnet2/model.py` | 用同一包装层契约构造前端（hop 320、forgetting norm 249）和 Small 主干，设置 `model_name="ipdnet2_small"`。配置中的聚合倍率为 5。训练用片段 PIT-MSE，解码调用最小 MSE 分支，生成负 MSE 分数。 |

模型构造参数区分 `input_sample_rate=48000` 与 `stft_sample_rate=16000`。原生输出名中的 `ipd` 指聚合后的逐轨张量，`frame_ipd` 用于排查帧对齐和中间结果。两个包装类均将输入采样率、STFT 参数、重采样器参数、麦坐标和麦序、轨数、声速、池化和分数类型写入 `model.config`，继续使用已有的 checkpoint 序列化。

`dpipd_tracks.py` 的投影分支使用 `dot(q,p)/dot(q,q)`，其中 `q` 是候选模板，`p` 是预测轨。MSE 分支在每条轨、每个候选上对全部 `2F×P` 个分量求平均，取最小值及它的候选下标。向共享缓存暴露的分数为该最小值取负，两条路径均从同一个实际网格读取角度。 `pool_clip()` 用 `min(pool_factor, num_stft_frames - i*pool_factor)` 计算第 `i` 组权重，以实际 STFT 帧数归一化。 [当前模型 API](https://github.com/exitzhang/ssl-doa-baselines/blob/d5ddc6ebdde344e5535bd864c10767ac314b24e7/src/ssl_doa_baselines/api.py) · [共享几何函数](https://github.com/exitzhang/ssl-doa-baselines/blob/d5ddc6ebdde344e5535bd864c10767ac314b24e7/src/ssl_doa_baselines/geometry.py)

### 4.2 修改现有入口

| 文件 | 具体修改 |
| --- | --- |
| `src/ssl_doa_baselines/api.py` | 在 `_MODEL_NAMES` 和 `build_model()` 加 `ipdnet`、`ipdnet2_small` 的延迟导入分支。将模型配置按各包装类构造参数传入。在 `predict()` 用 `out["ipd"]` 逐轨解码，按 `threshold_abs` 过滤后返回公共检测格式。IPDnet2 的阈值接受负数。现有 `predict_native()`、`save_checkpoint()` 和 `load_model()` 可原样复用。 |
| `src/ssl_doa_baselines/cli.py` | 给 `fit --model` 增加两个键。让 `_read_audio()` 可接收期望输入采样率，校验当前已读出但丢弃的 WAV 采样率。对两种新模型在 `cmd_predict()` 和 `ManifestDataset` 传入 48 kHz；重采样由模型执行一次。独立 `cmd_fit()` 把配置的 `scheduler_gamma` 传入训练循环。 |
| `src/ssl_doa_baselines/training.py` | 在 `fit()` 新增可选 `scheduler_gamma` 与 `scheduler="exponential"`。采用 `ExponentialLR(optimizer, gamma=0.975)`，每完成一个 epoch 后 step。现有 cosine / none 行为照常。继续按验证 loss 写 best，两个新模型提供各自的 `training_loss()`。 |
| `pyproject.toml` | 将两个名称加入简介，增加 `ipdnet` 和 `ipdnet2` 可选依赖组。两者依赖 torch 和与训练环境匹配的 torchaudio，`ipdnet2` 复用 TF-Mamba 的 Mamba 依赖配方。更新 `all`，在 README 记录实际验收的 torch/torchaudio/CUDA 组合。 |
| `integrations/locata/adapter.py` | 保留 `build_angle_dataset()` 和 `angle_batch_collate()` 的变长角度。为两种模型核对仿真元数据的原采样率与长度，以及 DICIT 评估清单对应 WAV 的 48 kHz 和 16,000 点契约。增加 `cache_candidates()` 的原生逐轨分支，直接消费片段 `out["ipd"]`，与 GT 按 loader 顺序写缓存，然后复用 `sweep_candidates()`、`decoded_at_threshold()` 和公共 matcher。 `train_native()` 传递 `scheduler_gamma`。对新增候选分支，计时用 GPU 同步区分前端及推理、模板匹配，样本 ID 用 loader 的实际顺序。 |
| `integrations/locata/run.py` | 在 `_model_from_cfg()` 对两键调用已有麦阵解析，校验通道顺序和配置一致。 `cmd_smoke()` 使用配置中的 16,000 点波形、检查两种原生输出，并用允许候选进入的阈值走解码与 checkpoint 回读。 `cmd_train()` 沿用非 TF-Mamba 的 train-native 分支。 `_evaluate()` 保留候选路径，确保 IPDnet2 负阈值使用同一配置。 `cmd_eval()` 对这两键在正式 DICIT 模式加载验证阈值，并核对模型、配置及分数类型。在产物里记录输入与 STFT 两个采样率、FFT、hop、阈值方向和原始 MSE 阈值。 |
| `integrations/locata/train_then_eval.py` | 直接复用现有 wrapper。队列命令传各自的配置，产物沿用 `checkpoint.pth`、`validation.json`、`eval_dicit.json`。 |

[现有 CLI](https://github.com/exitzhang/ssl-doa-baselines/blob/d5ddc6ebdde344e5535bd864c10767ac314b24e7/src/ssl_doa_baselines/cli.py) · [现有 runner](https://github.com/exitzhang/ssl-doa-baselines/blob/d5ddc6ebdde344e5535bd864c10767ac314b24e7/integrations/locata/run.py) · [现有适配器](https://github.com/exitzhang/ssl-doa-baselines/blob/d5ddc6ebdde344e5535bd864c10767ac314b24e7/integrations/locata/adapter.py)

### 4.3 配置、队列、文档和验收文件

| 文件 | 具体修改 |
| --- | --- |
| `integrations/locata/configs/ipdnet_dicit.toml` | 新增配置。 `[model]` 设 `name="ipdnet"`、`win_len=512`、`n_fft=512`、`hop=256`、`forgetting_frames=280`、`hidden_size=128`、`pool_factor=12`。 `[train]` 用 Adam 与 `weight_decay=0.0`。 `[decode]` 用 `score_kind="ratio"`，阈值起始网格为 `[0.0,0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]`。 |
| `integrations/locata/configs/ipdnet2_small_dicit.toml` | 新增配置。 `[model]` 设 `name="ipdnet2_small"`、`win_len=512`、`n_fft=512`、`hop=320`、`forgetting_frames=249`、`dim_hidden=96`、`num_layers=8`、`attention="mamba(16,4)"`、`freq_compression=16`、`time_compression=5`、`time_compression_layer=0`。 `[train]` 用 AdamW 与 `weight_decay=0.01`。 `[decode]` 用 `score_kind="neg_mse"`，阈值起始网格为 `[-1.0,-0.8,-0.6,-0.5,-0.45,-0.4,-0.35,-0.3,-0.25,-0.2,-0.15,-0.1,-0.05,0.0]`。 |
| `integrations/locata/configs/data_local.example.toml` | 补注释说明两种模型共用现有三套数据根目录和麦阵来源。正式路径和数据清单仍取实际完成对照的配置。 |
| `queue/ipdnet_512_s42.json` | 新增单作业，ID 为 `ipdnet_s42`。沿用现有前台 `train_then_eval.py` 入口，传固定 LOCATA worktree、IPDnet 配置、同一 data config、seed 42 和 `runs/ipdnet_s42`。CUDA 设备由实际队列分配。 |
| `queue/ipdnet2_small_512_s42.json` | 新增单作业，ID 为 `ipdnet2_small_s42`。改用 IPDnet2-Small 配置、已验收的 Mamba 环境和 `runs/ipdnet2_small_s42`。同样在一个队列槽完成训练、阈值锁定和评估。 |
| `README.md` | 新增两行模型列表、可选依赖与从头训练用法。原生输出写明逐轨 DP-IPD 和各自活动评分。 |
| `THIRD_PARTY.md` | 逐文件登记上游固定提交、转入路径和移植改动，记录官方 README 的许可标注及随源码保留的声明。 |
| `RESULTS.md` | 新增两行待跑记录及对应 checkpoint、产物路径。验收后填写 F1/FAR、MAE 和分组结果，表注记录 48→16 kHz、512 点 FFT 及片段级监督。 |
| `tests/test_dpipd_frontend.py` | 后续实施时加频点和总帧数检查。覆盖 48→16 kHz、双模型 hop、20/17 帧、2/4 组及末组权重。用已知通道延迟检查相位方向和重采样后的通道一致性。 |
| `tests/test_dpipd_tracks.py` | 后续加有效轨与空轨轴序、0/1/2/3 源 PIT、换序不变性，验证 0°/90°/180°、近角双源和静音分数。 |
| `tests/test_ipdnet_integration.py` | 后续核对模型注册、checkpoint 重载和两个模型的候选缓存及阈值方向。IPDnet2 的实际 Mamba 前后向放在 GPU smoke，普通安装可运行 IPDnet。 |

两份 TOML 共享 `input_sample_rate=48000`、`stft_sample_rate=16000`、`n_mics=3`、`max_num_sources=3`、`n_az=181`。数据段设 `target_num_samples=16000`、`mic_indices=[5,6,8]`、`max_sources=3`。解码段设 `acc_thresholds=[5,10,15]`、`source_column=10`、`far_target=0.03`。当前 SRP-DNN 配方是预算起点，训练/验证增强和实际划分从已完成对照的 `config.json` 核对。 [现有 SRP-DNN 配方](https://github.com/exitzhang/ssl-doa-baselines/blob/d5ddc6ebdde344e5535bd864c10767ac314b24e7/integrations/locata/configs/srp_dnn_dicit.toml) · [现有队列清单](https://github.com/exitzhang/ssl-doa-baselines/blob/d5ddc6ebdde344e5535bd864c10767ac314b24e7/queue/srp_dnn_2026-11-12.json)

IPDnet2-Small 其余结构取作者公开入口的 `num_heads=4`、`dim_squeeze=8`、`kernel_size=(5,3)`、`conv_groups=(8,8)`、`norms=["LN","LN","GN","LN","LN","LN"]`、`rope=False`。 `num_freqs=256` 由前端实际频点数得到。 [作者 Small 配置](https://github.com/Audio-WestlakeU/FN-SSL/blob/76fcb281be92caf068c712dfb015e354f437260f/IPDnet2/run_IPDnet2.py#L103)

首轮沿用现有外部基线的 **100 epoch 上限、6 次验证未改善的早停、seed 42**。以 SRP-DNN 配方的 batch 8 和每两 epoch 验证为共同起点，核对实际看到的样本数和优化步数。保留作者的 Adam / AdamW、起始 LR 0.0005 和指数衰减 0.975。现有 `training.fit()` 只支持 cosine 或无调度，需增加显式配置的 exponential 分支。显存不足时记录实际 batch 与有效步数，先验证梯度累计能否维持共享有效 batch。按仿真验证 PIT-MSE 选 checkpoint，按同一仿真验证集最大 F1@10°（并列选较低 FAR）锁定活动阈值。 [现有 SRP-DNN 配方](https://github.com/exitzhang/ssl-doa-baselines/blob/d5ddc6ebdde344e5535bd864c10767ac314b24e7/integrations/locata/configs/srp_dnn_dicit.toml) · [公共工作点逻辑](https://github.com/exitzhang/ssl-doa-baselines/blob/d5ddc6ebdde344e5535bd864c10767ac314b24e7/integrations/locata/adapter.py#L358)

落地后从基线仓库启动。两条正式作业在队列清单中调用同一个 wrapper，传各自的配置与运行名。

```bash
python integrations/locata/train_then_eval.py \
  --locata-root /path/to/wt/locata-dev \
  --config integrations/locata/configs/ipdnet_dicit.toml \
  --data-config /path/to/data_local.toml \
  --output runs/ipdnet_s42 --seed 42 --device-index 0
```

## 5. 验收与入表

1. **源代码与数值验收。** 无 LOCATA 的环境验证模型包可导入。与作者主干在相同输入和权重下对拍可对齐的中间量。校验 STFT 的频率轴、参考麦相位正负、训练和推理共用前端、模板 0°/90°/180°、Bessel 空轨频率与麦对顺序，以及输出时序末组。对理想 DP-IPD 检查静音、单源、近角双源与三源以及调换目标顺序的 PIT。
2. **训练验收。** 用少量 0/1/2/3 源片段做前后向和小集过拟合，检查三轨确实能输出三源。确认 IPDnet2 的 Mamba CUDA 路径、验证 loss、checkpoint 重载和计分链路。首轮先用 FP32 对拍，再启用 AMP 并检查数值。只在脚本通过这些检查后进入正式队列。
3. **报告验收。** DICIT 沿用冻结的公共测试片段，报告验证锁定点的 F1@10°、P/R、实际 FAR、匹配 MAE，以及 1/2/3 源分组。测试曲线单独报 R@FAR 3% 和同点 MAE，实测点夹住 3% 才插值。若引入实录阈值校准，按原始录音划分，在共同新协议下重评比较行。若测试集有静音片段，另报静音片段误报占比并保留公共 FAR 计数口径。计时覆盖前端、模型和原生模板匹配，GPU 预热并同步。
4. **结果记录。** 保存 `checkpoint.pth`、`validation.json`、`eval_dicit.json` 和 `config.json`，记录两仓提交、作者固定提交、环境、样本清单和三麦几何。任一模型通过三源验收且完成正式评估后，即可填写其主表行。若某模型只完成两源适配，在共同的 1–2 源子集报告。 [现有结果口径](https://github.com/exitzhang/ssl-doa-baselines/blob/d5ddc6ebdde344e5535bd864c10767ac314b24e7/RESULTS.md) · [固定提交的公共计分器](https://github.com/exitzhang/LOCATA/blob/d5e7c5d8ebea3d1356eb937e2f56ebeff3251bde/engine/sweep_eval.py) · [现有队列 wrapper](https://github.com/exitzhang/ssl-doa-baselines/blob/d5ddc6ebdde344e5535bd864c10767ac314b24e7/integrations/locata/train_then_eval.py)
