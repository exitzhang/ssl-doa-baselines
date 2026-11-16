# IPDnet 与 IPDnet2-Small 审查整改方案

本方案对应 `ssl-doa-baselines/feat/external-baselines@ee72073` 的四项审查发现。修改落在 `ssl-doa-baselines`，评估继续调用已固定版本的 LOCATA 计分接口。 [审查版本](https://github.com/exitzhang/ssl-doa-baselines/tree/ee720734bdbf4dbabba88dd53c7d2fa252f2a23e)

| 优先级 | 问题 | 修改完成后 |
| --- | --- | --- |
| P1 | DICIT 直接评估可以在测试集选阈值 | 正式工作点只使用同一 checkpoint 在仿真验证集上锁定的阈值 |
| P2 | 加载 checkpoint 后，评估数据的麦序和麦对几何未校验 | 错麦序或错几何在前向前报错 |
| P2 | IPDnet2 独立预测默认用 0 作为负 MSE 阈值 | 使用显式提供或由配套验证文件读取的阈值 |
| P3 | IPDnet 参数量手填有误，主表计时范围混用 | 从实际重建的模型数参数，同列采用相同端到端计时范围 |

## 1. 锁住 DICIT 评估阈值

**修改 `integrations/locata/run.py`。**

- 提取用于写入验证结果的流程，供 `cmd_train()` 和单独执行的 `eval --split val --output /path/to/validation.json` 共用。以仿真验证集最大 F1@10° 选阈值，并列取较低 FAR。保存 `locked.key`、数值阈值、`score_kind`、分数比较方向、模型键、checkpoint 的 SHA-256 和选点角度容差。使用 checkpoint 内容摘要作为身份，复制或移动文件后仍能匹配。
- 在 `cmd_eval()` 中对这两个模型的 DICIT 评估，在加载数据和前向前检查 `--validation`，并核对记录中的模型键、checkpoint 摘要、分数类型、分数方向和选点容差。缺文件、字段不全或身份不一致时，提示用该 checkpoint 单独跑仿真验证，生成新的验证文件。
- DICIT 保留完整测试阈值曲线。把已锁定的数值阈值并入本次评估的阈值网格，避免配置改过扫描点或 JSON 键格式变化时丢失该点。单点 `metrics` 取这个阈值在 DICIT 上的结果，`working_point` 保留验证来源。测试曲线只用于曲线指标，含 `R@FAR 3%`。

**新增 `src/ssl_doa_baselines/working_point.py`。** 只存放 checkpoint 摘要计算、验证文件读取及身份校验，供 runner 和独立 CLI 共用。阈值扫描和 F1 选点继续放在基线适配器。

**复用 `integrations/locata/train_then_eval.py`。** 该 wrapper 已将训练生成的 `validation.json` 传给 DICIT 评估。核对现有队列入口继续使用这一文件路径。 [现有阈值分支](https://github.com/exitzhang/ssl-doa-baselines/blob/ee720734bdbf4dbabba88dd53c7d2fa252f2a23e/integrations/locata/run.py#L449) · [队列 wrapper](https://github.com/exitzhang/ssl-doa-baselines/blob/ee720734bdbf4dbabba88dd53c7d2fa252f2a23e/integrations/locata/train_then_eval.py#L63)

**验收。** DICIT 缺少验证文件时，在任何前向前失败。使用另一模型或同模型另一 epoch 的验证文件时失败。用两个小型评分样例刻意设置“验证最佳阈值”和“测试最佳阈值”不同，确认单点指标使用前者。同时核对测试曲线仍包含两处工作点。

## 2. 校验固定阵列、麦序和实际输入

**修改 `integrations/locata/adapter.py` 与 `integrations/locata/run.py`。**

- 在适配器新增统一的数据契约检查。读取实际将传进 DataLoader 的原始 `mic_indices`，检查数量、非负、互异与边界。三麦这轮默认 `[5,6,8]`，顺序按训练配置固定。使用同一个顺序切片米制坐标。
- `cmd_train()` 在启动优化前，对比 model、训练集与验证集的选中阵列。用 `p_other - p_ref` 比较有方向的参考麦对向量，容许浮点容差及相同阵列整体平移。参考麦由模型配置确定。将 `mic_indices`、`mic_positions` 和 `ref_mic` 写入 checkpoint 可追溯的配置及运行记录。现有 `train_native()` 的导出 checkpoint 已保存 `config["data"]`，直接复用其中的 `mic_indices`，并通过现有 `model_config` 读取坐标和参考麦。
- 在 `cmd_eval()` 复用同一检查，对比 checkpoint 的通道顺序与本次 loader 的通道顺序，再比对用于该数据集的参考麦对向量。例如 checkpoint 为 `[5,6,8]` 时，拒绝 `[8,6,5]`。DICIT 数据没有坐标元数据时，使用当前对照已确认的几何来源，记录来源和选中的坐标。继续检查 WAV 头的原始采样率、片段长度和通道数。
- 评估读取 checkpoint 自带的有效配置作为模型依据，`eval_dicit.json` 也记录实际加载的模型几何与实际 loader 麦序。若当前 API 只返回模型实例，可让 runner 通过一次 checkpoint 加载同时取得随包元数据，沿用现有仅返回模型的公开 API 调用方式。 [现有模型重建](https://github.com/exitzhang/ssl-doa-baselines/blob/ee720734bdbf4dbabba88dd53c7d2fa252f2a23e/src/ssl_doa_baselines/api.py#L128) · [麦坐标解析](https://github.com/exitzhang/ssl-doa-baselines/blob/ee720734bdbf4dbabba88dd53c7d2fa252f2a23e/integrations/locata/run.py#L49) · [现有契约检查](https://github.com/exitzhang/ssl-doa-baselines/blob/ee720734bdbf4dbabba88dd53c7d2fa252f2a23e/integrations/locata/adapter.py#L176)

**验收。** 同麦序和坐标通过。交换第一、第三支麦克风后拒绝。坐标同时平移时通过，改变一对麦克风间距时拒绝。先以合成小样例验证，再对冻结的 train、val 和 DICIT 清单执行一次开跑前检查。

## 3. 明确独立推理的活动阈值

**修改 `src/ssl_doa_baselines/api.py`。** 对 `ipdnet` 和 `ipdnet2_small` 的 `predict()` 路径，要求 `decode["threshold_abs"]` 为明确、有限的数值。若缺少，在模型前向前提示传入验证阈值。按模型的活动分数类型校验输入。IPDnet 使用投影比值，IPDnet2-Small 使用负 MSE，`-0.4` 表示接受 MSE 不大于 0.4 的轨道。 `predict_native()` 继续提供无需阈值的原生输出。 [现有预测分支](https://github.com/exitzhang/ssl-doa-baselines/blob/ee720734bdbf4dbabba88dd53c7d2fa252f2a23e/src/ssl_doa_baselines/api.py#L220)

**修改 `src/ssl_doa_baselines/cli.py` 与 `README.md`。** 独立 CLI 继续支持 `--threshold-abs=-0.4` 这类显式参数，再补 `--validation PATH` 读取已锁定点。复用第 1 节的 checkpoint 身份与分数类型校验，避免 runner 和 CLI 各维护一套格式。两种来源同时传入时提示用户只保留一个。README 给出用配套验证文件预测的例子。 [现有独立 CLI](https://github.com/exitzhang/ssl-doa-baselines/blob/ee720734bdbf4dbabba88dd53c7d2fa252f2a23e/src/ssl_doa_baselines/cli.py#L62)

**验收。** 构造已知角度的理想 DP-IPD，并加入少量偏差，使最小 MSE 大于零。显式负阈值和配套验证文件得到同一批检测结果。缺阈值时给出可操作的错误，错配验证文件时拒绝。额外检查分数等于阈值时与 `sweep_candidates()` 的 `>=` 规则一致。

## 4. 修正参数量与计时

**修改 `integrations/locata/adapter.py` 与 `integrations/locata/run.py`。**

- 从实际重建并加载的模型计算 `sum(p.numel() for p in model.parameters())`，以整数 `num_params` 写进评估 JSON。按当前 IPDnet 的隐藏维 128、三麦三轨配置，静态预期是 **725,248**。正式表值从评估产物提取，IPDnet2-Small 亦采用同一规则。
- 新增口径明确的 `inference_ms_per_clip`。统一计时从设备上的波形开始，到锁定阈值下生成每段方位集合结束，包含方法自有的前端、骨干、模板匹配和筛选。所有方法在同一环境使用相同的 batch、预热、GPU 同步和重复次数，按实际片段数归一化。公共评分、读盘和完整阈值扫描另计。保留已有前端、骨干、模板分项便于定位耗时。

**修改 `RESULTS.md`。** 将 IPDnet 的 0.35 M 改为待核实，或从产物填约 0.73 M。主表只在各方法补齐相同计时口径后填统一耗时列。已有 `Forward ms` 可保留在历史记录，并标注测量范围。 [IPDnet 主干](https://github.com/exitzhang/ssl-doa-baselines/blob/ee720734bdbf4dbabba88dd53c7d2fa252f2a23e/src/ssl_doa_baselines/ipdnet/backbone.py#L55) · [分段计时](https://github.com/exitzhang/ssl-doa-baselines/blob/ee720734bdbf4dbabba88dd53c7d2fa252f2a23e/integrations/locata/adapter.py#L538) · [现有主表](https://github.com/exitzhang/ssl-doa-baselines/blob/ee720734bdbf4dbabba88dd53c7d2fa252f2a23e/RESULTS.md#L70)

**验收。** checkpoint 重载后的参数量与训练启动日志相符。用含不足整批的测试集核对耗时分母确为累计片段数。检查 IPDnet 的前端及模板匹配均落在端到端计时范围内。

## 5. 落地顺序

1. 先改验证阈值绑定和麦阵契约。在 `tests/test_ipdnet_integration.py` 增加缺验证文件、跨 checkpoint 错配、麦序交换和阵列整体平移的回归用例。
2. 再改独立预测和结果记录，补显式阈值、负 MSE 比较边界、checkpoint 重载参数量的验收。运行两个模型的 GPU smoke 和小批原生 train/val/eval。
3. 对已产出的 checkpoint 核实训练数据和阵列记录。记录齐全时，直接补跑仿真验证生成新版阈值文件，再重评 DICIT 和基准耗时。只有发现训练时使用了错误的阵列或通道顺序，才需要重训受影响的模型。
