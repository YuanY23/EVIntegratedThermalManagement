# 参数辨识与 V&V

## 定位

本模块把“参数从哪里来、能否被数据辨识、在独立工况上是否有效”纳入同一条证据链。当前仓库没有车型台架或实车原始数据，因此现阶段结果只属于 **synthetic truth recovery（合成真值回辨）与方法验证**，不属于特定车型标定，也不属于真实数据模型确认。

## 参数治理

`configs/parameter_registry.yaml` 是参数元数据的唯一注册表。每项参数包含：

- 名称与适用模型；
- SI 单位、默认值和物理/工程边界；
- 参数来源与可信等级；
- 是否允许辨识，以及对应的 `default_config.yaml` 路径（如适用）。

当前首批可辨识参数为电池核心热容、表面热容和核心—表面热导。液路阻力、泵曲线和换热器 UA 已先注册，但要在一维液路网络接入整车后再做联合辨识，避免在即将被替换的固定阻力模型上得到失效参数。

`IntegratedSimulator` 保持原有调用兼容，同时允许注入注册名称组成的电池参数覆盖集；未覆盖值继续来自类型化项目配置。

## 观测数据合同

CSV 必须包含一个 `dataset_id` 和一个 maturity 标签（`synthetic` 或 `measured`），并提供 episode、时间、产热、冷却液温度、冷却 UA、核心/表面温度及各温度测量标准差。每个 episode 内时间严格递增，数值必须有限，测量不确定度必须为正。

训练 episode 与 holdout episode 通过 `ObservationDataset.subset()` 显式分离；验证集不参与拟合。外部台架数据可以复用同一合同，但只有数据来源、传感器精度、同步与工况覆盖完成审计后，才能把 maturity 标为 `measured`。

## 辨识方法与诊断

模型沿用电池双节点热平衡：

```text
C_core * dT_core/dt = Q_gen - G_cs * (T_core - T_surface)
C_surface * dT_surface/dt = G_cs * (T_core - T_surface)
                         - UA_coolant * (T_surface - T_coolant) + Q_external
```

目标函数是按测量标准差归一化后的核心与表面温度残差平方和，使用有界 Powell 优化；所有候选值均受注册表边界约束。报告同时给出：

- 参数估计、近似标准误差与 95% 区间；
- 残差时序、优化调用次数和代价值；
- 数值 Jacobian 秩、缩放条件数和触边参数；
- 观测不足、秩亏、病态或触边时的失败诊断。

这一区间反映当前灰箱目标函数附近的局部统计不确定度，不应替代台架重复性、传感器系统误差和模型结构误差评估。

## 四层 V&V 口径

1. **代码验证**：单元测试检查边界、单位、序列化、状态方程方向与失败路径。
2. **数值验证**：固定步长离散、确定性种子、守恒量与求解诊断检查。
3. **参数辨识验证**：用隐藏真值生成带噪训练数据，确认受界算法可回辨，并在未参与拟合的 episode 上降低 RMSE。
4. **模型确认**：必须使用独立的真实台架/实车数据；当前未完成，报告中的 `model_confirmation` 固定为 `false`。

## 灵敏度与不确定性

局部分析使用参数中心差分弹性，输出对峰值核心温度、峰值表面温度和冷却液排热量的归一化影响。全局分析在注册边界内做固定种子的均匀抽样，输出 Spearman 排序和 P05/P50/P95 指标区间。接入一维液路后，同一接口将扩展至泵耗、支路流量和压降指标。

## 运行

```powershell
D:\anaconda\python.exe experiments\run_parameter_identification.py
D:\anaconda\python.exe -m pytest -q tests/test_calibration_parameters.py tests/test_parameter_identification.py tests/test_validation_and_sensitivity.py
```

输出位于 `results/calibration`，其中 `maturity_statement.json` 是结论边界的机器可读声明；参数表、残差、holdout 指标以及局部/全局灵敏度表可直接用于项目汇报和面试说明。
