# 温度与SOC硬约束下的充电时间—预热能耗—相对老化三目标优化

## 为什么使用 Pareto，而不是直接加权原始量

到站预热存在至少四个互相冲突的结果：到站温度偏离理想区间、快充时间、预热附件能耗和相对老化损伤。本实现把温度和SOC作为硬约束及状态审计，把快充时间、预热附件能耗和相对老化作为三个Pareto目标。分钟、kWh和damage没有可直接相加的共同量纲，因此流程先做硬约束筛选，再求非支配集：

```text
candidate policy
  → same route + same fast-charge plant
  → SOC / temperature / power / termination audit
  → feasible candidates only
  → Pareto(time, preconditioning energy, aging)
```

温度不是被任意权重“买掉”的软目标：目标温度本身是决策变量，到站温度和核心峰值进入约束审计与结果着色。只有在 Pareto 集内选择一个展示用工程推荐点时，才对充电时间、预热能耗和相对老化分别做 0–1 归一化，并采用 0.45/0.20/0.35 的可见权重。权重不影响 Pareto 集，招聘答辩时可以直接展示不同偏好如何改变推荐点。

为避免数值噪声制造大量“几乎一样”的非支配点，前沿采用工程分辨率：1 min、0.1 kWh 和 `5e-6` 相对 damage；落在同一分辨单元的候选只保留稳定排序的一个代表点。

## 决策变量与约束

首版离线优化使用可解释的分段策略，而不是难以说明的逐秒黑箱控制：

- 距到站多少秒开始；
- 目标电池温度；
- 最大预热/预冷热功率。

候选必须满足：充电正常到达目标 SOC、到站 SOC 不低于 0.10、核心峰值不高于 50°C、路线功率闭合残差小于 `1e-6 W`。所有候选复用与无预热、规则预热完全相同的电池、充电接受功率、附件和老化模型。

## 基准、推荐与回退

`none` 和 `rule` 永远保留在候选表中。优化推荐点必须来自可行 Pareto 集；路线预览无效或路线取消时，优化不再搜索并确定性回退 `none`。若所有候选违反到站 SOC 或热安全约束，同样返回带原因的基准回退，而不是放松安全边界。

`joint_optimization_candidates.csv` 保存全部候选，`constraint_audit.csv` 保存每项硬约束，`joint_optimization_pareto.csv` 只含非支配解，`joint_optimization_recommended.csv` 保存三个场景的工程推荐点。鲁棒性表对低温推荐策略施加环境 ±5°C、到站时间 ±5 min 和站端 100 kW 降额。

## 结论边界

老化参数、快充接受功率图和热参数尚未由特定电芯/整车数据确认。当前 Pareto 前沿用于证明方法、接口和策略权衡，不能宣称某车型能够获得同等寿命收益。接入真实数据时，应通过参数注册和独立 holdout V&V 更新参数，再重算整个候选集。

## 运行

```powershell
D:\anaconda\python.exe experiments\run_joint_optimization.py
D:\anaconda\python.exe -m pytest -q tests/test_joint_optimization.py
```

图 `results/optimization/joint_optimization_pareto.png` 用横轴表示充电时间、纵轴表示相对老化、颜色表示到站核心温度、点大小表示预热能耗，星号为归一化工程推荐点。
