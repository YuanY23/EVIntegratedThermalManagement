# 后续AI学习指南

## 1. 学习目标

任何后续AI接手本仓库时，都应把它理解为一个已经完成系统升级的“纯电动汽车整车集成热管理、参数辨识、架构选型、快充到站预热与联合优化”项目，而不是旧版的单一LSTM预测项目。

## 2. 阅读顺序与参考地图

唯一具有约束力的编号阅读顺序定义在根目录`AGENTS.md`的“Source of truth”中，本文件不维护第二份顺序。读完本文件后，应返回该列表继续下一项。

完成必读列表后，如需专题深入，可按问题选择：

- 紧凑正式结果：`docs/results_summary.md`；
- 参数辨识：`docs/parameter_identification.md`；
- 架构选型：`docs/architecture_comparison.md`；
- 快充预热：`docs/fast_charge_preconditioning.md`；
- 联合优化：`docs/joint_optimization.md`；
- 实现与复现：`src/ev_thermal/`和`experiments/`。

只有用户明确询问历史版本时，才读取`project_delivery/archive/legacy_v1/`。

## 3. 事实优先级

发生冲突时使用以下优先级：

```text
机器manifest/CSV/JSON
    > current目录交付文档
    > docs专题文档
    > README说明
    > archive历史材料
```

关键机器事实源：

- `results/logs/run_manifest.json`；
- `results/logs/upgrade_manifest.json`；
- `models/test_metrics.json`；
- `results/tables/strategy_comparison.csv`；
- `results/calibration/maturity_statement.json`；
- `results/architecture/architecture_summary.json`；
- `results/charging/preconditioning_comparison.csv`；
- `results/optimization/joint_optimization_summary.json`；
- `results/optimization/joint_optimization_recommended.csv`。

## 4. 当前版本锚点

- 正式运行：`final-reviewed-formal-v2`；
- plant SHA：`d3c0ea7429cd7292d88f2a46f2d4b42a63108c5de92c9bd1e7527affce68525f`；
- 正式数据：24 episodes；
- 策略结果：6工况×2策略；
- 升级证据：参数、架构、充电、优化四个suite，共44项；
- 联合优化：186个候选，各工况合计33个工程分辨率Pareto代表点（低温11、温和12、高温10）。

如果源码、配置或实验脚本已经变化，应先运行验证器；不得继续把上述SHA当成新代码的结果。

## 5. 正确理解项目

### 5.1 项目主体

项目主体是物理模型和整车系统集成，包括：

- 驾驶工况和纵向动力学；
- 电驱效率、损耗和热状态；
- 电池电气、SOC、产热和双节点热状态；
- 座舱2R2C；
- 一维液压网络、泵、冷板、散热器和换热器；
- 状态机、PID和执行器动态；
- 参数辨识与V&V；
- 架构和规格选型；
- 到站预热、快充和相对老化；
- Pareto联合优化。

LSTM是前瞻信息模块，不是整个项目的主体，也不直接输出执行器控制量。

### 5.2 结果应该怎样解释

- 预测策略平均降低温度但增加能耗，说明存在权衡；
- 参数辨识结果只证明合成真值回辨方法；
- 共享热汇只是当前指标下排名第一；
- 规则预热更快，但更耗能且相对damage更高；
- 联合推荐点是公开权重下的Pareto折中；
- 守恒闭合证明内部账本一致，不证明实车精度。

## 6. 禁止形成的错误结论

后续AI不得：

1. 使用archive中的旧数字覆盖当前结果；
2. 将quick运行当作正式运行；
3. 把合成数据写成台架或实车数据；
4. 把相对damage写成容量衰减百分比；
5. 宣称共享热汇是量产绝对最优；
6. 宣称预测控制一定降低能耗；
7. 混用不同plant SHA生成的模型与结果；
8. 在没有重跑实验时手工修改正式结果数字。

## 7. 回答用户问题时的最低核验

### 用户问结果

先读取对应CSV/JSON，再回答，不只引用报告中的二手数字。

### 用户问简历

必须带“仿真”“合成观测”“相对老化”或“当前模型假设”等边界词；优先使用`项目介绍.md`第8节。

### 用户问技术细节

先查专题文档，再查`src/ev_thermal/`对应实现；方程与代码不一致时以代码和当前测试为准，并指出文档需要更新。

### 用户要求继续升级

改动`src/ev_thermal/`、`configs/`或`experiments/`后，global plant SHA会变化，当前正式运行和四个suite会一起失效。必须重跑正式基线及参数、架构、充电、优化四个suite，重建upgrade manifest并通过验证器，再更新交付文档。

## 8. 当前复现命令

```powershell
D:\anaconda\python.exe -m pytest -q
D:\anaconda\python.exe experiments\verify_artifacts.py
```

如果只想理解项目，不应先重跑耗时正式实验；先读取已经验证的manifest和结果。只有源码发生改变或用户明确要求重新生成时才重跑。

## 9. 文档维护协议

一次新的正式升级完成后，应同时更新：

1. `results/logs/run_manifest.json`与upgrade manifest；
2. `project_delivery/current/数据与结果索引.md`；
3. `project_delivery/current/项目研究报告.md`；
4. `project_delivery/current/项目介绍.md`；
5. 本文件中的版本锚点；
6. `project_delivery/项目交付文档.md`和`project_delivery/README.md`；
7. 根目录`AGENTS.md`和`README.md`中的版本与入口；
8. `docs/results_summary.md`及结论受影响的专题文档。

被替换的运行和文档应移入新版本归档目录，不应直接删除，也不能继续保留在current中。

### 变更影响矩阵

| 变更类型 | 必须执行 |
|---|---|
| 仅current/README文字、无数字变化 | 链接检查、术语与机器数字一致性检查 |
| 仅测试文件 | 运行相关测试；plant SHA不变 |
| `src/ev_thermal/`、`configs/`或`experiments/` | 重跑formal和四个suite，重建upgrade manifest，执行完整验收，更新全部版本锚点 |
| 机器结果数字变化 | 禁止手工改结果；从实验重新生成，并同步研究报告、介绍、数据索引、交付文档和专题文档 |
