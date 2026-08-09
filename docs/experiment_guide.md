# 实验复现指南

## 测试

```powershell
D:\anaconda\python.exe -m pytest -q
```

## 快速全流程

```powershell
D:\anaconda\python.exe experiments\run_all.py --quick
```

用于检查数据、训练、模型加载、策略仿真、指标表、图表和 manifest 是否全部生成。每次运行都会写入独立的 `artifacts/runs/<run_id>`，并更新 `artifacts/latest/quick.json`；quick 运行不会覆盖 `data/processed`、`models` 和 `results` 中已发布的正式结果。

## 正式实验

```powershell
D:\anaconda\python.exe experiments\run_all.py
D:\anaconda\python.exe experiments\verify_artifacts.py
```

正式运行先在隔离目录中生成完整结果。只有 24 个 episode、6×2 场景策略矩阵、有限数值、热平衡、模型指标一致性、精确图表集合和全部文件哈希均通过校验，才会发布到兼容路径：表格和图位于 `results`，模型位于 `models`，正式运行指针位于 `artifacts/latest/formal.json`。

`results/logs/run_manifest.json` 保存运行 profile、状态、配置与 plant 哈希、随机种子、训练设置、场景集合、指标和产物 SHA-256。`models/model_manifest.json` 将模型权重绑定到生成它的配置和 plant 版本；plant 结构升级后必须重新训练，不能静默复用旧模型。

分步调试入口也不会直接写正式目录：`generate_dataset.py` 创建隔离 run，后续 `train_predictor.py` 和 `run_comparison.py` 必须显式传入该 `--run-root`。推荐使用 `run_all.py` 完成可发布实验。

## 升级实验

```powershell
D:\anaconda\python.exe experiments\run_parameter_identification.py
D:\anaconda\python.exe experiments\run_architecture_comparison.py
D:\anaconda\python.exe experiments\run_preconditioning_comparison.py
D:\anaconda\python.exe experiments\run_joint_optimization.py
D:\anaconda\python.exe experiments\build_upgrade_manifest.py
D:\anaconda\python.exe experiments\verify_artifacts.py
```

四个入口依次生成参数/V&V、三架构与规格、快充预热基准、联合优化结果。每个入口先生成独立suite manifest，完整绑定输入审计表、轨迹、摘要与图；`build_upgrade_manifest.py` 再将44个升级产物哈希绑定到最新正式 run 与 plant。最后一个验证命令同时检查正式模型链和升级证据链。源码、参数注册表或实验生成器改变后，旧 formal、suite manifest 与 upgrade manifest 都会被拒绝，必须重新生成。

## 结果解释

比较策略时应同时观察热安全、舒适性和能耗，不能只看峰值温度。预测提前冷却可能增加泵耗，却降低后续强化冷却或压缩机峰值；余热回收可能改善低温能耗，但受可用余热和温差限制。所有结论应以多个工况均值和逐工况结果共同支持。
