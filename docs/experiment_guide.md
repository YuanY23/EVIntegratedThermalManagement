# 实验复现指南

## 测试

```powershell
D:\anaconda\python.exe -m pytest -q
```

## 快速全流程

```powershell
D:\anaconda\python.exe experiments\run_all.py --quick
```

用于检查数据、训练、模型加载、策略仿真、指标表、图表和manifest是否全部生成。

## 正式实验

```powershell
D:\anaconda\python.exe experiments\generate_dataset.py --episodes 24 --duration 1200
D:\anaconda\python.exe experiments\train_predictor.py
D:\anaconda\python.exe experiments\run_comparison.py --duration 1800
```

或者执行 `D:\anaconda\python.exe experiments\run_all.py`。正式结果位于 `results/tables` 和 `results/figures`，模型位于 `models`。`results/logs/run_manifest.json` 保存配置哈希、随机种子、episode数量、最佳epoch和预测指标。

## 结果解释

比较策略时应同时观察热安全、舒适性和能耗，不能只看峰值温度。预测提前冷却可能增加泵耗，却降低后续强化冷却或压缩机峰值；余热回收可能改善低温能耗，但受可用余热和温差限制。所有结论应以多个工况均值和逐工况结果共同支持。

