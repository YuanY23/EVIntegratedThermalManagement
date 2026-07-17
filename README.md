# 纯电动汽车整车集成能量与热管理仿真

项目题目：**融合数据驱动热负荷预测的纯电动汽车整车集成热管理系统建模与能量优化**。

本项目面向能源与动力工程、清洁能源技术方向的硕士研究与热管理岗位求职。主体是整车系统级物理建模：由驾驶循环计算轮端需求、电驱功率与损耗，建立电池、电机、逆变器、乘员舱、冷却液回路、冷板、散热器、液液换热器、热泵、PTC和余热回收模型。LSTM仅预测未来300秒三类热负荷，并调整规则控制的启用时机和阈值，不直接输出执行器占空比。

## 项目特点

- 纵向动力学包含惯性、滚阻、风阻、坡度和制动能量回收；
- 电池采用等效电路、SOC/温度相关内阻、Bernardi产热和核心/表面双节点；
- 电驱采用转速-负荷效率面和电机/逆变器独立热容节点；
- 液路根据冷却液物性、Darcy压降、泵曲线与系统阻力曲线交点计算实际流量；
- 冷板根据Re、Pr、Nu计算对流换热，散热器和换热器采用epsilon-NTU；
- 座舱采用空气/内饰2R2C模型，热泵采用带温升与低温退化的准稳态模型；
- 状态机处理热模式和回路优先级，局部PID处理连续执行器请求；
- LSTM同时预测电池产热、电驱余热和座舱净热负荷，数据按episode划分避免泄漏；
- 输出热安全、舒适性、泵/风扇/压缩机/PTC能耗、COP、余热利用、百公里电耗与等效续航。

## 环境与运行

```powershell
cd E:\EVIntegratedThermalManagement
D:\anaconda\python.exe -m pip install -e .
D:\anaconda\python.exe -m pytest -q
D:\anaconda\python.exe experiments\run_all.py --quick
D:\anaconda\python.exe experiments\run_all.py
```

快速流程用于检查完整链路；不带 `--quick` 的流程生成24个多工况episode、训练正式模型并运行六类工况的双策略对比。

## 目录

```text
configs/                 参数与训练配置
src/ev_thermal/          物理模型、控制、预测、仿真与图表
experiments/             可复现实验入口
tests/                   单元和集成测试
data/processed/          仿真训练数据
models/                  LSTM、标准化器和训练记录
results/tables/          指标表
results/figures/         论文图
results/logs/            运行清单
docs/                    方程、控制、预测和实验说明
```

## 建模边界

这是整车热管理系统级集总参数模型，不是三维CFD、制冷剂两相瞬态模型或实车标定模型。默认参数用于方法验证和趋势研究；用于具体车型前，应由试验台架、供应商性能图和实车CAN数据重新标定，并进行参数灵敏度与不确定性分析。

