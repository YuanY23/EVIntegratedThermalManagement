# 快充到站预热、功率账本与相对老化

## Mission 与状态

快充研究把任务分为 `en-route → arrival → fast-charging → charge-complete / safe-stop`。路线阶段提供剩余时间、预计到站 SOC、环境温度以及预览有效性；规则策略仅在路线仍激活、预览有效、SOC 留有安全余量且温度收益足够时启动。

当前可解释基准为：

- `none`：不进行专门到站预热/预冷；
- `rule`：低温以 20°C 为目标预热，高温以 30°C 为目标预冷，功率受剩余时间和 5/6 kW 执行器上限约束；
- `optimized`：在联合优化阶段生成，始终与前两者同场比较。

预览缺失、路线取消或 SOC 储备不足时，规则策略确定性退出。路线仿真显式记录请求牵引功率、热管理之后的可用牵引功率、实际牵引功率和端功率闭合残差，防止附件功率只记账却不反馈到车辆可用功率。

## 快充接受功率

充电采用充电为正的工程输出、传入电池模型时转换为负端功率。实际接受功率取以下约束的最小值：

```text
P_bat,accepted = min(P_station*eta_charger - P_aux,
                     P_temperature(T_core),
                     P_SOC_taper(SOC),
                     P_current_voltage(I_max, OCV, R),
                     P_to_target_SOC)
```

低温与高温均降额；中等 SOC 为恒功率近似，高 SOC 进入连续 taper。达到目标 SOC 正常结束，核心温度达到 50°C 硬上限则进入 `safe_stop_temperature`。充电期间低温 PTC、冷却、泵和基础附件持续计入。

功率账本分别记录：

```text
P_grid = P_charger_loss + P_aux + P_battery_terminal_in
P_dc_bus = P_aux + P_battery_terminal_in
```

因此表格中的站端请求、电网实际功率、充电机损失、DC 母线功率、附件功率、实际入电池功率和削减功率不会混用。

## 相对老化

老化是独立 damage accumulator，不改变电池 `BatteryState`，避免把尚未确认的 SOH 模型扩散到整车所有状态。循环损伤与日历损伤分开，显式依赖温度、SOC、C-rate 和 Ah-throughput：

```text
D_cycle ∝ Ah_throughput * Arrhenius(T) * stress_SOC * (1 + C_rate^n)
D_calendar ∝ dt * Arrhenius(T) * stress_SOC
```

这些系数尚未由具体电芯寿命数据辨识，因此 damage 只用于同初始条件、同终止 SOC 策略的相对比较，不能解释为容量衰减百分比或寿命里程。

## 场景与鲁棒性

正式基准覆盖低温到站、温和到站和高温到站，并另外检查预览缺失、路线取消、充电站降额和晚选择充电站。每个事件要求 SOC 单调上升、数值有限、电池/充电桩限制不被突破且功率账本闭合。

## 运行

```powershell
D:\anaconda\python.exe experiments\run_preconditioning_comparison.py
D:\anaconda\python.exe -m pytest -q tests/test_fast_charging.py tests/test_preconditioning.py tests/test_battery_aging.py
```

结果位于 `results/charging`。`preconditioning_comparison.csv` 是策略摘要，route/charge 时序用于检查状态、限功率原因和账本，`preconditioning_robustness.csv` 记录回退场景。联合优化阶段将在相同物理事件上生成 Pareto 前沿。
