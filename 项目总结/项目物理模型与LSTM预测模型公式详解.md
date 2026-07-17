# 纯电动汽车整车集成热管理项目：物理模型与LSTM预测模型公式详解

## 0. 文档说明

本文集中整理项目中使用的全部主要物理模型、控制模型、能量评价模型和LSTM热负荷预测模型。公式采用标准LaTeX语法，支持MathJax或KaTeX的Markdown阅读器可以直接渲染。

建议学习顺序：

1. 先掌握符号约定和整车能量流；
2. 学习纵向动力学、电驱功率和电池电气模型；
3. 学习电池、电驱和座舱热平衡；
4. 学习管路、水泵、冷板和换热器；
5. 学习热泵、PTC、chiller和余热回收；
6. 学习状态机、PID和执行器动态；
7. 最后学习LSTM数据窗口、门控方程和预测增强策略。

---

# 1. 总体系统、符号与正负号

## 1.1 整车能量与热量传递关系

整车模型的主要能量路径为：

$$
\text{电池电能}
\rightarrow
\text{电驱直流功率}
\rightarrow
\text{电机机械功率}
\rightarrow
\text{轮端功率}
$$

电驱和电池损耗转化为热量：

$$
\text{电池损耗}\rightarrow \dot Q_{\mathrm{bat}}
$$

$$
\text{电机/逆变器损耗}
\rightarrow
\dot Q_{\mathrm{motor}}+\dot Q_{\mathrm{inv}}
$$

部件热量经冷却液输送至散热器、chiller、座舱余热回收支路或环境。

## 1.2 功率正负号

本项目统一规定：

$$
P>0 \quad \text{表示电池放电或系统对外输出功率}
$$

$$
P<0 \quad \text{表示再生制动或电池充电}
$$

电池电流采用相同约定：

$$
I>0 \quad \text{放电}, \qquad I<0 \quad \text{充电}
$$

热流正方向通常定义为从高温部件流向冷却液或从热源流入储能节点。每个模型会单独说明。

## 1.3 常用符号

| 符号 | 含义 | SI单位 |
|---|---|---|
| $m$ | 质量 | kg |
| $v$ | 车速或流速 | m/s |
| $a$ | 加速度 | m/s² |
| $F$ | 力 | N |
| $P$ | 功率 | W |
| $E$ | 能量 | J或kWh |
| $T$ | 温度 | K或°C |
| $C$ | 集总热容 | J/K |
| $c_p$ | 比热容 | J/(kg·K) |
| $\dot m$ | 质量流量 | kg/s |
| $\dot Q$ | 热流率 | W |
| $UA$ | 总传热系数与面积乘积 | W/K |
| $h$ | 对流换热系数 | W/(m²·K) |
| $\rho$ | 密度 | kg/m³ |
| $\mu$ | 动力黏度 | Pa·s |
| $k$ | 导热系数 | W/(m·K) |
| $\eta$ | 效率 | - |
| $u$ | 归一化控制命令 | - |

## 1.4 角度与温度换算

道路坡度角由角度转换为弧度：

$$
\theta_{\mathrm{rad}}
=
\theta_{\mathrm{deg}}\frac{\pi}{180}
$$

摄氏温度转换为绝对温度：

$$
T_{\mathrm K}=T_{^\circ\mathrm C}+273.15
$$

涉及Carnot COP和电池熵热时必须使用绝对温度。

---

# 2. 驾驶循环与加速度

## 2.1 离散车速序列

驾驶循环给出离散车速：

$$
v_k=v(t_k),\qquad t_k=k\Delta t
$$

本项目正式仿真时间步为：

$$
\Delta t=5\ \mathrm s
$$

## 2.2 加速度

内部使用数值梯度计算加速度。中心差分形式可表示为：

$$
a_k
\approx
\frac{v_{k+1}-v_{k-1}}{2\Delta t}
$$

端点采用单边差分：

$$
a_0\approx\frac{v_1-v_0}{\Delta t}
$$

$$
a_N\approx\frac{v_N-v_{N-1}}{\Delta t}
$$

加速度是惯性力、电机功率和瞬时产热的重要输入。车速序列中的高频噪声会通过差分被放大，因此真实数据应用前通常需要低通滤波或平滑处理。

---

# 3. 整车纵向动力学模型

## 3.1 惯性力

根据牛顿第二定律：

$$
F_{\mathrm{inertia}}=ma
$$

加速时 $a>0$，惯性力需求为正；减速时 $a<0$，可能产生再生制动功率。

## 3.2 滚动阻力

$$
F_{\mathrm{roll}}
=
mgC_{\mathrm{rr}}\cos\theta
$$

其中：

- $m=1950\ \mathrm{kg}$；
- $g=9.81\ \mathrm{m/s^2}$；
- $C_{\mathrm{rr}}=0.0105$。

车辆接近静止时，模型将滚动阻力功率置零，以避免停车阶段产生虚假能耗。

## 3.3 空气阻力

$$
F_{\mathrm{aero}}
=
\frac{1}{2}\rho_{\mathrm{air}}C_dA_fv^2
$$

项目参数：

$$
\rho_{\mathrm{air}}=1.225\ \mathrm{kg/m^3}
$$

$$
C_d=0.27,\qquad A_f=2.35\ \mathrm{m^2}
$$

空气阻力与车速平方成正比，因此空气阻力功率与车速三次方近似成正比：

$$
P_{\mathrm{aero}}
=F_{\mathrm{aero}}v
\propto v^3
$$

这解释了高速工况能耗明显增加的原因。

## 3.4 坡度阻力

$$
F_{\mathrm{grade}}
=
mg\sin\theta
$$

小坡度时可近似：

$$
\sin\theta\approx\tan\theta\approx i_{\mathrm{grade}}
$$

因此：

$$
F_{\mathrm{grade}}
\approx
mgi_{\mathrm{grade}}
$$

持续爬坡会产生持续正功率需求，是坡道高负荷工况电耗较高的主要原因。

## 3.5 总牵引力

$$
F_{\mathrm{trac}}
=
F_{\mathrm{inertia}}
+F_{\mathrm{roll}}
+F_{\mathrm{aero}}
+F_{\mathrm{grade}}
$$

展开为：

$$
F_{\mathrm{trac}}
=
ma
+mgC_{\mathrm{rr}}\cos\theta
+\frac{1}{2}\rho_{\mathrm{air}}C_dA_fv^2
+mg\sin\theta
$$

## 3.6 轮端功率

$$
P_{\mathrm{wheel}}
=
F_{\mathrm{trac}}v
$$

当 $P_{\mathrm{wheel}}>0$ 时车辆需要驱动功率；当 $P_{\mathrm{wheel}}<0$ 时车辆处于减速或下坡状态。

## 3.7 电机转速

车轮角速度：

$$
\omega_{\mathrm{wheel}}
=
\frac{v}{r_{\mathrm{wheel}}}
$$

电机角速度：

$$
\omega_{\mathrm{motor}}
=
i_{\mathrm{final}}\omega_{\mathrm{wheel}}
$$

转速：

$$
n_{\mathrm{motor}}
=
\omega_{\mathrm{motor}}\frac{60}{2\pi}
$$

项目参数：

$$
r_{\mathrm{wheel}}=0.34\ \mathrm m,qquad
i_{\mathrm{final}}=9.1
$$

## 3.8 电机转矩

机械功率满足：

$$
P_{\mathrm{mech}}
=
T_{\mathrm{motor}}\omega_{\mathrm{motor}}
$$

因此：

$$
T_{\mathrm{motor}}
=
\frac{P_{\mathrm{mech}}}{\omega_{\mathrm{motor}}}
$$

低速时对分母设置下限：

$$
T_{\mathrm{motor}}
=
\frac{P_{\mathrm{mech}}}
{\max(|\omega_{\mathrm{motor}}|,\omega_{\min})}
$$

用于避免零速附近的数值发散。

## 3.9 驱动与再生功率限制

驱动时：

$$
P_{\mathrm{drive,eq}}
=
\min\left(
\frac{P_{\mathrm{wheel}}}{\eta_{\mathrm{drivetrain}}},
P_{\mathrm{traction,max}}
\right)
$$

再生时：

$$
P_{\mathrm{regen,eq}}
=
\max\left(
P_{\mathrm{wheel}}\eta_{\mathrm{drivetrain}},
-P_{\mathrm{regen,max}}
\right)
$$

其中：

$$
\eta_{\mathrm{drivetrain}}=0.97
$$

$$
P_{\mathrm{traction,max}}=180\ \mathrm{kW}
$$

$$
P_{\mathrm{regen,max}}=80\ \mathrm{kW}
$$

---

# 4. 电机与逆变器效率、损耗和温升模型

## 4.1 归一化负荷与转速

$$
\lambda_P
=
\operatorname{clip}
\left(
\frac{|P_{\mathrm{mech}}|}{180000},
0,1
\right)
$$

$$
\lambda_n
=
\operatorname{clip}
\left(
\frac{|n_{\mathrm{motor}}|}{14000},
0,1
\right)
$$

## 4.2 解析效率面

项目使用以下系统级解析效率面：

$$
\eta_{\mathrm{edrive}}
=
0.965
-0.07(\lambda_P-0.65)^2
-0.05(\lambda_n-0.55)^2
-0.04e^{-8\lambda_P}
$$

最终限幅：

$$
\eta_{\mathrm{edrive}}
=
\operatorname{clip}
(\eta_{\mathrm{edrive}},0.78,0.97)
$$

该公式表达以下规律：

- 中高负荷效率较高；
- 过低负荷时固定损耗占比增大，效率降低；
- 转速偏离高效区域时效率下降；
- 效率不会超过物理边界。

## 4.3 驱动工况功率流

当 $P_{\mathrm{mech}}\ge 0$：

$$
P_{\mathrm{dc}}
=
\frac{P_{\mathrm{mech}}}{\eta_{\mathrm{edrive}}}
$$

电驱总损耗：

$$
\dot Q_{\mathrm{loss}}
=
P_{\mathrm{dc}}-P_{\mathrm{mech}}
$$

## 4.4 再生工况功率流

当 $P_{\mathrm{mech}}<0$：

$$
P_{\mathrm{dc}}
=
P_{\mathrm{mech}}\eta_{\mathrm{edrive}}
$$

损耗为：

$$
\dot Q_{\mathrm{loss}}
=
\left|
P_{\mathrm{mech}}-P_{\mathrm{dc}}
\right|
$$

## 4.5 电机与逆变器损耗分配

$$
\dot Q_{\mathrm{motor,loss}}
=
0.72\dot Q_{\mathrm{loss}}
$$

$$
\dot Q_{\mathrm{inv,loss}}
=
0.28\dot Q_{\mathrm{loss}}
$$

该比例是系统级假设，真实车辆可使用电机铜耗、铁耗、机械损耗和逆变器开关损耗图替换。

## 4.6 部件到冷却液换热

电机：

$$
\dot Q_{\mathrm{motor\rightarrow cool}}
=
UA_{\mathrm{motor}}
(T_{\mathrm{motor}}-T_{\mathrm{cool,p}})
$$

逆变器：

$$
\dot Q_{\mathrm{inv\rightarrow cool}}
=
UA_{\mathrm{inv}}
(T_{\mathrm{inv}}-T_{\mathrm{cool,p}})
$$

总动力系统UA随流量增加：

$$
UA_{\mathrm{edrive}}
=
900
\min\left(
\frac{\dot m_{\mathrm{powertrain}}}{0.25},1
\right)
$$

模型将其中72%分配给电机，28%分配给逆变器：

$$
UA_{\mathrm{motor}}=0.72UA_{\mathrm{edrive}}
$$

$$
UA_{\mathrm{inv}}=0.28UA_{\mathrm{edrive}}
$$

## 4.7 电机温升方程

$$
C_{\mathrm{motor}}
\frac{dT_{\mathrm{motor}}}{dt}
=
\dot Q_{\mathrm{motor,loss}}
-\dot Q_{\mathrm{motor\rightarrow cool}}
$$

$$
C_{\mathrm{motor}}
=85000\ \mathrm{J/K}
$$

## 4.8 逆变器温升方程

$$
C_{\mathrm{inv}}
\frac{dT_{\mathrm{inv}}}{dt}
=
\dot Q_{\mathrm{inv,loss}}
-\dot Q_{\mathrm{inv\rightarrow cool}}
$$

$$
C_{\mathrm{inv}}
=28000\ \mathrm{J/K}
$$

## 4.9 动力系统进入冷却液的热量

只有通过部件边界传出的热量进入冷却液：

$$
\dot Q_{\mathrm{edrive\rightarrow coolant}}
=
\dot Q_{\mathrm{motor\rightarrow cool}}
+\dot Q_{\mathrm{inv\rightarrow cool}}
$$

不能直接令冷却液得热等于全部损耗，否则会同时在部件热容和冷却液中重复计算热量。

---

# 5. 动力电池电气模型

## 5.1 SOC限幅

用于OCV函数的SOC为：

$$
x
=
\operatorname{clip}(SOC,0.02,0.98)
$$

## 5.2 开路电压模型

$$
U_{\mathrm{oc}}
=
U_{\mathrm{nom}}
\left(
0.88+0.20x-0.04x^2
\right)
$$

其中：

$$
U_{\mathrm{nom}}=380\ \mathrm V
$$

## 5.3 温度修正内阻

温度修正因子：

$$
f_T
=
\exp
\left[
\operatorname{clip}
\left(
0.018(25-T_{\mathrm{core}}),
-0.5,1.2
\right)
\right]
$$

温度低于25 °C时，$f_T>1$，内阻增大。

## 5.4 低SOC修正

$$
f_{SOC}
=
1+
1.8
\frac{\max(0.15-SOC,0)^2}{0.15^2}
$$

当 $SOC\ge 0.15$ 时：

$$
f_{SOC}=1
$$

## 5.5 总等效内阻

$$
R
=
R_{\mathrm{nom}}f_Tf_{SOC}
$$

$$
R_{\mathrm{nom}}=0.065\ \Omega
$$

## 5.6 电池端电压

放电正方向下：

$$
U_{\mathrm{terminal}}
=
U_{\mathrm{oc}}-IR
$$

## 5.7 电池端功率

$$
P_{\mathrm{terminal}}
=
U_{\mathrm{terminal}}I
$$

代入端电压：

$$
P_{\mathrm{terminal}}
=
(U_{\mathrm{oc}}-IR)I
$$

整理得到：

$$
RI^2-U_{\mathrm{oc}}I+P_{\mathrm{terminal}}=0
$$

## 5.8 功率可实现边界

对于给定 $U_{\mathrm{oc}}$ 和 $R$，二次方程判别式必须非负：

$$
U_{\mathrm{oc}}^2-4RP\ge 0
$$

理论最大放电功率：

$$
P_{\mathrm{theory,max}}
=
\frac{U_{\mathrm{oc}}^2}{4R}
$$

模型使用安全系数0.95：

$$
P_{\mathrm{dis,max,actual}}
=
\min\left(
200000,
0.95\frac{U_{\mathrm{oc}}^2}{4R}
\right)
$$

最大充电功率：

$$
P_{\mathrm{chg,max}}=100000\ \mathrm W
$$

## 5.9 电流低根

二次方程两个根中，系统选取低电流稳定根：

$$
I
=
\frac{
U_{\mathrm{oc}}
-\sqrt{U_{\mathrm{oc}}^2-4RP}
}{2R}
$$

## 5.10 电池等效容量

$$
C_{\mathrm{Ah}}
=
\frac{E_{\mathrm{capacity,kWh}}\times1000}
{U_{\mathrm{nom}}}
$$

对于75 kWh、380 V电池：

$$
C_{\mathrm{Ah}}
\approx
197.37\ \mathrm{Ah}
$$

## 5.11 SOC连续方程

$$
\frac{dSOC}{dt}
=
-\frac{I}{3600C_{\mathrm{Ah}}}
$$

显式离散：

$$
SOC_{k+1}
=
\operatorname{clip}
\left(
SOC_k
-\frac{I_k\Delta t}{3600C_{\mathrm{Ah}}},
0,1
\right)
$$

---

# 6. 动力电池产热模型

## 6.1 Bernardi总产热

$$
\dot Q_{\mathrm{bat}}
=
I^2R
-IT\frac{dU_{\mathrm{oc}}}{dT}
$$

其中 $T$ 必须使用开尔文：

$$
T=T_{\mathrm{core},^\circ\mathrm C}+273.15
$$

## 6.2 不可逆欧姆热

$$
\dot Q_{\mathrm{irr}}
=
I^2R
$$

欧姆热始终非负，并与电流平方成正比，因此功率尖峰会显著增加电池热负荷。

## 6.3 可逆熵热

$$
\dot Q_{\mathrm{rev}}
=
-IT\frac{dU_{\mathrm{oc}}}{dT}
$$

项目包级熵系数：

$$
\frac{dU_{\mathrm{oc}}}{dT}
=
-0.035\ \mathrm{V/K}
$$

放电时 $I>0$，在该负熵系数下可逆项为正。充电时可逆热方向可能变化，因此瞬时总热负荷可能出现负值。

## 6.4 总热量分解

$$
\dot Q_{\mathrm{bat}}
=
\dot Q_{\mathrm{irr}}
+\dot Q_{\mathrm{rev}}
$$

---

# 7. 电池核心-表面双节点热模型

## 7.1 核心到表面导热

$$
\dot Q_{\mathrm{core\rightarrow surf}}
=
G_{\mathrm{cs}}
(T_{\mathrm{core}}-T_{\mathrm{surf}})
$$

$$
G_{\mathrm{cs}}=380\ \mathrm{W/K}
$$

当核心温度高于表面温度时，该热流为正。

## 7.2 表面到冷却液换热

$$
\dot Q_{\mathrm{surf\rightarrow cool}}
=
UA_{\mathrm{coldplate}}
(T_{\mathrm{surf}}-T_{\mathrm{cool,b}})
$$

## 7.3 核心节点能量方程

$$
C_{\mathrm{core}}
\frac{dT_{\mathrm{core}}}{dt}
=
\dot Q_{\mathrm{bat}}
-\dot Q_{\mathrm{core\rightarrow surf}}
$$

$$
C_{\mathrm{core}}=380000\ \mathrm{J/K}
$$

## 7.4 表面节点能量方程

$$
C_{\mathrm{surf}}
\frac{dT_{\mathrm{surf}}}{dt}
=
\dot Q_{\mathrm{core\rightarrow surf}}
-\dot Q_{\mathrm{surf\rightarrow cool}}
+\dot Q_{\mathrm{heater}}
$$

$$
C_{\mathrm{surf}}=95000\ \mathrm{J/K}
$$

加热器热量进入表面节点，因为PTC或加热液路从电池包边界向内部传热。

## 7.5 显式离散

核心：

$$
T_{\mathrm{core},k+1}
=
T_{\mathrm{core},k}
+\frac{\Delta t}{C_{\mathrm{core}}}
\left(
\dot Q_{\mathrm{bat},k}
-\dot Q_{\mathrm{core\rightarrow surf},k}
\right)
$$

表面：

$$
T_{\mathrm{surf},k+1}
=
T_{\mathrm{surf},k}
+\frac{\Delta t}{C_{\mathrm{surf}}}
\left(
\dot Q_{\mathrm{core\rightarrow surf},k}
-\dot Q_{\mathrm{surf\rightarrow cool},k}
+\dot Q_{\mathrm{heater},k}
\right)
$$

## 7.6 双节点模型意义

核心-表面温差反映内部热滞后：

$$
\Delta T_{\mathrm{core-surf}}
=
T_{\mathrm{core}}-T_{\mathrm{surf}}
$$

单节点模型只能给出平均温度，双节点模型能够表示核心产热到表面散热之间的动态延迟，同时保持较低计算量。

---

# 8. 乘员舱2R2C热模型

## 8.1 节点定义

座舱具有两个温度状态：

$$
T_{\mathrm{air}} \quad \text{座舱空气温度}
$$

$$
T_{\mathrm{int}} \quad \text{内饰和车身等效温度}
$$

热容：

$$
C_{\mathrm{air}}=65000\ \mathrm{J/K}
$$

$$
C_{\mathrm{int}}=650000\ \mathrm{J/K}
$$

## 8.2 环境到内饰传热

$$
\dot Q_{\mathrm{env\rightarrow int}}
=
UA_{\mathrm{env}}
(T_{\mathrm{amb}}-T_{\mathrm{int}})
$$

$$
UA_{\mathrm{env}}=110\ \mathrm{W/K}
$$

## 8.3 渗透风和新风等效负荷

$$
\dot Q_{\mathrm{infiltration}}
=
UA_{\mathrm{inf}}
(T_{\mathrm{amb}}-T_{\mathrm{air}})
$$

$$
UA_{\mathrm{inf}}=35\ \mathrm{W/K}
$$

## 8.4 太阳辐射负荷

$$
\dot Q_{\mathrm{solar}}
=
I_{\mathrm{solar}}
A_{\mathrm{solar}}
\alpha_{\mathrm{solar}}
$$

$$
A_{\mathrm{solar}}=3.2\ \mathrm{m^2}
$$

$$
\alpha_{\mathrm{solar}}=0.55
$$

分配到内饰和空气：

$$
\dot Q_{\mathrm{solar,int}}
=0.65\dot Q_{\mathrm{solar}}
$$

$$
\dot Q_{\mathrm{solar,air}}
=0.35\dot Q_{\mathrm{solar}}
$$

## 8.5 乘员显热

$$
\dot Q_{\mathrm{occupant}}
=
N_{\mathrm{occ}}q_{\mathrm{person}}
$$

$$
q_{\mathrm{person}}=90\ \mathrm{W/person}
$$

## 8.6 内饰到空气换热

$$
\dot Q_{\mathrm{int\rightarrow air}}
=
UA_{\mathrm{air-int}}
(T_{\mathrm{int}}-T_{\mathrm{air}})
$$

$$
UA_{\mathrm{air-int}}=180\ \mathrm{W/K}
$$

## 8.7 内饰节点方程

$$
C_{\mathrm{int}}
\frac{dT_{\mathrm{int}}}{dt}
=
\dot Q_{\mathrm{env\rightarrow int}}
+0.65\dot Q_{\mathrm{solar}}
-\dot Q_{\mathrm{int\rightarrow air}}
$$

## 8.8 空气节点方程

$$
C_{\mathrm{air}}
\frac{dT_{\mathrm{air}}}{dt}
=
\dot Q_{\mathrm{infiltration}}
+0.35\dot Q_{\mathrm{solar}}
+\dot Q_{\mathrm{occupant}}
+\dot Q_{\mathrm{int\rightarrow air}}
+\dot Q_{\mathrm{HVAC}}
$$

规定：

$$
\dot Q_{\mathrm{HVAC}}>0 \quad \text{座舱制热}
$$

$$
\dot Q_{\mathrm{HVAC}}<0 \quad \text{座舱制冷}
$$

## 8.9 未调节座舱净热负荷

LSTM预测目标之一为：

$$
\dot Q_{\mathrm{cabin,load}}
=
\dot Q_{\mathrm{env\rightarrow int}}
+\dot Q_{\mathrm{infiltration}}
+\dot Q_{\mathrm{solar}}
+\dot Q_{\mathrm{occupant}}
$$

该负荷不包含HVAC热量，用于表示环境与乘员造成的外部热需求。

---

# 9. 冷却液热物性模型

## 9.1 适用温度

模型温度先限幅：

$$
T_f
=
\operatorname{clip}(T,-30,100)
$$

## 9.2 密度

$$
\rho_f
=
1068-0.52(T_f-20)
$$

单位为 $\mathrm{kg/m^3}$。

## 9.3 比热

$$
c_{p,f}
=
3440+3.2(T_f-20)
$$

单位为 $\mathrm{J/(kg\cdot K)}$。

## 9.4 动力黏度

$$
\mu_f
=
0.0042
\exp[-0.032(T_f-20)]
$$

单位为 $\mathrm{Pa\cdot s}$。低温时黏度明显升高，导致压降增加。

## 9.5 导热系数

$$
k_f
=
0.385+0.00045(T_f-20)
$$

单位为 $\mathrm{W/(m\cdot K)}$。

## 9.6 Prandtl数

$$
Pr
=
\frac{c_{p,f}\mu_f}{k_f}
$$

$Pr$ 表示动量扩散和热扩散的相对强弱，是Nusselt关联式的重要输入。

---

# 10. 管路流动与压降模型

## 10.1 圆管流通面积

$$
A_{\mathrm{flow}}
=
\frac{\pi D^2}{4}
$$

## 10.2 体积流量

$$
\dot V
=
\frac{\dot m}{\rho_f}
$$

## 10.3 平均流速

$$
u_f
=
\frac{\dot V}{A_{\mathrm{flow}}}
=
\frac{\dot m}{\rho_fA_{\mathrm{flow}}}
$$

## 10.4 Reynolds数

$$
Re
=
\frac{\rho_fu_fD}{\mu_f}
$$

## 10.5 层流Darcy摩阻系数

当 $Re<2300$：

$$
f
=
\frac{64}{Re}
$$

## 10.6 湍流Swamee-Jain摩阻系数

当 $Re\ge 2300$：

$$
f
=
\frac{0.25}
{\left[
\log_{10}
\left(
\frac{\varepsilon_r}{3.7D}
+\frac{5.74}{Re^{0.9}}
\right)
\right]^2}
$$

其中 $\varepsilon_r$ 为管壁绝对粗糙度，默认：

$$
\varepsilon_r=1.5\times10^{-6}\ \mathrm m
$$

## 10.7 动压

$$
q_{\mathrm{dyn}}
=
\frac{1}{2}\rho_fu_f^2
$$

## 10.8 沿程压降

$$
\Delta p_{\mathrm{pipe}}
=
f\frac{L}{D}
\frac{\rho_fu_f^2}{2}
$$

## 10.9 局部压降

$$
\Delta p_{\mathrm{local}}
=
K_{\mathrm{local}}
\frac{\rho_fu_f^2}{2}
$$

## 10.10 总压降

$$
\Delta p_{\mathrm{total}}
=
\left(
f\frac{L}{D}+K_{\mathrm{local}}
\right)
\frac{\rho_fu_f^2}{2}
$$

湍流时压降通常近似与流量平方成正比，因此高流量会快速增加泵耗。

---

# 11. 水泵工作点与泵耗模型

## 11.1 泵相似定律

归一化泵转速为 $n\in[0,1]$。

停流压升：

$$
\Delta p_{0}(n)
=
\Delta p_{0,\mathrm{nom}}n^2
$$

零扬程最大流量：

$$
\dot m_{\max}(n)
=
\dot m_{\max,\mathrm{nom}}n
$$

项目参数：

$$
\Delta p_{0,\mathrm{nom}}=85000\ \mathrm{Pa}
$$

$$
\dot m_{\max,\mathrm{nom}}=0.45\ \mathrm{kg/s}
$$

## 11.2 泵扬程曲线

$$
\Delta p_{\mathrm{pump}}(\dot m,n)
=
\Delta p_0(n)
\left[
1-
\left(
\frac{\dot m}{\dot m_{\max}(n)}
\right)^2
\right]
$$

## 11.3 系统阻力曲线

$$
\Delta p_{\mathrm{system}}
=
K_{\mathrm{sys}}\dot m^2
$$

项目中电池和动力回路使用不同系统阻力系数：

$$
K_{\mathrm{sys,b}}=360000
\ \mathrm{Pa/(kg/s)^2}
$$

$$
K_{\mathrm{sys,p}}=260000
\ \mathrm{Pa/(kg/s)^2}
$$

## 11.4 工作点方程

实际流量满足：

$$
\Delta p_{\mathrm{pump}}(\dot m,n)
=
\Delta p_{\mathrm{system}}(\dot m)
$$

该非线性方程通过数值求根得到 $\dot m^*$。

## 11.5 泵效率

定义流量比：

$$
\phi
=
\frac{\dot m^*}{\dot m_{\max}(n)}
$$

经验效率：

$$
\eta_{\mathrm{pump}}
=
\max
\left[
0.18,
0.52\left(1-0.8(\phi-0.65)^2\right)
\right]
$$

## 11.6 液压功率

$$
P_{\mathrm{hyd}}
=
\Delta p\dot V
=
\Delta p\frac{\dot m}{\rho_f}
$$

## 11.7 泵电功率

$$
P_{\mathrm{pump}}
=
\frac{P_{\mathrm{hyd}}}{\eta_{\mathrm{pump}}}
$$

---

# 12. 电池冷板对流换热模型

## 12.1 流道参数

$$
N_{\mathrm{ch}}=12
$$

$$
D_h=0.0035\ \mathrm m
$$

$$
A_{\mathrm{ch}}=8.0\times10^{-6}\ \mathrm{m^2}
$$

$$
L_{\mathrm{ch}}=0.7\ \mathrm m
$$

$$
A_{\mathrm{ht}}=1.8\ \mathrm{m^2}
$$

$$
R_{\mathrm{wall}}=0.002\ \mathrm{K/W}
$$

## 12.2 单流道质量流量

假设并联流道均匀分流：

$$
\dot m_{\mathrm{ch}}
=
\frac{\dot m_{\mathrm{total}}}{N_{\mathrm{ch}}}
$$

## 12.3 单流道流速

$$
u_{\mathrm{ch}}
=
\frac{\dot m_{\mathrm{ch}}}
{\rho_fA_{\mathrm{ch}}}
$$

## 12.4 冷板Reynolds数

$$
Re_{\mathrm{cp}}
=
\frac{\rho_fu_{\mathrm{ch}}D_h}{\mu_f}
$$

## 12.5 层流Nusselt数

充分发展、定热流边界近似：

$$
Nu=4.36,qquad Re<2300
$$

## 12.6 湍流摩阻系数

$$
f
=
\left(0.79\ln Re-1.64\right)^{-2}
$$

## 12.7 Gnielinski关联式

$$
Nu
=
\frac{
(f/8)(Re-1000)Pr
}
{
1+12.7\sqrt{f/8}
\left(Pr^{2/3}-1\right)
}
$$

## 12.8 对流换热系数

$$
h
=
\frac{Nu\,k_f}{D_h}
$$

## 12.9 对流热阻

$$
R_{\mathrm{conv}}
=
\frac{1}{hA_{\mathrm{ht}}}
$$

## 12.10 冷板总UA

$$
UA_{\mathrm{cp}}
=
\frac{1}
{R_{\mathrm{wall}}+R_{\mathrm{conv}}}
$$

## 12.11 冷却液热容量率

$$
C_f
=
\dot m c_{p,f}
$$

## 12.12 冷板有效度

将电池表面视为大热容量恒温边界：

$$
\varepsilon_{\mathrm{cp}}
=
1-
\exp\left(-\frac{UA_{\mathrm{cp}}}{C_f}\right)
$$

## 12.13 冷板换热量

$$
\dot Q_{\mathrm{cp}}
=
\varepsilon_{\mathrm{cp}}C_f
(T_{\mathrm{surf}}-T_{\mathrm{cool,in}})
$$

## 12.14 冷却液出口温度

$$
T_{\mathrm{cool,out}}
=
T_{\mathrm{cool,in}}
+\frac{\dot Q_{\mathrm{cp}}}{C_f}
$$

如果电池表面高于入口冷却液：

$$
T_{\mathrm{cool,in}}
<T_{\mathrm{cool,out}}
<T_{\mathrm{surf}}
$$

## 12.15 冷板压降

$$
\Delta p_{\mathrm{cp}}
=
\left(
f\frac{L_{\mathrm{ch}}}{D_h}+K_{\mathrm{cp}}
\right)
\frac{\rho_fu_{\mathrm{ch}}^2}{2}
$$

项目局部阻力系数：

$$
K_{\mathrm{cp}}=2.5
$$

---

# 13. epsilon-NTU换热器模型

## 13.1 热侧和冷侧热容量率

$$
C_h=\dot m_hc_{p,h}
$$

$$
C_c=\dot m_cc_{p,c}
$$

## 13.2 最小和最大热容量率

$$
C_{\min}=\min(C_h,C_c)
$$

$$
C_{\max}=\max(C_h,C_c)
$$

## 13.3 热容量率比

$$
C_r
=
\frac{C_{\min}}{C_{\max}}
$$

## 13.4 传热单元数

$$
NTU
=
\frac{UA}{C_{\min}}
$$

## 13.5 逆流换热器有效度

当 $C_r\ne1$：

$$
\varepsilon
=
\frac{
1-\exp[-NTU(1-C_r)]
}
{
1-C_r\exp[-NTU(1-C_r)]
}
$$

当 $C_r\approx1$：

$$
\varepsilon
=
\frac{NTU}{1+NTU}
$$

## 13.6 最大理论换热量

$$
\dot Q_{\max}
=
C_{\min}(T_{h,\mathrm{in}}-T_{c,\mathrm{in}})
$$

## 13.7 实际换热量

$$
\dot Q
=
\varepsilon\dot Q_{\max}
$$

## 13.8 出口温度

热侧：

$$
T_{h,\mathrm{out}}
=
T_{h,\mathrm{in}}-
\frac{\dot Q}{C_h}
$$

冷侧：

$$
T_{c,\mathrm{out}}
=
T_{c,\mathrm{in}}+
\frac{\dot Q}{C_c}
$$

---

# 14. 空气散热器模型

## 14.1 空气质量流量

$$
\dot m_{\mathrm{air}}
=
0.12
+0.035v_{\mathrm{vehicle}}
+0.85u_{\mathrm{fan}}
$$

三部分分别代表基础自然风、高速迎风和风扇强制风。

## 14.2 空气侧热容量率

$$
C_{\mathrm{air}}
=
\dot m_{\mathrm{air}}c_{p,\mathrm{air}}
$$

$$
c_{p,\mathrm{air}}=1006\ \mathrm{J/(kg\cdot K)}
$$

## 14.3 冷却液侧热容量率

$$
C_{\mathrm{cool}}
=
\dot m_{\mathrm{cool}}c_{p,f}
$$

## 14.4 随空气流量变化的UA

$$
UA_{\mathrm{rad}}
=
UA_{\mathrm{nom}}
\left[
0.25
+0.75
\left(
\frac{\dot m_{\mathrm{air}}}{1.82}
\right)^{0.65}
\right]
$$

动力系统散热器：

$$
UA_{\mathrm{nom,p}}=850\ \mathrm{W/K}
$$

电池散热器：

$$
UA_{\mathrm{nom,b}}=620\ \mathrm{W/K}
$$

## 14.5 散热器换热量

根据epsilon-NTU模型：

$$
\dot Q_{\mathrm{rad}}
=
\varepsilon_{\mathrm{rad}}
C_{\min}
(T_{\mathrm{cool,in}}-T_{\mathrm{amb}})
$$

当冷却液低于环境时，$\dot Q_{\mathrm{rad}}$ 可以为负，表示环境向冷却液加热。实际系统可通过旁通阀避免不需要的反向换热。

## 14.6 风扇功率

$$
P_{\mathrm{fan}}
=
P_{\mathrm{fan,max}}u_{\mathrm{fan}}^3
$$

$$
P_{\mathrm{fan,max}}=420\ \mathrm W
$$

立方关系来自风机相似定律，说明高风扇转速的边际能耗很高。

---

# 15. 液液换热器模型

液液换热器同样使用epsilon-NTU方法。

热侧热容量率：

$$
C_h=\dot m_hc_{p,h}
$$

冷侧热容量率：

$$
C_c=\dot m_cc_{p,c}
$$

默认：

$$
UA_{\mathrm{llhx}}=900\ \mathrm{W/K}
$$

换热量：

$$
\dot Q_{h\rightarrow c}
=
\varepsilon C_{\min}
(T_{h,\mathrm{in}}-T_{c,\mathrm{in}})
$$

出口温度分别为：

$$
T_{h,\mathrm{out}}
=
T_{h,\mathrm{in}}
-\frac{\dot Q_{h\rightarrow c}}{C_h}
$$

$$
T_{c,\mathrm{out}}
=
T_{c,\mathrm{in}}
+\frac{\dot Q_{h\rightarrow c}}{C_c}
$$

---

# 16. 准稳态热泵模型

## 16.1 基本关系

制热性能系数：

$$
COP_{\mathrm{heat}}
=
\frac{\dot Q_{\mathrm{heat}}}{P_{\mathrm{comp}}}
$$

制冷性能系数：

$$
COP_{\mathrm{cool}}
=
\frac{|\dot Q_{\mathrm{cool}}|}{P_{\mathrm{comp}}}
$$

## 16.2 制热冷热端温度

$$
T_h
=
\max(T_{\mathrm{sink}}+273.15+5,275)
$$

$$
T_c
=
\min(T_{\mathrm{amb}}+273.15-3,T_h-3)
$$

## 16.3 Carnot制热COP

$$
COP_{\mathrm{Carnot,heat}}
=
\frac{T_h}{T_h-T_c}
$$

## 16.4 低温结霜退化因子

$$
f_{\mathrm{frost}}
=
\begin{cases}
0.68, & T_{\mathrm{amb}}<-10^\circ\mathrm C\\
0.82, & -10^\circ\mathrm C\le T_{\mathrm{amb}}<2^\circ\mathrm C\\
1.00, & T_{\mathrm{amb}}\ge2^\circ\mathrm C
\end{cases}
$$

## 16.5 实际制热COP

$$
COP_{\mathrm{heat}}
=
\operatorname{clip}
\left(
\eta_{\mathrm{Carnot}}
COP_{\mathrm{Carnot,heat}}
f_{\mathrm{frost}},
1.0,4.5
\right)
$$

$$
\eta_{\mathrm{Carnot}}=0.42
$$

## 16.6 环境相关制热能力

$$
\dot Q_{\mathrm{heat,max}}(T_{\mathrm{amb}})
=
7000
\max
\left[
0.45,
\min\left(
1,
\frac{T_{\mathrm{amb}}+30}{40}
\right)
\right]
$$

在 $-20^\circ\mathrm C$ 时，能力约为：

$$
7000\times0.45=3150\ \mathrm W
$$

## 16.7 实际制热量

$$
\dot Q_{\mathrm{heat}}
=
\min
\left(
\dot Q_{\mathrm{request}},
\dot Q_{\mathrm{heat,max}}
\right)
$$

## 16.8 制冷端温度

$$
T_c
=
\max(T_{\mathrm{sink}}+273.15-7,268)
$$

$$
T_h
=
\max(T_{\mathrm{amb}}+273.15+8,T_c+3)
$$

## 16.9 Carnot制冷COP

$$
COP_{\mathrm{Carnot,cool}}
=
\frac{T_c}{T_h-T_c}
$$

## 16.10 实际制冷COP

$$
COP_{\mathrm{cool}}
=
\operatorname{clip}
\left(
0.42COP_{\mathrm{Carnot,cool}},
1.2,4.2
\right)
$$

## 16.11 制冷能力限制

$$
|\dot Q_{\mathrm{cool}}|
\le
6500\ \mathrm W
$$

## 16.12 压缩机电功率

$$
P_{\mathrm{comp}}
=
\frac{|\dot Q_{\mathrm{thermal}}|}{COP}
$$

该模型属于系统级准稳态模型，不计算制冷剂压力、焓值、过热度、过冷度和两相换热器瞬态。

---

# 17. PTC电加热模型

## 17.1 PTC热效率

$$
\eta_{\mathrm{PTC}}=0.97
$$

## 17.2 PTC热量

$$
\dot Q_{\mathrm{PTC}}
=
\eta_{\mathrm{PTC}}P_{\mathrm{PTC}}
$$

## 17.3 电池PTC

电池PTC最大热请求约为：

$$
\dot Q_{\mathrm{bat,heater,max}}
=4500\ \mathrm W
$$

实际电功率：

$$
P_{\mathrm{bat,heater}}
=
\frac{4500u_{\mathrm{bat,PTC}}}{0.97}
$$

电池获得热量：

$$
\dot Q_{\mathrm{bat,heater}}
=
0.97P_{\mathrm{bat,heater}}
$$

## 17.4 座舱辅助PTC

$$
P_{\mathrm{cabin,PTC}}
=
5000u_{\mathrm{cabin,PTC}}
$$

$$
\dot Q_{\mathrm{cabin,PTC}}
=
0.97P_{\mathrm{cabin,PTC}}
$$

极寒时，电池PTC和座舱PTC可以同时运行，但会显著增加附件能耗。

---

# 18. 电池chiller模型

## 18.1 chiller制冷请求

$$
\dot Q_{\mathrm{chiller,request}}
=
-6000u_{\mathrm{chiller}}
$$

负号表示从电池冷却液中移除热量。

## 18.2 实际移除热量

热泵模型输出为负制冷量，因此定义：

$$
\dot Q_{\mathrm{chiller,removed}}
=
\max(-\dot Q_{\mathrm{HP,chiller}},0)
$$

## 18.3 chiller电功率

$$
P_{\mathrm{chiller}}
=
\frac{\dot Q_{\mathrm{chiller,removed}}}
{COP_{\mathrm{cool}}}
$$

## 18.4 环境温度需求因子

一般冷却模式下：

$$
f_{\mathrm{amb,chiller}}
=
\operatorname{clip}
\left(
\frac{T_{\mathrm{amb}}-22}{12},
0,1
\right)
$$

$$
u_{\mathrm{chiller}}
=
u_{\mathrm{bat,cooling}}
f_{\mathrm{amb,chiller}}
$$

环境温度较低时优先使用空气散热器；环境温度高时增加chiller需求。

---

# 19. 动力系统余热回收模型

## 19.1 可用电驱余热

可用余热不是全部电驱损耗，而是实际进入动力冷却液的热量：

$$
\dot Q_{\mathrm{waste,available}}
=
\max
\left(
\dot Q_{\mathrm{edrive\rightarrow coolant}},0
\right)
$$

## 19.2 座舱制热需求

$$
\dot Q_{\mathrm{cabin,heat,request}}
=
\max(\dot Q_{\mathrm{HVAC,request}},0)
$$

## 19.3 实际余热回收

$$
\dot Q_{\mathrm{waste,recovered}}
=
u_{\mathrm{waste,valve}}
\min
\left(
\dot Q_{\mathrm{waste,available}},
\dot Q_{\mathrm{cabin,heat,request}}
\right)
$$

余热回收必须同时满足有热源和有热需求。

---

# 20. 共享压缩机能力分配

## 20.1 原始请求

座舱压缩机请求：

$$
u_c\in[0,1]
$$

电池chiller请求：

$$
u_b\in[0,1]
$$

## 20.2 总请求

$$
u_{\Sigma}=u_c+u_b
$$

## 20.3 缩放因子

$$
s
=
\frac{1}{\max(u_{\Sigma},1)}
$$

## 20.4 实际分配

$$
u_{c,\mathrm{actual}}=su_c
$$

$$
u_{b,\mathrm{actual}}=su_b
$$

因此：

$$
u_{c,\mathrm{actual}}
+u_{b,\mathrm{actual}}
\le1
$$

该约束防止座舱和电池同时使用100%压缩机能力。

---

# 21. 冷却液集总热容模型

## 21.1 电池冷却液热容

电池回路冷却液等效质量为：

$$
m_{\mathrm{cool,b}}=3.5\ \mathrm{kg}
$$

热容为：

$$
C_{\mathrm{cool,b}}
=
m_{\mathrm{cool,b}}c_{p,f}
$$

## 21.2 电池冷却液能量方程

$$
C_{\mathrm{cool,b}}
\frac{dT_{\mathrm{cool,b}}}{dt}
=
\dot Q_{\mathrm{bat\rightarrow cool}}
-\dot Q_{\mathrm{bat,rad}}
-\dot Q_{\mathrm{chiller}}
-UA_{\mathrm{amb,b}}
(T_{\mathrm{cool,b}}-T_{\mathrm{amb}})
$$

其中：

$$
UA_{\mathrm{amb,b}}=90\ \mathrm{W/K}
$$

## 21.3 动力系统冷却液热容

$$
m_{\mathrm{cool,p}}=5.5\ \mathrm{kg}
$$

$$
C_{\mathrm{cool,p}}
=
m_{\mathrm{cool,p}}c_{p,f}
$$

## 21.4 动力系统冷却液方程

$$
C_{\mathrm{cool,p}}
\frac{dT_{\mathrm{cool,p}}}{dt}
=
\dot Q_{\mathrm{edrive\rightarrow coolant}}
-\dot Q_{\mathrm{powertrain,rad}}
-\dot Q_{\mathrm{waste,recovered}}
$$

这两个液体节点用于描述热量在部件产生后不会瞬间排出，而是可以暂时储存在冷却液中的动态过程。

---

# 22. 基准状态机控制模型

## 22.1 座舱温差

$$
e_{\mathrm{cabin}}
=
T_{\mathrm{set}}-T_{\mathrm{cabin}}
$$

$$
T_{\mathrm{set}}=24^\circ\mathrm C
$$

死区：

$$
|e_{\mathrm{cabin}}|\le0.8^\circ\mathrm C
$$

死区内不产生新的大幅HVAC热请求。

## 22.2 电池冷却滞环

开启一般冷却：

$$
T_{\mathrm{bat}}
\ge
T_{\mathrm{cool,on}}
$$

基准值：

$$
T_{\mathrm{cool,on}}=34^\circ\mathrm C
$$

解除冷却锁存：

$$
T_{\mathrm{bat}}
\le
T_{\mathrm{cool,on}}-2^\circ\mathrm C
$$

## 22.3 强化冷却

$$
T_{\mathrm{bat}}
\ge
T_{\mathrm{high}}=39^\circ\mathrm C
$$

强化冷却时电池泵、风扇和chiller具有较高优先级。

## 22.4 电池预热

$$
T_{\mathrm{bat}}
<
T_{\mathrm{heat,on}}=12^\circ\mathrm C
$$

加热严重度：

$$
u_{\mathrm{bat,heater}}
=
\min
\left[
\frac{T_{\mathrm{heat,on}}-T_{\mathrm{bat}}}{10}
+0.25,
1
\right]
$$

## 22.5 极寒座舱PTC需求

环境低温严重度：

$$
s_{\mathrm{cold}}
=
\operatorname{clip}
\left(
\frac{5-T_{\mathrm{amb}}}{25},
0,1
\right)
$$

座舱PTC请求：

$$
u_{\mathrm{cabin,PTC}}
=
s_{\mathrm{cold}}
\operatorname{clip}
\left(
\frac{\dot Q_{\mathrm{cabin,request}}}{6500},
0,1
\right)
$$

## 22.6 余热回收启用条件

基准余热门槛：

$$
\dot Q_{\mathrm{waste,min}}
=1200\ \mathrm W
$$

当：

$$
\dot Q_{\mathrm{waste,available}}
>
\dot Q_{\mathrm{waste,min}}
$$

且座舱存在正制热需求时，优先启用余热回收。

---

# 23. PID控制模型

## 23.1 误差

$$
e_k=r_k-y_k
$$

对于冷却控制，使用反向误差，使温度超过设定值时输出增加：

$$
e_{\mathrm{cool},k}
=
y_k-r_k
$$

## 23.2 比例项

$$
P_k=K_pe_k
$$

## 23.3 积分项

$$
I_k
=
K_i
\sum_{j=0}^{k}e_j\Delta t
$$

积分状态：

$$
z_k
=
\operatorname{clip}
(z_{k-1}+e_k\Delta t,
-z_{\max},z_{\max})
$$

## 23.4 微分项

$$
D_k
=
K_d
\frac{e_k-e_{k-1}}{\Delta t}
$$

当前项目 $K_d=0$，避免离散工况噪声放大。

## 23.5 未限幅输出

$$
u_k^*
=
K_pe_k+K_iz_k+K_d\frac{e_k-e_{k-1}}{\Delta t}
$$

## 23.6 限幅输出

$$
u_k
=
\operatorname{clip}
(u_k^*,u_{\min},u_{\max})
$$

## 23.7 条件积分抗饱和

如果：

$$
u_k=u_{\max}\quad\text{且}\quad e_k>0
$$

则暂停进一步正向积分。

如果：

$$
u_k=u_{\min}\quad\text{且}\quad e_k<0
$$

则暂停进一步负向积分。

## 23.8 PID参数

| 控制对象 | $K_p$ | $K_i$ | $K_d$ |
|---|---:|---:|---:|
| 电池冷却 | 0.12 | 0.004 | 0 |
| 动力系统冷却 | 0.07 | 0.002 | 0 |
| 座舱温度 | 0.20 | 0.003 | 0 |

---

# 24. 执行器一阶动态模型

## 24.1 连续模型

$$
\tau\frac{dx}{dt}+x=u
$$

等价为：

$$
\frac{dx}{dt}
=
\frac{u-x}{\tau}
$$

其中：

- $u$ 为控制器目标命令；
- $x$ 为实际执行器状态；
- $\tau$ 为时间常数。

## 24.2 精确离散

假设一个时间步内命令保持不变：

$$
x_{k+1}
=
x_k
+\left(
1-e^{-\Delta t/\tau}
\right)
(u_k-x_k)
$$

## 24.3 项目时间常数

| 执行器 | $\tau$ |
|---|---:|
| 电池泵 | 8 s |
| 动力泵 | 8 s |
| 风扇 | 6 s |
| 座舱压缩机 | 20 s |
| 电池chiller | 20 s |
| 电池/通用PTC | 5 s |
| 座舱PTC | 5 s |

## 24.4 模型性质

当 $u$ 保持常数时：

$$
\lim_{t\rightarrow\infty}x(t)=u
$$

如果 $x_0,u\in[0,1]$，则精确离散更新保持：

$$
x_k\in[0,1]
$$

---

# 25. LSTM数据窗口模型

## 25.1 离散时间序列

设每个采样点特征向量为：

$$
\mathbf{x}_t\in\mathbb{R}^{11}
$$

目标热负荷向量为：

$$
\mathbf{y}_t
=
\begin{bmatrix}
\dot Q_{\mathrm{battery},t}\\
\dot Q_{\mathrm{powertrain},t}\\
\dot Q_{\mathrm{cabin},t}
\end{bmatrix}
\in\mathbb{R}^{3}
$$

## 25.2 历史窗口

历史长度：

$$
H=60
$$

对应：

$$
H\Delta t=60\times5=300\ \mathrm s
$$

输入张量：

$$
\mathbf{X}_t
=
\begin{bmatrix}
\mathbf{x}_{t-H+1}\\
\mathbf{x}_{t-H+2}\\
\vdots\\
\mathbf{x}_{t}
\end{bmatrix}
\in\mathbb{R}^{60\times11}
$$

## 25.3 预测窗口

预测长度：

$$
F=60
$$

目标张量：

$$
\mathbf{Y}_t
=
\begin{bmatrix}
\mathbf{y}_{t+1}\\
\mathbf{y}_{t+2}\\
\vdots\\
\mathbf{y}_{t+F}
\end{bmatrix}
\in\mathbb{R}^{60\times3}
$$

## 25.4 输入特征向量

$$
\mathbf{x}_t
=
\begin{bmatrix}
v_t,
T_{\mathrm{amb},t},
T_{\mathrm{bat},t},
T_{\mathrm{motor},t},
T_{\mathrm{cabin},t},
P_{\mathrm{bat},t},
u_{\mathrm{pump},t},
t,
\dot Q_{\mathrm{bat},t},
\dot Q_{\mathrm{powertrain},t},
\dot Q_{\mathrm{cabin},t}
\end{bmatrix}^{\!T}
$$

历史热负荷只使用当前和过去时刻，因此不属于未来标签泄漏。

---

# 26. 数据标准化模型

## 26.1 特征均值与标准差

对训练episode中的第 $j$ 个特征：

$$
\mu_j
=
\frac{1}{N_{\mathrm{train}}}
\sum_{i=1}^{N_{\mathrm{train}}}
x_{ij}
$$

$$
\sigma_j
=
\sqrt{
\frac{1}{N_{\mathrm{train}}}
\sum_{i=1}^{N_{\mathrm{train}}}
(x_{ij}-\mu_j)^2
}
$$

## 26.2 标准化

$$
\tilde x_{ij}
=
\frac{x_{ij}-\mu_j}{\sigma_j}
$$

输出目标使用独立标准化器：

$$
\tilde y_{ik}
=
\frac{y_{ik}-\mu_{y,k}}{\sigma_{y,k}}
$$

## 26.3 反标准化

$$
\hat y_{ik}
=
\tilde{\hat y}_{ik}\sigma_{y,k}
+\mu_{y,k}
$$

标准化器只在训练episode上拟合，验证和测试数据不得参与均值与标准差估计。

---

# 27. LSTM单元门控方程

## 27.1 输入、隐藏状态与记忆状态

在时间步 $t$：

$$
\mathbf{x}_t\in\mathbb{R}^{d_x}
$$

$$
\mathbf{h}_{t-1}\in\mathbb{R}^{d_h}
$$

$$
\mathbf{c}_{t-1}\in\mathbb{R}^{d_h}
$$

项目隐藏维度：

$$
d_h=48
$$

## 27.2 遗忘门

$$
\mathbf{f}_t
=
\sigma
\left(
W_f\mathbf{x}_t
+U_f\mathbf{h}_{t-1}
+\mathbf{b}_f
\right)
$$

遗忘门决定保留多少旧记忆：

$$
0<\mathbf{f}_t<1
$$

## 27.3 输入门

$$
\mathbf{i}_t
=
\sigma
\left(
W_i\mathbf{x}_t
+U_i\mathbf{h}_{t-1}
+\mathbf{b}_i
\right)
$$

## 27.4 候选记忆

$$
\tilde{\mathbf{c}}_t
=
\tanh
\left(
W_c\mathbf{x}_t
+U_c\mathbf{h}_{t-1}
+\mathbf{b}_c
\right)
$$

## 27.5 记忆状态更新

$$
\mathbf{c}_t
=
\mathbf{f}_t\odot\mathbf{c}_{t-1}
+\mathbf{i}_t\odot\tilde{\mathbf{c}}_t
$$

其中 $\odot$ 表示逐元素乘法。

## 27.6 输出门

$$
\mathbf{o}_t
=
\sigma
\left(
W_o\mathbf{x}_t
+U_o\mathbf{h}_{t-1}
+\mathbf{b}_o
\right)
$$

## 27.7 隐藏状态

$$
\mathbf{h}_t
=
\mathbf{o}_t\odot\tanh(\mathbf{c}_t)
$$

## 27.8 激活函数

Sigmoid：

$$
\sigma(z)
=
\frac{1}{1+e^{-z}}
$$

双曲正切：

$$
\tanh(z)
=
\frac{e^z-e^{-z}}{e^z+e^{-z}}
$$

LSTM通过门控结构保留长时间尺度信息，适合描述热惯性和驾驶负荷历史。

---

# 28. 编码器-解码器LSTM模型

## 28.1 编码器

编码器依次读取60步历史：

$$
(\mathbf{h}_t^{(l)},\mathbf{c}_t^{(l)})
=
\operatorname{LSTM}^{(l)}
(\mathbf{h}_t^{(l-1)},
\mathbf{h}_{t-1}^{(l)},
\mathbf{c}_{t-1}^{(l)})
$$

其中 $l=1,2$ 表示两层LSTM。

最终上下文向量：

$$
\mathbf{z}
=
\mathbf{h}_{H}^{(2)}
$$

## 28.2 解码器初始热负荷

解码器不是从零开始，而是使用最后一个历史热负荷：

$$
\hat{\mathbf y}_0
=
\tilde{\mathbf y}_{t}
$$

这相当于使用持久性模型作为初始基线。

## 28.3 解码器输入

第 $j$ 个预测步输入：

$$
\mathbf d_j
=
\begin{bmatrix}
\hat{\mathbf y}_{j-1}\\
\mathbf z
\end{bmatrix}
$$

维度为：

$$
d_d=3+48=51
$$

## 28.4 解码器状态更新

$$
(\mathbf h_j^d,\mathbf c_j^d)
=
\operatorname{LSTM}_{\mathrm{dec}}
(\mathbf d_j,
\mathbf h_{j-1}^d,
\mathbf c_{j-1}^d)
$$

## 28.5 输出投影

第一线性层：

$$
\mathbf a_j
=
W_1\mathbf h_j^d+\mathbf b_1
$$

ReLU：

$$
\mathbf r_j
=
\max(\mathbf a_j,0)
$$

第二线性层：

$$
\Delta\hat{\mathbf y}_j
=
W_2\mathbf r_j+\mathbf b_2
$$

## 28.6 残差预测

$$
\hat{\mathbf y}_j
=
\hat{\mathbf y}_{j-1}
+0.2\Delta\hat{\mathbf y}_j
$$

残差结构让网络学习热负荷相对于前一步的变化，而不是每一步重新预测绝对水平。

## 28.7 输出张量

$$
\hat{\mathbf Y}
=
\begin{bmatrix}
\hat{\mathbf y}_1\\
\hat{\mathbf y}_2\\
\vdots\\
\hat{\mathbf y}_{60}
\end{bmatrix}
\in\mathbb{R}^{60\times3}
$$

批量输入时：

$$
\hat{\mathbf Y}_{\mathrm{batch}}
\in
\mathbb{R}^{B\times60\times3}
$$

---

# 29. Teacher Forcing与自回归预测

## 29.1 Teacher Forcing

训练时，以概率 $p_{\mathrm{TF}}$ 使用真实上一时刻目标作为解码器输入：

$$
\mathbf y_{j-1}^{\mathrm{input}}
=
\begin{cases}
\mathbf y_{j-1}, & r<p_{\mathrm{TF}}\\
\hat{\mathbf y}_{j-1}, & r\ge p_{\mathrm{TF}}
\end{cases}
$$

其中：

$$
r\sim\mathcal U(0,1)
$$

## 29.2 Teacher Forcing比例衰减

$$
p_{\mathrm{TF}}(e)
=
\max
\left[
0,
0.5
\left(
1-\frac{e}{E_{\max}}
\right)
\right]
$$

其中 $e$ 为当前epoch，$E_{\max}=35$。

## 29.3 推理阶段

推理阶段：

$$
p_{\mathrm{TF}}=0
$$

每一步都使用模型自身上一时刻输出，以真实反映自回归误差累积。

---

# 30. LSTM训练损失与优化

## 30.1 标准化MSE损失

批量大小为 $B$，预测长度为 $F=60$，目标数为 $K=3$：

$$
\mathcal L_{\mathrm{MSE}}
=
\frac{1}{BFK}
\sum_{b=1}^{B}
\sum_{j=1}^{F}
\sum_{k=1}^{K}
\left(
\tilde{\hat y}_{b,j,k}
-\tilde y_{b,j,k}
\right)^2
$$

在标准化空间训练可以避免量级较大的目标主导损失。

## 30.2 AdamW概念形式

一阶矩估计：

$$
\mathbf m_t
=
\beta_1\mathbf m_{t-1}
+(1-\beta_1)\mathbf g_t
$$

二阶矩估计：

$$
\mathbf v_t
=
\beta_2\mathbf v_{t-1}
+(1-\beta_2)\mathbf g_t^2
$$

偏差修正：

$$
\hat{\mathbf m}_t
=
\frac{\mathbf m_t}{1-\beta_1^t}
$$

$$
\hat{\mathbf v}_t
=
\frac{\mathbf v_t}{1-\beta_2^t}
$$

AdamW参数更新可写为：

$$
\boldsymbol\theta_{t+1}
=
\boldsymbol\theta_t
-\alpha
\frac{\hat{\mathbf m}_t}
{\sqrt{\hat{\mathbf v}_t}+\epsilon}
-\alpha\lambda\boldsymbol\theta_t
$$

项目参数：

$$
\alpha=0.001,qquad
\lambda=10^{-5}
$$

## 30.3 梯度裁剪

如果总梯度范数超过1：

$$
\mathbf g
\leftarrow
\mathbf g
\frac{1}{\|\mathbf g\|_2}
$$

一般形式：

$$
\mathbf g
\leftarrow
\mathbf g
\min
\left(
1,
\frac{g_{\max}}{\|\mathbf g\|_2}
\right)
$$

$$
g_{\max}=1.0
$$

## 30.4 早停

如果验证损失连续 $p$ 个epoch未改善，则停止训练：

$$
p=6
$$

保存验证损失最小的模型参数：

$$
\boldsymbol\theta^*
=
\arg\min_{\boldsymbol\theta_e}
\mathcal L_{\mathrm{val}}(e)
$$

最终正式模型最佳epoch为35。

---

# 31. 预测评价指标

## 31.1 平均绝对误差

$$
MAE
=
\frac{1}{N}
\sum_{i=1}^{N}
|\hat y_i-y_i|
$$

## 31.2 均方根误差

$$
RMSE
=
\sqrt{
\frac{1}{N}
\sum_{i=1}^{N}
(\hat y_i-y_i)^2
}
$$

RMSE对较大误差更敏感。

## 31.3 决定系数

$$
R^2
=
1-
\frac{
\sum_i(\hat y_i-y_i)^2
}
{
\sum_i(y_i-\bar y)^2
}
$$

解释：

- $R^2=1$：完美预测；
- $R^2=0$：与预测测试集均值相当；
- $R^2<0$：比均值预测更差。

## 31.4 不同预测时距误差

60 s对应第12个未来采样点：

$$
j_{60}=\frac{60}{5}=12
$$

180 s：

$$
j_{180}=36
$$

300 s：

$$
j_{300}=60
$$

对每个时距计算三个目标的平均绝对误差：

$$
MAE_{\mathrm{all},j}
=
\frac{1}{3N}
\sum_{n=1}^{N}
\sum_{k=1}^{3}
|\hat y_{n,j,k}-y_{n,j,k}|
$$

---

# 32. 预测增强阈值模型

## 32.1 未来电池热负荷严重度

设LSTM预测未来300 s电池热负荷峰值为：

$$
\dot Q_{\mathrm{bat,peak}}^{\mathrm{pred}}
=
\max_{1\le j\le60}
\hat{\dot Q}_{\mathrm{bat},j}
$$

严重度：

$$
s_{\mathrm{heat}}
=
\operatorname{clip}
\left(
\frac{
\dot Q_{\mathrm{bat,peak}}^{\mathrm{pred}}-500
}{3500},
0,1
\right)
$$

## 32.2 一般冷却阈值

$$
T_{\mathrm{cool,on}}^{\mathrm{pred}}
=
\max
\left(
28,
34-6s_{\mathrm{heat}}
\right)
$$

预测峰值越高，冷却越早启动；最低阈值限制为28 °C。

## 32.3 强化冷却阈值

$$
T_{\mathrm{high}}^{\mathrm{pred}}
=
\max
\left(
36,
39-2s_{\mathrm{heat}}
\right)
$$

## 32.4 余热门槛

设未来座舱平均负荷：

$$
\bar Q_{\mathrm{cabin}}^{\mathrm{pred}}
=
\frac{1}{60}
\sum_{j=1}^{60}
\hat Q_{\mathrm{cabin},j}
$$

当：

$$
\bar Q_{\mathrm{cabin}}^{\mathrm{pred}}>1000\ \mathrm W
$$

余热回收门槛降低400 W，但不低于700 W：

$$
Q_{\mathrm{waste,min}}^{\mathrm{pred}}
=
\max(700,1200-400)
=800\ \mathrm W
$$

其他情况下保持1200 W。

## 32.5 AI不直接控制执行器

预测模型输出映射为阈值：

$$
\hat{\mathbf Y}
\rightarrow
\left{
T_{\mathrm{cool,on}},
T_{\mathrm{high}},
Q_{\mathrm{waste,min}}
\right}
$$

实际执行器仍由：

$$
\text{实际温度}
+\text{状态机}
+\text{PID}
+\text{执行器动态}
$$

共同决定。

---

# 33. 预测有效性与故障降级模型

## 33.1 历史长度条件

$$
N_{\mathrm{history}}\ge60
$$

历史不足时：

$$
\mathrm{valid}=0
$$

## 33.2 输出形状

$$
\hat{\mathbf Y}
\in
\mathbb{R}^{60\times3}
$$

## 33.3 有限性条件

$$
\forall i,j,
\quad
\hat Y_{ij}\in\mathbb{R}
$$

不允许NaN或Inf。

## 33.4 工程范围

$$
\max_{i,j}|\hat Y_{ij}|
\le100000\ \mathrm W
$$

## 33.5 降级逻辑

$$
\mathrm{valid}=0
\quad\Longrightarrow\quad
\boldsymbol\theta_{\mathrm{threshold}}
=
\boldsymbol\theta_{\mathrm{baseline}}
$$

其中基准阈值为不可变副本，防止预测阈值累积漂移。

---

# 34. 整车电功率模型

## 34.1 附件总功率

$$
P_{\mathrm{aux}}
=
P_{\mathrm{pump,b}}
+P_{\mathrm{pump,p}}
+P_{\mathrm{fan}}
+P_{\mathrm{comp,cabin}}
+P_{\mathrm{chiller}}
+P_{\mathrm{PTC,b}}
+P_{\mathrm{PTC,cabin}}
$$

## 34.2 电池总请求功率

$$
P_{\mathrm{bat,request}}
=
P_{\mathrm{edrive,dc}}
+P_{\mathrm{aux}}
$$

电池等效电路根据该功率求实际电流和端功率。

## 34.3 电功率平衡残差

$$
r_e
=
P_{\mathrm{bat,terminal}}
-\left(
P_{\mathrm{edrive,dc}}
+P_{\mathrm{aux}}
\right)
$$

---

# 35. 热量守恒模型

## 35.1 一般热储能方程

对任一集总热节点：

$$
\frac{dE_{\mathrm{th}}}{dt}
=
\sum\dot Q_{\mathrm{in}}
-\sum\dot Q_{\mathrm{out}}
$$

如果热容常数：

$$
E_{\mathrm{th}}=CT
$$

因此：

$$
C\frac{dT}{dt}
=
\sum\dot Q_{\mathrm{in}}
-\sum\dot Q_{\mathrm{out}}
$$

## 35.2 子系统热残差

$$
r_{\mathrm{th},i}
=
\frac{\Delta E_{\mathrm{stored},i}}{\Delta t}
-\dot Q_{\mathrm{boundary},i}
$$

项目分别计算：

$$
i\in
\left{
\mathrm{battery},
\mathrm{edrive},
\mathrm{cabin},
\mathrm{coolant,b},
\mathrm{coolant,p}
\right}
$$

## 35.3 电池整体热残差

$$
\frac{
C_{\mathrm{core}}\Delta T_{\mathrm{core}}
+C_{\mathrm{surf}}\Delta T_{\mathrm{surf}}
}{\Delta t}
-
\left(
\dot Q_{\mathrm{bat}}
+\dot Q_{\mathrm{heater}}
-\dot Q_{\mathrm{bat\rightarrow cool}}
\right)
$$

核心到表面的内部热流在整体求和中相互抵消。

## 35.4 电驱整体热残差

$$
\frac{
C_{\mathrm{motor}}\Delta T_{\mathrm{motor}}
+C_{\mathrm{inv}}\Delta T_{\mathrm{inv}}
}{\Delta t}
-
\left(
\dot Q_{\mathrm{loss}}
-\dot Q_{\mathrm{edrive\rightarrow coolant}}
\right)
$$

## 35.5 座舱整体热残差

$$
\frac{
C_{\mathrm{air}}\Delta T_{\mathrm{air}}
+C_{\mathrm{int}}\Delta T_{\mathrm{int}}
}{\Delta t}
-
\left(
\dot Q_{\mathrm{env}}
+\dot Q_{\mathrm{solar}}
+\dot Q_{\mathrm{occupant}}
+\dot Q_{\mathrm{HVAC}}
\right)
$$

空气和内饰之间的内部换热相互抵消。

## 35.6 总热残差

$$
r_{\mathrm{th,total}}
=
\sum_i r_{\mathrm{th},i}
$$

## 35.7 归一化热平衡误差

$$
\varepsilon_{\mathrm{th}}
=
\frac{
\sum_k|r_{\mathrm{th,total},k}|\Delta t
}
{
\sum_k\dot Q_{\mathrm{throughput},k}\Delta t
}
\times100\%
$$

数值守恒只能证明代码内部能量路径一致，不能代替实验标定和模型确认。

---

# 36. 能耗、续航与舒适性评价模型

## 36.1 行驶距离

$$
S
=
\frac{1}{1000}
\int_0^{t_f}v(t)dt
$$

离散梯形积分：

$$
S
\approx
\frac{1}{1000}
\sum_{k=0}^{N-1}
\frac{v_k+v_{k+1}}{2}
\Delta t
$$

单位为km。

## 36.2 净电池能耗

$$
E_{\mathrm{net,kWh}}
=
\frac{1}{3.6\times10^6}
\sum_kP_{\mathrm{bat,total},k}\Delta t
$$

负再生功率会降低净能耗。

## 36.3 附件能耗

$$
E_{\mathrm{aux,kWh}}
=
\frac{1}{3.6\times10^6}
\sum_kP_{\mathrm{aux},k}\Delta t
$$

## 36.4 百公里电耗

$$
EC_{100}
=
100\frac{E_{\mathrm{net,kWh}}}{S}
$$

单位为 $\mathrm{kWh/100km}$。

## 36.5 等效续航

$$
R_{\mathrm{eq}}
=
100
\frac{E_{\mathrm{battery,kWh}}}{EC_{100}}
$$

$$
E_{\mathrm{battery,kWh}}=75\ \mathrm{kWh}
$$

## 36.6 座舱温度误差

$$
e_{T,k}
=
T_{\mathrm{cabin},k}
-T_{\mathrm{set}}
$$

## 36.7 座舱舒适RMSE

$$
RMSE_{\mathrm{cabin}}
=
\sqrt{
\frac{1}{N}
\sum_{k=1}^{N}e_{T,k}^2
}
$$

该指标包含从初始温度升至舒适区的整个过程，因此极寒启动工况RMSE会明显较大。

## 36.8 余热回收能量

$$
E_{\mathrm{waste,kWh}}
=
\frac{1}{3.6\times10^6}
\sum_k
\dot Q_{\mathrm{waste,recovered},k}
\Delta t
$$

---

# 37. 显式时间积分与稳定性

## 37.1 一般状态方程

$$
\frac{dx}{dt}=f(x,u,t)
$$

显式Euler离散：

$$
x_{k+1}
=
x_k+\Delta t f(x_k,u_k,t_k)
$$

## 37.2 热系统时间常数

对于一阶热节点：

$$
C\frac{dT}{dt}=\dot Q-UA(T-T_\infty)
$$

其特征时间常数：

$$
\tau_{\mathrm{th}}
=
\frac{C}{UA}
$$

时间步应明显小于系统最小重要时间常数。项目选择5 s，并通过有限性、范围和能量平衡测试验证稳定性。

## 37.3 数值限幅

项目对以下变量设置边界：

$$
SOC\in[0,1]
$$

$$
u\in[0,1]
$$

$$
\varepsilon\in[0,1]
$$

$$
\eta_{\mathrm{edrive}}\in[0.78,0.97]
$$

$$
COP_{\mathrm{heat}}\in[1.0,4.5]
$$

$$
COP_{\mathrm{cool}}\in[1.2,4.2]
$$

限幅用于保证经验模型不在适用范围外产生非物理值，但不能替代真实部件约束标定。

---

# 38. 各模型之间的输入输出关系

## 38.1 纵向动力学

输入：

$$
\{v,a,\theta\}
$$

输出：

$$
\{F_{\mathrm{trac}},P_{\mathrm{wheel}},
n_{\mathrm{motor}},T_{\mathrm{motor}}\}
$$

## 38.2 电驱模型

输入：

$$
\{P_{\mathrm{mech}},n_{\mathrm{motor}},
T_{\mathrm{cool,p}},\dot m_p\}
$$

输出：

$$
\{P_{\mathrm{dc}},\dot Q_{\mathrm{motor}},
\dot Q_{\mathrm{inv}},T_{\mathrm{motor}},T_{\mathrm{inv}}\}
$$

## 38.3 电池模型

输入：

$$
\{P_{\mathrm{dc}},P_{\mathrm{aux}},SOC,
T_{\mathrm{core}},T_{\mathrm{surf}},T_{\mathrm{cool,b}}\}
$$

输出：

$$
\{I,U_{\mathrm{terminal}},SOC^+,
\dot Q_{\mathrm{bat}},T_{\mathrm{core}}^+,
T_{\mathrm{surf}}^+\}
$$

## 38.4 热流体回路

输入：

$$
\{u_{\mathrm{pump}},u_{\mathrm{fan}},u_{\mathrm{chiller}},
T_{\mathrm{component}},T_{\mathrm{amb}}\}
$$

输出：

$$
\{\dot m,\Delta p,P_{\mathrm{pump}},
\dot Q_{\mathrm{cp}},\dot Q_{\mathrm{rad}},
\dot Q_{\mathrm{chiller}},T_{\mathrm{cool}}^+\}
$$

## 38.5 座舱模型

输入：

$$
\{T_{\mathrm{amb}},I_{\mathrm{solar}},N_{\mathrm{occ}},
\dot Q_{\mathrm{HVAC}}\}
$$

输出：

$$
\{T_{\mathrm{air}}^+,T_{\mathrm{int}}^+,
\dot Q_{\mathrm{cabin,load}}\}
$$

## 38.6 LSTM模型

输入：

$$
\mathbf X_t\in\mathbb R^{60\times11}
$$

输出：

$$
\hat{\mathbf Y}_t\in\mathbb R^{60\times3}
$$

## 38.7 预测控制层

输入：

$$
\{\hat Q_{\mathrm{bat,peak}},
\bar Q_{\mathrm{powertrain}},
\bar Q_{\mathrm{cabin}},
\mathrm{valid}\}
$$

输出：

$$
\{T_{\mathrm{cool,on}},T_{\mathrm{high}},
Q_{\mathrm{waste,min}}\}
$$

---

# 39. 正式模型结果与公式理解

## 39.1 最终预测指标

| 目标 | MAE | RMSE | $R^2$ |
|---|---:|---:|---:|
| 电池产热 | 396.86 W | 589.53 W | 0.539 |
| 电驱余热 | 427.95 W | 537.26 W | 0.567 |
| 座舱净热负荷 | 153.75 W | 206.34 W | 0.981 |

## 39.2 为什么座舱预测更容易

座舱2R2C方程具有明显低通特性：

$$
\tau_{\mathrm{cabin}}
\sim
\frac{C}{UA}
$$

较大热容会平滑高频变化。电池热量则包含：

$$
\dot Q_{\mathrm{irr}}=I^2R
$$

电流尖峰经过平方后更突出，因此更难用历史状态预测未来高频变化。

## 39.3 预测时距误差

$$
MAE_{60\mathrm s}=337.93\ \mathrm W
$$

$$
MAE_{180\mathrm s}=381.19\ \mathrm W
$$

$$
MAE_{300\mathrm s}=396.23\ \mathrm W
$$

误差随时距增加，但未发生明显发散。

## 39.4 温度-能耗权衡

六工况平均：

$$
\Delta T_{\mathrm{bat,max}}
=-0.280^\circ\mathrm C
$$

$$
\Delta EC_{100}
=+0.0874\ \mathrm{kWh/100km}
$$

$$
\Delta R_{\mathrm{eq}}
=-2.41\ \mathrm{km}
$$

说明提前冷却以附件能耗换取热安全裕度。

---

# 40. 模型假设、适用范围与学习重点

## 40.1 纵向动力学假设

- 只考虑一维纵向运动；
- 不考虑轮胎滑移、横摆和侧倾；
- 空气密度为常数；
- 传动效率使用集总值。

## 40.2 电池模型假设

- 电池包等效为单一电压源和内阻；
- 不描述电化学极化支路；
- 核心和表面各自温度均匀；
- 参数未针对具体车型标定；
- 不包含老化和热失控。

## 40.3 电驱模型假设

- 效率面为解析近似；
- 电机与逆变器各采用一个热节点；
- 损耗分配比例固定；
- 不描述定子、转子和永磁体内部温差。

## 40.4 热流体模型假设

- 冷却液物性由经验关联式给出；
- 并联冷板流道均匀分流；
- 换热器采用准稳态epsilon-NTU；
- 冷却液回路使用集总温度；
- 不考虑气泡、泄漏、相变和结冰。

## 40.5 热泵模型假设

- 不求解制冷剂压力和焓；
- 使用Carnot修正COP；
- 低温结霜通过经验因子表示；
- 能力分配为系统级近似。

## 40.6 LSTM模型假设

- 训练episode能够覆盖部署分布；
- 历史物理估算热负荷在线可获得；
- 采样周期固定为5 s；
- 预测时域固定为300 s；
- 仿真数据与实车之间存在sim-to-real差异。

## 40.7 学习时必须区分的三个概念

### 数值守恒

$$
\text{储能变化}=\text{净边界能量}
$$

证明代码没有明显丢能或重复计能。

### 物理合理性

温度、流量、功率、效率和热流方向符合工程规律。

### 实车准确性

需要通过台架和道路数据进行参数标定与模型确认。数值守恒不能自动证明实车准确性。

---

# 41. 推荐的模型学习练习

## 41.1 纵向动力学练习

给定：

$$
m=1950\ \mathrm{kg},
\quad v=20\ \mathrm{m/s},
\quad a=0,
\quad \theta=0
$$

分别计算滚阻、风阻、总牵引力和轮端功率，再比较5°上坡结果。

## 41.2 电池电流练习

给定：

$$
U_{\mathrm{oc}}=390\ \mathrm V,
\quad R=0.065\ \Omega,
\quad P=60000\ \mathrm W
$$

使用二次方程低根计算电流，并与简单近似 $I=P/U_{\mathrm{oc}}$ 比较。

## 41.3 电池产热练习

根据上题电流，计算：

$$
I^2R
$$

再加入：

$$
-IT\frac{dU_{\mathrm{oc}}}{dT}
$$

比较不可逆热和可逆热占比。

## 41.4 管路压降练习

改变质量流量，计算 $Re$、$f$ 和 $\Delta p$，验证湍流区域压降近似随流量平方增加。

## 41.5 冷板练习

分别取低流量和高流量，计算 $Re$、$Nu$、$h$、$UA$、有效度、换热量和泵功，分析边际换热收益。

## 41.6 热泵练习

分别取环境温度：

$$
-20^\circ\mathrm C,
\quad 0^\circ\mathrm C,
\quad 10^\circ\mathrm C
$$

计算Carnot COP、结霜修正、实际COP和最大制热能力，解释极寒PTC介入原因。

## 41.7 LSTM窗口练习

画出一个长度为150步的episode，标出：

$$
60\ \text{步历史}
+60\ \text{步预测}
$$

并说明为什么窗口不能跨episode、为什么StandardScaler只能拟合训练episode。

## 41.8 预测阈值练习

分别令预测电池热峰值为：

$$
500,\ 2000,\ 4000,\ 8000\ \mathrm W
$$

计算 $s_{\mathrm{heat}}$、一般冷却阈值和强化冷却阈值，观察限幅作用。

---

# 42. 总结

整个项目可以归纳为四组核心方程：

## 42.1 功率来源

$$
\text{驾驶循环}
\rightarrow
F_{\mathrm{trac}}
\rightarrow
P_{\mathrm{wheel}}
\rightarrow
P_{\mathrm{dc}}
\rightarrow
P_{\mathrm{battery}}
$$

## 42.2 热量来源

$$
\dot Q_{\mathrm{battery}}
=
I^2R-IT\frac{dU_{\mathrm{oc}}}{dT}
$$

$$
\dot Q_{\mathrm{edrive}}
=
|P_{\mathrm{electrical}}-P_{\mathrm{mechanical}}|
$$

$$
\dot Q_{\mathrm{cabin,load}}
=
\dot Q_{\mathrm{environment}}
+\dot Q_{\mathrm{solar}}
+\dot Q_{\mathrm{occupant}}
$$

## 42.3 热量传递

$$
C\frac{dT}{dt}
=
\sum\dot Q_{\mathrm{in}}
-\sum\dot Q_{\mathrm{out}}
$$

$$
Re\rightarrow Nu\rightarrow h\rightarrow UA
\rightarrow\varepsilon\rightarrow\dot Q
$$

## 42.4 预测增强控制

$$
\mathbf X_{t-59:t}
\xrightarrow{\mathrm{LSTM}}
\hat{\mathbf Y}_{t+1:t+60}
\xrightarrow{\mathrm{bounded\ scheduling}}
\text{状态机+PID+执行器动态}
$$

项目的核心思想不是用AI替代物理方程，而是让物理模型负责能量守恒、传热和温升，让LSTM负责提供未来热负荷信息，再由可解释控制器决定是否值得提前干预。
