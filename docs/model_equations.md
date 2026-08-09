# 物理模型与方程说明

## 1. 整车纵向动力学

牵引力由惯性力、滚动阻力、空气阻力和坡度阻力组成：

```text
F_trac = m*a + m*g*Crr*cos(theta) + 0.5*rho*Cd*A*v^2 + m*g*sin(theta)
P_wheel = F_trac*v
```

驱动时轮端功率除以传动效率得到电机机械侧需求；制动时根据传动效率和最大回收功率限制再生能量。项目规定正功率为电池放电，负功率为回充。

## 2. 电池电气与产热

端电压和功率关系为 `U_t = OCV - I*R`、`P_t = U_t*I`。给定端功率后求解二次方程的低电流根。SOC由库仑积分计算：

```text
dSOC/dt = -I / (3600*C_Ah)
```

Bernardi集总产热为：

```text
Q_bat = I^2*R - I*T*dOCV/dT
```

第一项为不可逆欧姆热，第二项为可逆熵热。核心和表面节点满足：

```text
C_core*dT_core/dt = Q_bat - G_cs*(T_core-T_surface)
C_surface*dT_surface/dt = G_cs*(T_core-T_surface) - Q_coldplate
```

## 3. 电驱损耗与温升

电机效率由归一化转速和负荷构成的二维效率面给出。驱动时 `P_dc=P_mech/eta`，回收时 `P_dc=P_mech*eta`。两侧功率差全部转化为电机铜铁损耗和逆变器开关/导通损耗。两个部件分别采用单热容节点并与动力冷却液换热。

## 4. 座舱2R2C

空气节点和内饰节点分别储存显热。环境通过车身传热进入内饰，通过渗透风直接作用空气；太阳得热按比例进入两个节点，乘员显热和HVAC直接作用空气。模型能体现空气温度响应快、内饰温度响应慢的热惯性差异。

## 5. 液路与泵

冷却液速度、Reynolds数和Darcy-Weisbach压降为：

```text
u = m_dot/(rho*A)
Re = rho*u*D_h/mu
DeltaP = (f*L/D_h + sum(K))*rho*u^2/2
```

层流采用 `f=64/Re`，湍流采用Swamee-Jain显式近似。水泵扬程曲线与全部命名部件压降之和求交点：

```text
DeltaP_pump(n,m_dot) = sum(DeltaP_pipe + DeltaP_local + DeltaP_plate
                           + DeltaP_valve + DeltaP_heat_exchanger)
```

泵功为 `P=DeltaP*(m_dot/rho)/eta`，因此增加流量同时增加换热和泵耗。求解器输出各部件 pressure budget 与闭合残差；不存在交点、关闭阀和非法压降不会被静默裁剪为伪工作点。

## 6. 冷板与换热器

冷板按Re区分层流/湍流，使用定热流边界Nu或Gnielinski关联式：

```text
h = Nu*k/D_h
UA = 1/(R_wall + 1/(h*A))
epsilon = 1-exp(-UA/(m_dot*cp))
Q = epsilon*m_dot*cp*(T_surface-T_coolant,in)
```

散热器和液液换热器采用逆流epsilon-NTU模型。散热器空气侧流量由车速迎风和风扇叠加，风扇功率近似与转速三次方成正比。液液换热器在两回路热账本中使用同一个 `Q`，一侧为负、另一侧为正，避免耦合热量重复计算。

## 7. 热泵、PTC与余热

热泵COP由Carnot COP乘系统效率得到，再施加低温结霜退化、能力边界和COP上下限。该模型适合整车能量研究，但不描述制冷剂压力、过热度、两相换热器动态。PTC按固定电热效率换算。电驱余热仅在温度和需求允许时通过换热支路用于座舱制热。

电池液路包含空气散热器和制冷剂chiller两条排热路径。环境温度较低时优先使用低能耗散热器；高温环境或强化冷却时，chiller从电池冷却液吸热。座舱换热和电池chiller共享归一化压缩机能力，需求之和超过1时按比例分配，避免重复计算压缩机容量和功率。

## 8. 适用范围

模型重点是系统能量流、部件温度和控制策略相对比较。默认经验参数不应解释为特定车型标定值。需要实车应用时应标定效率图、内阻、热容、UA、泵曲线、压缩机图和座舱热参数，并对数值模型进行台架与道路验证。
