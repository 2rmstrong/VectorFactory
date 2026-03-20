# ⚡ Armstrong-VectorFactory (阿姆向量工厂)
> **A High-Performance Quant Research Framework | Vectorized Alpha Generation Engine**
> 
> "Everything is a Vector. Every Alpha is a Signal."

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9+-green.svg)
![Engine](https://img.shields.io/badge/engine-Polars%20%7C%20DuckDB-orange.svg)
![Market](https://img.shields.io/badge/market-A--Share%20T%2B1-red.svg)
![Computing](https://img.shields.io/badge/Compute-GPU%20Accelerated-green.svg)

## 🏗️ 架构哲学 (The Philosophy)
**VectorFactory** 是由 Armstrong 独立搭建的高性能量化研究实验室。

在非线性的金融市场中，传统的 `for-loop` 回测早已无法满足海量数据的投研需求。

本项目全面弃用 Pandas，将底层架构迁移至 **Polars (基于 Rust 的 SIMD 向量化引擎)** 与 **DuckDB (列式 OLAP 数据库)**。

通过将 A 股市场的微观行为转化为高维特征向量，实现了从数据清洗、信号挖掘到 T+1 实盘回测的闭环。硬件要求不高。

数据上，只需要准备门槛最低的 **tushare API** 。

---

## 📂 目录拓扑 (Repository Structure)

```text
Armstrong-VectorFactory/
├── core/                  # 核心算力层：统一的事件驱动与向量化撮合底座
├── data_pipeline/         # 数据流管道：外部 API 接入与 DuckDB 增量更新层
├── strategies/            # 向量工厂：策略信号提取脚本
└── backtests/             # 实验沙盒：策略的历史回放与资金曲线测算
```

## 🚀 第一批核心战术矩阵 (Strategies)
本库首批开源了 9 套涵盖不同市场失效模式的核心量化模型。

万变不离其宗。可自行优化参数和资金管理模型。

所有逻辑均已完成 A 股特有的涨跌停流动性限制与非对称交易成本适配。

## 💡 核心 Alpha 提示 (基于 10 年以上的数据回测)
经过 10 年以上数据样本回测验证，在当前 A 股市场的微观结构下，本矩阵中有效性最强、夏普比率最高的两套模型为：

SM-05 (Fama-French 小市值底座)：在 A 股，“微小市值+低估值”因子的超额收益（Alpha）极其显著且具备长期持续性，是截面选股的最强底座因子之一。

SM-07 (Pairs Trading 协整配对)：利用强相关资产的均值回归特性，在震荡市与熊市中表现出极强的抗跌能力，贡献了极为稀缺的“市场中性 Alpha”。

## 📡 策略信号说明 (Signal Engine)

工厂内的每一套策略，都是对市场特定特征的数学剥离。以下为第一批 **4 大类**、共 **9 个模块**的核心触发逻辑：
### 1. 均值回归与统计套利
#### SM-05 (Fama-French 价值与小盘)
基于账面市值（BM）比与市值大小（Size）进行二维正交切分，做多小盘价值（Small Value），剔除大盘成长泡沫，获取 A 股最强的小盘溢价。

#### SM-07 (协整配对套利)
定期扫描全市场，对高相关性资产组合（策略中为 2 只个股）进行 Engle-Granger 协整检验，当价差（Spread）的 Z-Score 偏离度超过 ±2 时，做空高估标的、做多低估标的，赚取绝对收益。

由于涉及全市场标的的截面两两校验，计算压力极大。本项目底层算子已针对 **GPU 加速** 优化，推荐在具备 **CUDA 环境** 的终端上运行，以实现分钟级别的全场扫描。

#### SM-03 (布林带极限反转)
捕捉波动率失真。当股价向下跌穿 MA(20) - 2 * StdDev，且伴随恐慌盘涌出，在极值点进行左侧博弈，回归均线平仓。

### 2. 动量与趋势捕获
#### SM-01 (海龟交易法)
提取时序动量。当 Close > Max(High, 20) 触发多头突破，通过 ATR 动态计算建仓头寸，跌破 10 日低点无条件止损。

#### SM-02 (Dual Thrust)
捕捉短线波动率扩张。基于前 N 日的极值构建非对称震荡区间，一旦开盘价加上 Range 乘数被突破，即刻顺势切入。

#### SM-06 (截面动量效应)
强者恒强模型。剥离出过去 x 个交易日绝对收益率位于全市场头部 10% 的动能标的，进行等权轮动。

### 3. 左侧逆势与事件驱动、
#### SM-04 (RSI 底背离)
动能衰竭探测器。寻找价格创出阶段新低，但 RSI 指标未能创新低的背离奇点，确立左侧买点。

#### SM-09 (PEAD 盈余漂移)
对财报公布日的超预期程度进行量化。当实际利润击穿一致预期时，捕捉市场定价滞后产生的连续向上漂移利润。

### 4. 宏观资产配置
#### SM-08 (风险平价 Risk Parity)
放弃主观预测，基于各资产标的（股、债、黄金）的历史协方差矩阵，按 **倒数波动率（Inverse Volatility）** 分配权重，实现投资组合的整体风险系数平衡。

## 🛠️ 技术范式 (Tech Stack & Features)
SIMD 加速计算：全链路采用 Polars 的并行计算能力，告别 for 循环。

冷数据极速流转：门槛低，基于 DuckDB 列式存储，千万级日线/Tick 数据实现毫秒级索引。

无未来函数 (No Look-ahead Bias)：严格的 shift(1) 信号生成机制，确保回测所见即实盘所得。

## ⚡ 极速点火 (Quick Start)
```bash
# [STEP 1] 获取阿姆向量工厂核心源码
git clone https://github.com/2rmstrong/VectorFactory.git

# [STEP 2] 自动化部署高性能计算依赖 (Polars, DuckDB, etc.)
pip install -r requirements.txt

# [STEP 3] 启动验证 (SM-05: 小市值底座)
python backtests/backtest-05.py
```

## 🌑 开发者宣言 (Developer Manifesto)
这里没有保证盈利的圣杯，只有对代码效率和数学逻辑的绝对偏执。

第一批仅涉及最基础策略，未来会对一些有效的主观策略进行量化开发、开源。

开源策略的所有参数和资金模型都可以自行优化。

如果你在这个工厂里找到了灵感，欢迎点亮 🌟 Star。

## ⚠️ 免责声明 (Disclaimer)
本项目提供的所有策略代码及回测框架仅供技术交流与学术研究。

量化交易具有极高风险，过去的净值曲线不代表未来收益，请勿直接用于实盘资金操作。作者不对任何交易亏损负责。