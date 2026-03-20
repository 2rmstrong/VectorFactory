"""
风险平价 · 基于历史波动率倒数的动态权重分配（Risk Parity）
策略与引擎分离：本模块只负责生成权重与波动率指标，不涉及资金扣减。

核心逻辑：
- 资产池：沪深300ETF、黄金ETF、十年期国债ETF（可配置）
- 滚动年化波动率 → 逆波动率 → 月末次日截面归一化得 target_weight
- 防未来函数：调仓日使用的 inv_vol 为 shift(1) 后的“昨日”波动率
"""
from __future__ import annotations

import math
import polars as pl

# -------------------------------------------------------------------------
# 全局配置（若数据库代码后缀不同，可在此修改）
# -------------------------------------------------------------------------
# 510300 沪深300ETF / 518880 黄金ETF / 511260 十年期国债ETF
TARGET_ASSETS: list[str] = ["510300.SH", "518880.SH", "511260.SH"]
VOL_WINDOW: int = 60       # 波动率计算周期（约 3 个月）
REBALANCE_FREQ: str = "M"  # 月末调仓（M = 每月首个交易日生效）


def generate_risk_parity_weights(df: pl.DataFrame, **params) -> pl.DataFrame:
    """
    根据日线数据生成风险平价权重与波动率。

    输入：必须包含 ts_code, trade_date, close。仅处理 TARGET_ASSETS 中的标的。
    输出：ts_code, trade_date, close, vol, target_weight（长表，时间已对齐）。

    防未来函数：inv_vol 经 shift(1) 后仅在每月首个交易日参与截面归一化，
    当日开盘调仓时只使用“昨日收盘”算出的波动率。
    """
    assets = list(params.get("TARGET_ASSETS", TARGET_ASSETS))
    vol_window = int(params.get("VOL_WINDOW", VOL_WINDOW))

    # -------------------------------------------------------------------------
    # 1) 过滤资产池并排序
    # -------------------------------------------------------------------------
    need = ["ts_code", "trade_date", "close"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"输入缺少列: {missing}")

    df = (
        df.filter(pl.col("ts_code").is_in(assets))
        .select(need)
        .sort(["trade_date", "ts_code"])
    )
    if df.is_empty():
        return pl.DataFrame(schema={"ts_code": pl.Utf8, "trade_date": pl.Utf8, "close": pl.Float64, "vol": pl.Float64, "target_weight": pl.Float64})

    # -------------------------------------------------------------------------
    # 2) Pivot 宽表 → 前向填充缺失 → Melt 回长表（三只资产同日对齐）
    # -------------------------------------------------------------------------
    wide = df.pivot(
        values="close",
        index="trade_date",
        on="ts_code",
        aggregate_function="first",
    )
    # 按列前向填充（某 ETF 停牌导致的缺失）
    fill_cols = [c for c in wide.columns if c != "trade_date"]
    wide = wide.with_columns(
        [pl.col(c).fill_null(strategy="forward") for c in fill_cols]
    )
    # 再反向填充首部可能仍为 null 的（首日无数据时）
    wide = wide.with_columns(
        [pl.col(c).fill_null(strategy="backward") for c in fill_cols]
    )

    long = wide.unpivot(
        index="trade_date",
        on=fill_cols,
        variable_name="ts_code",
        value_name="close",
    ).sort(["ts_code", "trade_date"])

    # -------------------------------------------------------------------------
    # 3) 日收益率、年化波动率、逆波动率（防未来：inv_vol 整体 shift(1)）
    # -------------------------------------------------------------------------
    prev_close = pl.col("close").shift(1).over("ts_code")
    daily_ret = (pl.col("close") / prev_close - 1.0).alias("daily_ret")

    long = long.with_columns([daily_ret])
    vol_raw = (
        pl.col("daily_ret")
        .rolling_std(vol_window)
        .over("ts_code")
        .alias("vol_raw")
    )
    long = long.with_columns([vol_raw])
    # 年化波动率
    vol = (pl.col("vol_raw") * math.sqrt(252)).alias("vol")
    long = long.with_columns([vol])

    # 逆波动率：仅 vol > 0 时取倒数，否则 null
    inv_vol = (
        pl.when(pl.col("vol") > 1e-12)
        .then(1.0 / pl.col("vol"))
        .otherwise(None)
        .alias("inv_vol")
    )
    long = long.with_columns([inv_vol])
    # 【防未来函数】调仓日只能用“昨日”的 inv_vol
    inv_vol_lag = pl.col("inv_vol").shift(1).over("ts_code").alias("inv_vol_lag")
    long = long.with_columns([inv_vol_lag])

    # -------------------------------------------------------------------------
    # 4) 月末次日 = 每月首个交易日，仅该日做截面归一化
    # -------------------------------------------------------------------------
    unique_dates = (
        long.select("trade_date")
        .unique()
        .sort("trade_date")
        .with_columns(pl.col("trade_date").str.slice(0, 6).alias("ym"))
    )
    first_of_month = (
        unique_dates.group_by("ym")
        .agg(pl.col("trade_date").min().alias("rebal_date"))
        .select("rebal_date")
    )
    rebal_list = first_of_month.to_series().to_list()
    is_rebal = pl.col("trade_date").is_in(rebal_list).alias("is_rebal")
    long = long.with_columns([is_rebal])

    # 截面归一化：仅调仓日 target_weight = inv_vol_lag / sum(inv_vol_lag) over trade_date
    sum_inv = pl.col("inv_vol_lag").sum().over("trade_date")
    target_weight = (
        pl.when(pl.col("is_rebal") & (sum_inv > 1e-12))
        .then(pl.col("inv_vol_lag") / sum_inv)
        .otherwise(None)
        .alias("target_weight")
    )
    long = long.with_columns([target_weight])

    # -------------------------------------------------------------------------
    # 5) 状态传递：月初权重整月保持（按 ts_code 前向填充）
    # -------------------------------------------------------------------------
    target_weight_ff = pl.col("target_weight").forward_fill().over("ts_code").alias("target_weight")
    long = long.with_columns([target_weight_ff])

    # -------------------------------------------------------------------------
    # 6) 返回约定列（去掉中间列）
    # -------------------------------------------------------------------------
    return long.select(
        [
            "ts_code",
            "trade_date",
            "close",
            "vol",
            "target_weight",
        ]
    )


if __name__ == "__main__":
    # 最小示例：三只 ETF 约 80 个交易日，验证 vol 与 target_weight
    import random
    random.seed(42)
    assets = ["510300.SH", "518880.SH", "511260.SH"]
    # 生成连续交易日（去掉周末，简化用 60 天）
    dates = []
    for m in (1, 2, 3):
        for d in range(1, 21):
            dates.append(f"2024{m:02d}{d:02d}")
    rows = []
    for i, d in enumerate(dates):
        for j, code in enumerate(assets):
            # 三只不同波动：股 > 黄金 > 债
            sig = [0.012, 0.010, 0.004][j]
            r = random.gauss(0.0002, sig)
            if i == 0:
                close = 10.0
            else:
                prev = [r for r in rows if r["ts_code"] == code and r["trade_date"] == dates[i - 1]]
                close = prev[0]["close"] * (1 + r) if prev else 10.0
            rows.append({"ts_code": code, "trade_date": d, "close": round(close, 4)})
    demo = pl.DataFrame(rows).sort(["trade_date", "ts_code"])
    out = generate_risk_parity_weights(demo, VOL_WINDOW=20)
    print(out.tail(24))
