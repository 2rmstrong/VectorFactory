"""
海龟交易法 · A股定制版（纯多头 + 唐奇安通道 + ATR）
策略与引擎分离：本模块只负责生成信号与仓位参数，不涉及资金扣减。
"""
from __future__ import annotations

import polars as pl

# 可调参数（便于回测时覆盖）
DONCHIAN_ENTRY_WINDOW = 20  # 入场通道：过去 N 日最高价
DONCHIAN_EXIT_WINDOW = 10   # 出场通道：过去 N 日最低价
ATR_WINDOW = 20             # ATR 周期


def generate_turtle_signals(df: pl.DataFrame) -> pl.DataFrame:
    """
    根据日线数据生成海龟交易法信号与仓位状态。

    输入：必须包含 ts_code, trade_date, open, high, low, close。
    输出：原表 + donchian_upper, donchian_lower, atr, entry_signal, exit_signal, position_status。
    """
    # -------------------------------------------------------------------------
    # 1. 数据预处理：按股票分组、按日期升序（rolling/shift 依赖顺序）
    # -------------------------------------------------------------------------
    df = df.sort(["ts_code", "trade_date"])

    # -------------------------------------------------------------------------
    # 2. 唐奇安通道（无未来函数：用 shift(1) 使今日只能看到昨日通道）
    # -------------------------------------------------------------------------
    # 入场线：过去 20 日最高价，再下移一天 → 今日收盘突破“昨日”的 20 日高点才入场
    donchian_upper_raw = pl.col("high").rolling_max(DONCHIAN_ENTRY_WINDOW).over("ts_code")
    donchian_upper = donchian_upper_raw.shift(1).over("ts_code").alias("donchian_upper")

    # 出场线：过去 10 日最低价，再下移一天 → 今日收盘跌破“昨日”的 10 日低点才出场
    donchian_lower_raw = pl.col("low").rolling_min(DONCHIAN_EXIT_WINDOW).over("ts_code")
    donchian_lower = donchian_lower_raw.shift(1).over("ts_code").alias("donchian_lower")

    # -------------------------------------------------------------------------
    # 3. True Range 与 ATR（核心灵魂：用于动态仓位/风险控制）
    # -------------------------------------------------------------------------
    prev_close = pl.col("close").shift(1).over("ts_code")
    tr = pl.max_horizontal(
        pl.col("high") - pl.col("low"),
        (pl.col("high") - prev_close).abs(),
        (pl.col("low") - prev_close).abs(),
    ).alias("tr")
    atr = tr.rolling_mean(ATR_WINDOW).over("ts_code").alias("atr")

    # -------------------------------------------------------------------------
    # 4. 纯多头信号：入场 / 出场 / 持仓状态
    # -------------------------------------------------------------------------
    # 入场：今日收盘 > 昨日 20 日高点（比较的是“已平移”的 donchian_upper）
    entry_signal = (pl.col("close") > pl.col("donchian_upper")).cast(pl.Int8).alias("entry_signal")
    # 出场：今日收盘 < 昨日 10 日低点
    exit_signal = (pl.col("close") < pl.col("donchian_lower")).cast(pl.Int8).alias("exit_signal")

    # 持仓状态：入场=1，出场=0，否则延续前一日状态（前向填充）
    # 先得到“状态变化点”：仅在有明确入场/出场时为 1/0，其余为 null
    raw_state = (
        pl.when(pl.col("entry_signal") == 1)
        .then(1)
        .when(pl.col("exit_signal") == 1)
        .then(0)
        .otherwise(None)
        .cast(pl.Int8)
    )
    # 按 ts_code 组内前向填充，首日无状态视为 0（空仓）
    position_status = raw_state.forward_fill().over("ts_code").fill_null(0).alias("position_status")

    # -------------------------------------------------------------------------
    # 5. 一次性 with_columns，保证列顺序清晰
    # -------------------------------------------------------------------------
    return (
        df.with_columns([donchian_upper, donchian_lower])
        .with_columns([tr, atr])
        .with_columns([entry_signal, exit_signal])
        .with_columns([position_status])
        .drop("tr")  # 若需保留 TR 列可删掉本行
    )


if __name__ == "__main__":
    # 最小示例：构造几条数据验证
    demo = pl.DataFrame({
        "ts_code": ["000001.SZ"] * 50,
        "trade_date": [f"202401{(i+1):02d}" for i in range(50)],
        "open": [10.0 + i * 0.01 for i in range(50)],
        "high": [10.5 + i * 0.01 for i in range(50)],
        "low": [9.8 + i * 0.01 for i in range(50)],
        "close": [10.2 + i * 0.01 for i in range(50)],
    })
    out = generate_turtle_signals(demo)
    print(out.tail(15))
