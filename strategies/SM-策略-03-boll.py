"""
布林带极限反转 · A股定制版（超卖极值 + 均值回归 + ATR 硬止损）
策略与引擎分离：本模块只负责生成信号与仓位参数，不涉及资金扣减。
"""
from __future__ import annotations

import polars as pl

# 可调参数（便于回测时覆盖）
BOLL_WINDOW = 20  # 中轨周期（MA）
K_LOWER = 2.5     # 下轨乘数（捕捉极度恐慌）
ATR_WINDOW = 20   # ATR 周期
STOP_ATR_MULT = 1.5  # 硬止损：入场价下方 1.5*ATR


def generate_boll_reversal_signals(df: pl.DataFrame) -> pl.DataFrame:
    """
    根据日线数据生成“布林带极限反转”信号与持仓状态。

    输入：必须包含 ts_code, trade_date, open, high, low, close。
    输出：原表 + mb, lower_band, atr, entry_signal, exit_signal, position_status。
    """
    # -------------------------------------------------------------------------
    # 1) 数据预处理：按股票分组、按日期升序（rolling/shift 依赖顺序）
    # -------------------------------------------------------------------------
    df = df.sort(["ts_code", "trade_date"])

    # -------------------------------------------------------------------------
    # 2) 布林带（无未来函数：指标用 shift(1) 使今日只能看到“昨日”的指标）
    #    mb：close 的 BOLL_WINDOW 均线
    #    std：close 的 BOLL_WINDOW 滚动标准差
    #    lower_band：mb - K_LOWER * std
    # -------------------------------------------------------------------------
    mb_raw = pl.col("close").rolling_mean(BOLL_WINDOW).over("ts_code")
    std_raw = pl.col("close").rolling_std(BOLL_WINDOW).over("ts_code")
    mb = mb_raw.shift(1).over("ts_code").alias("mb")
    lower_band = (mb_raw - pl.lit(K_LOWER) * std_raw).shift(1).over("ts_code").alias("lower_band")

    # -------------------------------------------------------------------------
    # 3) True Range 与 ATR（照抄海龟逻辑：用于动态止损）
    # -------------------------------------------------------------------------
    prev_close = pl.col("close").shift(1).over("ts_code")
    tr = pl.max_horizontal(
        pl.col("high") - pl.col("low"),
        (pl.col("high") - prev_close).abs(),
        (pl.col("low") - prev_close).abs(),
    ).alias("tr")
    atr = tr.rolling_mean(ATR_WINDOW).over("ts_code").alias("atr")

    # -------------------------------------------------------------------------
    # 4) 入场/出场信号（向量化）
    #    入场（entry_signal）：
    #      昨日 close < 昨日 lower_band（超卖） 且 今日 close > 今日 open（收红盘，拒绝接飞刀）
    #    出场（exit_signal）初版：
    #      今日 close > 昨日 mb（均值回归目标完成）
    # -------------------------------------------------------------------------
    prev_close2 = pl.col("close").shift(1).over("ts_code")
    entry_signal = (
        (prev_close2 < pl.col("lower_band")) & (pl.col("close") > pl.col("open"))
    ).cast(pl.Int8).alias("entry_signal")
    exit_base = (pl.col("close") > pl.col("mb")).cast(pl.Int8)

    # -------------------------------------------------------------------------
    # 5) 动态止损：追踪每次入场成本价，并在 exit_signal 中加入硬止损
    #    entry_price：仅在入场日记录 close，其余为 null，随后组内 forward_fill 追踪“最近一次入场价”
    #    硬止损：close < entry_price - 1.5 * atr
    # -------------------------------------------------------------------------
    entry_price = (
        pl.when(pl.col("entry_signal") == 1)
        .then(pl.col("close"))
        .otherwise(None)
        .forward_fill()
        .over("ts_code")
        .alias("entry_price")
    )
    # ATR 极小值保护（避免 entry_price - 1.5*atr 失真）
    atr_safe = pl.max_horizontal(pl.col("atr"), pl.col("close") * 1e-4).replace(0.0, None)
    stop_loss = (pl.col("close") < (pl.col("entry_price") - pl.lit(STOP_ATR_MULT) * atr_safe)).cast(pl.Int8)
    exit_signal = (exit_base | stop_loss).cast(pl.Int8).alias("exit_signal")

    # -------------------------------------------------------------------------
    # 6) 持仓状态机：入场=1，出场=0，否则延续前一日状态（前向填充）
    # -------------------------------------------------------------------------
    raw_state = (
        pl.when(pl.col("entry_signal") == 1)
        .then(1)
        .when(pl.col("exit_signal") == 1)
        .then(0)
        .otherwise(None)
        .cast(pl.Int8)
    )
    position_status = raw_state.forward_fill().over("ts_code").fill_null(0).alias("position_status")

    # -------------------------------------------------------------------------
    # 7) 一次性 with_columns，保证列顺序清晰
    # -------------------------------------------------------------------------
    return (
        df.with_columns([mb, lower_band])
        .with_columns([tr, atr])
        .with_columns([entry_signal])
        .with_columns([entry_price])
        .with_columns([exit_signal])
        .with_columns([position_status])
        .drop("tr")  # 若需保留 TR 列可删掉本行
    )


if __name__ == "__main__":
    # 最小示例：构造几条数据验证
    demo = pl.DataFrame(
        {
            "ts_code": ["000001.SZ"] * 80,
            "trade_date": [f"202401{(i+1):02d}" for i in range(80)],
            "open": [10.0 + i * 0.02 for i in range(80)],
            "high": [10.2 + i * 0.02 for i in range(80)],
            "low": [9.8 + i * 0.02 for i in range(80)],
            "close": [10.0 + i * 0.02 + (0.1 if (i % 7 == 0) else -0.05) for i in range(80)],
        }
    )
    out = generate_boll_reversal_signals(demo)
    print(out.tail(15))

