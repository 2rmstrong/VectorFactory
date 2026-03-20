"""
RSI 底背离 · A股定制版（超卖区动能衰竭 + 价格创新低的左侧背离确认，纯多头）
策略与引擎分离：本模块只负责生成信号与仓位参数，不涉及资金扣减。
"""
from __future__ import annotations

import polars as pl

# 可调参数（便于回测时覆盖）
RSI_PERIOD = 14        # RSI 周期
LOOKBACK_WINDOW = 20   # 背离观察窗口
RSI_OVERSOLD = 30      # 超卖阈值：只在极度超卖区域内的背离才有效
ATR_WINDOW = 20        # ATR 周期（用于动态止损）
STOP_ATR_MULT = 1.5    # 硬止损：入场价下方 1.5*ATR


def generate_rsi_divergence_signals(df: pl.DataFrame) -> pl.DataFrame:
    """
    根据日线数据生成 RSI 底背离信号与持仓状态（纯多头）。

    输入：必须包含 ts_code, trade_date, open, high, low, close。
    输出：原表 + rsi, atr, entry_signal, exit_signal, position_status。

    说明：
    - RSI 使用 Wilder 平滑（等价于 alpha = 1/RSI_PERIOD 的 EMA 递推）
    - 底背离使用“滚动极值比对法”，避免复杂局部极值探测
    """
    # -------------------------------------------------------------------------
    # 1) 数据预处理：按股票分组、按日期升序（rolling/shift 依赖顺序）
    # -------------------------------------------------------------------------
    df = df.sort(["ts_code", "trade_date"])

    # -------------------------------------------------------------------------
    # 2) RSI（Wilder 平滑）与 ATR（照抄海龟逻辑）
    # -------------------------------------------------------------------------
    prev_close = pl.col("close").shift(1).over("ts_code")
    price_diff = (pl.col("close") - prev_close).alias("price_diff")

    gain = pl.when(pl.col("price_diff") > 0).then(pl.col("price_diff")).otherwise(0.0).alias("gain")
    loss = pl.when(pl.col("price_diff") < 0).then((-pl.col("price_diff"))).otherwise(0.0).alias("loss")

    # Wilder 平滑：alpha = 1/RSI_PERIOD，adjust=False
    alpha = 1.0 / float(RSI_PERIOD)
    avg_gain = pl.col("gain").ewm_mean(alpha=alpha, adjust=False).over("ts_code").alias("_avg_gain")
    avg_loss = pl.col("loss").ewm_mean(alpha=alpha, adjust=False).over("ts_code").alias("_avg_loss")

    rs = (pl.col("_avg_gain") / pl.col("_avg_loss").replace(0.0, None)).alias("_rs")
    rsi = (
        (100.0 - (100.0 / (1.0 + pl.col("_rs"))))
        .replace([float("inf"), float("-inf")], None)
        .fill_nan(None)
        .clip(0.0, 100.0)
        .alias("rsi")
    )

    tr = pl.max_horizontal(
        pl.col("high") - pl.col("low"),
        (pl.col("high") - prev_close).abs(),
        (pl.col("low") - prev_close).abs(),
    ).alias("tr")
    atr = tr.rolling_mean(ATR_WINDOW).over("ts_code").alias("atr")

    # 注意：Polars eager 的 with_columns 同批次引用新列可能失败，因此分两段落地 _rs → rsi
    df = (
        df.with_columns([price_diff])
        .with_columns([gain, loss])
        .with_columns([avg_gain, avg_loss])
        .with_columns([rs])
        .with_columns([rsi])
        .with_columns([tr, atr])
    )

    # -------------------------------------------------------------------------
    # 3) 底背离核心：滚动极值比对法（窗口极值均 shift(1) 防未来函数）
    #    past_min_close：过去窗口最低价（不含今日）
    #    past_min_rsi  ：过去窗口最低 RSI（不含今日）
    #    条件A：价格创新低  close <= past_min_close
    #    条件B：动能未创新低 rsi > past_min_rsi
    #    条件C：发生于超卖区 past_min_rsi < RSI_OVERSOLD
    # -------------------------------------------------------------------------
    past_min_close = (
        pl.col("close")
        .rolling_min(LOOKBACK_WINDOW)
        .over("ts_code")
        .shift(1)
        .over("ts_code")
        .alias("past_min_close")
    )
    past_min_rsi = (
        pl.col("rsi")
        .rolling_min(LOOKBACK_WINDOW)
        .over("ts_code")
        .shift(1)
        .over("ts_code")
        .alias("past_min_rsi")
    )

    bullish_divergence = (
        (pl.col("close") <= pl.col("past_min_close"))
        & (pl.col("rsi") > pl.col("past_min_rsi"))
        & (pl.col("past_min_rsi") < pl.lit(RSI_OVERSOLD))
    ).alias("bullish_divergence")

    # -------------------------------------------------------------------------
    # 4) 信号生成（适配 A 股 T+1：策略信号由当日收盘确认）
    #    入场：发生底背离 + 今日收红盘（close > open）
    #    出场：RSI>70（止盈）或 跌破 entry_price - 1.5*ATR（硬止损）
    # -------------------------------------------------------------------------
    entry_signal = (
        (pl.col("bullish_divergence")) & (pl.col("close") > pl.col("open"))
    ).cast(pl.Int8).alias("entry_signal")

    entry_price = (
        pl.when(pl.col("entry_signal") == 1)
        .then(pl.col("close"))
        .otherwise(None)
        .forward_fill()
        .over("ts_code")
        .alias("entry_price")
    )

    atr_safe = pl.max_horizontal(pl.col("atr"), pl.col("close") * 1e-4).replace(0.0, None)
    take_profit = (pl.col("rsi") > 70.0).cast(pl.Int8)
    stop_loss = (pl.col("close") < (pl.col("entry_price") - pl.lit(STOP_ATR_MULT) * atr_safe)).cast(pl.Int8)
    exit_signal = (take_profit | stop_loss).cast(pl.Int8).alias("exit_signal")

    # -------------------------------------------------------------------------
    # 5) 持仓状态机：入场=1，出场=0，否则延续前一日状态（前向填充）
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
    # 6) 返回：只保留对外需要的核心列（内部中间列可在此删除）
    # -------------------------------------------------------------------------
    return (
        df.with_columns([past_min_close, past_min_rsi])
        .with_columns([bullish_divergence])
        .with_columns([entry_signal])
        .with_columns([entry_price])
        .with_columns([exit_signal])
        .with_columns([position_status])
        .drop(["price_diff", "gain", "loss", "_avg_gain", "_avg_loss", "_rs", "tr"])
    )


if __name__ == "__main__":
    # 最小示例：构造几条数据验证
    n = 120
    demo = pl.DataFrame(
        {
            "ts_code": ["000001.SZ"] * n,
            "trade_date": [f"2023{(i+1):04d}" for i in range(n)],
            "open": [10.0 + (i * 0.01) for i in range(n)],
            "high": [10.2 + (i * 0.01) for i in range(n)],
            "low": [9.8 + (i * 0.01) for i in range(n)],
            # 人为加入波动与“创新低后反弹”的形态
            "close": [10.0 + (i * 0.01) - (0.6 if (60 <= i <= 65) else 0.0) + (0.2 if (i == 66) else 0.0) for i in range(n)],
        }
    )
    out = generate_rsi_divergence_signals(demo)
    print(out.tail(20))

