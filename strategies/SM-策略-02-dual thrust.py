"""
Dual Thrust · A股日线区间突破（纯多头）
策略与引擎分离：本模块只负责生成信号与持仓状态，不涉及资金扣减。
"""
from __future__ import annotations

import polars as pl


def generate_dual_thrust_signals(
    df: pl.DataFrame,
    N: int = 5,
    K1: float = 0.5,
    K2: float = 0.5,
) -> pl.DataFrame:
    """
    根据日线数据生成 Dual Thrust 的上下轨与持仓状态（纯多头版本）。

    输入：必须包含 ts_code, trade_date, open, high, low, close。
    输出：原表 + Range, buy_line, sell_line, entry_signal, exit_signal, position_status。

    关键避坑（防未来函数）：
    - HH/LC/HC/LL 这些 N 日极值必须先 rolling，再 shift(1) 下移一位；
      这样 “今天的触发区间 Range” 只使用到“昨天及以前”的既定事实。
    """
    # -------------------------------------------------------------------------
    # 1) 数据预处理：rolling/shift 依赖顺序，必须按 ts_code 分组并按日期升序
    # -------------------------------------------------------------------------
    df = df.sort(["ts_code", "trade_date"])

    # -------------------------------------------------------------------------
    # 2) 计算 N 日极值，并 shift(1) 避免未来函数
    #    HH: 过去 N 日最高价的最大值
    #    LC: 过去 N 日收盘价的最小值
    #    HC: 过去 N 日收盘价的最大值
    #    LL: 过去 N 日最低价的最小值
    # -------------------------------------------------------------------------
    hh = (
        pl.col("high")
        .rolling_max(N)
        .over("ts_code")
        .shift(1)
        .over("ts_code")
        .alias("_hh")
    )
    lc = (
        pl.col("close")
        .rolling_min(N)
        .over("ts_code")
        .shift(1)
        .over("ts_code")
        .alias("_lc")
    )
    hc = (
        pl.col("close")
        .rolling_max(N)
        .over("ts_code")
        .shift(1)
        .over("ts_code")
        .alias("_hc")
    )
    ll = (
        pl.col("low")
        .rolling_min(N)
        .over("ts_code")
        .shift(1)
        .over("ts_code")
        .alias("_ll")
    )

    # -------------------------------------------------------------------------
    # 3) Range = Max(HH - LC, HC - LL)
    #    注意：Range 基于“已 shift(1) 的极值”，因此今天的 Range 不含今天的数据
    # -------------------------------------------------------------------------
    range_expr = (
        pl.max_horizontal(
            (pl.col("_hh") - pl.col("_lc")),
            (pl.col("_hc") - pl.col("_ll")),
        )
        .replace([float("inf"), float("-inf")], None)
        .fill_nan(None)
        .alias("Range")
    )

    # -------------------------------------------------------------------------
    # 4) 信号轨道：开盘价 +/− 系数 * Range
    #    buy_line  = open + K1 * Range
    #    sell_line = open - K2 * Range
    # -------------------------------------------------------------------------
    buy_line = (pl.col("open") + pl.lit(K1) * pl.col("Range")).alias("buy_line")
    sell_line = (pl.col("open") - pl.lit(K2) * pl.col("Range")).alias("sell_line")

    # -------------------------------------------------------------------------
    # 5) 信号与持仓状态（纯多头）
    #    entry_signal：close > buy_line
    #    exit_signal ：close < sell_line
    #    position_status：entry=1、exit=0、否则 forward_fill 延续
    # -------------------------------------------------------------------------
    entry_signal = (pl.col("close") > pl.col("buy_line")).cast(pl.Int8).alias("entry_signal")
    exit_signal = (pl.col("close") < pl.col("sell_line")).cast(pl.Int8).alias("exit_signal")

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
    # 6) 一次性链式追加，保持结构清晰（并删除内部临时列）
    # -------------------------------------------------------------------------
    return (
        df.with_columns([hh, lc, hc, ll])
        .with_columns([range_expr])
        .with_columns([buy_line, sell_line])
        .with_columns([entry_signal, exit_signal])
        .with_columns([position_status])
        .drop(["_hh", "_lc", "_hc", "_ll"])
    )


if __name__ == "__main__":
    # 最小示例：构造少量数据验证列是否生成
    demo = pl.DataFrame(
        {
            "ts_code": ["000001.SZ"] * 20,
            "trade_date": [f"202401{(i+1):02d}" for i in range(20)],
            "open": [10.0 + i * 0.05 for i in range(20)],
            "high": [10.2 + i * 0.05 for i in range(20)],
            "low": [9.8 + i * 0.05 for i in range(20)],
            "close": [10.1 + i * 0.05 for i in range(20)],
        }
    )
    out = generate_dual_thrust_signals(demo, N=5, K1=0.5, K2=0.5)
    print(out.tail(8))
