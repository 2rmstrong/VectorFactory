"""
截面动量效应 · A股定制版（强者恒强，纯多头）
策略与引擎分离：本模块只负责生成信号与持仓状态，不涉及资金扣减。
"""
from __future__ import annotations

import polars as pl

# 可调参数（便于回测时覆盖）
LOOKBACK_WINDOW = 250  # 动量观察期
SKIP_WINDOW = 20       # 剔除最近 1 个月
TOP_N_ENTRY = 100      # 截面前 N 触发买入
TOP_N_EXIT = 200       # 跌出前 N（缓冲带）触发卖出


def generate_momentum_signals(df: pl.DataFrame, **params) -> pl.DataFrame:
    """
    生成截面动量（Cross-Sectional Momentum）信号与持仓状态（纯多头）。

    输入：必须包含 ts_code, trade_date, open, close。
    输出：原表 + momentum_return, final_rank, entry_signal, exit_signal, position_status

    关键要求：
    - 计算 momentum_return 时使用 .over("ts_code") 的时序 shift，避免跨股污染
    - 截面排名使用 .over("trade_date")，每日日终只用当日截面做横向打分
    - 防未来：排名完成后把 rank_mom shift(1).over("ts_code") → final_rank
    """
    lookback = int(params.get("LOOKBACK_WINDOW", LOOKBACK_WINDOW))
    skip = int(params.get("SKIP_WINDOW", SKIP_WINDOW))
    top_entry = int(params.get("TOP_N_ENTRY", TOP_N_ENTRY))
    top_exit = int(params.get("TOP_N_EXIT", TOP_N_EXIT))

    # -------------------------------------------------------------------------
    # 1) 数据预处理：按股票分组、按日期升序（rolling/shift 依赖顺序）
    # -------------------------------------------------------------------------
    df = df.sort(["ts_code", "trade_date"])

    # -------------------------------------------------------------------------
    # 2) 时序动量：从 250 天前到 20 天前的累计收益率
    #    momentum_return = close.shift(SKIP) / close.shift(LOOKBACK) - 1
    # -------------------------------------------------------------------------
    close_skip = pl.col("close").shift(skip).over("ts_code")
    close_lookback = pl.col("close").shift(lookback).over("ts_code")
    momentum_return = (close_skip / close_lookback - 1.0).alias("momentum_return")

    # -------------------------------------------------------------------------
    # 3) 截面排名打分：按 trade_date 独立计算
    #    收益率越高越靠前 → descending=True，rank=1 为最高动量
    #    防未来：将 rank_mom 进行 shift(1).over("ts_code") → final_rank
    # -------------------------------------------------------------------------
    # 注意：必须对“已落地的列”做截面 rank
    # 否则 Polars 可能把 rank 当作“嵌套窗口表达式”处理，导致整截面 rank 结果变为 null。
    # 修复：先落地 momentum_return，再用 pl.col("momentum_return") 做 over("trade_date")。
    df = df.with_columns([momentum_return])
    rank_mom = pl.col("momentum_return").rank(method="dense", descending=True).over("trade_date").alias("rank_mom")
    df = df.with_columns([rank_mom])
    df = df.with_columns(pl.col("rank_mom").shift(1).over("ts_code").alias("final_rank"))

    # -------------------------------------------------------------------------
    # 4) 目标池/剔除池（用昨日 final_rank 决定今日买卖）
    #    target_pool：昨日 final_rank <= TOP_N_ENTRY
    #    drop_pool  ：昨日 final_rank >  TOP_N_EXIT
    # -------------------------------------------------------------------------
    df = df.with_columns(
        [
            (pl.col("final_rank") <= pl.lit(top_entry)).cast(pl.Int8).alias("target_pool"),
            (pl.col("final_rank") > pl.lit(top_exit)).cast(pl.Int8).alias("drop_pool"),
        ]
    )

    # -------------------------------------------------------------------------
    # 5) 信号生成与状态机（标准状态机风格）
    #    entry_signal：target_pool==1 且 昨日不在 target_pool（空仓简化条件）
    #    exit_signal ：drop_pool==1
    # -------------------------------------------------------------------------
    prev_target = pl.col("target_pool").shift(1).over("ts_code")
    entry_signal = (
        (pl.col("target_pool") == 1) & ((prev_target == 0) | prev_target.is_null())
    ).cast(pl.Int8).alias("entry_signal")
    exit_signal = (pl.col("drop_pool") == 1).cast(pl.Int8).alias("exit_signal")

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
    # 6) 返回（保留用于对接回测底座的列）
    # -------------------------------------------------------------------------
    return (
        df.with_columns([entry_signal, exit_signal])
        .with_columns([position_status])
        .drop(["rank_mom"])  # 只保留 final_rank 供对接
    )


if __name__ == "__main__":
    # 最小示例：构造两只股票的日线数据，验证截面排名与状态机可运行
    demo = pl.DataFrame(
        {
            "ts_code": ["A"] * 260 + ["B"] * 260,
            "trade_date": [f"2020{(i+1):03d}" for i in range(260)] * 2,
            "open": [10.0 + i * 0.01 for i in range(260)] + [10.0 + i * 0.008 for i in range(260)],
            "close": [10.0 + i * 0.012 for i in range(260)] + [10.0 + i * 0.009 for i in range(260)],
        }
    )

    out = generate_momentum_signals(demo, LOOKBACK_WINDOW=60, SKIP_WINDOW=10, TOP_N_ENTRY=1, TOP_N_EXIT=1)
    # 只打印末尾几天，观察 final_rank / entry / exit / position_status
    print(out.tail(20).select(["trade_date", "ts_code", "momentum_return", "final_rank", "entry_signal", "exit_signal", "position_status"]))

