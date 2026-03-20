"""
Fama-French 小盘价值因子 · A股定制版（截面排名选股，纯多头）
策略与引擎分离：本模块只负责生成信号与仓位状态，不涉及资金扣减。
"""
from __future__ import annotations

import polars as pl

# 可调参数（便于回测时覆盖）
TOP_N_ENTRY = 100  # 进入池：排名前 N
TOP_N_EXIT = 150   # 退出池：跌出前 N（缓冲带降低换手）


def generate_ff_signals(
    df: pl.DataFrame,
    TOP_N_ENTRY: int = TOP_N_ENTRY,
    TOP_N_EXIT: int = TOP_N_EXIT,
) -> pl.DataFrame:
    """
    根据截面因子（小盘 + 低 PB）生成选股信号与持仓状态（纯多头）。

    输入：必须包含 ts_code, trade_date, open, close, total_mv, pb。
    可选：volume（如存在则剔除 volume==0 的停牌/极度枯竭标的）。

    输出：原表 + final_rank_lag, target_pool, drop_pool, entry_signal, exit_signal, position_status。

    关键要求：
    - 截面排名必须使用 over("trade_date")
    - 防未来函数：final_rank 计算后 shift(1).over("ts_code")，只能用昨天排名决定今天买卖
    """
    # -------------------------------------------------------------------------
    # 1) 数据预处理：按日期/股票排序（shift 依赖顺序）
    # -------------------------------------------------------------------------
    df = df.sort(["ts_code", "trade_date"])

    # -------------------------------------------------------------------------
    # 2) 数据清洗（极度重要）
    #    - 剔除 PB<=0 的资不抵债股
    #    - 若存在 volume 列，则剔除 volume==0 的停牌/流动性枯竭
    # -------------------------------------------------------------------------
    df = df.filter(pl.col("pb") > 0)
    if "volume" in df.columns:
        df = df.filter(pl.col("volume") > 0)

    # -------------------------------------------------------------------------
    # 3) 截面因子打分（每日独立 over("trade_date")）
    #    rank_mv：市值越小越靠前（升序）
    #    rank_pb：PB 越低越靠前（升序）
    #    combined_score：两者相加
    #    final_rank：对 combined_score 再次升序排名（越小越优）
    # -------------------------------------------------------------------------
    rank_mv = pl.col("total_mv").rank(method="dense").over("trade_date").alias("rank_mv")
    rank_pb = pl.col("pb").rank(method="dense").over("trade_date").alias("rank_pb")
    combined_score = (pl.col("rank_mv") + pl.col("rank_pb")).alias("combined_score")
    final_rank = pl.col("combined_score").rank(method="dense").over("trade_date").alias("final_rank")

    # 防未来函数：只能用“昨日”的 final_rank 来决定“今日”的买卖
    final_rank_lag = pl.col("final_rank").shift(1).over("ts_code").alias("final_rank_lag")

    # -------------------------------------------------------------------------
    # 4) 目标池/剔除池（用昨日排名）
    #    target_pool：昨日 final_rank <= TOP_N_ENTRY
    #    drop_pool  ：昨日 final_rank >  TOP_N_EXIT
    # -------------------------------------------------------------------------
    target_pool = (pl.col("final_rank_lag") <= pl.lit(int(TOP_N_ENTRY))).cast(pl.Int8).fill_null(0).alias("target_pool")
    drop_pool = (pl.col("final_rank_lag") > pl.lit(int(TOP_N_EXIT))).cast(pl.Int8).fill_null(0).alias("drop_pool")

    # -------------------------------------------------------------------------
    # 5) 信号生成 + 状态机（纯多头）
    #    entry_signal：今日进入 target_pool 且 昨日不在 target_pool（避免重复买入信号）
    #    exit_signal ：今日触发 drop_pool
    # -------------------------------------------------------------------------
    prev_target = pl.col("target_pool").shift(1).over("ts_code")
    entry_signal = (
        (pl.col("target_pool") == 1) & ((prev_target == 0) | prev_target.is_null())
    ).cast(pl.Int8).alias("entry_signal")
    exit_signal = (pl.col("drop_pool") == 1).cast(pl.Int8).alias("exit_signal")

    # 标准状态机：入场=1，出场=0，否则延续前一日状态（前向填充）
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
    # 6) 输出：一次性追加必要列（保留 rank 列便于验证/调参）
    # -------------------------------------------------------------------------
    # 注意：Polars eager 的 with_columns 同批次引用新列可能失败，因此分步落地 combined_score → final_rank
    return (
        df.with_columns([rank_mv, rank_pb])
        .with_columns([combined_score])
        .with_columns([final_rank])
        .with_columns([final_rank_lag])
        .with_columns([target_pool, drop_pool])
        .with_columns([entry_signal, exit_signal])
        .with_columns([position_status])
    )


if __name__ == "__main__":
    # 最小示例：两天 3 只股票的截面排名验证
    demo = pl.DataFrame(
        {
            "ts_code": ["A", "B", "C", "A", "B", "C"],
            "trade_date": ["20240102", "20240102", "20240102", "20240103", "20240103", "20240103"],
            "open": [10, 10, 10, 10, 10, 10],
            "close": [10, 10, 10, 10, 10, 10],
            "total_mv": [300, 200, 100, 310, 190, 120],
            "pb": [1.5, 0.8, 1.0, 1.6, 0.7, 1.1],
        }
    )
    out = generate_ff_signals(demo, TOP_N_ENTRY=1, TOP_N_EXIT=2)
    print(out.sort(["trade_date", "final_rank"]).select(["trade_date", "ts_code", "rank_mv", "rank_pb", "final_rank", "final_rank_lag", "target_pool", "entry_signal", "exit_signal", "position_status"]))

