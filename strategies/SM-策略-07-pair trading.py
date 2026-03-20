"""
协整均值回归 · 配对交易（Cointegration Pairs Trading）
策略与引擎分离：本模块只负责生成信号与持仓状态，不涉及资金扣减。

核心逻辑：
- 对齐两只标的 trade_date
- 120 日滚动 OLS（用 cov/var 向量化近似）得到动态对冲比例 Beta
- Spread = close_a - Beta * close_b
- 20 日滚动 Z-Score 标准化
- 阈值穿越入场、回归 0 轴附近平仓

状态管理：
- allow_shorting=True：position_status ∈ {1(多A空B), -1(空A多B), 0(空仓)}
- allow_shorting=False：仅做多 A（不做空 B），position_status ∈ {1, 0}
"""

from __future__ import annotations

import polars as pl
from statsmodels.tsa.stattools import coint

# 可调参数（便于回测时覆盖）
ROLL_BETA_WINDOW = 120  # OLS 回归窗口：过去 120 日
Z_WINDOW = 20           # Z-Score 标准化窗口
ENTRY_Z = 2.0           # 入场阈值
EXIT_BAND = 0.5         # 平仓带：[-0.5, 0.5]
COINT_PVALUE_TH = 0.05  # 协整检验阈值


def generate_pairs_signals(
    df_a: pl.DataFrame,
    df_b: pl.DataFrame,
    *,
    allow_shorting: bool = True,
    **params,
) -> pl.DataFrame:
    """
    基于协整均值回归的配对交易信号。

    输入：
    - df_a/df_b：必须包含 trade_date, close。可选包含 ts_code（会保留在输出列名中）。

    输出：
    - trade_date, close_a, close_b, beta, spread, z_score
    - entry_short_a_long_b, entry_long_a_short_b, exit_signal
    - position_status
    - coint_tstat, coint_pvalue（整段样本的协整检验结果，供上层引擎/回测过滤）
    """
    beta_window = int(params.get("ROLL_BETA_WINDOW", ROLL_BETA_WINDOW))
    z_window = int(params.get("Z_WINDOW", Z_WINDOW))
    entry_z = float(params.get("ENTRY_Z", ENTRY_Z))
    exit_band = float(params.get("EXIT_BAND", EXIT_BAND))
    p_th = float(params.get("COINT_PVALUE_TH", COINT_PVALUE_TH))

    # -------------------------------------------------------------------------
    # 1) 预处理：trade_date 对齐 + 按日期升序（rolling/shift 依赖顺序）
    # -------------------------------------------------------------------------
    cols_a = [c for c in ["trade_date", "close", "ts_code"] if c in df_a.columns]
    cols_b = [c for c in ["trade_date", "close", "ts_code"] if c in df_b.columns]

    rename_a = {"close": "close_a"}
    rename_b = {"close": "close_b"}
    if "ts_code" in cols_a:
        rename_a["ts_code"] = "ts_code_a"
    if "ts_code" in cols_b:
        rename_b["ts_code"] = "ts_code_b"

    a = df_a.select(cols_a).rename(rename_a)
    b = df_b.select(cols_b).rename(rename_b)

    df = a.join(b, on="trade_date", how="inner").sort("trade_date")

    # 极端数据保护：去除缺失/非数
    df = df.with_columns(
        [
            pl.col("close_a").cast(pl.Float64).fill_nan(None),
            pl.col("close_b").cast(pl.Float64).fill_nan(None),
        ]
    ).drop_nulls(["close_a", "close_b", "trade_date"])

    # -------------------------------------------------------------------------
    # 2) 协整检验（整段样本一次计算）：非协整默认不出信号
    # -------------------------------------------------------------------------
    coint_tstat = None
    coint_pvalue = None
    if df.height >= max(beta_window, z_window) + 5:
        s_a = df.get_column("close_a").to_numpy()
        s_b = df.get_column("close_b").to_numpy()
        try:
            tstat, pvalue, _ = coint(s_a, s_b)
            coint_tstat = float(tstat)
            coint_pvalue = float(pvalue)
        except Exception:
            coint_tstat = None
            coint_pvalue = None

    is_coint = (coint_pvalue is not None) and (coint_pvalue < p_th)

    df = df.with_columns(
        [
            pl.lit(coint_tstat).cast(pl.Float64).alias("coint_tstat"),
            pl.lit(coint_pvalue).cast(pl.Float64).alias("coint_pvalue"),
        ]
    )

    # -------------------------------------------------------------------------
    # 3) 动态对冲比例 Beta（滚动 OLS：beta = cov(a,b) / var(b)）
    #    为确保 Polars eager 不出现同批次引用问题，分步落地中间列。
    # -------------------------------------------------------------------------
    mean_a = pl.col("close_a").rolling_mean(beta_window).alias("_mean_a")
    mean_b = pl.col("close_b").rolling_mean(beta_window).alias("_mean_b")
    mean_ab = (pl.col("close_a") * pl.col("close_b")).rolling_mean(beta_window).alias("_mean_ab")
    mean_bb = (pl.col("close_b") * pl.col("close_b")).rolling_mean(beta_window).alias("_mean_bb")

    df = df.with_columns([mean_a, mean_b, mean_ab, mean_bb])

    cov_ab = (pl.col("_mean_ab") - pl.col("_mean_a") * pl.col("_mean_b")).alias("_cov_ab")
    var_b = (pl.col("_mean_bb") - pl.col("_mean_b") * pl.col("_mean_b")).alias("_var_b")
    beta = (pl.col("_cov_ab") / pl.col("_var_b").replace(0.0, None)).alias("beta")

    df = df.with_columns([cov_ab, var_b]).with_columns([beta])

    spread = (pl.col("close_a") - pl.col("beta") * pl.col("close_b")).alias("spread")
    df = df.with_columns([spread])

    # -------------------------------------------------------------------------
    # 4) Z-Score 标准化（rolling mean/std）
    # -------------------------------------------------------------------------
    spread_mean = pl.col("spread").rolling_mean(z_window).alias("_spread_mean")
    spread_std = pl.col("spread").rolling_std(z_window).alias("_spread_std")
    df = df.with_columns([spread_mean, spread_std])

    z_score = (
        (pl.col("spread") - pl.col("_spread_mean")) / pl.col("_spread_std").replace(0.0, None)
    ).alias("z_score")
    df = df.with_columns([z_score])

    # -------------------------------------------------------------------------
    # 5) 信号逻辑：阈值穿越入场 + 回到 0 轴附近平仓
    # -------------------------------------------------------------------------
    z_prev = pl.col("z_score").shift(1).alias("_z_prev")
    df = df.with_columns([z_prev])

    entry_short_a_long_b = (
        (pl.col("z_score") > pl.lit(entry_z)) & (pl.col("_z_prev") <= pl.lit(entry_z))
    ).cast(pl.Int8).fill_null(0).alias("entry_short_a_long_b")
    entry_long_a_short_b = (
        (pl.col("z_score") < pl.lit(-entry_z)) & (pl.col("_z_prev") >= pl.lit(-entry_z))
    ).cast(pl.Int8).fill_null(0).alias("entry_long_a_short_b")

    exit_signal = (
        (pl.col("z_score") >= pl.lit(-exit_band)) & (pl.col("z_score") <= pl.lit(exit_band))
    ).cast(pl.Int8).fill_null(0).alias("exit_signal")

    df = df.with_columns([entry_short_a_long_b, entry_long_a_short_b, exit_signal])

    # -------------------------------------------------------------------------
    # 6) 状态管理：position_status ∈ {1, -1, 0}
    #    allow_shorting=False：仅允许“做多A（不对冲B）”，退化为相对强弱均值回归。
    # -------------------------------------------------------------------------
    if allow_shorting:
        raw_state = (
            pl.when(pl.col("entry_long_a_short_b") == 1)
            .then(1)
            .when(pl.col("entry_short_a_long_b") == 1)
            .then(-1)
            .when(pl.col("exit_signal") == 1)
            .then(0)
            .otherwise(None)
            .cast(pl.Int8)
        )
        position_status = raw_state.forward_fill().fill_null(0).alias("position_status")
    else:
        entry_long_only = pl.col("entry_long_a_short_b").alias("_entry_long_only")
        df = df.with_columns([entry_long_only])
        raw_state = (
            pl.when(pl.col("_entry_long_only") == 1)
            .then(1)
            .when(pl.col("exit_signal") == 1)
            .then(0)
            .otherwise(None)
            .cast(pl.Int8)
        )
        position_status = raw_state.forward_fill().fill_null(0).alias("position_status")

        # 退化模式下，不输出“做空A/做多B”入场信号
        df = df.with_columns(
            [
                pl.lit(0).cast(pl.Int8).alias("entry_short_a_long_b"),
                pl.col("_entry_long_only").cast(pl.Int8).fill_null(0).alias("entry_long_a_short_b"),
            ]
        )

    df = df.with_columns([position_status])

    # -------------------------------------------------------------------------
    # 7) 协整过滤：非协整时全部信号清零 + 空仓
    # -------------------------------------------------------------------------
    if not is_coint:
        df = df.with_columns(
            [
                pl.lit(0).cast(pl.Int8).alias("entry_short_a_long_b"),
                pl.lit(0).cast(pl.Int8).alias("entry_long_a_short_b"),
                pl.lit(0).cast(pl.Int8).alias("exit_signal"),
                pl.lit(0).cast(pl.Int8).alias("position_status"),
            ]
        )

    # 输出整理：删掉内部中间列
    drop_cols = [c for c in df.columns if c.startswith("_")]
    return df.drop(drop_cols)


if __name__ == "__main__":
    # 最小示例：构造一对“协整”的价格序列验证（A ≈ 1.5*B + 噪声）
    import numpy as np

    n = 260
    rng = np.random.default_rng(7)
    b = 100 + np.cumsum(rng.normal(0, 1.0, size=n))
    a = 1.5 * b + rng.normal(0, 2.0, size=n)

    demo_a = pl.DataFrame(
        {
            "trade_date": [f"2024{(i+1):03d}" for i in range(n)],
            "close": a.tolist(),
        }
    )
    demo_b = pl.DataFrame(
        {
            "trade_date": [f"2024{(i+1):03d}" for i in range(n)],
            "close": b.tolist(),
        }
    )

    out = generate_pairs_signals(demo_a, demo_b, allow_shorting=True)
    print(out.tail(25).select(["trade_date", "close_a", "close_b", "beta", "spread", "z_score", "entry_long_a_short_b", "entry_short_a_long_b", "exit_signal", "position_status", "coint_pvalue"]))
