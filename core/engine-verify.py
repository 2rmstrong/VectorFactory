from __future__ import annotations

import os
import time
import warnings
from datetime import datetime

import polars as pl

from engine import EngineConfig, FireControlEngine

warnings.simplefilter("ignore", category=DeprecationWarning)


def _mb(n_bytes: int) -> float:
    return n_bytes / (1024 * 1024)


def main() -> None:
    duckdb_path = os.getenv("DUCKDB_PATH", "shiming_daily_base.duckdb")

    start = "20180101"
    # 默认到今天（右开区间）；可用环境变量覆盖：VERIFY_END=YYYYMMDD
    end = os.getenv("VERIFY_END") or datetime.now().strftime("%Y%m%d")

    cfg = EngineConfig(duckdb_path=duckdb_path)
    engine = FireControlEngine(cfg)

    print("=== verify_engine.py ===")
    print(f"DuckDB: {duckdb_path}")
    print(f"Range:  {start} ~ {end} (left-closed, right-open by filter)")

    # -----------------------
    # Medium stress test
    # -----------------------
    t0 = time.perf_counter()
    lf = engine.scan().filter((pl.col("trade_date") >= start) & (pl.col("trade_date") < end))
    # Polars >= 1.25: streaming 参数已弃用，改用 engine="streaming"
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Sortedness of columns cannot be checked*")
        df = lf.collect(engine="streaming")
    t1 = time.perf_counter()

    print("\n## 压测耗时")
    print(f'collect(engine="streaming") elapsed: {t1 - t0:.3f} sec')

    # -----------------------
    # Basic health check
    # -----------------------
    print("\n## 基础体检")
    rows, cols = df.shape
    try:
        mem_bytes = int(df.estimated_size())
    except Exception:
        # fallback for older polars
        mem_bytes = int(df.to_arrow().nbytes)

    print(f"rows: {rows:,}")
    print(f"cols: {cols:,}")
    print(f"estimated memory: {_mb(mem_bytes):.2f} MB")

    # -----------------------
    # Adjustment spot check
    # -----------------------
    print("\n## 复权抽查（随机 1 只股票，最新 5 日）")
    if rows == 0:
        print("⚠ 数据为空，跳过抽查与信号统计。")
        return

    ts_code = (
        df.select(pl.col("ts_code").unique().sample(n=1, with_replacement=False).first())
        .to_series()
        .item()
    )

    spot = (
        df.filter(pl.col("ts_code") == ts_code)
        .select(["trade_date", "close", "adj_close", cfg.adj_factor_col])
        .sort("trade_date")
        .tail(5)
    )
    print(f"ts_code = {ts_code}")
    print(spot)

    # -----------------------
    # Signal diagnostics
    # -----------------------
    print("\n## 雷达信号探伤（全市场 6 年）")
    sig_cols = [c for c in df.columns if c.startswith("sig_")]
    if not sig_cols:
        print("⚠ 未发现 sig_* 信号列。")
        return

    # 触发次数：bool->int 再 sum；频率=触发/总行数
    agg_exprs = [pl.col(c).cast(pl.Int64).sum().alias(c) for c in sig_cols]
    counts = df.select(agg_exprs)

    report = (
        counts.unpivot(variable_name="signal", value_name="triggers")
        .with_columns(
            pl.col("triggers").cast(pl.Int64),
            (pl.col("triggers") / pl.lit(rows)).alias("frequency"),
        )
        .sort("triggers", descending=True)
    )

    # 让打印更直观
    pl.Config.set_tbl_rows(min(len(sig_cols), 50))
    pl.Config.set_tbl_cols(10)
    print(report)


if __name__ == "__main__":
    main()

