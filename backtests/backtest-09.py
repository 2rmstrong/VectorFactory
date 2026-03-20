"""
PEAD 事件驱动回测入口（与 backtest-01..08 命名一致）

优先读取你已生成的信号文件（默认 `SM-策略-09-pead_signals.parquet`）并回测：
- 必备列：ts_code, trade_date, open, close, entry_signal, event_open（或 entry_open）
- 可选列：daily_ret（用于同日多信号按强度排序；缺失会自动补）
"""
from __future__ import annotations

import os
import sys
from datetime import datetime

import polars as pl

_root = os.path.dirname(os.path.abspath(__file__))
if _root not in sys.path:
    sys.path.insert(0, _root)

from sm_backtest_09_pead import BacktestConfig, _ensure_signal_columns, plot_nav, print_stats, run_backtest


def _load_signal_file(path: str) -> pl.DataFrame:
    if path.lower().endswith(".parquet"):
        # 大文件优化：只读取回测需要列，避免一次性把无关列装进内存
        cols = ["ts_code", "trade_date", "open", "close", "daily_ret", "entry_signal", "event_open", "entry_open"]
        lf = pl.scan_parquet(path)
        names = lf.collect_schema().names()
        return lf.select([c for c in cols if c in names]).collect()
    if path.lower().endswith(".csv"):
        # CSV 通常不大；若很大可改 scan_csv
        return pl.read_csv(path)
    raise ValueError("仅支持 .parquet 或 .csv")


if __name__ == "__main__":
    cfg = BacktestConfig(
        start_date="20150101",
        end_date=None,
        initial_capital=10_000_000,
        max_positions=30,
        hold_days_exit=40,
        sell_cost=0.0013,
        buy_cost=0.0003,
    )

    default_signal = os.path.join(_root, "SM-策略-09-pead_signals.parquet")
    signal_path = os.getenv("PEAD_SIGNAL_PATH", default_signal)
    if not os.path.isabs(signal_path):
        signal_path = os.path.join(_root, signal_path)
    # 若用户环境变量指向不存在文件，则回退到默认文件名
    if not os.path.isfile(signal_path) and os.path.isfile(default_signal):
        signal_path = default_signal

    if os.path.isfile(signal_path):
        df_signal = _ensure_signal_columns(_load_signal_file(signal_path))
        print(
            f"已加载信号表: {signal_path}，行数={df_signal.height:,}，股票数={df_signal['ts_code'].n_unique():,}"
        )
    else:
        # 兜底：合成数据仅用于验证流程
        from datetime import timedelta

        n_days = 500
        codes = ["000001.SZ", "000002.SZ", "600000.SH"]
        base = datetime(2020, 1, 1)
        dates = []
        d = base
        while len(dates) < n_days:
            if d.weekday() < 5:
                dates.append(d.strftime("%Y%m%d"))
            d += timedelta(days=1)

        rows = []
        for c in codes:
            prev_c = None
            for i, dt in enumerate(dates):
                open_p = 10.0 + i * 0.01 + (hash(c) % 100) * 0.001
                close_p = open_p * (1.0 + 0.002 * (i % 5 - 2))
                daily_ret = (close_p / prev_c - 1.0) if prev_c and prev_c > 0 else 0.0
                prev_c = close_p
                rows.append(
                    {
                        "ts_code": c,
                        "trade_date": dt,
                        "open": open_p,
                        "close": close_p,
                        "daily_ret": daily_ret,
                        "entry_signal": 1 if (i > 20 and i % 50 == 21 and c == codes[0]) else 0,
                        "entry_open": open_p,
                    }
                )
        df_signal = _ensure_signal_columns(pl.DataFrame(rows))
        print(f"未找到信号表文件：{signal_path}，已改用合成数据（仅用于验证流程）。")

    daily, meta = run_backtest(df_signal, cfg)
    print_stats(daily, meta, cfg)
    plot_nav(daily, path="pead_nav.png")
