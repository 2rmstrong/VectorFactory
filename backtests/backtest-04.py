"""
RSI 底背离 · 专属回测脚本（100 槽位，先到先得，T+1）

需求要点：
- 初始资金：10,000,000
- 100 个固定槽位，每槽预算 100,000（最多持仓 100 只）
- T+1 执行：T 日信号，T+1 以 open 成交；先卖出后买入
- 买入费率 0.0003；卖出费率 0.0013
- 并发买入排序：按 T 日 rsi 从小到大（越超卖越优先）
- 输出：净值曲线 + 图形 + 指标 + 监测数据
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime

_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo not in sys.path:
    sys.path.insert(0, _repo)
import project_paths as pp

import duckdb
import polars as pl


LOT_SIZE = 100


def _load_strategy():
    """
    动态加载 RSI 策略模块。
    兼容两种文件名：
    - sm_04_rsi_divergence.py（推荐）
    - SM-策略-04-rsi divergence.py（用户可能的旧/变体命名）
    """
    root = pp.strategies_dir()
    for name in ("sm_04_rsi_divergence.py", "SM-策略-04-rsi divergence.py", "SM-策略-04-rsi_divergence.py"):
        path = os.path.join(root, name)
        if os.path.isfile(path):
            from importlib.util import spec_from_file_location, module_from_spec

            spec = spec_from_file_location("_rsi_div", path)
            mod = module_from_spec(spec)
            spec.loader.exec_module(mod)
            return getattr(mod, "generate_rsi_divergence_signals")
    raise FileNotFoundError("未找到 RSI 策略文件：sm_04_rsi_divergence.py")


generate_rsi_divergence_signals = _load_strategy()


def load_from_duckdb(
    db_path: str,
    start_date: str = "20150101",
    end_date: str | None = None,
) -> pl.DataFrame:
    """
    从 shiming_daily_base.duckdb 读取 daily_data + adj_factor，计算前复权 OHLC。
    返回列：ts_code, trade_date, adj_open, adj_high, adj_low, adj_close
    """
    end = end_date or datetime.now().strftime("%Y%m%d")
    con = duckdb.connect(db_path, read_only=True)
    try:
        d = con.execute(
            "SELECT ts_code, trade_date, open, high, low, close FROM daily_data WHERE trade_date >= ? AND trade_date <= ?",
            [start_date, end],
        ).pl()
        a = con.execute(
            "SELECT ts_code, trade_date, adj_factor FROM adj_factor WHERE trade_date >= ? AND trade_date <= ?",
            [start_date, end],
        ).pl()
    finally:
        con.close()

    j = d.join(a, on=["ts_code", "trade_date"], how="inner").sort(["ts_code", "trade_date"])
    last_af = pl.col("adj_factor").last().over("ts_code")
    ratio = pl.col("adj_factor") / last_af
    return j.with_columns(
        (pl.col("open") * ratio).alias("adj_open"),
        (pl.col("high") * ratio).alias("adj_high"),
        (pl.col("low") * ratio).alias("adj_low"),
        (pl.col("close") * ratio).alias("adj_close"),
    ).select(["ts_code", "trade_date", "adj_open", "adj_high", "adj_low", "adj_close"])


def _unify_prices(df: pl.DataFrame) -> pl.DataFrame:
    """统一价格列名（优先使用复权价）。"""
    if "adj_close" in df.columns and "close" not in df.columns:
        df = df.with_columns(pl.col("adj_close").alias("close"))
    if "adj_open" in df.columns and "open" not in df.columns:
        df = df.with_columns(pl.col("adj_open").alias("open"))
    if "adj_high" in df.columns and "high" not in df.columns:
        df = df.with_columns(pl.col("adj_high").alias("high"))
    if "adj_low" in df.columns and "low" not in df.columns:
        df = df.with_columns(pl.col("adj_low").alias("low"))
    return df


@dataclass
class BacktestConfig:
    start_date: str = "20150101"
    end_date: str | None = None

    initial_capital: float = 10_000_000
    n_slots: int = 100
    slot_budget: float = 100_000

    buy_cost: float = 0.0003
    sell_cost: float = 0.0013

    t_plus_one: bool = True


def _calc_stats(daily: pl.DataFrame) -> dict[str, float]:
    if daily.is_empty():
        return {"total_return": 0.0, "ann_return": 0.0, "max_dd": 0.0, "sharpe": 0.0}

    equity = daily["total_equity"].fill_nan(0.0)
    nav = (equity / float(equity[0])).fill_nan(1.0).clip(1e-8, 1e12)
    total_return = float(nav[-1]) - 1.0

    peak = nav.cum_max()
    max_dd = float(((nav / peak) - 1.0).min())

    n_days = len(daily)
    years = n_days / 252.0 if n_days else 0.0
    ann_return = (float(nav[-1])) ** (1.0 / years) - 1.0 if years > 0 else total_return

    ret = daily["strategy_return"].fill_nan(0.0)
    mu = float(ret.mean())
    sigma = float(ret.std()) if ret.len() > 1 else 0.0
    sharpe = (mu / sigma * math.sqrt(252.0)) if sigma > 1e-12 else 0.0

    return {
        "total_return": total_return,
        "ann_return": ann_return,
        "max_dd": max_dd,
        "sharpe": sharpe,
    }


def run_backtest(df: pl.DataFrame, config: BacktestConfig | None = None) -> tuple[pl.DataFrame, dict]:
    """
    执行 RSI 底背离回测（100 槽位，T+1，先卖后买，按 RSI 排队）。
    """
    cfg = config or BacktestConfig()
    end = cfg.end_date or datetime.now().strftime("%Y%m%d")

    df = _unify_prices(df)
    df = df.filter((pl.col("trade_date") >= cfg.start_date) & (pl.col("trade_date") <= end))
    df = df.sort(["ts_code", "trade_date"])

    # 1) 生成策略信号（策略模块负责计算 rsi/entry/exit）
    df = generate_rsi_divergence_signals(df)

    # 2) T+1 执行：把 T 日信号与排序字段（T 日 rsi）shift 到 T+1 执行日
    if cfg.t_plus_one:
        df = df.with_columns(
            pl.col("entry_signal").shift(1).over("ts_code").fill_null(0).alias("actual_entry"),
            pl.col("exit_signal").shift(1).over("ts_code").fill_null(0).alias("actual_exit"),
            pl.col("rsi").shift(1).over("ts_code").alias("actual_rsi"),
        )
    else:
        df = df.with_columns(
            pl.col("entry_signal").fill_null(0).alias("actual_entry"),
            pl.col("exit_signal").fill_null(0).alias("actual_exit"),
            pl.col("rsi").alias("actual_rsi"),
        )

    # 3) 账户撮合：按日推进（路径依赖：现金/槽位/持仓）
    cash = float(cfg.initial_capital)
    holdings: dict[str, int] = {}  # ts_code -> shares（一个标的占用一个槽位）
    last_close: dict[str, float] = {}

    reject_lot_too_expensive = 0
    daily_records: list[dict] = []

    needed = ["ts_code", "open", "close", "actual_entry", "actual_exit", "actual_rsi"]

    for (trade_date,), day in df.group_by("trade_date", maintain_order=True):
        rows = list(day.select(needed).iter_rows(named=True))

        # 日初总资产（按昨收估值）
        equity_start = cash
        for code, sh in holdings.items():
            if sh <= 0:
                continue
            equity_start += sh * last_close.get(code, 0.0)

        # --------------------------
        # A) 先卖出：昨日 exit_signal -> 今日 open 全部卖出
        # --------------------------
        for r in rows:
            if int(r["actual_exit"]) != 1:
                continue
            code = r["ts_code"]
            sh = holdings.get(code, 0)
            if sh <= 0:
                continue
            sell_amt = sh * float(r["open"])
            cash += sell_amt * (1.0 - float(cfg.sell_cost))
            holdings.pop(code, None)

        # --------------------------
        # B) 后买入：昨日 entry_signal -> 今日 open 买入（先到先得，槽位不足按 RSI 排队）
        # --------------------------
        slots_left = max(0, int(cfg.n_slots) - sum(1 for _c, sh in holdings.items() if sh > 0))
        if slots_left > 0 and cash > cfg.slot_budget:
            candidates = []
            for r in rows:
                if int(r["actual_entry"]) != 1:
                    continue
                code = r["ts_code"]
                if holdings.get(code, 0) > 0:
                    continue
                candidates.append(r)

            # 排队：按昨日 rsi 从小到大（越超卖越优先）
            if len(candidates) > slots_left:
                candidates.sort(key=lambda x: (x["actual_rsi"] if x["actual_rsi"] is not None else float("inf")))
                candidates = candidates[:slots_left]
            else:
                candidates.sort(key=lambda x: (x["actual_rsi"] if x["actual_rsi"] is not None else float("inf")))

            for r in candidates:
                if slots_left <= 0:
                    break
                if cash <= cfg.slot_budget:
                    break
                price = float(r["open"])
                if price <= 0:
                    continue
                shares = math.floor(cfg.slot_budget / (price * LOT_SIZE)) * LOT_SIZE
                # 特例拦截：10 万连一手都买不起，直接跳过
                if shares <= 0:
                    reject_lot_too_expensive += 1
                    continue
                need_cash = shares * price * (1.0 + float(cfg.buy_cost))
                if need_cash > cash:
                    # 现金不够就跳过，让位给后面的票（仍然占用“排序名额”）
                    continue
                cash -= need_cash
                holdings[r["ts_code"]] = int(shares)
                slots_left -= 1

        # 更新收盘价缓存
        for r in rows:
            last_close[r["ts_code"]] = float(r["close"])

        # 日末市值与总资产（按今收）
        market_value = 0.0
        for code, sh in holdings.items():
            if sh <= 0:
                continue
            market_value += sh * last_close.get(code, 0.0)
        total_equity = cash + market_value

        daily_pnl = total_equity - equity_start
        daily_ret = (daily_pnl / equity_start) if equity_start > 1e-8 else 0.0
        daily_ret = max(-0.99, min(10.0, float(daily_ret)))

        daily_records.append(
            {
                "trade_date": trade_date,
                "cash": cash,
                "holdings_count": sum(1 for _c, sh in holdings.items() if sh > 0),
                "market_value": market_value,
                "total_equity": total_equity,
                "daily_pnl": daily_pnl,
                "strategy_return": daily_ret,
            }
        )

    daily = pl.DataFrame(daily_records).sort("trade_date")
    daily = daily.with_columns((pl.col("total_equity") / pl.col("total_equity").first()).alias("cum_nav"))

    meta = {
        "avg_holdings": float(daily["holdings_count"].mean()) if not daily.is_empty() else 0.0,
        "reject_lot_too_expensive": reject_lot_too_expensive,
    }
    return daily, meta


def plot_nav(daily: pl.DataFrame, path: str = "rsi_div_nav.png") -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("未安装 matplotlib，跳过绘图。")
        return

    d = daily.to_pandas()
    d["trade_date"] = d["trade_date"].astype(str)
    if d["trade_date"].str.len().max() == 8:
        d["trade_date"] = d["trade_date"].str[:4] + "-" + d["trade_date"].str[4:6] + "-" + d["trade_date"].str[6:8]
    d["trade_date"] = d["trade_date"].astype("datetime64[ns]")

    plt.figure(figsize=(12, 5))
    plt.plot(d["trade_date"], d["cum_nav"], lw=1.5)
    plt.title("RSI Divergence Strategy NAV (100 Slots)")
    plt.xlabel("Date")
    plt.ylabel("NAV")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"已保存: {path}")


if __name__ == "__main__":
    cfg = BacktestConfig(
        start_date="20150101",
        end_date=None,
        initial_capital=10_000_000,
        n_slots=100,
        slot_budget=100_000,
        buy_cost=0.0003,
        sell_cost=0.0013,
        t_plus_one=True,
    )

    db_path = pp.resolve_db_path(os.getenv("DUCKDB_PATH", "shiming_daily_base.duckdb"))
    if not os.path.isfile(db_path):
        raise FileNotFoundError(f"未找到数据库：{db_path}")

    df = load_from_duckdb(db_path, start_date=cfg.start_date, end_date=cfg.end_date)
    if df.is_empty():
        raise ValueError("DuckDB 查询结果为空")
    print(f"已从 {db_path} 加载 {len(df):,} 条日线，{df['ts_code'].n_unique()} 只标的。")

    daily, meta = run_backtest(df, cfg)
    stats = _calc_stats(daily)

    print("=== RSI 底背离回测结果（100 槽位）===")
    print(f"区间:         {daily['trade_date'][0]} ~ {daily['trade_date'][-1]}")
    print(f"总收益率:     {stats['total_return']:.4f}")
    print(f"年化收益率:   {stats['ann_return']:.4f}")
    print(f"最大回撤:     {stats['max_dd']:.4f}")
    print(f"夏普比率:     {stats['sharpe']:.4f}")
    print("--- 监测 ---")
    print(f"平均每日持仓: {meta['avg_holdings']:.2f}")
    print(f"买不起一手拒单次数: {meta['reject_lot_too_expensive']}")

    plot_nav(daily, path=pp.docs_plot_path("rsi_div_nav", daily, cfg.start_date, cfg.end_date))

