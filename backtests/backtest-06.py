"""
截面因子回测引擎 · sm_backtest_05_cross_sectional.py

适配你的 V3 截面打分策略输出：
ts_code, trade_date, open, close, final_rank, entry_signal, exit_signal

核心实现：
- T+1：用 trade_date 当天的 open 成交（signals 来自“昨日 final_rank/排名”）
- 动态 100 槽位池：单票目标分配 = 当前 total_equity / 100，实现复利滚动
- 按每日截面 final_rank 升序抢筹（越靠前越先买）
- 先卖出后买入；换手率与平均持仓用于缓冲带（100-200）验证
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import datetime

import duckdb
import polars as pl


LOT_SIZE = 100


def _load_cross_section_strategy():
    """
    动态加载截面策略模块（默认使用动量强者恒强模板）。
    目标：导入后获得一个可调用的信号函数，输入 df，输出含 final_rank/entry_signal/exit_signal 的 df。
    """
    root = os.path.dirname(os.path.abspath(__file__))

    # 默认：你当前的截面策略（也与你的描述最贴近）
    candidates = [
        "SM-策略-06-momentum.py",
        "sm_05_momentum.py",
        "sm_05_fama_french_value_size.py",
        "SM-策略-05-fama french value size.py",
    ]
    func_candidates = (
        "generate_momentum_signals",
        "generate_ff_signals",
        "generate_momentum_signals".lower(),
    )

    from importlib.util import spec_from_file_location, module_from_spec

    for name in candidates:
        path = os.path.join(root, name)
        if not os.path.isfile(path):
            continue
        spec = spec_from_file_location("_cross_sectional", path)
        mod = module_from_spec(spec)
        spec.loader.exec_module(mod)
        for fn in func_candidates:
            if hasattr(mod, fn):
                return getattr(mod, fn)
        # 若函数名不同：在模块内找任意 generate_*_signals
        for k, v in vars(mod).items():
            if callable(v) and k.startswith("generate_") and k.endswith("_signals"):
                return v

    raise FileNotFoundError("未找到可用的截面策略模块（momentum/FF）。")


generate_signals = _load_cross_section_strategy()


def load_from_duckdb(
    db_path: str,
    start_date: str = "20150101",
    end_date: str | None = None,
) -> pl.DataFrame:
    """
    读取并构建输入 DataFrame：
    - open/close：复权后的 OHLC（用于成交与截面收益计算）
    - 可选：total_mv/pb/volume（供价值/流动性过滤策略使用）
    """
    end = end_date or datetime.now().strftime("%Y%m%d")
    con = duckdb.connect(db_path, read_only=True)
    try:
        # daily_data（尽量带 volume，若表里字段不同会回退）
        try:
            d = con.execute(
                "SELECT ts_code, trade_date, open, high, low, close, vol FROM daily_data WHERE trade_date >= ? AND trade_date <= ?",
                [start_date, end],
            ).pl()
            d = d.rename({"vol": "volume"})
        except Exception:
            d = con.execute(
                "SELECT ts_code, trade_date, open, high, low, close FROM daily_data WHERE trade_date >= ? AND trade_date <= ?",
                [start_date, end],
            ).pl()

        # daily_basic（total_mv/pb）
        try:
            b = con.execute(
                "SELECT ts_code, trade_date, total_mv, pb FROM daily_basic WHERE trade_date >= ? AND trade_date <= ?",
                [start_date, end],
            ).pl()
        except Exception:
            b = None

        a = con.execute(
            "SELECT ts_code, trade_date, adj_factor FROM adj_factor WHERE trade_date >= ? AND trade_date <= ?",
            [start_date, end],
        ).pl()
    finally:
        con.close()

    j = d.join(a, on=["ts_code", "trade_date"], how="inner").sort(["ts_code", "trade_date"])
    last_af = pl.col("adj_factor").last().over("ts_code")
    ratio = pl.col("adj_factor") / last_af

    j = j.with_columns(
        (pl.col("open") * ratio).alias("adj_open"),
        (pl.col("close") * ratio).alias("adj_close"),
    )
    j = j.with_columns(pl.col("adj_open").alias("open"), pl.col("adj_close").alias("close"))

    if b is not None:
        j = j.join(b, on=["ts_code", "trade_date"], how="left")

    # 只保留后续必要列；策略若需要更多列，可自行在其函数内扩展
    keep = {"ts_code", "trade_date", "open", "close", "total_mv", "pb", "volume"}
    existing = [c for c in keep if c in j.columns]
    return j.select(existing)


@dataclass
class BacktestConfig:
    start_date: str = "20150101"
    end_date: str | None = None

    initial_capital: float = 10_000_000
    max_positions: int = 100

    buy_cost: float = 0.0003   # 万分之三（佣金+滑点）
    sell_cost: float = 0.0013  # 千分之1.3（含印花税+佣金）

    # 截面策略超参（在策略模块里决定 entry/exit）
    # 这里仅作占位，便于你将来在同一引擎中透传参数


def run_backtest(df: pl.DataFrame, config: BacktestConfig | None = None) -> tuple[pl.DataFrame, dict]:
    """
    截面因子回测（动态 100 槽位复利池）。

    重要假设：signals 已经满足你的口径：
    - entry_signal==1：表示昨日 final_rank <= TOP_N_ENTRY（用于今日 open 买入）
    - exit_signal==1 ：表示昨日 final_rank > TOP_N_EXIT  （用于今日 open 卖出）
    因此这里不再对 entry/exit 做 shift。
    """
    cfg = config or BacktestConfig()
    df = df.sort(["ts_code", "trade_date"])

    # 1) 信号生成（纯截面策略）
    df = generate_signals(df)

    # 2) 校验必要列
    need_cols = {"ts_code", "trade_date", "open", "close", "final_rank", "entry_signal", "exit_signal"}
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise KeyError(f"策略输出缺少列：{missing}（请检查你的截面策略模块输出字段）")

    end = cfg.end_date or df["trade_date"].max()
    df = df.filter((pl.col("trade_date") >= cfg.start_date) & (pl.col("trade_date") <= end))
    df = df.sort(["ts_code", "trade_date"])

    cash = float(cfg.initial_capital)
    current_positions: dict[str, dict[str, float | int]] = {}  # code -> {"shares": int, "cost": float}
    last_close: dict[str, float] = {}

    daily_records: list[dict] = []

    total_traded_double_side = 0.0

    for (trade_date,), day in df.group_by("trade_date", maintain_order=True):
        # 抽取当日截面
        rows = list(
            day.select(["ts_code", "open", "close", "final_rank", "entry_signal", "exit_signal"]).iter_rows(named=True)
        )

        open_by_code = {r["ts_code"]: float(r["open"]) for r in rows}
        close_by_code = {r["ts_code"]: float(r["close"]) for r in rows}
        entry_by_code = {r["ts_code"]: int(r["entry_signal"] or 0) for r in rows}
        exit_by_code = {r["ts_code"]: int(r["exit_signal"] or 0) for r in rows}
        rank_by_code = {r["ts_code"]: (r["final_rank"] if r["final_rank"] is not None else None) for r in rows}

        # 当日开盘前资产估值：现金 +（持仓 * 昨日收盘）
        equity_start = cash
        for code, pos in current_positions.items():
            sh = int(pos["shares"])
            equity_start += sh * last_close.get(code, 0.0)

        # 1) 卖出：若今日截面中该 code 的 exit_signal==1，则用今日 open 全卖
        traded_sell = 0.0
        for code in list(current_positions.keys()):
            if exit_by_code.get(code, 0) != 1:
                continue
            sh = int(current_positions[code]["shares"])
            price_open = open_by_code.get(code)
            if price_open is None or price_open <= 0:
                continue
            sell_amt = sh * price_open
            cash += sell_amt * (1.0 - float(cfg.sell_cost))
            traded_sell += sell_amt
            del current_positions[code]

        # 2) 买入：昨日 entry_signal==1 且当前不持仓
        free_slots = cfg.max_positions - len(current_positions)
        traded_buy = 0.0

        # 目标分配：在“卖出后”的最新 total_equity（仍用昨日收盘估值剩余持仓）
        equity_after_sell = cash
        for code, pos in current_positions.items():
            sh = int(pos["shares"])
            equity_after_sell += sh * last_close.get(code, 0.0)

        target_alloc = equity_after_sell / float(cfg.max_positions) if cfg.max_positions > 0 else 0.0

        if free_slots > 0 and target_alloc > 0 and cash > 0:
            candidates = []
            for code in entry_by_code.keys():
                if entry_by_code.get(code, 0) != 1:
                    continue
                if code in current_positions:
                    continue
                candidates.append(code)

            if candidates:
                # 优中选优：按昨日 final_rank（越小越靠前）
                candidates.sort(key=lambda c: (rank_by_code.get(c) if rank_by_code.get(c) is not None else float("inf")))
                for code in candidates:
                    if free_slots <= 0:
                        break
                    if cash <= 0:
                        break
                    if code not in open_by_code:
                        continue
                    price_open = open_by_code[code]
                    if price_open <= 0:
                        continue

                    # 买入股数：floor(target_alloc / (open * 100)) * 100
                    shares = int(math.floor(target_alloc / (price_open * LOT_SIZE)) * LOT_SIZE)
                    # 特例：买不起一手则跳过，不占用槽位
                    if shares <= 0:
                        continue

                    buy_amt = shares * price_open
                    need_cash = buy_amt * (1.0 + float(cfg.buy_cost))
                    if need_cash > cash:
                        continue

                    cash -= need_cash
                    current_positions[code] = {"shares": shares, "cost": need_cash}
                    free_slots -= 1
                    traded_buy += buy_amt

        # 日末 Mark-to-Market：用今日 close 估值
        market_value = 0.0
        for code, pos in current_positions.items():
            sh = int(pos["shares"])
            market_value += sh * close_by_code.get(code, last_close.get(code, 0.0))

        total_equity = cash + market_value
        daily_pnl = total_equity - equity_start
        daily_ret = (daily_pnl / equity_start) if equity_start > 1e-8 else 0.0
        daily_ret = max(-0.99, min(10.0, float(daily_ret)))

        holdings_count = len(current_positions)
        # 双边换手：当日买卖成交额合计 / 开盘前总资产
        traded_double = traded_buy + traded_sell
        daily_turnover = traded_double / equity_start if equity_start > 1e-8 else 0.0
        total_traded_double_side += traded_double

        daily_records.append(
            {
                "trade_date": trade_date,
                "cash": cash,
                "holdings_count": holdings_count,
                "market_value": market_value,
                "total_equity": total_equity,
                "daily_pnl": daily_pnl,
                "strategy_return": daily_ret,
                "daily_turnover": daily_turnover,
            }
        )

        # 更新 last_close：用于次日开盘前估值
        for code, px in close_by_code.items():
            last_close[code] = float(px)

    daily = pl.DataFrame(daily_records).sort("trade_date")
    daily = daily.with_columns(
        (1.0 + pl.col("strategy_return")).cum_prod().alias("cum_nav")
    )

    # 评估指标
    meta = {}
    meta["avg_holdings"] = float(daily["holdings_count"].mean()) if not daily.is_empty() else 0.0
    meta["annual_turnover"] = float(daily["daily_turnover"].mean()) * 252.0 if not daily.is_empty() else 0.0
    return daily, meta


def _calc_performance(daily: pl.DataFrame) -> dict[str, float]:
    if daily.is_empty():
        return {"total_return": 0.0, "ann_return": 0.0, "max_dd": 0.0, "sharpe": 0.0}

    nav = daily["cum_nav"].fill_nan(1.0).clip(1e-8, 1e12)
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

    return {"total_return": total_return, "ann_return": ann_return, "max_dd": max_dd, "sharpe": sharpe}


def plot_nav(daily: pl.DataFrame, path: str = "cross_sectional_nav.png") -> None:
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

    plt.figure(figsize=(12, 6))
    plt.plot(d["trade_date"], d["cum_nav"], lw=1.5, label="NAV")
    plt.title("Cross-Sectional Factor Strategy NAV (Dynamic 100 Slots)")
    plt.xlabel("Date")
    plt.ylabel("Cumulative NAV")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"已保存: {path}")


if __name__ == "__main__":
    cfg = BacktestConfig(
        start_date="20150101",
        end_date=None,
        initial_capital=10_000_000,
        max_positions=100,
        buy_cost=0.0003,
        sell_cost=0.0013,
    )

    db_path = os.getenv("DUCKDB_PATH", "shiming_daily_base.duckdb")
    if not os.path.isabs(db_path):
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), db_path)
    if not os.path.isfile(db_path):
        raise FileNotFoundError(f"未找到数据库：{db_path}")

    base = load_from_duckdb(db_path, start_date=cfg.start_date, end_date=cfg.end_date)
    if base.is_empty():
        raise ValueError("DuckDB 查询结果为空")

    daily, meta = run_backtest(base, cfg)
    perf = _calc_performance(daily)

    print("=== 截面因子回测结果（动态 100 槽位）===")
    print(f"总收益率:     {perf['total_return']:.4f}")
    print(f"年化收益率:   {perf['ann_return']:.4f}")
    print(f"最大回撤:     {perf['max_dd']:.4f}")
    print(f"夏普比率:     {perf['sharpe']:.4f}")
    print("--- 因子监测 ---")
    print(f"平均每日持仓: {meta.get('avg_holdings', 0.0):.2f}")
    print(f"年化双边换手率: {meta.get('annual_turnover', 0.0):.4f}")

    plot_nav(daily, path="cross_sectional_nav.png")

