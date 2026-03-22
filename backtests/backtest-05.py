"""
Fama-French 小盘价值因子 · 专属回测脚本（动态 100 槽位复利池，T+1 换血）

需求要点：
- 初始资金：10,000,000
- Max Positions = 100（动态槽位）
- 复利分配：每次买入目标分配金额 = 当日开盘前总资产 / 100
- T+1 执行：T 日信号，T+1 以 open 成交；先卖出后买入
- 买入费率 0.0003；卖出费率 0.0013
- 并发买入排序：按 T 日 final_rank 从小到大（越小越优先）
- 输出：净值曲线 + 图形 + 指标 + 平均持仓数 + 年化换手率
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
    动态加载 FF 策略模块（位于仓库 strategies/ 目录）。
    兼容两种文件名：
    - sm_05_fama_french_value_size.py（推荐）
    - SM-策略-05-fama french value size.py（用户可能的命名）
    """
    root = pp.strategies_dir()
    for name in (
        "sm_05_fama_french_value_size.py",
        "SM-策略-05-fama french value size.py",
        "SM-策略-05-fama_french_value_size.py",
    ):
        path = os.path.join(root, name)
        if os.path.isfile(path):
            from importlib.util import spec_from_file_location, module_from_spec

            spec = spec_from_file_location("_ff", path)
            mod = module_from_spec(spec)
            spec.loader.exec_module(mod)
            return getattr(mod, "generate_ff_signals")
    raise FileNotFoundError(
        f"未找到策略文件（已在 strategies/ 查找）：sm_05_fama_french_value_size.py / SM-策略-05-fama french value size.py，目录={root}"
    )


generate_ff_signals = _load_strategy()


def load_from_duckdb(
    db_path: str,
    start_date: str = "20150101",
    end_date: str | None = None,
) -> pl.DataFrame:
    """
    从 shiming_daily_base.duckdb 读取：
    - daily_data：open/high/low/close（以及可选 vol）
    - daily_basic：total_mv, pb
    - adj_factor：复权因子
    并计算前复权 OHLC。

    返回列（至少）：
    ts_code, trade_date, open, close, total_mv, pb
    """
    end = end_date or datetime.now().strftime("%Y%m%d")
    con = duckdb.connect(db_path, read_only=True)
    try:
        # 先尝试带 vol（有些表字段叫 vol）
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

        b = con.execute(
            "SELECT ts_code, trade_date, total_mv, pb FROM daily_basic WHERE trade_date >= ? AND trade_date <= ?",
            [start_date, end],
        ).pl()
        a = con.execute(
            "SELECT ts_code, trade_date, adj_factor FROM adj_factor WHERE trade_date >= ? AND trade_date <= ?",
            [start_date, end],
        ).pl()
    finally:
        con.close()

    j = d.join(b, on=["ts_code", "trade_date"], how="inner").join(a, on=["ts_code", "trade_date"], how="inner")
    j = j.sort(["ts_code", "trade_date"])

    # 前复权（以该股最新交易日 adj_factor 为分母）
    last_af = pl.col("adj_factor").last().over("ts_code")
    ratio = pl.col("adj_factor") / last_af
    out = j.with_columns(
        (pl.col("open") * ratio).alias("adj_open"),
        (pl.col("high") * ratio).alias("adj_high"),
        (pl.col("low") * ratio).alias("adj_low"),
        (pl.col("close") * ratio).alias("adj_close"),
    )

    # 回测统一用 open/close（复权）
    out = out.with_columns(
        pl.col("adj_open").alias("open"),
        pl.col("adj_close").alias("close"),
    )

    keep = ["ts_code", "trade_date", "open", "close", "total_mv", "pb"]
    if "volume" in out.columns:
        keep.append("volume")
    return out.select(keep)


@dataclass
class BacktestConfig:
    start_date: str = "20150101"
    end_date: str | None = None

    initial_capital: float = 10_000_000
    max_positions: int = 100

    buy_cost: float = 0.0003
    sell_cost: float = 0.0013

    t_plus_one: bool = True

    # 策略参数（透传）
    top_n_entry: int = 100
    top_n_exit: int = 150


def _calc_stats(daily: pl.DataFrame) -> dict[str, float]:
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


def run_backtest(df: pl.DataFrame, config: BacktestConfig | None = None) -> tuple[pl.DataFrame, dict]:
    """
    执行 FF 小盘价值回测（动态 100 槽位复利池）。
    """
    cfg = config or BacktestConfig()
    end = cfg.end_date or datetime.now().strftime("%Y%m%d")

    df = df.filter((pl.col("trade_date") >= cfg.start_date) & (pl.col("trade_date") <= end))
    df = df.sort(["ts_code", "trade_date"])

    # 1) 生成策略信号（截面策略，策略内部会处理 pb/volume 等过滤）
    df = generate_ff_signals(df, TOP_N_ENTRY=cfg.top_n_entry, TOP_N_EXIT=cfg.top_n_exit)

    # 2) T+1 执行：T 日信号与排序字段 shift 到 T+1 执行日
    if cfg.t_plus_one:
        df = df.with_columns(
            pl.col("entry_signal").shift(1).over("ts_code").fill_null(0).alias("actual_entry"),
            pl.col("exit_signal").shift(1).over("ts_code").fill_null(0).alias("actual_exit"),
            pl.col("final_rank").shift(1).over("ts_code").alias("actual_rank"),
        )
    else:
        df = df.with_columns(
            pl.col("entry_signal").fill_null(0).alias("actual_entry"),
            pl.col("exit_signal").fill_null(0).alias("actual_exit"),
            pl.col("final_rank").alias("actual_rank"),
        )

    # 3) 动态槽位账户：cash + holdings（ts_code->shares）
    cash = float(cfg.initial_capital)
    holdings: dict[str, int] = {}
    last_close: dict[str, float] = {}

    daily_records: list[dict] = []
    daily_turnovers: list[float] = []

    needed = ["ts_code", "open", "close", "actual_entry", "actual_exit", "actual_rank"]

    for (trade_date,), day in df.group_by("trade_date", maintain_order=True):
        rows = list(day.select(needed).iter_rows(named=True))

        # 开盘前总资产（按昨收估值）
        equity_start = cash
        for code, sh in holdings.items():
            if sh <= 0:
                continue
            equity_start += sh * last_close.get(code, 0.0)

        traded_gross = 0.0  # 当日成交额（买+卖，毛额，用于换手率）

        # --------------------------
        # A) 优胜劣汰：昨日 exit -> 今日 open 全部卖出
        # --------------------------
        for r in rows:
            if int(r["actual_exit"]) != 1:
                continue
            code = r["ts_code"]
            sh = holdings.get(code, 0)
            if sh <= 0:
                continue
            sell_amt = sh * float(r["open"])
            traded_gross += sell_amt
            cash += sell_amt * (1.0 - float(cfg.sell_cost))
            holdings.pop(code, None)

        # --------------------------
        # B) 新鲜血液：昨日 entry -> 今日 open 买入（按昨日 final_rank 升序）
        #     单票目标金额 = equity_start / max_positions（复利滚动）
        # --------------------------
        slots_left = max(0, int(cfg.max_positions) - sum(1 for _c, sh in holdings.items() if sh > 0))
        target_alloc = equity_start / float(cfg.max_positions) if cfg.max_positions > 0 else 0.0

        if slots_left > 0 and cash > target_alloc and target_alloc > 0:
            candidates = []
            for r in rows:
                if int(r["actual_entry"]) != 1:
                    continue
                code = r["ts_code"]
                if holdings.get(code, 0) > 0:
                    continue
                candidates.append(r)

            # 并发抢筹：按昨日 final_rank 从小到大（越靠前越优先）
            candidates.sort(key=lambda x: (x["actual_rank"] if x["actual_rank"] is not None else float("inf")))
            if len(candidates) > slots_left:
                candidates = candidates[:slots_left]

            for r in candidates:
                if slots_left <= 0:
                    break
                if cash <= target_alloc:
                    break
                price = float(r["open"])
                if price <= 0:
                    continue
                shares = math.floor(target_alloc / (price * LOT_SIZE)) * LOT_SIZE
                if shares <= 0:
                    continue
                buy_amt = shares * price
                need_cash = buy_amt * (1.0 + float(cfg.buy_cost))
                if need_cash > cash:
                    continue
                traded_gross += buy_amt
                cash -= need_cash
                holdings[r["ts_code"]] = int(shares)
                slots_left -= 1

        # 更新收盘价缓存
        for r in rows:
            last_close[r["ts_code"]] = float(r["close"])

        # 收盘市值与总资产
        market_value = 0.0
        for code, sh in holdings.items():
            if sh <= 0:
                continue
            market_value += sh * last_close.get(code, 0.0)
        total_equity = cash + market_value

        daily_pnl = total_equity - equity_start
        daily_ret = (daily_pnl / equity_start) if equity_start > 1e-8 else 0.0
        daily_ret = max(-0.99, min(10.0, float(daily_ret)))

        daily_turn = (traded_gross / equity_start) if equity_start > 1e-8 else 0.0
        daily_turnovers.append(daily_turn)

        daily_records.append(
            {
                "trade_date": trade_date,
                "cash": cash,
                "holdings_count": sum(1 for _c, sh in holdings.items() if sh > 0),
                "market_value": market_value,
                "total_equity": total_equity,
                "daily_pnl": daily_pnl,
                "strategy_return": daily_ret,
                "daily_turnover": daily_turn,
            }
        )

    daily = pl.DataFrame(daily_records).sort("trade_date")
    daily = daily.with_columns(
        (pl.col("total_equity") / pl.col("total_equity").first()).alias("cum_nav")
    )

    avg_holdings = float(daily["holdings_count"].mean()) if not daily.is_empty() else 0.0
    ann_turnover = float(daily["daily_turnover"].mean()) * 252.0 if not daily.is_empty() else 0.0

    meta = {
        "avg_holdings": avg_holdings,
        "annual_turnover": ann_turnover,
    }
    return daily, meta


def plot_nav(daily: pl.DataFrame, path: str = "ff_nav.png") -> None:
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
    plt.title("Fama-French Value+Size NAV (Dynamic 100 Slots)")
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
        max_positions=100,
        buy_cost=0.0003,
        sell_cost=0.0013,
        t_plus_one=True,
        top_n_entry=100,
        top_n_exit=150,
    )

    db_path = pp.resolve_db_path(os.getenv("DUCKDB_PATH", "shiming_daily_base.duckdb"))
    if not os.path.isfile(db_path):
        raise FileNotFoundError(f"未找到数据库：{db_path}")

    df = load_from_duckdb(db_path, start_date=cfg.start_date, end_date=cfg.end_date)
    if df.is_empty():
        raise ValueError("DuckDB 查询结果为空（或 daily_basic 缺失 total_mv/pb 字段）")
    print(f"已从 {db_path} 加载 {len(df):,} 条日线，{df['ts_code'].n_unique()} 只标的。")

    daily, meta = run_backtest(df, cfg)
    stats = _calc_stats(daily)

    print("=== FF 小盘价值回测结果（动态 100 槽位）===")
    print(f"区间:         {daily['trade_date'][0]} ~ {daily['trade_date'][-1]}")
    print(f"总收益率:     {stats['total_return']:.4f}")
    print(f"年化收益率:   {stats['ann_return']:.4f}")
    print(f"最大回撤:     {stats['max_dd']:.4f}")
    print(f"夏普比率:     {stats['sharpe']:.4f}")
    print("--- 因子监测 ---")
    print(f"平均每日持仓: {meta['avg_holdings']:.2f}")
    print(f"年化换手率:   {meta['annual_turnover']:.4f}")

    plot_nav(daily, path=pp.docs_plot_path("ff_nav", daily, cfg.start_date, cfg.end_date))

