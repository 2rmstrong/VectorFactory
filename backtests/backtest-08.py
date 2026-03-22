"""
风险平价策略 · 回测框架（独立入口）
导入 SM-策略-08-risk parity.py 的 generate_risk_parity_weights，按目标权重月末调仓、T+1 开盘成交。
初始资金 1000 万，从 DuckDB 读取三只 ETF（510300/518880/511260）前复权日线。
默认回测区间：2015 年至今（start_date=20150101, end_date=None）。
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

# ----------------- 策略加载 -----------------
def _load_strategy():
    root = pp.strategies_dir()
    for name in (
        "SM-策略-08-risk parity.py",
        "sm_08_risk_parity.py",
    ):
        path = os.path.join(root, name)
        if os.path.isfile(path):
            from importlib.util import spec_from_file_location, module_from_spec
            spec = spec_from_file_location("_risk_parity", path)
            mod = module_from_spec(spec)
            spec.loader.exec_module(mod)
            return getattr(mod, "generate_risk_parity_weights"), getattr(mod, "TARGET_ASSETS", ["510300.SH", "518880.SH", "511260.SH"])
    raise FileNotFoundError("未找到策略文件：SM-策略-08-risk parity.py 或 sm_08_risk_parity.py")

generate_risk_parity_weights, TARGET_ASSETS = _load_strategy()

# ETF 最小交易单位（手）
LOT_SIZE = 100


def load_from_duckdb(
    db_path: str,
    start_date: str,
    end_date: str | None,
    assets: list[str],
) -> pl.DataFrame:
    """
    从 DuckDB 读取指定 ETF 的 daily_data + adj_factor，计算前复权 open/close。
    返回列：ts_code, trade_date, open, close。
    """
    end = end_date or datetime.now().strftime("%Y%m%d")
    placeholders = ",".join("?" * len(assets))
    con = duckdb.connect(db_path, read_only=True)
    try:
        d = con.execute(
            f"SELECT ts_code, trade_date, open, close FROM daily_data WHERE trade_date >= ? AND trade_date <= ? AND ts_code IN ({placeholders})",
            [start_date, end] + list(assets),
        ).pl()
        a = con.execute(
            f"SELECT ts_code, trade_date, adj_factor FROM adj_factor WHERE trade_date >= ? AND trade_date <= ? AND ts_code IN ({placeholders})",
            [start_date, end] + list(assets),
        ).pl()
    finally:
        con.close()

    j = d.join(a, on=["ts_code", "trade_date"], how="inner").sort(["ts_code", "trade_date"])
    last_af = pl.col("adj_factor").last().over("ts_code")
    ratio = pl.col("adj_factor") / last_af
    return j.with_columns(
        (pl.col("open") * ratio).alias("open"),
        (pl.col("close") * ratio).alias("close"),
    ).select(["ts_code", "trade_date", "open", "close"])


@dataclass
class BacktestConfig:
    start_date: str = "20150101"   # 回测起点，默认 2015 年
    end_date: str | None = None    # 终点，None 表示至今
    initial_capital: float = 10_000_000
    cost_roundtrip: float = 0.0003   # 单边万三
    t_plus_one: bool = True
    # 策略参数透传
    vol_window: int = 60


def run_backtest(df: pl.DataFrame, config: BacktestConfig | None = None) -> tuple[pl.DataFrame, dict]:
    """
    风险平价回测：按 target_weight 每月首个交易日以开盘价调仓，其余日持有。
    df 须含 ts_code, trade_date, open, close；策略输出增加 vol, target_weight。
    """
    cfg = config or BacktestConfig()
    end = cfg.end_date or datetime.now().strftime("%Y%m%d")
    df = df.filter(
        (pl.col("trade_date") >= cfg.start_date) & (pl.col("trade_date") <= end)
    ).sort(["ts_code", "trade_date"])

    # 1) 策略权重（策略输出无 open，需与原始数据合并保留 open 用于成交）
    df_weights = generate_risk_parity_weights(df, VOL_WINDOW=cfg.vol_window)
    df = df.join(
        df_weights.select(["ts_code", "trade_date", "vol", "target_weight"]),
        on=["ts_code", "trade_date"],
        how="left",
    )
    # 调仓日当日的 target_weight 已是基于昨日波动率算出，直接用于当日开盘调仓
    df = df.with_columns(pl.col("target_weight").alias("exec_weight"))

    # 每月首个交易日为调仓日（与策略一致）
    unique_dates = df.select("trade_date").unique().sort("trade_date")
    ym = unique_dates.with_columns(pl.col("trade_date").str.slice(0, 6).alias("ym"))
    rebal_dates = set(
        ym.group_by("ym").agg(pl.col("trade_date").min()).select("trade_date").to_series().to_list()
    )

    # 3) 按日迭代
    cash = float(cfg.initial_capital)
    hold_shares: dict[str, int] = {}
    last_close: dict[str, float] = {}
    daily_records: list[dict] = []

    dates_sorted = df.select("trade_date").unique().sort("trade_date").to_series().to_list()
    for i, trade_date in enumerate(dates_sorted):
        day = df.filter(pl.col("trade_date") == trade_date)
        rows = list(day.select(["ts_code", "open", "close", "exec_weight"]).iter_rows(named=True))
        open_px = {r["ts_code"]: float(r["open"]) for r in rows if r["open"] is not None}
        close_px = {r["ts_code"]: float(r["close"]) for r in rows if r["close"] is not None}
        weight = {r["ts_code"]: (float(r["exec_weight"]) if r["exec_weight"] is not None else None) for r in rows}

        # 开盘前总资产（昨收估值）
        equity_start = cash
        for code, n in hold_shares.items():
            if n > 0:
                equity_start += n * last_close.get(code, 0.0)

        if trade_date in rebal_dates and i > 0:
            # 调仓日：按 exec_weight 以开盘价再平衡
            valid = [(c, w) for c, w in weight.items() if w is not None and w > 0 and c in open_px and open_px[c] > 0]
            if valid and sum(w for _, w in valid) > 0.999:
                # 先全部卖出
                for code in list(hold_shares.keys()):
                    n = hold_shares[code]
                    if n <= 0:
                        continue
                    p = open_px.get(code)
                    if p and p > 0:
                        cash += n * p * (1.0 - cfg.cost_roundtrip)
                    hold_shares[code] = 0
                hold_shares = {c: 0 for c, _ in valid}
                # 按权重分配
                total = cash  # 卖出后全部是现金
                for code, w in valid:
                    p = open_px[code]
                    if p <= 0:
                        continue
                    target_val = total * w
                    target_shares = int(target_val / p / LOT_SIZE) * LOT_SIZE
                    if target_shares < LOT_SIZE:
                        continue
                    need = target_shares * p * (1.0 + cfg.cost_roundtrip)
                    if need > cash:
                        target_shares = int(cash / (p * (1.0 + cfg.cost_roundtrip)) / LOT_SIZE) * LOT_SIZE
                        if target_shares < LOT_SIZE:
                            continue
                        need = target_shares * p * (1.0 + cfg.cost_roundtrip)
                    cash -= need
                    hold_shares[code] = target_shares
        elif i == 0 and len(hold_shares) == 0:
            # 首日：若为调仓日且有权重则建仓，否则等首次调仓日
            valid = [(c, w) for c, w in weight.items() if w is not None and w > 0 and c in open_px and open_px[c] > 0]
            if valid and sum(w for _, w in valid) > 0.999 and trade_date in rebal_dates:
                total = cash
                hold_shares = {c: 0 for c, _ in valid}
                for code, w in valid:
                    p = open_px[code]
                    if p <= 0:
                        continue
                    target_val = total * w
                    target_shares = int(target_val / p / LOT_SIZE) * LOT_SIZE
                    if target_shares < LOT_SIZE:
                        continue
                    need = target_shares * p * (1.0 + cfg.cost_roundtrip)
                    if need > cash:
                        target_shares = int(cash / (p * (1.0 + cfg.cost_roundtrip)) / LOT_SIZE) * LOT_SIZE
                        if target_shares < LOT_SIZE:
                            continue
                        need = target_shares * p * (1.0 + cfg.cost_roundtrip)
                    cash -= need
                    hold_shares[code] = target_shares

        for code, c in close_px.items():
            last_close[code] = c

        equity_end = cash
        for code, n in hold_shares.items():
            if n > 0:
                equity_end += n * close_px.get(code, last_close.get(code, 0.0))

        daily_ret = (equity_end - equity_start) / equity_start if equity_start > 1e-8 else 0.0
        daily_ret = max(-0.99, min(10.0, daily_ret))
        daily_records.append({
            "trade_date": trade_date,
            "equity_start": equity_start,
            "equity_end": equity_end,
            "strategy_return": daily_ret,
        })

    daily = pl.DataFrame(daily_records).sort("trade_date")
    daily = daily.with_columns(
        pl.col("strategy_return").fill_nan(0.0).clip(-0.99, 10.0)
    ).with_columns(
        (1.0 + pl.col("strategy_return")).cum_prod().alias("cum_nav")
    )
    meta = {"assets": list(hold_shares.keys())}
    return daily, meta


def print_stats(daily: pl.DataFrame, initial_capital: float = 10_000_000) -> None:
    if daily.is_empty():
        print("无交易日数据。")
        return
    nav = daily["cum_nav"]
    nav = nav.fill_nan(1.0).clip(1e-8, 1e10)
    total_return = float(nav[-1]) - 1.0
    peak = nav.cum_max()
    max_dd = float(((nav / peak) - 1.0).min())
    win_rate = float((daily["strategy_return"] > 0).mean())
    n_days = len(daily)
    years = n_days / 252.0 if n_days else 0.0
    ann_return = (float(nav[-1])) ** (1.0 / years) - 1.0 if years > 0 else total_return
    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0.0
    ret = daily["strategy_return"].fill_nan(0.0)
    sigma = float(ret.std()) if ret.len() > 1 else 0.0
    sharpe = (float(ret.mean()) / sigma * math.sqrt(252.0)) if sigma > 1e-12 else 0.0

    print("=== 风险平价回测结果（月末调仓，1000万）===")
    print(f"总收益率:     {total_return:.4f}")
    print(f"年化收益:     {ann_return:.4f}")
    print(f"最大回撤:     {max_dd:.4f}")
    print(f"Calmar 比率:  {calmar:.4f}")
    print(f"夏普比率:     {sharpe:.4f}")
    print(f"胜率(日):     {win_rate:.4f}")


def plot_nav(daily: pl.DataFrame, path: str = "risk_parity_nav.png") -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("未安装 matplotlib，跳过绘图。")
        return
    if daily.is_empty():
        return
    d = daily.to_pandas()
    d["trade_date"] = d["trade_date"].astype(str)
    if d["trade_date"].str.len().max() == 8:
        d["trade_date"] = d["trade_date"].str[:4] + "-" + d["trade_date"].str[4:6] + "-" + d["trade_date"].str[6:8]
    d["trade_date"] = d["trade_date"].astype("datetime64[ns]")
    plt.figure(figsize=(12, 5))
    plt.plot(d["trade_date"], d["cum_nav"], lw=1.5)
    plt.title("Risk Parity (510300/518880/511260) NAV")
    plt.xlabel("Date")
    plt.ylabel("NAV")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"已保存: {path}")


if __name__ == "__main__":
    config = BacktestConfig(
        start_date="20150101",
        end_date=None,
        initial_capital=10_000_000,
        cost_roundtrip=0.0003,
        t_plus_one=True,
        vol_window=60,
    )

    db_path = pp.resolve_db_path(os.getenv("DUCKDB_PATH", "shiming_daily_base.duckdb"))

    if os.path.isfile(db_path):
        try:
            df = load_from_duckdb(db_path, config.start_date, config.end_date, TARGET_ASSETS)
            if df.is_empty():
                raise ValueError("DuckDB 查询结果为空（可能不包含三只 ETF）")
            print(f"已从 {db_path} 加载 {len(df):,} 条日线，标的: {TARGET_ASSETS}")
        except Exception as e:
            print(f"从 DuckDB 加载失败（{e}），改用合成数据。")
            df = None
    else:
        print(f"未找到数据库 {db_path}，改用合成数据。")
        df = None

    if df is None:
        # 合成三只 ETF 日线：2015 年至今，按真实日历只取工作日（模拟交易日）
        import random
        from datetime import timedelta
        random.seed(42)
        start = datetime(2015, 1, 1)
        end = datetime(2026, 12, 31)
        dates = []
        d = start
        while d <= end:
            if d.weekday() < 5:  # 0=Mon .. 4=Fri
                dates.append(d.strftime("%Y%m%d"))
            d += timedelta(days=1)
        rows = []
        for i, dt in enumerate(dates):
            for j, code in enumerate(TARGET_ASSETS):
                sig = [0.012, 0.010, 0.005][j]
                r = random.gauss(0.0003, sig)
                if i == 0:
                    close = 4.0 + j * 2.0
                else:
                    prev = [x for x in rows if x["ts_code"] == code and x["trade_date"] == dates[i - 1]]
                    close = prev[0]["close"] * (1 + r) if prev else 4.0 + j * 2.0
                rows.append({"ts_code": code, "trade_date": dt, "open": close * 0.998, "close": close})
        df = pl.DataFrame(rows)

    daily, meta = run_backtest(df, config)
    print_stats(daily, initial_capital=config.initial_capital)
    plot_nav(daily, path=pp.docs_plot_path("risk_parity_nav", daily, config.start_date, config.end_date))
